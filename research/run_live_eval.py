#!/usr/bin/env python3
"""Seeded ALFWorld evaluation against an already-running native SGLang server.

Exact sampled token IDs are carried forward; assistant text is parsed solely for
action execution and measurements, never retokenized as the next prompt.
No retries, no generation service management, and no awareness insertion.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dressage.recipes.alfworld.cue_study import VERSION, inject_cue
from dressage.recipes.alfworld.tools import (
    ALFWORLD_SYSTEM_PROMPT, ALFWORLD_INITIAL_USER_TEMPLATE, ALFWORLD_TOOL_SCHEMAS,
    extract_task_text, first_command, format_admissible, format_invalid_response,
    format_tool_response, make_env,
)
from dressage.proxy.tito.tito_tokenizer import Qwen35TITOTokenizer
from dressage.proxy.reasoning_parser import parse_qwen3_reasoning
from dressage.proxy.tool_call_parser import parse_qwen3_5_tool_calls, parse_hermes_tool_calls
from dressage.training.structured_sft import get_generator
from research.score_evaluations import score
from dressage.recipes.alfworld import tools as alf_tools


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            sha.update(block)
    return sha.hexdigest()


def checkpoint_identity(model_path):
    path = Path(model_path).resolve(strict=True)
    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        raise ValueError("Expected local HF safetensors files for checkpoint identity")
    files = weights + [path / "config.json", path / "tokenizer.json", path / "tokenizer_config.json"]
    if (path / "generation_config.json").exists():
        files.append(path / "generation_config.json")
    return {"path": str(path), "files": {file.name: {"sha256": digest(file), "bytes": file.stat().st_size}
                                        for file in files}}


def turn_seed(episode_seed, turn):
    # Keep the manifest's unsigned 32-bit seed for turn zero; subsequent turns
    # share a deterministic seed schedule across the cue/no-cue pair.
    return (int(episode_seed) + int(turn)) % (2**32)


def validate_runtime_limits(limits, context):
    if limits.get('schema') != 'cue-study-runtime-limits-v1':
        raise ValueError('Missing supported runtime-limit schema; refusing guessed limits')
    for key in ('context_length', 'max_input_tokens_exclusive', 'reserved_tokens'):
        if type(limits.get(key)) is not int:
            raise ValueError(f'Invalid runtime limit: {key}')
    if (limits['context_length'] != context or
            not 1 < limits['max_input_tokens_exclusive'] < context or
            not 0 <= limits['reserved_tokens'] < context - 1 or
            limits.get('allow_auto_truncate') is not False):
        raise ValueError('Incompatible runtime limits or automatic truncation enabled')
    return limits


def remaining_context(context, input_length, limits):
    validate_runtime_limits(limits, context)
    # The scheduler rejects INPUT >= its own threshold, independently of the
    # configured total context. Conservatively keep input+output within that
    # threshold too, avoiding scheduler-side silent output-budget clipping.
    if input_length >= limits['max_input_tokens_exclusive']:
        return 0
    total = min(context - 1 - limits['reserved_tokens'],
                limits['max_input_tokens_exclusive'])
    return max(0, total - input_length)


def sampled_ids(response):
    output = response.get("output_ids")
    if output is None:
        entries = response.get("meta_info", {}).get("output_token_logprobs")
        if not isinstance(entries, list):
            raise ValueError("Server did not return original sampled token IDs")
        output = [entry[1] for entry in entries]
    if not isinstance(output, list) or not output or any(type(token) is not int for token in output):
        raise ValueError("Missing/malformed sampled IDs; refusing decode/re-encode fallback")
    count = response.get("meta_info", {}).get("completion_tokens")
    if count is not None and count != len(output):
        raise ValueError("Sampled ID count differs from server completion_tokens")
    return output


def finish_type(response):
    if response.get("error"):
        raise RuntimeError(f"Generation server error: {response['error']}")
    finish = response.get("meta_info", {}).get("finish_reason")
    kind = finish.get("type") if isinstance(finish, dict) else finish
    if kind not in ("stop", "length"):
        raise RuntimeError(f"Unsuccessful/unknown generation finish: {finish!r}")
    return kind


def validate_render_environment():
    if os.environ.get("ALFWORLD_NAV_SPLIT", "0") != "0" or alf_tools._NAV_SPLIT:
        raise ValueError("Evaluation requires ALFWORLD_NAV_SPLIT=0")
    if os.environ.get("ALFWORLD_DEMANGLE", "1") != "1":
        raise ValueError("Evaluation requires ALFWORLD_DEMANGLE=1")
    return {"ALFWORLD_NAV_SPLIT": "0", "ALFWORLD_DEMANGLE": "1"}


def assistant_message(raw, turn):
    parsed = parse_qwen3_reasoning(raw)
    if "</think>" not in raw:
        # Pinned generation prefix opens <think>. An unfinished reasoning span is
        # still reasoning and must not be counted as missing awareness evidence.
        return {"role": "assistant", "reasoning_content": raw, "content": ""}, []
    content, calls = parse_qwen3_5_tool_calls(parsed.text)
    if not calls:
        content, calls = parse_hermes_tool_calls(parsed.text)
    for index, call in enumerate(calls or []):
        call["id"] = f"eval_turn_{turn}_call_{index}"
    return {"role": "assistant", "reasoning_content": parsed.reasoning_content or "",
            "content": content or "", **({"tool_calls": calls} if calls else {})}, calls or []


class NativeServer:
    def __init__(self, url, timeout=600):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def request(self, path, payload=None):
        request = urllib.request.Request(self.url + path,
                    data=None if payload is None else json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read(8192).decode(errors='replace')
            raise RuntimeError(f'Server HTTP {error.code} at {path}: {detail}') from error


def boundary_preflight(server, context, limits, token_id):
    """Real requests beyond the old failure and at the new safe input boundary.

    Synthetic infrastructure checks, not evaluation episodes. No retries or
    auto-truncation. Flush their cache before seeded task generation begins.
    """
    validate_runtime_limits(limits, context)
    threshold = limits['max_input_tokens_exclusive']
    if threshold < context - 6:
        raise RuntimeError('Server memory pool cannot support requested full native context')
    report = {'runtime_limits': limits, 'requests': [], 'passed': False}
    for length in sorted({18429, threshold - 1}):
        if remaining_context(context, length, limits) < 1:
            raise RuntimeError('Boundary preflight length is not accepted by client guard')
        started = time.monotonic()
        response = server.request('/generate', {
            'input_ids': [token_id] * length,
            'sampling_params': {'max_new_tokens': 1, 'temperature': 0.,
                                'sampling_seed': 9001, 'skip_special_tokens': False},
            'return_logprob': True, 'logprob_start_len': length - 1,
            'rid': f'context-preflight-{length}',
        })
        finish_type(response)
        output = sampled_ids(response)
        if len(output) != 1 or response.get('meta_info', {}).get('prompt_tokens') != length:
            raise RuntimeError('Boundary request token accounting mismatch or truncation')
        report['requests'].append({'input_tokens': length, 'output_tokens': len(output),
                                   'seconds': time.monotonic() - started})
    # Rejected lengths must terminate locally, without sending an HTTP request.
    report['guard_rejected_lengths'] = [threshold, threshold + 1, context]
    if any(remaining_context(context, n, limits) != 0 for n in report['guard_rejected_lengths']):
        raise RuntimeError('Client did not reject server input boundary')
    with urllib.request.urlopen(server.url + '/flush_cache', timeout=60) as response:
        if response.status != 200:
            raise RuntimeError('Synthetic boundary cache was not flushed')
        report['cache_flush'] = response.read().decode()
    report['passed'] = True
    return report


def evaluate_episode(row, args, tokenizer, server):
    metadata = row["metadata"]
    game = args.game_root / metadata["game_file"]
    env = make_env(str(game.resolve(strict=True)), args.max_steps)
    messages, turns = [], []
    success, invalid, executed, truncated = False, 0, 0, False
    stop_reason = "llm_turn_limit"
    context_ids = None
    tito = Qwen35TITOTokenizer(tokenizer)
    try:
        obs, infos = env.reset()
        admissible = list(infos.get("admissible_commands") or [])
        initial = ALFWORLD_INITIAL_USER_TEMPLATE.format(
            task_text=extract_task_text(obs, row["prompt"]), observation=(obs or "").strip()[:2000],
            admissible_commands=format_admissible(admissible))
        messages = [{"role": "system", "content": ALFWORLD_SYSTEM_PROMPT},
                    {"role": "user", "content": inject_cue(initial, metadata["cue_condition"])}]
        context_ids = tokenizer.apply_chat_template(messages, tools=ALFWORLD_TOOL_SCHEMAS,
                        add_generation_prompt=True, preserve_thinking=True, tokenize=True, return_dict=False)
        for turn in range(args.max_steps):
            remaining = remaining_context(args.context, len(context_ids), args.runtime_limits)
            if remaining <= 0:
                truncated, stop_reason = True, "context_exhausted"
                break
            budget = min(args.max_new_tokens, remaining)
            seed = turn_seed(metadata["requested_generation_seed"], turn)
            payload = {"input_ids": context_ids, "sampling_params": {
                "max_new_tokens": budget, "temperature": args.temperature, "top_p": 1.0,
                "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0,
                "frequency_penalty": 0.0, "presence_penalty": 0.0,
                "sampling_seed": seed, "skip_special_tokens": False, "no_stop_trim": True},
                "return_logprob": True, "return_text_in_logprobs": False,
                "logprob_start_len": len(context_ids) - 1, "stream": False,
                "rid": f"{metadata['instance_id']}-turn-{turn}"}
            response = server.request("/generate", payload)
            finish_kind = finish_type(response)
            if finish_kind == "length":
                truncated = True
            output = sampled_ids(response)
            if len(output) > budget:
                raise ValueError("Server exceeded the requested generation token budget")
            raw = tokenizer.decode(output, skip_special_tokens=False)
            # Delimiter trimming is ONLY for the parser; context keeps original IDs.
            parsed_raw = raw.removesuffix("<|im_end|>").removesuffix("<|endoftext|>")
            assistant, calls = assistant_message(parsed_raw, turn)
            messages.append(assistant)
            prefix_ids = list(context_ids) + output
            finish = response.get("meta_info", {}).get("finish_reason")
            turns.append({"turn": turn, "sampling_seed": seed, "sampling_params": payload["sampling_params"], "input_ids": list(context_ids),
                          "output_ids": output, "finish_reason": finish, "raw_text": raw})
            call_id, command, error = first_command(assistant["content"], calls)
            tool_id = call_id or f"eval_turn_{turn}_invalid"
            if command is None or command not in admissible:
                invalid += 1
                feedback = format_invalid_response(obs, admissible,
                    error or f"command {command!r} is not in admissible commands")
            else:
                obs, _, done, infos = env.step(command)
                executed += 1
                admissible = list(infos.get("admissible_commands") or [])
                feedback = format_tool_response(obs, admissible)
                success = bool(infos.get("won", False))
                if success or done:
                    messages.append({"role": "tool", "tool_call_id": tool_id, "content": feedback})
                    step_limit = bool(infos.get("TimeLimit.truncated", False)) or executed >= args.max_steps
                    stop_reason = "success" if success else ("environment_step_limit" if step_limit else "environment_done")
                    break
            before_feedback = copy.deepcopy(messages)
            messages.append({"role": "tool", "tool_call_id": tool_id, "content": feedback})
            if finish_kind == "length":
                truncated, stop_reason = True, "generation_length"
                break
            context_ids = tito.merge_tokens(old_messages=before_feedback, new_messages=messages,
                            pretokenized_token_ids=prefix_ids, tools=ALFWORLD_TOOL_SCHEMAS)
            if context_ids[:len(prefix_ids)] != prefix_ids:
                raise AssertionError("TITO merge changed generated token prefix")
    finally:
        env.close()
    record = {**metadata, "session_id": f"eval-{metadata['instance_id']}",
              "segment_index": 0, "segment_count": 1, "task_success": success,
              "invalid_action_count": invalid, "num_steps": executed,
              "truncated": truncated, "stop_reason": stop_reason, "messages": messages,
              "num_llm_turns": len(turns), "checkpoint_label": args.checkpoint_label,
              "generation_seed_policy": "sampling_seed=(requested_generation_seed+turn)%2**32"}
    return record, turns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--game-root", type=Path, default=Path("/workspace/tito-dressage"))
    parser.add_argument("--server", default="http://127.0.0.1:30000")
    parser.add_argument("--expected-model-path", required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--allow-heldout-test", action="store_true")
    parser.add_argument("--dedicated-server", action="store_true",
                        help="Confirm this server is exclusive to this evaluation, with no hot weight updates")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new; retries must not silently replace previous episodes")
    if not args.dedicated_server:
        parser.error("A dedicated static-checkpoint server is required; acknowledge with --dedicated-server")
    render_environment = validate_render_environment()
    if digest(args.manifest) != args.manifest_sha256:
        parser.error("Manifest SHA mismatch")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not rows or len({row["metadata"]["instance_id"] for row in rows}) != len(rows):
        parser.error("Empty/duplicate evaluation manifest")
    for row in rows:
        meta = row["metadata"]
        if meta["study_version"] != VERSION:
            parser.error("Wrong study version")
        if meta["study_split"].startswith("test") and not args.allow_heldout_test:
            parser.error("Held-out test evaluation requires explicit --allow-heldout-test")
        if "requested_generation_seed" not in meta:
            parser.error("Missing frozen generation seed")
    server = NativeServer(args.server)
    info = server.request("/get_model_info")
    if str(info.get("model_path")) != args.expected_model_path:
        parser.error(f"Wrong served model: {info.get('model_path')}")
    identity = checkpoint_identity(args.expected_model_path)
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    server_info = server.request("/get_server_info")
    required_server = {"enable_deterministic_inference": True, "sampling_backend": "pytorch",
                       "context_length": args.context, "max_running_requests": 1,
                       "speculative_algorithm": None}
    for key, value in required_server.items():
        if server_info.get(key) != value:
            parser.error(f"Server {key}={server_info.get(key)!r}; expected {value!r}")
    args.runtime_limits = validate_runtime_limits(server.request('/cue-study/limits'), args.context)
    generator = get_generator(args.tokenizer)
    tokenizer = generator.tokenizer
    tokenizer.chat_template = generator.template
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = Path(args.expected_model_path) / "generation_config.json"
    (args.output / "config.json").write_text(json.dumps({**vars(args), "model_info": info, "server_info": server_info,
        "checkpoint_identity": identity, "checkpoint_identity_sha256": identity_sha,
        "render_environment": render_environment,
        "pinned_template_sha256": hashlib.sha256(generator.template.encode()).hexdigest(),
        "generation_config": json.loads(config_path.read_text()) if config_path.exists() else None,
        "manifest_sha256": digest(args.manifest)}, indent=2, default=str) + "\n")
    print('Running native-context boundary preflight before task episodes', flush=True)
    boundary = boundary_preflight(server, args.context, args.runtime_limits,
                                  tokenizer.encode('x', add_special_tokens=False)[0])
    (args.output / 'boundary_preflight.json').write_text(json.dumps(boundary, indent=2) + '\n')
    print('Native-context boundary preflight passed; starting frozen development episodes', flush=True)
    records = []
    with (args.output / "records.jsonl").open("x") as out, (args.output / "token_traces.jsonl").open("x") as trace:
        for index, row in enumerate(rows):
            started = time.monotonic()
            if server.request("/get_model_info") != info:
                raise RuntimeError("Server checkpoint identity changed during evaluation")
            if server.request('/cue-study/limits') != args.runtime_limits:
                raise RuntimeError('Server runtime limits changed during evaluation')
            # Transport/infrastructure exceptions stop the run. Never label a
            # server failure as model failure or selectively retry bad outcomes.
            record, turns = evaluate_episode(row, args, tokenizer, server)
            if server.request("/get_model_info") != info:
                raise RuntimeError("Server checkpoint identity changed during episode")
            record["checkpoint_identity_sha256"] = identity_sha
            out.write(json.dumps(record) + "\n")
            out.flush()
            trace.write(json.dumps({"instance_id": record["instance_id"], "turns": turns}) + "\n")
            trace.flush()
            records.append(record)
            print(f"{index + 1}/{len(rows)} success={record['task_success']} steps={record['num_steps']} "
                  f"stop={record['stop_reason']} seconds={time.monotonic()-started:.1f}", flush=True)
    report = score(rows, records)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
