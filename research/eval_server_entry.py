"""Dedicated SGLang launcher exposing actual runtime limits read-only.

No installed SGLang files are modified. The tokenizer manager receives the
scheduler's real input threshold at startup, including memory-pool constraints.
"""
from __future__ import annotations

import os
import sys


def main():
    from sglang.srt.entrypoints import http_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    @http_server.app.get('/cue-study/limits')
    async def study_limits():
        manager = http_server._global_state.tokenizer_manager
        return {
            'schema': 'cue-study-runtime-limits-v1',
            'context_length': manager.context_len,
            'max_input_tokens_exclusive': manager.max_req_input_len,
            'reserved_tokens': manager.num_reserved_tokens,
            'allow_auto_truncate': manager.server_args.allow_auto_truncate,
        }

    args = prepare_server_args(sys.argv[1:])
    if args.grpc_mode or args.encoder_only or args.tokenizer_worker_num != 1:
        raise ValueError('Dedicated evaluation requires single-tokenizer HTTP generation')
    try:
        http_server.launch_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == '__main__':
    main()
