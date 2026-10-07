"""CPU-only ALFWorld catalog export. Run on a node containing the game files.

Writes JSONL to stdout; never edits games or starts an environment/model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

FAMILIES = {
    "pick_and_place_simple": "pick_and_place",
    "look_at_obj_in_light": "examine",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "pick_two_obj_and_place": "pick_two",
}


def catalog(root: Path):
    for split in ("train", "valid_seen", "valid_unseen"):
        for path in sorted((root / "json_2.1.1" / split).rglob("game.tw-pddl")):
            game = json.loads(path.read_text())
            traj = json.loads(path.with_name("traj_data.json").read_text())
            family = FAMILIES.get(traj.get("task_type"))
            task_group = path.parent.parent.name
            reason = None
            if game.get("solvable") is not True:
                reason = "not_marked_solvable"
            elif "Sliced" in task_group or "movable" in task_group:
                reason = "unsupported_sliced_or_movable"
            elif family is None:
                reason = "unsupported_task_family"
            identity = {k: game.get(k) for k in ("pddl_domain", "pddl_problem", "grammar")}
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            relative = path.relative_to(root).as_posix()
            annotations = traj.get("turk_annotations", {}).get("anns", [])
            yield {
                "game_id": relative,
                "game_file": "examples/data/alfworld/alfworld_data/" + relative,
                "official_split": split,
                "task_family": family,
                "task_group": task_group,
                "scene": traj.get("scene", {}).get("floor_plan"),
                "problem_sha256": fingerprint,
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "walkthrough_steps": len(game["walkthrough"]) if isinstance(game.get("walkthrough"), list) else None,
                "prompt": annotations[0].get("task_desc", "") if annotations else "",
                "eligible": reason is None,
                "exclusion_reason": reason,
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ssh", choices=("spark1", "spark2"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.ssh:
        if not args.output:
            parser.error("--ssh requires --output")
        # This branch streams this source to a CPU-only remote Python process.
        # The fixed project path is not user-controlled shell syntax.
        if str(args.root) != "/workspace/tito-dressage/examples/data/alfworld/alfworld_data":
            parser.error("Remote catalog root must be the inspected ALFWorld path")
        result = subprocess.run([
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", args.ssh,
            "docker exec -i tito env PYTHONDONTWRITEBYTECODE=1 python3 - --root " + str(args.root),
        ], input=Path(__file__).read_text(), text=True, capture_output=True, check=True)
        for line in result.stdout.splitlines():
            json.loads(line)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(result.stdout)
        print(f"Exported {len(result.stdout.splitlines())} catalog rows to {args.output}")
    else:
        for row in catalog(args.root):
            print(json.dumps(row, sort_keys=True))
