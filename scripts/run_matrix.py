"""Run the drift matrix. Example (pilot):

    python scripts/run_matrix.py --run-id pilot-001 --models nano,super --prompts neutral,cautious --reps 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from permissionlab.experiment import ExperimentError, run_matrix  # noqa: E402
from permissionlab.io import LabelValidationError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--models", default="nano,super,ultra")
    p.add_argument("--prompts", default="neutral,cautious")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--tasks", default="", help="comma-separated task ids (default: all)")
    p.add_argument("--backend", choices=["nebius", "smoke"], default="nebius",
                   help="smoke = offline scripted agent, no API calls")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--retry-errors", action="store_true", help="re-run episodes that ended in agent_error")
    p.add_argument("--allow-unlabeled", action="store_true", help="skip the label check (smoke tests only)")
    args = p.parse_args(argv)
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    try:
        run_dir = run_matrix(args.run_id, models=split(args.models), prompts=split(args.prompts),
                             repetitions=args.reps, backend=args.backend, concurrency=args.concurrency,
                             task_ids=split(args.tasks) or None, retry_errors=args.retry_errors,
                             allow_unlabeled=args.allow_unlabeled, max_steps=args.max_steps)
    except (ExperimentError, LabelValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"episodes: {run_dir / 'episodes.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
