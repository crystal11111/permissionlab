"""Score a run and apply the pre-registered GO / PIVOT / STOP rule.

    python scripts/analyze.py runs/pilot-001
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from permissionlab.analysis import AnalysisError, analyze  # noqa: E402
from permissionlab.experiment import ExperimentError  # noqa: E402
from permissionlab.io import LabelValidationError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--labels", type=Path, default=ROOT / "data" / "labels_adjudicated.csv")
    p.add_argument("--n-boot", type=int)
    args = p.parse_args(argv)
    try:
        report = analyze(args.run_dir, labels_path=args.labels, n_boot=args.n_boot)
    except (AnalysisError, ExperimentError, LabelValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(report.read_text())
    print(f"written to {report.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
