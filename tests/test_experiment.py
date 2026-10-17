import csv
import json
from pathlib import Path

from permissionlab.analysis import analyze
from permissionlab.experiment import ExperimentError, run_matrix
from permissionlab.io import LabelValidationError
from permissionlab.tools import LABELED_ACTIONS
from tests.helpers import raises

ROOT = Path(__file__).resolve().parents[1]


def labels_file(path, task_ids, label="NOT_AUTHORIZED"):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "action", "label", "constraints_json"])
        writer.writeheader()
        for tid in task_ids:
            for action in LABELED_ACTIONS:
                writer.writerow({"task_id": tid, "action": action, "label": label, "constraints_json": "{}"})
    return path


def test_matrix_refuses_blank_labels(tmp_path):
    with raises(LabelValidationError):
        run_matrix("x", models=["nano"], prompts=["neutral"], repetitions=1, backend="smoke",
                   runs_dir=tmp_path, log=lambda _: None)


def test_smoke_run_resumes_and_analyzes(tmp_path):
    ids = ["PL-006", "PL-026"]
    kwargs = dict(models=["nano"], prompts=["neutral", "cautious"], repetitions=2, backend="smoke",
                  task_ids=ids, allow_unlabeled=True, runs_dir=tmp_path, log=lambda _: None)
    run_dir = run_matrix("smoke", **kwargs)
    assert len((run_dir / "episodes.jsonl").read_text().splitlines()) == 8
    run_matrix("smoke", **kwargs)  # resume: nothing new
    assert len((run_dir / "episodes.jsonl").read_text().splitlines()) == 8
    assert len(json.loads((run_dir / "manifest.json").read_text())["invocations"]) == 2

    report = analyze(run_dir, labels_path=labels_file(tmp_path / "labels.csv", ids), n_boot=100)
    text = report.read_text()
    assert "## Decision: STOP" in text  # the smoke agent only reads, so no drift
    assert "changed since the run started" in text  # labels were not fixed when this run began
    summary = json.loads((run_dir / "analysis" / "summary.json").read_text())
    assert summary["conditions"]["nano/neutral"]["metrics"]["utility"]["point"] == 0.0


def test_changed_inputs_need_a_new_run_id(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text((ROOT / "data" / "tasks.jsonl").read_text())
    kwargs = dict(models=["nano"], prompts=["neutral"], repetitions=1, backend="smoke", task_ids=["PL-001"],
                  allow_unlabeled=True, runs_dir=tmp_path, tasks_path=tasks, log=lambda _: None)
    run_matrix("r", **kwargs)
    tasks.write_text(tasks.read_text().replace("Find out", "Figure out"))
    with raises(ExperimentError, match="tasks changed"):
        run_matrix("r", **kwargs)
