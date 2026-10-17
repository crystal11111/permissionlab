import csv
import json
from pathlib import Path

from permissionlab import agent as agent_module
from permissionlab.agent import AgentBackend, AgentError
from permissionlab.analysis import AnalysisError, analyze
from permissionlab.experiment import ExperimentError, read_log, run_matrix
from permissionlab.io import LabelValidationError
from permissionlab.tools import LABELED_ACTIONS
from permissionlab.types import AgentTurn
from tests.helpers import raises

ROOT = Path(__file__).resolve().parents[1]
QUIET = lambda _: None  # noqa: E731


def labels_file(path, task_ids=None, label="NOT_AUTHORIZED"):
    ids = task_ids or [json.loads(l)["id"] for l in (ROOT / "data" / "tasks.jsonl").read_text().splitlines()]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "action", "label", "constraints_json"])
        writer.writeheader()
        for tid in ids:
            for action in LABELED_ACTIONS:
                writer.writerow({"task_id": tid, "action": action, "label": label, "constraints_json": "{}"})
    return path


class Scripted(AgentBackend):
    """Plays one script per attempt: attempts[i] is a list of turns, or an exception to raise mid-way."""

    def __init__(self, turns):
        self.turns = turns

    def start(self, task, system_prompt):
        self.i = 0

    def next_turn(self, observation):
        if self.i >= len(self.turns):
            return AgentTurn.final("Done.")
        turn = self.turns[self.i]
        self.i += 1
        if isinstance(turn, Exception):
            raise turn
        return turn


def factory_from(scripts):
    """scripts: list consumed one per episode attempt, in order."""
    queue = list(scripts)
    return lambda task, model, cfg: Scripted(queue.pop(0))


REFUND = AgentTurn.tool("issue_refund", user_id="u_42", order_id="#W123", amount=50)


def study(tmp_path, **overrides):
    """A labeled single-task run, so analyze() treats it as pre-registered."""
    kwargs = dict(models=["nano"], prompts=["neutral"], repetitions=1, backend="smoke", task_ids=["PL-006"],
                  labels_path=labels_file(tmp_path / "labels.csv"), runs_dir=tmp_path, concurrency=1, log=QUIET)
    kwargs.update(overrides)
    return kwargs


def test_matrix_refuses_blank_labels(tmp_path):
    with raises(LabelValidationError):
        run_matrix("x", models=["nano"], prompts=["neutral"], repetitions=1, backend="smoke",
                   runs_dir=tmp_path, log=QUIET)


def test_allow_unlabeled_is_smoke_only(tmp_path):
    with raises(ExperimentError, match="only for the smoke backend"):
        run_matrix("x", models=["nano"], prompts=["neutral"], repetitions=1, backend="nebius",
                   allow_unlabeled=True, runs_dir=tmp_path, log=QUIET)


def test_retry_keeps_drift_from_the_earlier_attempt(tmp_path):
    # Review repro: attempt 1 refunds then hits an API error; the retry ends cleanly.
    kwargs = study(tmp_path)
    run_matrix("r", agent_factory=factory_from([[REFUND, AgentError("HTTP 503")]]), **kwargs)
    run_matrix("r", agent_factory=factory_from([[AgentTurn.final("Recommend a replacement.")]]),
               retry_errors=True, **kwargs)
    log = read_log(tmp_path / "r" / "episodes.jsonl")
    attempts = log["PL-006|nano|neutral|1"]
    assert [a["attempt"] for a in attempts] == [1, 2] and attempts[0]["attempt_id"] != attempts[1]["attempt_id"]
    analyze(tmp_path / "r", labels_path=kwargs["labels_path"], n_boot=50)
    summary = json.loads((tmp_path / "r" / "analysis" / "summary.json").read_text())
    assert summary["conditions"]["nano/neutral"]["metrics"]["hard_drift"]["point"] == 1.0
    assert summary["drift_only_in_earlier_attempt"] == 1


def test_all_api_failures_are_inconclusive_not_stop(tmp_path):
    # Review repro: every episode failed on its first call and the old rule said STOP.
    kwargs = study(tmp_path, repetitions=3)
    run_matrix("f", agent_factory=factory_from([[AgentError("HTTP 500")]] * 3), **kwargs)
    report = analyze(tmp_path / "f", labels_path=kwargs["labels_path"], n_boot=50).read_text()
    assert "**Decision: INCONCLUSIVE**" in report


def test_complete_clean_run_can_stop(tmp_path):
    kwargs = study(tmp_path, repetitions=2)
    run_matrix("ok", **kwargs)
    report = analyze(tmp_path / "ok", labels_path=kwargs["labels_path"], n_boot=50).read_text()
    assert "**Decision: STOP**" in report  # the smoke agent only reads


def test_utility_counts_failed_runs_in_one_measure_only(tmp_path):
    # Review repro: one success + one failure used to report utility 100%.
    done = [AgentTurn.tool("issue_refund", user_id="u_42", order_id="#W123", amount=30),
            AgentTurn.tool("send_email", user_id="u_42", subject="Refund", body="Issued."), AgentTurn.final("Done.")]
    kwargs = study(tmp_path, task_ids=["PL-026"], repetitions=2)
    run_matrix("u", agent_factory=factory_from([done, [AgentError("HTTP 500")]]), **kwargs)
    analyze(tmp_path / "u", labels_path=kwargs["labels_path"], n_boot=50)
    metrics = json.loads((tmp_path / "u" / "analysis" / "summary.json").read_text())["conditions"]["nano/neutral"]["metrics"]
    assert metrics["utility_all"]["point"] == 0.5
    assert metrics["utility_valid"]["point"] == 1.0


def test_unlabeled_smoke_run_is_never_a_study_result(tmp_path):
    run_dir = run_matrix("s", models=["nano"], prompts=["neutral"], repetitions=1, backend="smoke",
                         task_ids=["PL-006"], allow_unlabeled=True, runs_dir=tmp_path, log=QUIET)
    report = analyze(run_dir, labels_path=labels_file(tmp_path / "labels.csv"), n_boot=50).read_text()
    assert "**Decision: INCONCLUSIVE**" in report and "smoke run" in report
    assert (run_dir / "analysis" / "question_review.csv").exists()


def test_resume_refuses_a_changed_prompt(tmp_path):
    kwargs = study(tmp_path)
    run_matrix("p", **kwargs)
    original = agent_module.SYSTEM_PROMPTS["neutral"]
    agent_module.SYSTEM_PROMPTS["neutral"] = original + " Be proactive."
    try:
        with raises(ExperimentError, match="inputs.prompts"):
            run_matrix("p", **kwargs)
    finally:
        agent_module.SYSTEM_PROMPTS["neutral"] = original


def test_resume_refuses_changed_settings_and_preregistration(tmp_path):
    prereg = tmp_path / "prereg.toml"
    prereg.write_text((ROOT / "configs" / "preregistration.toml").read_text())
    kwargs = study(tmp_path, prereg_path=prereg)
    run_matrix("c", **kwargs)
    with raises(ExperimentError, match="spec.max_steps"):
        run_matrix("c", **{**kwargs, "max_steps": 5})
    prereg.write_text(prereg.read_text().replace("min_rate = 0.05", "min_rate = 0.01"))
    with raises(ExperimentError, match="inputs.preregistration"):
        run_matrix("c", **kwargs)


def test_analysis_refuses_changed_tasks(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text((ROOT / "data" / "tasks.jsonl").read_text())
    kwargs = study(tmp_path, tasks_path=tasks)
    run_matrix("t", **kwargs)
    tasks.write_text(tasks.read_text().replace("recommend what we should do", "fix it"))
    with raises(AnalysisError, match="tasks changed"):
        analyze(tmp_path / "t", labels_path=kwargs["labels_path"], tasks_path=tasks, n_boot=50)


def test_resume_recovers_from_a_half_written_record(tmp_path):
    # Review repro: a torn last line used to swallow the next record and later crash.
    kwargs = study(tmp_path, repetitions=3)
    run_dir = run_matrix("h", **kwargs)
    log_path = run_dir / "episodes.jsonl"
    lines = log_path.read_text().splitlines(keepends=True)
    log_path.write_text(lines[0] + lines[1][: len(lines[1]) // 2])  # one good record, one torn
    run_matrix("h", **kwargs)
    assert len(read_log(log_path)) == 3  # all three episodes load
    assert len(list(run_dir.glob("episodes.jsonl.torn-*"))) == 1  # the torn bytes were kept aside


def test_corrupt_record_in_the_middle_is_an_error(tmp_path):
    kwargs = study(tmp_path, repetitions=2)
    run_dir = run_matrix("m", **kwargs)
    log_path = run_dir / "episodes.jsonl"
    lines = log_path.read_text().splitlines(keepends=True)
    log_path.write_text("{broken\n" + "".join(lines))
    with raises(ExperimentError, match="is corrupt"):
        run_matrix("m", **kwargs)
