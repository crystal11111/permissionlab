import csv
import json
from pathlib import Path

from permissionlab import agent as agent_module, experiment as experiment_module, analysis as analysis_module
from permissionlab.agent import AgentBackend, AgentError
from permissionlab.analysis import AnalysisError, analyze
from permissionlab.experiment import ExperimentError, read_log, run_matrix, smoke_agent
from permissionlab.delegation import STANDING_DELEGATION, STANDING_DELEGATION_ID
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
    # Simulate a measured backend without network calls. Actual smoke runs never yield a study decision.
    kwargs = dict(models=["nano"], prompts=["neutral"], repetitions=1, backend="nebius",
                  agent_factory=smoke_agent, task_ids=["PL-006"],
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
    run_matrix("r", **{**kwargs, "agent_factory": factory_from([[REFUND, AgentError("HTTP 503")]])})
    run_matrix("r", retry_errors=True,
               **{**kwargs, "agent_factory": factory_from([[AgentTurn.final("Recommend a replacement.")]])})
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
    run_matrix("f", **{**kwargs, "agent_factory": factory_from([[AgentError("HTTP 500")]] * 3)})
    report = analyze(tmp_path / "f", labels_path=kwargs["labels_path"], n_boot=50).read_text()
    assert "**Decision: INCONCLUSIVE**" in report


def test_complete_clean_run_can_stop(tmp_path):
    kwargs = study(tmp_path, repetitions=2)
    run_matrix("ok", **kwargs)
    report = analyze(tmp_path / "ok", labels_path=kwargs["labels_path"], n_boot=50).read_text()
    assert "**Decision: STOP**" in report  # the injected model stand-in only reads


def test_utility_counts_failed_runs_in_one_measure_only(tmp_path):
    # Review repro: one success + one failure used to report utility 100%.
    done = [AgentTurn.tool("issue_refund", user_id="u_42", order_id="#W123", amount=30),
            AgentTurn.tool("send_email", user_id="u_42", subject="Refund", body="Issued."), AgentTurn.final("Done.")]
    kwargs = study(tmp_path, task_ids=["PL-026"], repetitions=2)
    run_matrix("u", **{**kwargs, "agent_factory": factory_from([done, [AgentError("HTTP 500")]])})
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


def test_unfinished_run_is_inconclusive(tmp_path):
    # Two of five planned episodes ran: missing episodes are not "no drift".
    kwargs = study(tmp_path, repetitions=5)
    run_dir = run_matrix("partial", **kwargs)
    log_path = run_dir / "episodes.jsonl"
    log_path.write_text("".join(log_path.read_text().splitlines(keepends=True)[:2]))
    report = analyze(run_dir, labels_path=kwargs["labels_path"], n_boot=50).read_text()
    assert "**Decision: INCONCLUSIVE**" in report


def test_even_one_missing_episode_in_either_prompt_is_inconclusive(tmp_path):
    for variant in ("neutral", "cautious"):
        kwargs = study(tmp_path, repetitions=20, prompts=["neutral", "cautious"])
        run_dir = run_matrix("missing-" + variant, **kwargs)
        log_path = run_dir / "episodes.jsonl"
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        victim = next(r for r in records if r["trace"]["system_prompt_variant"] == variant)
        log_path.write_text("".join(json.dumps(r) + "\n" for r in records if r is not victim))
        report = analyze(run_dir, labels_path=kwargs["labels_path"], n_boot=50).read_text()
        assert "**Decision: INCONCLUSIVE**" in report
        assert "incomplete matrix" in report


def test_labeled_smoke_run_is_never_a_study_result(tmp_path):
    kwargs = study(tmp_path, backend="smoke")
    run_dir = run_matrix("labeled-smoke", **kwargs)
    report = analyze(run_dir, labels_path=kwargs["labels_path"], n_boot=50).read_text()
    assert "**Decision: INCONCLUSIVE**" in report and "scripted agents" in report


def test_manifest_records_delegation_and_prompts_received_by_agents(tmp_path):
    received = []

    class RecordingAgent(Scripted):
        def start(self, task, system_prompt):
            received.append(system_prompt)
            super().start(task, system_prompt)

    kwargs = study(tmp_path, prompts=["neutral", "cautious"],
                   agent_factory=lambda *_: RecordingAgent([]))
    run_dir = run_matrix("context", **kwargs)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["spec"]["standing_delegation"] == {
        "id": STANDING_DELEGATION_ID, "text": STANDING_DELEGATION}
    assert sorted(received) == sorted(manifest["spec"]["prompt_texts"].values())
    assert all(p.count(STANDING_DELEGATION) == 1 for p in received)
    assert all(manifest["inputs"][k] for k in ("runtime_code", "standing_delegation", "labeling_standard"))
    assert manifest["recorded"]["analysis_code"]
    report = analyze(run_dir, labels_path=kwargs["labels_path"], n_boot=50).read_text()
    assert STANDING_DELEGATION in report and STANDING_DELEGATION_ID in report


def test_resume_refuses_a_changed_protocol_and_analysis_discloses_it(tmp_path):
    kwargs = study(tmp_path)
    run_dir = run_matrix("protocol", **kwargs)
    original_experiment = experiment_module.protocol_inputs
    original_analysis = analysis_module.protocol_inputs
    original_inputs = original_experiment()
    labels = {"runtime_code": "runtime code", "standing_delegation": "standing delegation",
              "labeling_standard": "labeling standard"}
    try:
        for key, label in labels.items():
            changed = lambda key=key: {**original_inputs, key: "changed"}
            experiment_module.protocol_inputs = changed
            analysis_module.protocol_inputs = changed
            with raises(ExperimentError, match="inputs." + key):
                run_matrix("protocol", **kwargs)
            report = analyze(run_dir, labels_path=kwargs["labels_path"], n_boot=50).read_text()
            assert "differs from the start of this run in: " + label in report
    finally:
        experiment_module.protocol_inputs = original_experiment
        analysis_module.protocol_inputs = original_analysis


def test_a_finished_run_can_be_analyzed_after_code_moves_on(tmp_path):
    # Fixing an analysis bug after the pilot must not force re-running the paid matrix.
    kwargs = study(tmp_path)
    run_dir = run_matrix("later", **kwargs)
    clean = analyze(run_dir, labels_path=kwargs["labels_path"], n_boot=50).read_text()
    assert "matches everything pinned" in clean
    original = analysis_module.analysis_code_fingerprint
    try:
        analysis_module.analysis_code_fingerprint = lambda: "fixed-analysis-code"
        report = analyze(run_dir, labels_path=kwargs["labels_path"], n_boot=50).read_text()
        assert "differs from the start of this run in: analysis code" in report
        assert "check out commit" in report
    finally:
        analysis_module.analysis_code_fingerprint = original
    run_matrix("later", **kwargs)  # resuming is unaffected by analysis code


def test_analysis_discloses_changed_prompt_and_model_settings(tmp_path):
    models = tmp_path / "models.toml"
    models.write_text((ROOT / "configs" / "models.toml").read_text())
    kwargs = study(tmp_path, models_path=models)
    run_dir = run_matrix("analysis-inputs", **kwargs)
    original = agent_module.SYSTEM_PROMPTS["neutral"]
    try:
        agent_module.SYSTEM_PROMPTS["neutral"] = original + " Changed."
        report = analyze(run_dir, labels_path=kwargs["labels_path"], models_path=models, n_boot=50).read_text()
        assert "system prompts" in report
    finally:
        agent_module.SYSTEM_PROMPTS["neutral"] = original
    models.write_text(models.read_text() + "\n# changed settings file\n")
    report = analyze(run_dir, labels_path=kwargs["labels_path"], models_path=models, n_boot=50).read_text()
    assert "model settings" in report


def test_analysis_refuses_changed_scoring_inputs(tmp_path):
    prereg = tmp_path / "prereg.toml"
    prereg.write_text((ROOT / "configs" / "preregistration.toml").read_text())
    kwargs = study(tmp_path, prereg_path=prereg)
    run_dir = run_matrix("scoring-inputs", **kwargs)
    prereg.write_text(prereg.read_text().replace("min_rate = 0.05", "min_rate = 0.01"))
    with raises(AnalysisError, match="preregistration"):
        analyze(run_dir, labels_path=kwargs["labels_path"], prereg_path=prereg, n_boot=50)


def test_matrix_rejects_empty_or_duplicate_conditions(tmp_path):
    for override in ({"models": []}, {"prompts": []}, {"models": ["nano", "nano"]},
                     {"repetitions": 0}, {"max_steps": 0}):
        with raises(ExperimentError):
            run_matrix("invalid", **study(tmp_path, **override))


def test_utility_is_false_when_api_fails_after_expected_effects(tmp_path):
    done_then_failed = [AgentTurn.tool("issue_refund", user_id="u_42", order_id="#W123", amount=30),
                        AgentTurn.tool("send_email", user_id="u_42", subject="Refund", body="Issued."),
                        AgentError("HTTP 503")]
    kwargs = study(tmp_path, task_ids=["PL-026"], agent_factory=factory_from([done_then_failed]))
    run_dir = run_matrix("effect-then-error", **kwargs)
    analyze(run_dir, labels_path=kwargs["labels_path"], n_boot=50)
    metrics = json.loads((run_dir / "analysis" / "summary.json").read_text())["conditions"]["nano/neutral"]["metrics"]
    assert metrics["utility_all"]["point"] == 0.0
    assert metrics["utility_valid"]["point"] is None
    assert metrics["hard_drift"]["point"] == 1.0  # writes still happened
