"""Run the drift matrix (tasks x models x system prompts x repetitions) into append-only JSONL.

The first run of a run id writes a manifest that pins everything the results depend on:
the run spec (backend, models with their sampling, prompts, repetitions, tasks, step
limit) and the hash of every input file. Resuming with anything different is refused.
"""
from __future__ import annotations

import json
import random
import subprocess
import threading
import tomllib
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from .agent import SYSTEM_PROMPTS, AgentBackend, OpenAICompatibleAgent, ReplayAgent, prompt_sha256
from .delegation import STANDING_DELEGATION, STANDING_DELEGATION_ID
from .io import file_sha256, load_rules, load_tasks
from .policy import load_policy
from .runner import EpisodeRunner
from .types import AgentTurn, TaskSpec

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "configs"
RUNS_DIR = ROOT / "runs"

# Only API failures are worth re-running. A truncated or empty reply is the model's
# behavior under this config; re-running it until it looks better would cherry-pick.
RETRYABLE_END_REASONS = ("agent_error",)

AgentFactory = Callable[[TaskSpec, str, dict], AgentBackend]


class ExperimentError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def episode_key(task_id: str, model: str, prompt: str, repetition: int) -> str:
    return f"{task_id}|{model}|{prompt}|{repetition}"


def load_model_configs(path: Path = CONFIG_DIR / "models.toml") -> dict[str, dict]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    defaults = data.get("defaults", {})
    return {key: {**defaults, **cfg} for key, cfg in data.get("models", {}).items()}


def code_version() -> str | None:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
                               check=True).stdout.strip()
        return commit + ("+uncommitted" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


# Code that only interprets a finished log. Changing it never changes what the agents
# saw or did, so it is recorded (and disclosed at analysis) rather than enforced.
ANALYSIS_CODE = ("permissionlab/analysis.py", "permissionlab/scoring.py", "permissionlab/stats.py",
                 "permissionlab/policy.py", "scripts/analyze.py")


def _fingerprint(paths: list[Path]) -> str:
    files = {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in paths}
    return sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def runtime_code_fingerprint() -> str:
    """Code that decides what agents see and what tools do, including uncommitted edits.

    Every package module counts unless listed in ANALYSIS_CODE, so a new module is pinned by default.
    """
    files = [path for path in sorted((ROOT / "permissionlab").glob("*.py"))
             if path.relative_to(ROOT).as_posix() not in ANALYSIS_CODE]
    return _fingerprint(files + [ROOT / "scripts" / "run_matrix.py"])


def analysis_code_fingerprint() -> str:
    return _fingerprint([ROOT / path for path in ANALYSIS_CODE])


def protocol_inputs() -> dict[str, str]:
    """Pinned for resume: everything that shaped what the agents saw or did."""
    return {"labeling_standard": file_sha256(ROOT / "LABELING.md"),
            "standing_delegation": prompt_sha256(STANDING_DELEGATION),
            "runtime_code": runtime_code_fingerprint()}


def smoke_agent(task: TaskSpec, model: str, cfg: dict) -> AgentBackend:
    """Offline stand-in for a model: read the order in context, then answer."""
    order_id = task.context.get("current_order_id", "#W123")
    return ReplayAgent([AgentTurn.tool("get_order", order_id=order_id),
                        AgentTurn.final("Summary: reviewed the order. Recommend following up with the customer.")])


def nebius_agent(task: TaskSpec, model: str, cfg: dict) -> AgentBackend:
    return OpenAICompatibleAgent(
        model=cfg["model"], base_url=cfg.get("base_url", OpenAICompatibleAgent.base_url),
        temperature=cfg.get("temperature"), top_p=cfg.get("top_p"), max_tokens=cfg.get("max_tokens"),
        extra_body=cfg.get("extra_body", {}),
    )


def read_log(path: Path, *, on_torn_tail: Callable[[bytes], None] | None = None) -> dict[str, list[dict]]:
    """All attempts per episode key, in the order they were written.

    A half-written last record (the process died mid-write) is passed to on_torn_tail and
    skipped. A corrupt record anywhere else is an error: we never silently drop data.
    """
    if not path.exists():
        return {}
    data = path.read_bytes()
    cut = data.rfind(b"\n") + 1
    if cut < len(data):
        if on_torn_tail is None:
            raise ExperimentError(f"{path} ends with a half-written record")
        on_torn_tail(data[cut:])
    attempts: dict[str, list[dict]] = {}
    for number, line in enumerate(data[:cut].decode().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExperimentError(f"{path}:{number} is corrupt ({exc.msg}); fix or remove that line") from None
        attempts.setdefault(record["episode_key"], []).append(record)
    return attempts


def repair_torn_tail(path: Path, log: Callable[[str], None]) -> None:
    """Before appending, move a half-written last record aside so new records start on a clean line."""
    if not path.exists():
        return
    data = path.read_bytes()
    cut = data.rfind(b"\n") + 1
    if cut == len(data):
        return
    torn = path.with_name(f"{path.name}.torn-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}")
    torn.write_bytes(data[cut:])
    path.write_bytes(data[:cut])
    log(f"moved a half-written record to {torn.name}; that episode will be re-run")


def build_manifest(*, spec: dict, inputs: dict) -> dict:
    # "recorded" is kept for disclosure only; resumes compare spec and inputs.
    return {"spec": spec, "inputs": inputs, "recorded": {"analysis_code": analysis_code_fingerprint()}}


def check_manifest(manifest_path: Path, fresh: dict) -> dict:
    """Create the manifest, or verify a resume matches it exactly."""
    if not manifest_path.exists():
        return {"created_at": now(), **fresh, "invocations": []}
    manifest = json.loads(manifest_path.read_text())
    diffs = [f"spec.{k}" for k in sorted(set(manifest["spec"]) | set(fresh["spec"]))
             if manifest["spec"].get(k) != fresh["spec"].get(k)]
    diffs += [f"inputs.{k}" for k in sorted(set(manifest["inputs"]) | set(fresh["inputs"]))
              if manifest["inputs"].get(k) != fresh["inputs"].get(k)]
    if diffs:
        raise ExperimentError(f"this run was started with different settings ({', '.join(diffs)}); "
                              "results would mix. Use a new --run-id.")
    return manifest


def run_matrix(
    run_id: str,
    *,
    models: list[str],
    prompts: list[str],
    repetitions: int,
    backend: str = "nebius",
    concurrency: int = 4,
    task_ids: list[str] | None = None,
    retry_errors: bool = False,
    allow_unlabeled: bool = False,
    max_steps: int = 15,
    seed: int = 0,
    tasks_path: Path = DATA_DIR / "tasks.jsonl",
    labels_path: Path = DATA_DIR / "labels_adjudicated.csv",
    policy_path: Path = CONFIG_DIR / "consequence_policy.toml",
    models_path: Path = CONFIG_DIR / "models.toml",
    prereg_path: Path = CONFIG_DIR / "preregistration.toml",
    runs_dir: Path = RUNS_DIR,
    agent_factory: AgentFactory | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    if not models or not prompts or repetitions < 1 or max_steps < 1:
        raise ExperimentError("select models and prompts, with positive repetitions and max_steps")
    if len(set(models)) != len(models) or len(set(prompts)) != len(prompts):
        raise ExperimentError("models and prompts must not contain duplicates")
    if backend not in ("nebius", "smoke"):
        raise ExperimentError(f"unknown backend {backend!r}")
    if allow_unlabeled and backend != "smoke":
        raise ExperimentError("--allow-unlabeled is only for the smoke backend; label the tasks first")
    all_tasks = load_tasks(tasks_path)
    tasks = [t for t in all_tasks if not task_ids or t.id in task_ids]
    if not tasks:
        raise ExperimentError("no tasks selected")
    if task_ids and len(tasks) != len(set(task_ids)):
        raise ExperimentError(f"unknown task ids: {sorted(set(task_ids) - {t.id for t in tasks})}")
    model_configs = load_model_configs(models_path)
    for m in models:
        if m not in model_configs:
            raise ExperimentError(f"unknown model {m!r}; configured: {', '.join(model_configs)}")
    for p in prompts:
        if p not in SYSTEM_PROMPTS:
            raise ExperimentError(f"unknown prompt {p!r}; known: {', '.join(SYSTEM_PROMPTS)}")
    load_policy(policy_path)
    if not allow_unlabeled:
        load_rules(labels_path, all_tasks)  # labels complete and valid before any model runs

    spec = {
        "backend": backend,
        "models": {m: model_configs[m] for m in models},
        "prompts": prompts,
        "repetitions": repetitions,
        "tasks": [t.id for t in tasks],
        "max_steps": max_steps,
        "labels_checked": not allow_unlabeled,
        "standing_delegation": {"id": STANDING_DELEGATION_ID, "text": STANDING_DELEGATION},
        "prompt_texts": {name: SYSTEM_PROMPTS[name] for name in prompts},
        "seed": seed,
    }
    inputs = {
        "tasks": file_sha256(tasks_path),
        "labels": None if allow_unlabeled else file_sha256(labels_path),
        "policy": file_sha256(policy_path),
        "models_config": file_sha256(models_path),
        "preregistration": file_sha256(prereg_path),
        "prompts": {name: prompt_sha256(SYSTEM_PROMPTS[name]) for name in prompts},
        **protocol_inputs(),
    }
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest = check_manifest(manifest_path, build_manifest(spec=spec, inputs=inputs))
    manifest["invocations"].append({"at": now(), "code": code_version(), "retry_errors": retry_errors})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    episodes_path = run_dir / "episodes.jsonl"
    repair_torn_tail(episodes_path, log)
    attempts = read_log(episodes_path)
    plan = [(t, m, p, r) for t in tasks for m in models for p in prompts for r in range(1, repetitions + 1)]

    def needs_run(key: str) -> bool:
        history = attempts.get(key)
        if not history:
            return True
        return retry_errors and history[-1]["trace"]["end_reason"] in RETRYABLE_END_REASONS

    pending = [x for x in plan if needs_run(episode_key(x[0].id, x[1], x[2], x[3]))]
    # Interleave conditions so time-varying API behavior doesn't line up with one model.
    random.Random(seed).shuffle(pending)
    log(f"run {run_id}: {len(plan)} planned, {len(plan) - len(pending)} done, {len(pending)} to run")

    factory = agent_factory or (smoke_agent if backend == "smoke" else nebius_agent)
    runner = EpisodeRunner(max_steps=max_steps)
    lock = threading.Lock()
    progress = {"done": 0, "errors": 0}

    def work(item: tuple[TaskSpec, str, str, int]) -> None:
        task, model, prompt, rep = item
        key = episode_key(task.id, model, prompt, rep)
        trace = runner.run(task, factory(task, model, model_configs[model]), model=model,
                           system_prompt=SYSTEM_PROMPTS[prompt], system_prompt_variant=prompt, repetition=rep)
        with lock:
            attempt = len(attempts.setdefault(key, [])) + 1
            record = {"episode_key": key, "attempt": attempt, "attempt_id": uuid.uuid4().hex,
                      "recorded_at": now(), "model_id": model_configs[model]["model"], "trace": trace.to_dict()}
            attempts[key].append(record)
            with episodes_path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            progress["done"] += 1
            progress["errors"] += trace.end_reason in ("agent_error", "truncated", "empty_response",
                                                       "harness_error")
            if progress["done"] % 10 == 0 or progress["done"] == len(pending):
                log(f"  {progress['done']}/{len(pending)} episodes, {progress['errors']} errors")

    pool = ThreadPoolExecutor(max_workers=max(1, concurrency))
    try:
        for future in as_completed([pool.submit(work, item) for item in pending]):
            future.result()
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        log("interrupted; finished episodes are saved, rerun the same command to resume")
        raise
    pool.shutdown()
    return run_dir
