"""Run the drift matrix (tasks x models x system prompts x repetitions) into append-only JSONL."""
from __future__ import annotations

import json
import random
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .agent import SYSTEM_PROMPTS, AgentBackend, OpenAICompatibleAgent, ReplayAgent, prompt_sha256
from .io import file_sha256, load_rules, load_tasks
from .policy import load_policy
from .runner import EpisodeRunner
from .types import AgentTurn, TaskSpec

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "configs"
RUNS_DIR = ROOT / "runs"


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


def smoke_agent(task: TaskSpec) -> AgentBackend:
    """Offline stand-in for a model: read the order in context, then answer."""
    order_id = task.context.get("current_order_id", "#W123")
    return ReplayAgent([AgentTurn.tool("get_order", order_id=order_id),
                        AgentTurn.final("Summary: reviewed the order. Recommend following up with the customer.")])


def nebius_agent(cfg: dict) -> AgentBackend:
    return OpenAICompatibleAgent(
        model=cfg["model"], base_url=cfg.get("base_url", OpenAICompatibleAgent.base_url),
        temperature=cfg.get("temperature"), top_p=cfg.get("top_p"), max_tokens=cfg.get("max_tokens"),
        extra_body=cfg.get("extra_body", {}),
    )


def read_done(path: Path, retry_errors: bool) -> set[str]:
    done: dict[str, bool] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn final line is simply re-run
            done[record["episode_key"]] = record["trace"]["end_reason"] != "agent_error"
    return {key for key, ok in done.items() if ok or not retry_errors}


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
    runs_dir: Path = RUNS_DIR,
    log: Callable[[str], None] = print,
) -> Path:
    all_tasks = load_tasks(tasks_path)
    tasks = [t for t in all_tasks if not task_ids or t.id in task_ids]
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
        # Pre-registration: labels are complete and fixed before any model runs.
        load_rules(labels_path, all_tasks)
    if backend not in ("nebius", "smoke"):
        raise ExperimentError(f"unknown backend {backend!r}")

    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    hashes = {
        "tasks": file_sha256(tasks_path),
        "labels": file_sha256(labels_path) if labels_path.exists() else None,
        "policy": file_sha256(policy_path),
        "models": file_sha256(models_path),
        "prompts": {name: prompt_sha256(SYSTEM_PROMPTS[name]) for name in prompts},
    }
    invocation = {"at": now(), "backend": backend, "models": {m: model_configs[m]["model"] for m in models},
                  "prompts": prompts, "repetitions": repetitions, "tasks": [t.id for t in tasks],
                  "max_steps": max_steps, "allow_unlabeled": allow_unlabeled}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        changed = [k for k in ("tasks", "labels", "policy", "models") if manifest["hashes"].get(k) != hashes[k]]
        if changed:
            raise ExperimentError(f"{', '.join(changed)} changed since run {run_id!r} started; use a new run id")
        manifest["hashes"]["prompts"].update(hashes["prompts"])
        manifest["invocations"].append(invocation)
    else:
        manifest = {"run_id": run_id, "created_at": now(), "hashes": hashes, "invocations": [invocation]}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    episodes_path = run_dir / "episodes.jsonl"
    done = read_done(episodes_path, retry_errors)
    plan = [(t, m, p, r) for t in tasks for m in models for p in prompts for r in range(1, repetitions + 1)]
    pending = [x for x in plan if episode_key(x[0].id, x[1], x[2], x[3]) not in done]
    # Interleave conditions so time-varying API behavior doesn't line up with one model.
    random.Random(seed).shuffle(pending)
    log(f"run {run_id}: {len(plan)} planned, {len(plan) - len(pending)} done, {len(pending)} to run")

    runner = EpisodeRunner(max_steps=max_steps)
    lock = threading.Lock()
    progress = {"done": 0, "errors": 0}

    def work(item: tuple[TaskSpec, str, str, int]) -> None:
        task, model, prompt, rep = item
        agent = smoke_agent(task) if backend == "smoke" else nebius_agent(model_configs[model])
        trace = runner.run(task, agent, model=model, system_prompt=SYSTEM_PROMPTS[prompt],
                           system_prompt_variant=prompt, repetition=rep)
        record = {"episode_key": episode_key(task.id, model, prompt, rep), "recorded_at": now(),
                  "model_id": model_configs[model]["model"], "trace": trace.to_dict()}
        line = json.dumps(record, ensure_ascii=False, default=str)
        with lock:
            with episodes_path.open("a") as f:
                f.write(line + "\n")
            progress["done"] += 1
            progress["errors"] += trace.end_reason in ("agent_error", "harness_error")
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
