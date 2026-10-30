"""Build a self-contained, key-free dashboard from an existing run. Does not run models or change labels.

    python scripts/build_dashboard.py --run-dir runs/pilot-case-v2-001
    python scripts/build_dashboard.py --run-dir runs/pilot-case-v2-001 --status

Without --run-dir, generate an honest waiting page with the clearly marked offline replay.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from permissionlab.experiment import read_log, ExperimentError
from permissionlab.io import load_tasks


def read_json(path):
    return json.loads(path.read_text())


def snapshot(run_dir: Path | None = None) -> dict:
    data = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": "waiting", "run_id": None, "backend": None, "progress": None,
            "decision": None, "conditions": {}, "episodes": [], "warnings": [],
            "label_basis": "See the run's labeling record; labels are not inferred by this dashboard."}
    if run_dir is None:
        demo = read_json(ROOT / "offline_demo_output.json")
        trace, score = demo["trace"], demo["score_with_illustrative_labels"]
        data["episodes"] = [{"key": "offline-replay", "task_id": trace["task_id"],
            "instruction": trace["prompt"], "model": "Offline replay", "prompt": "neutral",
            "hard_drift": score["hard_drift"], "soft_drift": score["soft_drift"],
            "end_reason": trace["end_reason"], "illustrative": True,
            "attempts": [{"number": 1, "calls": score["classified_calls"],
                          "end_reason": trace["end_reason"], "answer": trace.get("final_answer")}]}]
        return data
    manifest = read_json(run_dir / "manifest.json")
    spec = manifest["spec"]
    warnings = data["warnings"]
    logs = read_log(run_dir / "episodes.jsonl", on_torn_tail=lambda _: warnings.append(
        "The last log record is incomplete and is excluded from this snapshot."))
    planned_keys = {f"{t}|{m}|{p}|{r}" for t in spec["tasks"] for m in spec["models"]
                    for p in spec["prompts"] for r in range(1, spec["repetitions"] + 1)}
    unexpected = set(logs) - planned_keys
    if unexpected:
        raise ValueError("Episode log contains records outside the planned matrix")
    last = [records[-1]["trace"] for records in logs.values()]
    reasons = dict(Counter(t["end_reason"] for t in last))
    data.update(mode="pending_analysis", run_id=run_dir.name, backend=spec["backend"],
        progress={"planned": len(planned_keys), "recorded": len(logs),
            "attempts": sum(len(v) for v in logs.values()),
            "retried": sum(len(v) > 1 for v in logs.values()), "end_reasons": reasons,
            "api_errors": reasons.get("agent_error", 0), "truncated": reasons.get("truncated", 0),
            "last_recorded_at": max((r.get("recorded_at", "") for a in logs.values() for r in a), default=None)},
        models=list(spec["models"]), prompts=spec["prompts"],
        delegation=spec.get("standing_delegation"), inputs=manifest.get("inputs"))
    summary_path, scores_path = run_dir / "analysis/summary.json", run_dir / "analysis/scores.jsonl"
    if not summary_path.exists() or not scores_path.exists():
        return data
    summary = read_json(summary_path)
    scores = [json.loads(l) for l in scores_path.read_text().splitlines() if l.strip()]
    by_key = {s["episode_key"]: s for s in scores}
    fresh = summary.get("run_id") == run_dir.name and len(by_key) == len(scores) and set(by_key) == set(logs)
    fresh = fresh and all(by_key[k].get("attempts", 1) == len(v) and
                          by_key[k]["end_reason"] == v[-1]["trace"]["end_reason"] for k, v in logs.items())
    if not fresh:
        warnings.append("Analysis is older than the recorded attempts. Re-run analyze.py before showing rates.")
        return data
    conditions = summary.get("conditions", {})
    expected_conditions = {f"{m}/{p}" for m in spec["models"] for p in spec["prompts"]}
    if set(conditions) - expected_conditions:
        raise ValueError("Analysis contains an unexpected model/prompt condition")
    decision = summary["decision"]
    # Never promote incomplete or scripted data to a final study verdict.
    complete = set(logs) == planned_keys
    if not complete or spec["backend"] == "smoke":
        decision = {"decision": "INCONCLUSIVE", "why": "Incomplete or scripted run; no final study conclusion."}
    data.update(mode="analyzed", decision=decision, conditions=conditions,
                validity=summary.get("validity", {}), analysis_changes=summary.get("working_copy_differs_from_run_start", []))
    tasks = {t.id: t for t in load_tasks(ROOT / "data/tasks.jsonl")}
    from permissionlab.scoring import classify_calls
    from permissionlab.io import load_rules, file_sha256
    from permissionlab.policy import load_policy
    # Use scored calls for the final attempt; older-attempt tags require the exact original labels.
    label_file = ROOT / "data/labels_adjudicated.csv"
    original_labels = manifest.get("inputs", {}).get("labels")
    rules = None
    if (original_labels and label_file.exists() and file_sha256(label_file) == original_labels
            and file_sha256(ROOT / "configs/consequence_policy.toml") == manifest.get("inputs", {}).get("policy")):
        rules = load_rules(label_file, list(tasks.values()), require={s["task_id"] for s in scores})
    from permissionlab.types import EpisodeTrace
    for key, history in logs.items():
        score = by_key[key]
        attempts = []
        for i, record in enumerate(history):
            trace = record["trace"]
            if i == len(history) - 1:
                calls = score["classified_calls"]
            elif rules is not None:
                calls = classify_calls(EpisodeTrace.from_dict(trace), rules[trace["task_id"]], load_policy())
            else:
                calls = [{"action": e["name"], "arguments": e["arguments"], "status": e["status"],
                          "authority": None, "reason": "Earlier attempt; see original analysis."}
                         for e in trace["tool_events"]]
            attempts.append({"number": i + 1, "calls": calls, "end_reason": trace["end_reason"],
                             "answer": trace.get("question_text") or trace.get("final_answer")})
        data["episodes"].append({"key": key, "task_id": score["task_id"], "instruction": history[-1]["trace"]["prompt"],
            "model": score["model"], "prompt": score["prompt_variant"], "hard_drift": score["hard_drift"],
            "soft_drift": score["soft_drift"], "end_reason": score["end_reason"], "illustrative": False,
            "attempts": attempts})
    return data


def public_payload(data: dict) -> dict:
    """Publish only fields the UI needs, never raw message bodies or case notes."""
    out = {k: data[k] for k in ("generated_at", "mode", "run_id", "backend", "progress", "decision",
                               "warnings", "label_basis") if k in data}
    for key in ("models", "analysis_changes"):
        if key in data:
            out[key] = data[key]
    out["conditions"] = {key: {"n": value["n"], "metrics": value["metrics"]}
                         for key, value in data.get("conditions", {}).items()}
    tasks = {t.id: t.prompt for t in load_tasks(ROOT / "data/tasks.jsonl")}
    out["episodes"] = []
    for episode in data.get("episodes", []):
        item = {k: episode[k] for k in ("key", "task_id", "model", "prompt", "hard_drift", "soft_drift",
                                        "end_reason", "illustrative")}
        # The public, fixed task prompt is the source; don't copy arbitrary transcript text.
        item["instruction"] = tasks.get(item["task_id"], "Task instruction unavailable.")
        item["attempts"] = []
        for attempt in episode["attempts"]:
            calls = []
            for call in attempt["calls"]:
                args = call.get("arguments", {})
                safe_args = {}
                for field, pattern in (("user_id", r"[uU]_\d+"), ("order_id", r"#?[wW]\d+")):
                    value = args.get(field)
                    if isinstance(value, str) and re.fullmatch(pattern, value):
                        safe_args[field] = value
                if isinstance(args.get("amount"), (int, float)) and not isinstance(args["amount"], bool):
                    safe_args["amount"] = args["amount"]
                calls.append({"action": call["action"], "arguments": safe_args,
                              "status": call["status"], "authority": call.get("authority")})
            item["attempts"].append({"number": attempt["number"], "end_reason": attempt["end_reason"],
                                     "calls": calls})
        out["episodes"].append(item)
    return out


def render(data: dict) -> str:
    template = (ROOT / "scripts/dashboard/template.html").read_text()
    payload = json.dumps(public_payload(data), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    return template.replace("__PERMISSIONLAB_DATA__", payload)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "docs/index.html")
    p.add_argument("--status", action="store_true", help="Print a log snapshot; do not write the dashboard")
    p.add_argument("--label-basis", help="Public disclosure of who labeled this run and any missing adjudication")
    args = p.parse_args(argv)
    try:
        data = snapshot(args.run_dir)
        if args.label_basis:
            data["label_basis"] = args.label_basis
        if args.status:
            print(json.dumps({k: data[k] for k in ("run_id", "mode", "generated_at", "progress", "warnings")}, indent=2))
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            html = render(data)
            args.output.write_text(html)
            print(f"Dashboard: {args.output} ({data['mode']})")
            if args.output.resolve() == (ROOT / "docs/index.html").resolve():
                # GitHub Pages serves the repository root; mirror the page there so the
                # root URL shows the dashboard itself (it is self-contained, no relative links).
                (ROOT / "index.html").write_text(html)
                print(f"Mirrored to {ROOT / 'index.html'}")
        return 0
    except (OSError, ValueError, KeyError, ExperimentError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main())
