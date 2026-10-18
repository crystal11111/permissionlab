"""Check API connectivity, tool calling, and response termination before a pilot.

Runs one explicit task (PL-026: refund $30 and email the customer) per model and
checks more than HTTP success: the model must make a valid tool call that runs, and
finish with a real reply (not cut off by the token limit, not empty, no API error).
PASS is not a verdict on task completion or authorization. This development check
can run before labels are ready and writes nothing to runs/.

    export NEBIUS_API_KEY=...
    python scripts/check_api.py --models nano,super,ultra
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from permissionlab.agent import SYSTEM_PROMPTS  # noqa: E402
from permissionlab.experiment import load_model_configs, nebius_agent  # noqa: E402
from permissionlab.io import load_tasks  # noqa: E402
from permissionlab.runner import EpisodeRunner  # noqa: E402


def check(model: str, cfg: dict, task) -> list[str]:
    """Connection-check problems; an empty list does not certify study readiness."""
    t0 = time.monotonic()
    trace = EpisodeRunner(max_steps=15).run(task, nebius_agent(task, model, cfg), model=model,
                                           system_prompt=SYSTEM_PROMPTS["neutral"], system_prompt_variant="neutral")
    meta = trace.agent_metadata
    print(f"{model} ({cfg['model']}): {trace.end_reason} in {time.monotonic() - t0:.1f}s, "
          f"{meta.get('requests', 0)} requests, finish reasons {meta.get('finish_reasons')}, usage {meta.get('usage')}")
    for event in trace.tool_events:
        print(f"    {event.status:9} {event.name}({event.arguments})" + (f"  {event.error}" if event.error else ""))
    problems = []
    if trace.end_reason not in ("final", "question"):
        problems.append(f"ended with {trace.end_reason}: {trace.error}")
    if not any(e.status == "succeeded" for e in trace.tool_events):
        problems.append("no tool call ran successfully")
    if any(e.status == "invalid" for e in trace.tool_events):
        problems.append("made malformed tool calls (see above)")
    if not (trace.final_answer or trace.question_text):
        problems.append("no final reply")
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="nano,super,ultra")
    p.add_argument("--task", default="PL-026")
    args = p.parse_args(argv)
    configs = load_model_configs()
    task = next(t for t in load_tasks(ROOT / "data" / "tasks.jsonl") if t.id == args.task)
    failed = False
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        problems = check(model, configs[model], task)
        print(f"  {'PASS' if not problems else 'FAIL: ' + '; '.join(problems)}\n")
        failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
