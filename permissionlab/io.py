from __future__ import annotations

import csv
import json
import re
from hashlib import sha256
from pathlib import Path

from .tools import LABELED_ACTIONS, TOOL_SPECS
from .types import ActionRule, Authority, TaskSpec

LABEL_COLUMNS = ("task_id", "action", "label", "constraints_json")
_CONSTRAINT_KEY = re.compile(r"^(eq|in|max|max_total)_(\w+)$")


class LabelValidationError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        shown = "\n  ".join(problems[:25])
        more = f"\n  ... and {len(problems) - 25} more" if len(problems) > 25 else ""
        super().__init__(f"{len(problems)} label problem(s):\n  {shown}{more}")


def file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def load_tasks(path: str | Path) -> list[TaskSpec]:
    tasks = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                tasks.append(TaskSpec(**json.loads(line)))
    return tasks


def constraint_problems(action: str, constraints: object) -> list[str]:
    """Check a constraints object against the action's arguments and types."""
    if not isinstance(constraints, dict):
        return ["constraints_json must be a JSON object"]
    spec = TOOL_SPECS[action]
    types = spec.param_types
    problems = []
    for key, value in constraints.items():
        if key == "max_uses":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                problems.append("max_uses must be a non-negative integer")
            continue
        match = _CONSTRAINT_KEY.match(key)
        # max_total_x also matches (max)_(total_x); prefer the max_total reading.
        if key.startswith("max_total_"):
            kind, arg = "max_total", key.removeprefix("max_total_")
        elif match:
            kind, arg = match.group(1), match.group(2)
        else:
            problems.append(f"unknown constraint {key!r}")
            continue
        if arg not in types:
            problems.append(f"{key}: {action} has no argument {arg!r} (has {', '.join(types)})")
            continue
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if kind in ("max", "max_total"):
            if types[arg] != "number":
                problems.append(f"{key}: {arg} is not numeric")
            elif not number:
                problems.append(f"{key} must be a number")
        elif kind == "eq":
            if types[arg] == "number" and not number:
                problems.append(f"{key} must be a number")
            if types[arg] == "string" and not isinstance(value, str):
                problems.append(f"{key} must be a string")
        elif kind == "in":
            if not isinstance(value, list) or not value:
                problems.append(f"{key} must be a non-empty list")
    return problems


def load_rules(
    path: str | Path,
    tasks: list[TaskSpec],
    *,
    require: set[str] | None = None,
    allow_incomplete: bool = False,
) -> dict[str, dict[str, ActionRule]]:
    """Load and validate a label file. Raises LabelValidationError on any problem.

    `tasks` is the full study; `require` limits the completeness check to some of
    its tasks (e.g. the ones a pilot run used). Every required task must label every
    state-changing action exactly once. One row per (task, action) keeps rules from
    overlapping, so no precedence order is needed. allow_incomplete skips blank rows;
    use it only for inspection, never for scoring.
    """
    task_ids = {t.id for t in tasks}
    required = task_ids if require is None else set(require)
    problems: list[str] = []
    rules: dict[str, dict[str, ActionRule]] = {}
    present: set[tuple[str, str]] = set()  # (task, action) rows in the file, blank or not
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        missing_cols = [c for c in LABEL_COLUMNS if c not in (reader.fieldnames or [])]
        if missing_cols:
            raise LabelValidationError([f"missing column(s): {', '.join(missing_cols)}"])
        for line_no, row in enumerate(reader, start=2):
            where = f"line {line_no} ({row['task_id']}/{row['action']})"
            if row["task_id"] not in task_ids:
                problems.append(f"{where}: unknown task_id")
                continue
            if row["action"] not in TOOL_SPECS:
                problems.append(f"{where}: unknown action")
                continue
            present.add((row["task_id"], row["action"]))
            label = (row["label"] or "").strip()
            if not label:
                if not allow_incomplete and row["task_id"] in required:
                    problems.append(f"{where}: label is blank")
                continue
            try:
                authority = Authority(label)
            except ValueError:
                problems.append(f"{where}: invalid label {label!r} (use {', '.join(a.value for a in Authority)})")
                continue
            try:
                constraints = json.loads(row["constraints_json"] or "{}")
            except json.JSONDecodeError as exc:
                problems.append(f"{where}: constraints_json is not valid JSON ({exc.msg})")
                continue
            problems += [f"{where}: {p}" for p in constraint_problems(row["action"], constraints)]
            if authority is Authority.NOT_AUTHORIZED and constraints:
                problems.append(f"{where}: constraints only apply to AUTHORIZED or AMBIGUOUS")
            task_rules = rules.setdefault(row["task_id"], {})
            if row["action"] in task_rules:
                problems.append(f"{where}: duplicate row; each task labels each action once")
                continue
            task_rules[row["action"]] = ActionRule(row["action"], authority, constraints)

    if not allow_incomplete:
        for task in tasks:
            if task.id not in required:
                continue
            for action in LABELED_ACTIONS:
                if (task.id, action) not in present:
                    problems.append(f"{task.id}/{action}: no row in the label file")
    if problems:
        raise LabelValidationError(problems)
    return rules


def load_adjudicated_rules(path: str | Path, tasks: list[TaskSpec], **kwargs) -> dict[str, dict[str, ActionRule]]:
    """Older name for load_rules."""
    return load_rules(path, tasks, **kwargs)
