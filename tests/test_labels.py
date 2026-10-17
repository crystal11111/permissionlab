import csv
from pathlib import Path

from permissionlab.io import LabelValidationError, load_rules, load_tasks
from permissionlab.tools import LABELED_ACTIONS
from tests.helpers import raises

ROOT = Path(__file__).resolve().parents[1]
TASKS = load_tasks(ROOT / "data" / "tasks.jsonl")
HEADERS = ["task_id", "category", "prompt", "action", "label", "constraints_json", "notes"]


def write_labels(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({"category": "", "prompt": "", "notes": "", "constraints_json": "{}", **row})


def full_rows(label="NOT_AUTHORIZED"):
    return [{"task_id": t.id, "action": a, "label": label} for t in TASKS for a in LABELED_ACTIONS]


def test_blank_label_file_is_rejected():
    # Review repro: the shipped sheets are blank; loading them must fail, not return {}.
    with raises(LabelValidationError) as err:
        load_rules(ROOT / "data" / "labels_adjudicated.csv", TASKS)
    assert len(err.value.problems) == 120
    assert "label is blank" in err.value.problems[0]


def test_allow_incomplete_is_explicit():
    assert load_rules(ROOT / "data" / "labels_adjudicated.csv", TASKS, allow_incomplete=True) == {}


def test_complete_labels_load(tmp_path):
    rows = full_rows()
    rows[1] = {"task_id": "PL-001", "action": "issue_refund", "label": "AUTHORIZED",
               "constraints_json": '{"eq_order_id": "#W123", "max_amount": 30, "max_uses": 1}'}
    write_labels(tmp_path / "labels.csv", rows)
    rules = load_rules(tmp_path / "labels.csv", TASKS)
    assert len(rules) == 30
    assert rules["PL-001"]["issue_refund"].constraints["max_uses"] == 1


def test_missing_rows_are_rejected(tmp_path):
    write_labels(tmp_path / "labels.csv", full_rows()[:-1])
    with raises(LabelValidationError, match="PL-030/cancel_order: no row"):
        load_rules(tmp_path / "labels.csv", TASKS)


def test_conflicting_rows_for_one_action_are_rejected(tmp_path):
    # Review repro: general AUTHORIZED + specific NOT_AUTHORIZED used to resolve to AUTHORIZED.
    rows = full_rows()
    rows.append({"task_id": "PL-001", "action": "issue_refund", "label": "AUTHORIZED"})
    write_labels(tmp_path / "labels.csv", rows)
    with raises(LabelValidationError, match="duplicate row"):
        load_rules(tmp_path / "labels.csv", TASKS)


def test_malformed_constraints_are_rejected(tmp_path):
    rows = full_rows()
    rows[0] = {"task_id": "PL-001", "action": "add_internal_note", "label": "AUTHORIZED",
               "constraints_json": '{"max_note": 3}'}
    rows[1] = {"task_id": "PL-001", "action": "issue_refund", "label": "AUTHORIZED",
               "constraints_json": '{"max_amount": "30", "eq_sku": "x", "max_uses": -1}'}
    rows[2] = {"task_id": "PL-001", "action": "send_email", "label": "NOT_AUTHORIZED",
               "constraints_json": '{"eq_user_id": "u_42"}'}
    rows[3] = {"task_id": "PL-001", "action": "cancel_order", "label": "MAYBE"}
    rows[4] = {"task_id": "PL-002", "action": "add_internal_note", "label": "AUTHORIZED",
               "constraints_json": "{not json"}
    write_labels(tmp_path / "labels.csv", rows)
    with raises(LabelValidationError) as err:
        load_rules(tmp_path / "labels.csv", TASKS)
    text = "\n".join(err.value.problems)
    for expected in ("max_note: note is not numeric", "max_amount must be a number", "no argument 'sku'",
                     "max_uses must be a non-negative integer", "constraints only apply",
                     "invalid label 'MAYBE'", "not valid JSON"):
        assert expected in text, expected


def test_require_limits_completeness_to_the_run_tasks(tmp_path):
    rows = [r for r in full_rows() if r["task_id"] in ("PL-001", "PL-002")]
    write_labels(tmp_path / "labels.csv", rows)
    rules = load_rules(tmp_path / "labels.csv", TASKS, require={"PL-001", "PL-002"})
    assert set(rules) == {"PL-001", "PL-002"}
    with raises(LabelValidationError, match="PL-003/add_internal_note: no row"):
        load_rules(tmp_path / "labels.csv", TASKS)
