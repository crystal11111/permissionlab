import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("generate_study_assets", ROOT / "scripts" / "generate_study_assets.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def test_refuses_to_overwrite_without_reset(tmp_path):
    assert gen.main([], data_dir=tmp_path) == 0
    (tmp_path / "tasks.jsonl").write_text("edited\n")
    assert gen.main([], data_dir=tmp_path) == 1
    assert (tmp_path / "tasks.jsonl").read_text() == "edited\n"


def test_reset_never_discards_filled_labels(tmp_path):
    # Review repro: rerunning the generator used to blank the adjudicated labels.
    gen.main([], data_dir=tmp_path)
    path = tmp_path / "labels_adjudicated.csv"
    rows = list(csv.DictReader(path.open()))
    rows[0]["label"] = "NOT_AUTHORIZED"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    before = path.read_text()
    (tmp_path / "tasks.jsonl").write_text("edited\n")
    assert gen.main(["--reset"], data_dir=tmp_path) == 1
    assert path.read_text() == before
    assert (tmp_path / "tasks.jsonl").read_text() == "edited\n"  # all-or-nothing


def test_reset_regenerates_blank_files(tmp_path):
    gen.main([], data_dir=tmp_path)
    (tmp_path / "labels_A.csv").write_text("task_id,label\n")
    assert gen.main(["--reset"], data_dir=tmp_path) == 0
    assert len((tmp_path / "labels_A.csv").read_text().splitlines()) == 121


def test_tasks_reference_the_customers_own_order():
    for row in gen.build_tasks():
        ctx = row["context"]
        expected_owner = "u_77" if ctx["current_order_id"] == "#W777" else "u_42"
        assert ctx["current_user_id"] == expected_owner, row["id"]


def test_explicit_tasks_have_expected_effects():
    rows = gen.build_tasks()
    assert all(bool(r.get("expected_effects")) == (r["category"] == "explicit_execute") for r in rows)
    assert json.loads((ROOT / "data" / "tasks.jsonl").read_text().splitlines()[25])["expected_effects"]
