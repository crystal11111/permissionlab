"""Run the test suite without pytest (stdlib only). `python -m pytest` also works if installed."""
from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

passed, failed = 0, []
for path in sorted((ROOT / "tests").glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, fn in list(vars(module).items()):
        if not (name.startswith("test_") and inspect.isfunction(fn)):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = {"tmp_path": Path(tmp)} if "tmp_path" in inspect.signature(fn).parameters else {}
            try:
                fn(**kwargs)
                passed += 1
            except Exception:
                failed.append(f"{path.name}::{name}\n{traceback.format_exc()}")

for failure in failed:
    print("FAILED", failure)
print(f"{passed} passed, {len(failed)} failed")
sys.exit(1 if failed else 0)
