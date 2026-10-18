"""PermissionLab: measure authority drift in AI agents."""
import sys

if sys.version_info < (3, 11):
    raise SystemExit(
        f"PermissionLab needs Python 3.11 or newer; this is Python {sys.version.split()[0]}.\n"
        "Run with `python3` (check `python3 --version`), or with pyenv: `pyenv local 3.11.11`."
    )
