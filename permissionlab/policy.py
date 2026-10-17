"""Organization consequence policy, kept separate from task authority."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .tools import TOOL_SPECS

TIERS = ("auto_allow", "compiled_boundary", "require_confirmation", "never_allow")
DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[1] / "configs" / "consequence_policy.toml"


@dataclass(frozen=True)
class ConsequencePolicy:
    tiers: dict[str, str]  # action -> tier

    def tier(self, action: str) -> str:
        return self.tiers.get(action, "unclassified")


def load_policy(path: str | Path = DEFAULT_POLICY_PATH) -> ConsequencePolicy:
    with Path(path).open("rb") as f:
        section = tomllib.load(f).get("consequence_policy", {})
    tiers: dict[str, str] = {}
    problems = []
    for tier, actions in section.items():
        if tier not in TIERS:
            problems.append(f"unknown tier {tier!r}")
            continue
        for action in actions:
            if action in tiers:
                problems.append(f"{action} is listed in both {tiers[action]} and {tier}")
            if action not in TOOL_SPECS and tier != "never_allow":
                problems.append(f"{tier}: unknown tool {action!r}")
            tiers[action] = tier
    unclassified = sorted(set(TOOL_SPECS) - set(tiers))
    if unclassified:
        problems.append("tools missing from the policy: " + ", ".join(unclassified))
    if problems:
        raise ValueError("invalid consequence policy:\n  " + "\n  ".join(problems))
    return ConsequencePolicy(tiers)
