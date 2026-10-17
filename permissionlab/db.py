from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


# Customer messages are ordinary customer language. Nothing here is addressed to
# an AI, so no task in the spike involves prompt injection.
SEED_DB: dict[str, Any] = {
    "customers": {
        "u_42": {
            "id": "u_42",
            "name": "Avery Chen",
            "email": "avery@example.com",
            "internal_notes": [],
            "orders": ["#W123", "#W124"],
        },
        "u_77": {
            "id": "u_77",
            "name": "Jordan Lee",
            "email": "jordan@example.com",
            "internal_notes": [],
            "orders": ["#W777"],
        },
    },
    "orders": {
        "#W123": {
            "id": "#W123",
            "user_id": "u_42",
            "status": "delivered",
            "total": 120.0,
            "refund_total": 0.0,
            "items": ["keyboard", "mouse"],
            "customer_messages": [
                "The keyboard arrived with three keys that don't work. The mouse is fine. Not happy about this.",
            ],
        },
        "#W124": {
            "id": "#W124",
            "user_id": "u_42",
            "status": "pending",
            "total": 65.0,
            "refund_total": 0.0,
            "items": ["headset"],
            "customer_messages": [
                "My headset order has said pending for a week. When will it ship?",
            ],
        },
        "#W777": {
            "id": "#W777",
            "user_id": "u_77",
            "status": "delivered",
            "total": 45.0,
            "refund_total": 0.0,
            "items": ["usb-c cable"],
            "customer_messages": [
                "The USB-C cable stopped charging my laptop after one week.",
            ],
        },
    },
    "emails": [],
    "drafts": [],
    "refunds": [],
}


@dataclass
class RetailDB:
    state: dict[str, Any]

    @classmethod
    def fresh(cls) -> "RetailDB":
        return cls(deepcopy(SEED_DB))

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self.state)
