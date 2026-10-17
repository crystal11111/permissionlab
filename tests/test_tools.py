from permissionlab.db import RetailDB
from permissionlab.tools import (IdempotencyConflict, InvalidToolCall, LABELED_ACTIONS, RetailTools,
                                 canonical_idempotency_key, openai_tool_schemas)
from tests.helpers import raises


def test_idempotent_refund_is_one_logical_action():
    db = RetailDB.fresh()
    tools = RetailTools(db)
    args = {"user_id": "u_42", "order_id": "#W123", "amount": 30}
    key = canonical_idempotency_key("t1", "grant-1")
    first = tools.execute("issue_refund", args, idempotency_key=key)
    second = tools.execute("issue_refund", args, idempotency_key=key)
    assert first == second
    assert db.state["orders"]["#W123"]["refund_total"] == 30
    assert len(db.state["refunds"]) == 1


def test_reusing_a_key_for_a_different_action_is_rejected():
    db = RetailDB.fresh()
    tools = RetailTools(db)
    key = canonical_idempotency_key("t1", "grant-1")
    tools.execute("issue_refund", {"user_id": "u_42", "order_id": "#W123", "amount": 30}, idempotency_key=key)
    with raises(IdempotencyConflict):
        tools.execute("cancel_order", {"order_id": "#W124"}, idempotency_key=key)
    assert db.state["orders"]["#W124"]["status"] == "pending"  # not cancelled, not a stale refund result


def test_reusing_a_key_with_a_different_amount_is_rejected():
    tools = RetailTools(RetailDB.fresh())
    key = canonical_idempotency_key("t1", "grant-1")
    tools.execute("issue_refund", {"user_id": "u_42", "order_id": "#W123", "amount": 30}, idempotency_key=key)
    with raises(IdempotencyConflict):
        tools.execute("issue_refund", {"user_id": "u_42", "order_id": "#W123", "amount": 90}, idempotency_key=key)


def test_failed_call_does_not_burn_its_key():
    db = RetailDB.fresh()
    tools = RetailTools(db)
    key = canonical_idempotency_key("t1", "grant-1")
    with raises(Exception):
        tools.execute("cancel_order", {"order_id": "#W123"}, idempotency_key=key)  # delivered: refused
    db.state["orders"]["#W123"]["status"] = "pending"
    assert tools.execute("cancel_order", {"order_id": "#W123"}, idempotency_key=key)["status"] == "cancelled"


def test_malformed_arguments_are_rejected_before_running():
    tools = RetailTools(RetailDB.fresh())
    with raises(InvalidToolCall, match="amount must be a number"):
        tools.execute("issue_refund", {"user_id": "u_42", "order_id": "#W123", "amount": "30"})
    with raises(InvalidToolCall, match="unexpected reason"):
        tools.execute("cancel_order", {"order_id": "#W124", "reason": "x"})
    with raises(InvalidToolCall, match="Unknown tool"):
        tools.execute("delete_customer", {"user_id": "u_42"})


def test_labeled_actions_and_schemas_cover_the_state_changing_tools():
    assert LABELED_ACTIONS == ("add_internal_note", "issue_refund", "send_email", "cancel_order")
    assert len(openai_tool_schemas()) == 8
