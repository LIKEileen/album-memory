from datetime import datetime, timedelta, timezone

from album_memory.policies import DIMENSION_POLICIES, evaluate_gate, statement_is_safe


def evidence(event_id: str, at: datetime, source: str = "camera") -> dict:
    return {
        "event_id": event_id,
        "at": at,
        "source_kind": source,
        "source_reliability": 0.8,
        "evidence_type": "pixel",
        "user_presence": "confirmed",
    }


def test_all_fixed_dimensions_exist():
    assert set(DIMENSION_POLICIES) == {
        "S1", "S2", "S3", "S4", "S5", "S6", "S7",
        "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10",
    }


def test_long_term_gate_requires_independent_events_and_span():
    now = datetime.now(timezone.utc)
    gate = evaluate_gate(
        DIMENSION_POLICIES["L6"],
        [
            evidence("event-1", now - timedelta(days=120)),
            evidence("event-2", now - timedelta(days=60)),
            evidence("event-3", now),
        ],
    )
    assert gate["passed"] is True


def test_sensitive_inference_is_blocked():
    assert statement_is_safe("用户职业是医生") is False
    assert statement_is_safe("多次照片中可见自行车") is True
