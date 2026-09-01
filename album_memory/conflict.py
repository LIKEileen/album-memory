from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from album_memory.models import (
    ImageObservation,
    MediaAsset,
    ObservationFact,
    ProfileClaim,
)
from album_memory.providers import ChatProvider
from album_memory.text import normalize_fact_value

SOURCE_PRIORITY = {
    "user_confirmation": 5,
    "exif": 4,
    "provided_metadata": 3,
    "ocr": 2,
    "pixel": 1,
}

TIME_VARYING_PREDICATES = {
    "coarse_location_observed",
    "coarse_location_provided",
    "visible_state",
    "state_change",
    "action_observed",
    "observable_expression",
}

COEXISTING_PREDICATES = {
    "contains_object",
    "visible_text",
    "person_present",
    "topic_observed",
}


def resolve_fact_conflicts(
    session: Session,
    *,
    user_id: uuid.UUID,
    observation_id: uuid.UUID,
    provider: ChatProvider,
) -> list[dict[str, Any]]:
    """Link conflicting facts without overwriting their immutable payload."""
    new_facts = list(
        session.scalars(
            select(ObservationFact).where(
                ObservationFact.user_id == user_id,
                ObservationFact.observation_id == observation_id,
            )
        )
    )
    outcomes = []
    for new in new_facts:
        prior = list(
            session.scalars(
                select(ObservationFact).where(
                    ObservationFact.user_id == user_id,
                    ObservationFact.canonical_key == new.canonical_key,
                    ObservationFact.observation_id != observation_id,
                    ObservationFact.status.in_(["active", "disputed"]),
                )
            )
        )
        for old in prior:
            if normalize_fact_value(old.value_json) == normalize_fact_value(new.value_json):
                continue
            if new.predicate in COEXISTING_PREDICATES:
                outcomes.append(
                    {"fact_ids": [str(old.fact_id), str(new.fact_id)], "verdict": "coexistence"}
                )
                continue

            old_priority = SOURCE_PRIORITY.get(old.evidence_type, 0)
            new_priority = SOURCE_PRIORITY.get(new.evidence_type, 0)
            if new.predicate in TIME_VARYING_PREDICATES:
                old_at = _trusted_captured_at(session, old.observation_id)
                new_at = _trusted_captured_at(session, new.observation_id)
                if old_at is not None and new_at is not None:
                    if new_at >= old_at:
                        old.status = "superseded"
                        new.supersedes_fact_id = old.fact_id
                    else:
                        new.status = "superseded"
                        old.supersedes_fact_id = new.fact_id
                    outcomes.append(
                        {
                            "fact_ids": [str(old.fact_id), str(new.fact_id)],
                            "verdict": "evolution",
                            "reason": "trusted capture times establish temporal order",
                        }
                    )
                    continue
            if new_priority > old_priority and new.confidence >= old.confidence:
                old.status = "superseded"
                new.supersedes_fact_id = old.fact_id
                outcomes.append(
                    {"fact_ids": [str(old.fact_id), str(new.fact_id)], "verdict": "conflict"}
                )
                continue
            if old_priority > new_priority and old.confidence >= new.confidence:
                new.status = "rejected"
                outcomes.append(
                    {"fact_ids": [str(old.fact_id), str(new.fact_id)], "verdict": "conflict"}
                )
                continue

            result = provider.classify_conflict(
                [
                    json.dumps(old.value_json, ensure_ascii=False),
                    json.dumps(new.value_json, ensure_ascii=False),
                ],
                f"predicate={new.predicate}; sources={old.evidence_type},{new.evidence_type}",
            )
            verdict = result.get("verdict", "unresolved")
            old.status = "disputed"
            new.status = "disputed"
            outcomes.append(
                {
                    "fact_ids": [str(old.fact_id), str(new.fact_id)],
                    "verdict": verdict,
                    "resolver": "llm_advisory",
                    "reason": result.get("reason", ""),
                }
            )
    return outcomes


def _trusted_captured_at(session: Session, observation_id: uuid.UUID):
    row = session.execute(
        select(MediaAsset.captured_at, MediaAsset.captured_time_confidence)
        .join(ImageObservation, ImageObservation.asset_id == MediaAsset.asset_id)
        .where(ImageObservation.observation_id == observation_id)
    ).one_or_none()
    if row is None or row.captured_time_confidence < 0.5:
        return None
    return row.captured_at


def prepare_claim_version(
    session: Session,
    candidate: ProfileClaim,
) -> ProfileClaim:
    """Apply deterministic version/supersession rules before inserting a claim."""
    existing = session.scalar(
        select(ProfileClaim)
        .where(
            ProfileClaim.user_id == candidate.user_id,
            ProfileClaim.logical_key == candidate.logical_key,
            ProfileClaim.archived.is_(False),
            ProfileClaim.status.in_(["active", "candidate"]),
        )
        .order_by(ProfileClaim.version.desc())
        .limit(1)
        .with_for_update()
    )
    if existing is None:
        candidate.version = 1
        return candidate

    same_payload = (
        existing.statement == candidate.statement
        and existing.value_json == candidate.value_json
        and existing.status == candidate.status
    )
    if same_payload:
        existing.confidence = max(existing.confidence, candidate.confidence)
        existing.gate_check_json = candidate.gate_check_json
        existing.valid_from = candidate.valid_from
        existing.valid_to = candidate.valid_to
        existing.expires_at = candidate.expires_at
        existing.next_review_at = candidate.next_review_at
        existing.research_run_id = candidate.research_run_id
        return existing

    candidate.version = existing.version + 1
    candidate.supersedes_claim_id = existing.claim_id

    if existing.review_state == "human_confirmed" and candidate.review_state != "human_confirmed":
        candidate.status = "candidate"
        return candidate

    existing.status = "ended"
    existing.valid_to = candidate.valid_from
    existing.archived = True
    return candidate
