from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from album_memory.config import MemoryConfig
from album_memory.conflict import prepare_claim_version
from album_memory.contracts import ClaimView, ProfileSnapshot
from album_memory.enums import Horizon, RecordStatus, ReviewState
from album_memory.errors import ClaimTransitionError
from album_memory.models import (
    ClaimEvidence,
    Event,
    EventAsset,
    ImageObservation,
    MediaAsset,
    MemoryItem,
    ObservationFact,
    ProfileClaim,
    UserConfirmation,
)
from album_memory.policies import DIMENSION_POLICIES, evaluate_gate, statement_is_safe
from album_memory.providers import ChatProvider
from album_memory.text import normalize_fact_value, source_reliability, utcnow


class ProfileEngine:
    def __init__(self, config: MemoryConfig, provider: ChatProvider):
        self.config = config
        self.provider = provider

    def refresh_for_observation(
        self,
        session: Session,
        *,
        user_id: uuid.UUID,
        observation: ImageObservation,
        event: Event,
        research_run_id: uuid.UUID,
        refresh_long_term: bool,
    ) -> None:
        now = utcnow()
        self._expire_short_claims(session, user_id, now)
        safety = observation.safety_json or {}
        if (
            observation.is_sensitive
            or safety.get("blocked_from_profile", False)
            or safety.get("contains_minor") is True
        ):
            return
        for window_days in (7, 30):
            self._refresh_horizon(
                session,
                user_id=user_id,
                horizon=Horizon.SHORT,
                cutoff=now - timedelta(days=window_days),
                until=now,
                research_run_id=research_run_id,
                window_days=window_days,
            )
        if refresh_long_term:
            self._refresh_horizon(
                session,
                user_id=user_id,
                horizon=Horizon.LONG,
                cutoff=None,
                until=now,
                research_run_id=research_run_id,
                window_days=None,
            )
            self._refresh_visual_style(
                session,
                user_id=user_id,
                research_run_id=research_run_id,
                now=now,
            )

    def _refresh_horizon(
        self,
        session: Session,
        *,
        user_id: uuid.UUID,
        horizon: Horizon,
        cutoff,
        until,
        research_run_id: uuid.UUID,
        window_days: int | None,
    ) -> None:
        query = (
            select(ObservationFact, ImageObservation, MediaAsset, Event)
            .join(ImageObservation, ImageObservation.observation_id == ObservationFact.observation_id)
            .join(MediaAsset, MediaAsset.asset_id == ImageObservation.asset_id)
            .outerjoin(EventAsset, EventAsset.asset_id == MediaAsset.asset_id)
            .outerjoin(Event, Event.event_id == EventAsset.event_id)
            .where(
                ObservationFact.user_id == user_id,
                ObservationFact.is_sensitive.is_(False),
                ObservationFact.status == "active",
                ImageObservation.current.is_(True),
                MediaAsset.deleted_at.is_(None),
            )
        )
        if cutoff is not None:
            query = query.where(MediaAsset.captured_at >= cutoff)
        rows = list(session.execute(query))
        policies = [
            policy
            for policy in DIMENSION_POLICIES.values()
            if policy.horizon == horizon
        ]
        for policy in policies:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for fact, observation, asset, event in rows:
                if fact.predicate not in policy.allowed_signals:
                    continue
                key = normalize_fact_value(fact.value_json)
                grouped[key].append(
                    {
                        "fact": fact,
                        "event_id": event.event_id if event and event.status == "active" else None,
                        "at": asset.captured_at,
                        "source_kind": asset.source_kind,
                        "source_reliability": source_reliability(
                            asset.source_kind,
                            {fact.evidence_type},
                            observation.user_presence,
                            imported=asset.source_kind == "imported",
                        ),
                        "evidence_type": fact.evidence_type,
                        "user_presence": observation.user_presence,
                        "confidence": fact.confidence,
                        "value": fact.value_json,
                    }
                )
            for value_key, evidence in grouped.items():
                self._write_claim(
                    session,
                    user_id=user_id,
                    policy=policy,
                    value_key=value_key,
                    evidence=evidence,
                    valid_from=cutoff,
                    valid_to=until,
                    expires_at=(
                        until + timedelta(days=2) if horizon == Horizon.SHORT else None
                    ),
                    research_run_id=research_run_id,
                    window_days=window_days,
                )

    def _write_claim(
        self,
        session: Session,
        *,
        user_id: uuid.UUID,
        policy,
        value_key: str,
        evidence: list[dict[str, Any]],
        valid_from,
        valid_to,
        expires_at,
        research_run_id: uuid.UUID,
        window_days: int | None,
    ) -> None:
        gate = evaluate_gate(policy, evidence)
        if any(item.get("source_kind") == "imported" for item in evidence):
            gate["violations"].append("legacy_import_requires_canonical_evidence")
            gate["passed"] = False
        statement = self._statement(policy.dimension_id, value_key, window_days)
        safe = statement_is_safe(statement)
        if not safe:
            gate["violations"].append("unsafe_statement")
            gate["passed"] = False
        avg_confidence = sum(item["confidence"] for item in evidence) / max(1, len(evidence))
        event_factor = min(
            1.0,
            gate["independent_event_count"] / max(1, policy.min_independent_events),
        )
        confidence = min(
            policy.max_auto_confidence,
            max(0.1, 0.55 * avg_confidence + 0.45 * event_factor),
        )
        status = RecordStatus.ACTIVE if gate["passed"] else RecordStatus.CANDIDATE
        review = ReviewState.AUTO_PASSED if gate["passed"] else ReviewState.NEEDS_REVIEW
        logical_key = (
            f"{policy.dimension_id}:{window_days or 'long'}:{value_key[:120]}"
        )
        candidate = ProfileClaim(
            user_id=user_id,
            dimension_id=policy.dimension_id,
            horizon=policy.horizon.value,
            logical_key=logical_key,
            status=status.value,
            statement=statement,
            value_json={
                "signal": value_key,
                "window_days": window_days,
                "independent_event_count": gate["independent_event_count"],
            },
            confidence=round(confidence, 3),
            source_type="album",
            gate_check_json=gate,
            valid_from=valid_from,
            valid_to=valid_to,
            activated_at=utcnow() if status == RecordStatus.ACTIVE else None,
            expires_at=expires_at,
            next_review_at=expires_at if horizon == Horizon.SHORT else until + timedelta(days=90),
            review_state=review.value,
            research_run_id=research_run_id,
            archived=False,
        )
        prepared = prepare_claim_version(session, candidate)
        target = prepared
        if prepared is candidate:
            session.add(candidate)
            session.flush()
        for item in evidence:
            fact = item["fact"]
            self._add_claim_evidence_if_missing(
                session,
                ClaimEvidence(
                    user_id=user_id,
                    claim_id=target.claim_id,
                    evidence_type="observation_fact",
                    evidence_id=fact.fact_id,
                    role="support",
                    evidence_path="value_json",
                    rationale=f"{fact.predicate} directly supports this bounded claim",
                    weight=min(1.0, fact.confidence),
                    source_confidence_snapshot=fact.confidence,
                ),
            )

    def _refresh_visual_style(
        self,
        session: Session,
        *,
        user_id: uuid.UUID,
        research_run_id: uuid.UUID,
        now,
    ) -> None:
        policy = DIMENSION_POLICIES["L10"]
        rows = list(
            session.execute(
                select(MemoryItem, MediaAsset, Event)
                .join(MediaAsset, MediaAsset.asset_id == MemoryItem.asset_id)
                .outerjoin(EventAsset, EventAsset.asset_id == MediaAsset.asset_id)
                .outerjoin(Event, Event.event_id == EventAsset.event_id)
                .where(
                    MemoryItem.user_id == user_id,
                    MemoryItem.grain == "L2",
                    MemoryItem.archived.is_(False),
                    MediaAsset.source_kind == "camera",
                    MediaAsset.captured_at.is_not(None),
                )
            )
        )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for memory, asset, event in rows:
            style = (memory.structured_payload or {}).get("visual_style") or {}
            for key in ("composition", "lighting", "colors", "editing_signals"):
                for value in style.get(key, []) or []:
                    grouped[f"{key}:{value}"].append(
                        {
                            "memory": memory,
                            "event_id": event.event_id if event and event.status == "active" else None,
                            "at": asset.captured_at,
                            "source_kind": asset.source_kind,
                            "source_reliability": memory.source_reliability,
                            "evidence_type": "memory_item",
                            "user_presence": "unknown",
                            "confidence": memory.source_reliability,
                        }
                    )
        for value, evidence in grouped.items():
            gate = evaluate_gate(policy, evidence)
            if not gate["passed"]:
                continue
            logical_key = f"L10:long:{value[:120]}"
            statement = f"用户相机拍摄记录中重复出现“{value}”的视觉风格；该结论不代表产品或活动偏好。"
            candidate = ProfileClaim(
                user_id=user_id,
                dimension_id="L10",
                horizon="long",
                logical_key=logical_key,
                status="active",
                statement=statement,
                value_json={"visual_style": value},
                confidence=min(0.9, 0.6 + 0.05 * len(evidence)),
                source_type="album",
                gate_check_json=gate,
                valid_from=min(item["at"] for item in evidence),
                valid_to=max(item["at"] for item in evidence),
                activated_at=now,
                next_review_at=now + timedelta(days=90),
                review_state="auto_passed",
                research_run_id=research_run_id,
                archived=False,
            )
            prepared = prepare_claim_version(session, candidate)
            target = prepared
            if prepared is candidate:
                session.add(candidate)
                session.flush()
            for item in evidence:
                memory = item["memory"]
                self._add_claim_evidence_if_missing(
                    session,
                    ClaimEvidence(
                        user_id=user_id,
                        claim_id=target.claim_id,
                        evidence_type="memory_item",
                        evidence_id=memory.memory_id,
                        role="support",
                        evidence_path="structured_payload.visual_style",
                        rationale="camera-session visual style evidence",
                        weight=memory.source_reliability,
                        source_confidence_snapshot=memory.source_reliability,
                    ),
                )

    @staticmethod
    def _add_claim_evidence_if_missing(
        session: Session,
        evidence: ClaimEvidence,
    ) -> None:
        exists = session.scalar(
            select(ClaimEvidence.claim_evidence_id).where(
                ClaimEvidence.claim_id == evidence.claim_id,
                ClaimEvidence.evidence_type == evidence.evidence_type,
                ClaimEvidence.evidence_id == evidence.evidence_id,
                ClaimEvidence.role == evidence.role,
            )
        )
        if exists is None:
            session.add(evidence)

    @staticmethod
    def _statement(dimension_id: str, signal: str, window_days: int | None) -> str:
        scope = f"最近{window_days}天" if window_days else "跨时间的相册记录"
        templates = {
            "S1": f"{scope}出现了“{signal}”的过程状态线索；仅表示可见进展证据。",
            "S2": f"{scope}在多个独立事件中出现“{signal}”相关动作。",
            "S3": f"{scope}多次记录了“{signal}”相关内容；关注不等于喜欢、拥有或到访。",
            "S4": f"{scope}重复出现“{signal}”人物或群组节点；关系类型和亲密度未知。",
            "S5": f"{scope}在可信粗粒度记录中多次出现“{signal}”区域线索；不代表住所。",
            "S6": f"{scope}已确认活动在“{signal}”时间范围内重复出现；不推断作息或工作制。",
            "S7": f"{scope}出现“{signal}”可观察表达线索；不代表心理状态或诊断。",
            "L1": f"经明确确认或可靠文字证据记录角色“{signal}”。",
            "L2": f"跨事件重复识别到实体关联“{signal}”；不自动认定所有权。",
            "L3": f"跨事件重复出现人物或群组节点“{signal}”；关系名称未知。",
            "L4": f"跨多个日历阶段重复出现粗粒度区域“{signal}”；不代表住所。",
            "L5": f"跨周期的已确认活动在“{signal}”时间范围重复出现；不推断睡眠或工作制度。",
            "L6": f"跨时间、跨独立事件重复出现“{signal}”动作；频繁发生不等于喜欢或擅长。",
            "L7": f"多次出现“{signal}”步骤或成果证据；只表达证据化能力线索。",
            "L8": f"跨时间、跨来源模式重复记录“{signal}”主题；记录主题不等于具体偏好。",
            "L9": f"在可比较选项中重复记录选择属性“{signal}”；不推断预算或消费能力。",
        }
        return templates.get(dimension_id, f"{scope}记录到“{signal}”证据。")[:300]

    @staticmethod
    def _expire_short_claims(session: Session, user_id: uuid.UUID, now) -> None:
        claims = list(
            session.scalars(
                select(ProfileClaim).where(
                    ProfileClaim.user_id == user_id,
                    ProfileClaim.horizon == "short",
                    ProfileClaim.status == "active",
                    ProfileClaim.expires_at.is_not(None),
                    ProfileClaim.expires_at <= now,
                )
            )
        )
        for claim in claims:
            claim.status = "ended"
            claim.valid_to = claim.expires_at
            claim.archived = True

    def get_snapshot(
        self,
        session: Session,
        *,
        user_id: uuid.UUID,
        horizon: Horizon | None = None,
        dimensions: set[str] | None = None,
    ) -> ProfileSnapshot:
        query = select(ProfileClaim).where(
            ProfileClaim.user_id == user_id,
            ProfileClaim.archived.is_(False),
        )
        if horizon is not None:
            query = query.where(ProfileClaim.horizon == horizon.value)
        if dimensions:
            query = query.where(ProfileClaim.dimension_id.in_(dimensions))
        claims = list(
            session.scalars(
                query.order_by(
                    ProfileClaim.horizon,
                    ProfileClaim.dimension_id,
                    ProfileClaim.confidence.desc(),
                )
            )
        )
        views = [self.claim_view(session, claim) for claim in claims]
        return ProfileSnapshot(
            user_id=user_id,
            generated_at=utcnow(),
            short_term=[view for view in views if view.horizon == Horizon.SHORT],
            long_term=[view for view in views if view.horizon == Horizon.LONG],
        )

    def confirm_claim(
        self,
        session: Session,
        *,
        user_id: uuid.UUID,
        claim_id: uuid.UUID,
        note: str | None = None,
    ) -> ClaimView:
        claim = session.get(ProfileClaim, claim_id)
        if claim is None or claim.user_id != user_id:
            raise ClaimTransitionError("claim not found")
        if not statement_is_safe(claim.statement):
            raise ClaimTransitionError("unsafe claims cannot be confirmed into the profile")
        confirmation = UserConfirmation(
            user_id=user_id,
            target_type="profile_claim",
            target_id=claim_id,
            action="confirm",
            value_json={"statement": claim.statement},
            note=note,
        )
        session.add(confirmation)
        session.flush()
        competing = list(
            session.scalars(
                select(ProfileClaim).where(
                    ProfileClaim.user_id == user_id,
                    ProfileClaim.logical_key == claim.logical_key,
                    ProfileClaim.status == "active",
                    ProfileClaim.claim_id != claim.claim_id,
                )
            )
        )
        for item in competing:
            item.status = "ended"
            item.archived = True
        claim.status = "active"
        claim.review_state = "human_confirmed"
        claim.next_review_at = None if claim.horizon == "long" else claim.expires_at
        claim.activated_at = claim.activated_at or utcnow()
        claim.confidence = max(claim.confidence, 0.95)
        session.add(
            ClaimEvidence(
                user_id=user_id,
                claim_id=claim.claim_id,
                evidence_type="user_confirmation",
                evidence_id=confirmation.confirmation_id,
                role="support",
                rationale="user explicitly confirmed this bounded claim",
                weight=1.0,
                source_confidence_snapshot=1.0,
            )
        )
        session.flush()
        return self.claim_view(session, claim)

    def correct_claim(
        self,
        session: Session,
        *,
        user_id: uuid.UUID,
        claim_id: uuid.UUID,
        statement: str,
        value: dict[str, Any],
        note: str | None = None,
    ) -> ClaimView:
        old = session.get(ProfileClaim, claim_id)
        if old is None or old.user_id != user_id:
            raise ClaimTransitionError("claim not found")
        if not statement_is_safe(statement):
            raise ClaimTransitionError("correction contains a prohibited inference")
        confirmation = UserConfirmation(
            user_id=user_id,
            target_type="profile_claim",
            target_id=claim_id,
            action="correct",
            value_json={"statement": statement, "value": value},
            note=note,
        )
        session.add(confirmation)
        session.flush()
        old.status = "ended"
        old.archived = True
        old.valid_to = utcnow()
        corrected = ProfileClaim(
            user_id=user_id,
            dimension_id=old.dimension_id,
            horizon=old.horizon,
            logical_key=old.logical_key,
            status="active",
            statement=statement[:300],
            value_json=value,
            confidence=1.0,
            source_type="mixed",
            gate_check_json={
                "passed": True,
                "user_confirmed": True,
                "violations": [],
            },
            valid_from=utcnow(),
            activated_at=utcnow(),
            valid_to=old.valid_to,
            expires_at=old.expires_at,
            review_state="human_confirmed",
            next_review_at=None if old.horizon == "long" else old.expires_at,
            version=old.version + 1,
            supersedes_claim_id=old.claim_id,
            resolution_reason="user correction",
            archived=False,
        )
        session.add(corrected)
        session.flush()
        session.add(
            ClaimEvidence(
                user_id=user_id,
                claim_id=corrected.claim_id,
                evidence_type="user_confirmation",
                evidence_id=confirmation.confirmation_id,
                role="support",
                rationale="user supplied a correction",
                weight=1.0,
                source_confidence_snapshot=1.0,
            )
        )
        return self.claim_view(session, corrected)

    @staticmethod
    def claim_view(session: Session, claim: ProfileClaim) -> ClaimView:
        evidence = list(
            session.scalars(
                select(ClaimEvidence).where(ClaimEvidence.claim_id == claim.claim_id)
            )
        )
        return ClaimView(
            claim_id=claim.claim_id,
            dimension_id=claim.dimension_id,
            horizon=Horizon(claim.horizon),
            status=RecordStatus(claim.status),
            statement=claim.statement,
            value=claim.value_json,
            confidence=claim.confidence,
            valid_from=claim.valid_from,
            valid_to=claim.valid_to,
            expires_at=claim.expires_at,
            next_review_at=claim.next_review_at,
            review_state=ReviewState(claim.review_state),
            supersedes_claim_id=claim.supersedes_claim_id,
            resolution_reason=claim.resolution_reason,
            evidence_ids=[
                str(item.evidence_id) for item in evidence if item.role == "support"
            ],
            counter_evidence_ids=[
                str(item.evidence_id) for item in evidence if item.role == "contradict"
            ],
        )

    @staticmethod
    def apply_conflict_outcomes(
        session: Session,
        *,
        user_id: uuid.UUID,
        outcomes: list[dict[str, Any]],
    ) -> None:
        """Attach counter-evidence and downgrade only claims touched by a conflict."""
        for outcome in outcomes:
            if (
                outcome.get("verdict") in {"coexistence", "evolution"}
                and outcome.get("resolver") != "llm_advisory"
            ):
                continue
            fact_ids = [uuid.UUID(value) for value in outcome.get("fact_ids", [])]
            if len(fact_ids) != 2:
                continue
            facts = {
                fact.fact_id: fact
                for fact in session.scalars(
                    select(ObservationFact).where(
                        ObservationFact.user_id == user_id,
                        ObservationFact.fact_id.in_(fact_ids),
                    )
                )
            }
            links = list(
                session.scalars(
                    select(ClaimEvidence).where(
                        ClaimEvidence.user_id == user_id,
                        ClaimEvidence.evidence_type == "observation_fact",
                        ClaimEvidence.evidence_id.in_(fact_ids),
                        ClaimEvidence.role == "support",
                    )
                )
            )
            for link in links:
                claim = session.get(ProfileClaim, link.claim_id)
                if claim is None or claim.archived:
                    continue
                linked_fact = facts.get(link.evidence_id)
                other_id = next(value for value in fact_ids if value != link.evidence_id)
                other = facts.get(other_id)
                if other is None:
                    continue
                exists = session.scalar(
                    select(ClaimEvidence.claim_evidence_id).where(
                        ClaimEvidence.claim_id == claim.claim_id,
                        ClaimEvidence.evidence_type == "observation_fact",
                        ClaimEvidence.evidence_id == other_id,
                        ClaimEvidence.role == "contradict",
                    )
                )
                if exists is None:
                    session.add(
                        ClaimEvidence(
                            user_id=user_id,
                            claim_id=claim.claim_id,
                            evidence_type="observation_fact",
                            evidence_id=other_id,
                            role="contradict",
                            evidence_path="value_json",
                            rationale="conflicting immutable fact retained for review",
                            weight=min(1.0, other.confidence),
                            source_confidence_snapshot=other.confidence,
                        )
                    )
                if claim.review_state == "human_confirmed":
                    pass
                elif linked_fact and linked_fact.status in {"superseded", "rejected"}:
                    claim.status = "ended" if linked_fact.status == "superseded" else "rejected"
                    claim.archived = True
                elif other.status in {"superseded", "rejected"}:
                    pass
                else:
                    claim.status = "candidate"
                    claim.review_state = "needs_review"
                claim.resolution_reason = (
                    "counter-evidence retained; user confirmation prevails"
                    if claim.review_state == "human_confirmed"
                    else (
                        "lower-priority counter-evidence retained but not activated"
                        if other.status in {"superseded", "rejected"}
                        else "counter-evidence requires rule or human resolution"
                    )
                )
