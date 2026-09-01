from __future__ import annotations

import math
import uuid
from datetime import timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from album_memory.config import MemoryConfig
from album_memory.contracts import MaintenanceItem, MaintenanceReport
from album_memory.models import (
    ClaimEvidence,
    Event,
    MemoryItem,
    ObservationFact,
    ProfileClaim,
    User,
    UserConfirmation,
)
from album_memory.text import utcnow


class MaintenanceService:
    def __init__(self, config: MemoryConfig):
        self.config = config

    def run(
        self,
        session: Session,
        *,
        user_id: uuid.UUID | None = None,
        dry_run: bool = True,
    ) -> MaintenanceReport:
        now = utcnow()
        query = select(MemoryItem)
        if user_id is not None:
            query = query.where(MemoryItem.user_id == user_id)
        memories = list(session.scalars(query))
        protected = self._protected_memory_ids(session, user_id)
        report = MaintenanceReport(dry_run=dry_run, scanned_memories=len(memories))

        for memory in memories:
            if memory.archived or memory.memory_id in protected:
                continue
            strength, idle_days = self._strength(memory, now)
            if not dry_run:
                memory.strength = strength
            if (
                strength < self.config.maintenance.strength_threshold
                and idle_days >= self.config.maintenance.idle_days
            ):
                item = MaintenanceItem(
                    object_type="memory",
                    object_id=memory.memory_id,
                    action="archive",
                    reason=f"low strength and idle for {idle_days:.1f} days",
                    strength=strength,
                    applied=not dry_run,
                )
                report.items.append(item)
                if not dry_run:
                    memory.archived = True
                    memory.archive_reason = "maintenance_low_strength"

        expired_query = select(ProfileClaim).where(
            ProfileClaim.horizon == "short",
            ProfileClaim.status == "active",
            ProfileClaim.expires_at.is_not(None),
            ProfileClaim.expires_at <= now,
        )
        if user_id is not None:
            expired_query = expired_query.where(ProfileClaim.user_id == user_id)
        for claim in session.scalars(expired_query):
            report.items.append(
                MaintenanceItem(
                    object_type="claim",
                    object_id=claim.claim_id,
                    action="end",
                    reason="short-term claim expired and requires reevaluation",
                    applied=not dry_run,
                )
            )
            if not dry_run:
                claim.status = "ended"
                claim.archived = True
                claim.valid_to = claim.expires_at

        withdrawn_query = select(User).where(User.consent_state == "withdrawn")
        if user_id is not None:
            withdrawn_query = withdrawn_query.where(User.user_id == user_id)
        for user in session.scalars(withdrawn_query):
            report.items.append(
                MaintenanceItem(
                    object_type="user",
                    object_id=user.user_id,
                    action="freeze",
                    reason="consent withdrawn; online recall is disabled",
                    applied=not dry_run,
                )
            )
            if not dry_run:
                for memory in session.scalars(
                    select(MemoryItem).where(
                        MemoryItem.user_id == user.user_id,
                        MemoryItem.archived.is_(False),
                    )
                ):
                    memory.archived = True
                    memory.archive_reason = "consent_withdrawn"
                for claim in session.scalars(
                    select(ProfileClaim).where(
                        ProfileClaim.user_id == user.user_id,
                        ProfileClaim.archived.is_(False),
                    )
                ):
                    claim.archived = True
        return report

    def _strength(self, memory: MemoryItem, now) -> tuple[float, float]:
        reference = (
            memory.last_reinforced_at
            or memory.last_injected_at
            or memory.created_at
        )
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        idle_days = max(0.0, (now - reference).total_seconds() / 86400)
        decay = math.exp(-self.config.maintenance.decay_lambda * idle_days)
        frequency = 1.0 + self.config.maintenance.frequency_eta * math.log1p(
            memory.injection_count
        )
        evidence_count = len(
            (memory.structured_payload or {}).get("fact_ids", [])
        )
        evidence_factor = min(1.0, 0.7 + 0.1 * evidence_count)
        strength = min(
            1.0,
            max(0.0, memory.source_reliability * decay * frequency * evidence_factor),
        )
        return round(strength, 6), idle_days

    @staticmethod
    def _protected_memory_ids(
        session: Session,
        user_id: uuid.UUID | None,
    ) -> set[uuid.UUID]:
        claim_query = (
            select(ClaimEvidence.evidence_id)
            .join(ProfileClaim, ProfileClaim.claim_id == ClaimEvidence.claim_id)
            .where(
                ClaimEvidence.evidence_type == "memory_item",
                ClaimEvidence.role == "support",
                ProfileClaim.status == "active",
                ProfileClaim.archived.is_(False),
            )
        )
        if user_id is not None:
            claim_query = claim_query.where(ProfileClaim.user_id == user_id)
        protected = set(session.scalars(claim_query))

        fact_query = (
            select(MemoryItem.memory_id)
            .join(
                ObservationFact,
                ObservationFact.observation_id == MemoryItem.observation_id,
            )
            .join(
                ClaimEvidence,
                ClaimEvidence.evidence_id == ObservationFact.fact_id,
            )
            .join(ProfileClaim, ProfileClaim.claim_id == ClaimEvidence.claim_id)
            .where(
                ClaimEvidence.evidence_type == "observation_fact",
                ClaimEvidence.role == "support",
                ProfileClaim.status == "active",
                ProfileClaim.archived.is_(False),
                MemoryItem.archived.is_(False),
            )
        )
        if user_id is not None:
            fact_query = fact_query.where(MemoryItem.user_id == user_id)
        protected.update(session.scalars(fact_query))

        l3_query = (
            select(MemoryItem.memory_id)
            .join(Event, Event.event_id == MemoryItem.event_id)
            .where(
                MemoryItem.grain == "L3",
                Event.status == "active",
                MemoryItem.archived.is_(False),
            )
        )
        if user_id is not None:
            l3_query = l3_query.where(MemoryItem.user_id == user_id)
        protected.update(session.scalars(l3_query))

        confirmation_query = select(UserConfirmation.target_id).where(
            UserConfirmation.target_type == "memory_item",
            UserConfirmation.action == "confirm",
        )
        if user_id is not None:
            confirmation_query = confirmation_query.where(
                UserConfirmation.user_id == user_id
            )
        protected.update(session.scalars(confirmation_query))
        return protected
