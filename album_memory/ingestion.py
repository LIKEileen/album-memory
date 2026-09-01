from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from album_memory.config import MemoryConfig
from album_memory.contracts import AssetInput, ImageObservation, IngestResult, RegistrationResult
from album_memory.enums import ConsentState, JobStatus
from album_memory.errors import (
    ConsentRequiredError,
    IdempotencyConflictError,
    ObservationConflictError,
)
from album_memory.models import (
    ImageObservation as ObservationRow,
    MediaAsset,
    ObservationFact,
    ProcessingJob,
    User,
)
from album_memory.text import canonical_hash


class IngestionService:
    def __init__(self, config: MemoryConfig):
        self.config = config

    def register_user(
        self,
        session: Session,
        *,
        external_subject_key: str,
        display_name: str | None = None,
        timezone_name: str = "UTC",
        consent_state: ConsentState = ConsentState.PENDING,
        privacy_tier: int = 2,
        retention_until: datetime | None = None,
    ) -> RegistrationResult:
        user = session.scalar(
            select(User).where(User.external_subject_key == external_subject_key)
        )
        if user is None:
            user = User(
                external_subject_key=external_subject_key,
                display_name=display_name,
                timezone=timezone_name,
                consent_state=consent_state.value,
                profile_injection_enabled=self.config.privacy.default_profile_injection_enabled,
                privacy_tier=privacy_tier,
                retention_until=retention_until,
            )
            session.add(user)
            session.flush()
        else:
            if display_name is not None:
                user.display_name = display_name
            user.timezone = timezone_name
            user.consent_state = consent_state.value
            user.privacy_tier = privacy_tier
            if retention_until is not None:
                user.retention_until = retention_until
        return RegistrationResult(
            user_id=user.user_id,
            consent_state=ConsentState(user.consent_state),
        )

    def ingest(
        self,
        session: Session,
        *,
        user_id: uuid.UUID,
        observation: ImageObservation,
        asset: AssetInput | None = None,
        idempotency_key: str | None = None,
    ) -> IngestResult:
        user = session.get(User, user_id)
        if user is None or user.erased_at is not None:
            raise ConsentRequiredError("user does not exist or has been erased")
        if user.consent_state != ConsentState.GRANTED:
            raise ConsentRequiredError("image memory processing requires granted consent")

        payload = observation.model_dump(mode="json")
        content_hash = canonical_hash(payload)
        request_key = (
            f"ingest:{user_id}:{idempotency_key}"
            if idempotency_key
            else f"ingest:{user_id}:{observation.observation_id}"
        )
        prior_job = session.scalar(
            select(ProcessingJob).where(ProcessingJob.idempotency_key == request_key)
        )
        if prior_job is not None:
            if prior_job.payload_json.get("content_hash") != content_hash:
                raise IdempotencyConflictError(
                    "the idempotency key was already used with different content"
                )
            prior_observation_id = uuid.UUID(prior_job.payload_json["observation_id"])
            prior_observation = session.get(ObservationRow, prior_observation_id)
            return IngestResult(
                user_id=user_id,
                asset_id=prior_observation.asset_id,
                observation_id=prior_observation.observation_id,
                observation_version=prior_observation.version,
                job_id=prior_job.job_id,
                idempotent_replay=True,
            )

        existing_observation = session.scalar(
            select(ObservationRow).where(
                ObservationRow.user_id == user_id,
                ObservationRow.external_observation_id == observation.observation_id,
            )
        )
        if existing_observation is not None:
            if existing_observation.content_hash != content_hash:
                raise ObservationConflictError(
                    "external observation ID is immutable and already has different content"
                )
            existing_job = session.scalar(
                select(ProcessingJob).where(
                    ProcessingJob.payload_json["observation_id"].astext
                    == str(existing_observation.observation_id)
                )
            )
            if existing_job is None:
                existing_job = self._new_job(
                    user_id, request_key, existing_observation.observation_id, content_hash
                )
                session.add(existing_job)
                session.flush()
            return IngestResult(
                user_id=user_id,
                asset_id=existing_observation.asset_id,
                observation_id=existing_observation.observation_id,
                observation_version=existing_observation.version,
                job_id=existing_job.job_id,
                idempotent_replay=True,
            )

        asset_input = asset or self._asset_from_observation(observation)
        external_asset_id = asset_input.external_asset_id or observation.asset_id
        asset_row = session.scalar(
            select(MediaAsset).where(
                MediaAsset.user_id == user_id,
                MediaAsset.external_asset_id == external_asset_id,
            )
        )
        if asset_row is None:
            asset_row = MediaAsset(
                user_id=user_id,
                external_asset_id=external_asset_id,
                storage_uri=asset_input.storage_uri,
                sha256=asset_input.sha256,
                mime_type=asset_input.mime_type,
                byte_size=asset_input.byte_size,
                source_kind=asset_input.source_kind.value,
                captured_at=asset_input.captured_at,
                captured_time_confidence=asset_input.captured_time_confidence,
                location_coarse=(
                    asset_input.location_coarse.model_dump(mode="json")
                    if asset_input.location_coarse
                    else None
                ),
                width=asset_input.width,
                height=asset_input.height,
                ingest_status="registered",
            )
            session.add(asset_row)
            session.flush()
        else:
            if (
                asset_row.sha256
                and asset_input.sha256
                and asset_row.sha256.lower() != asset_input.sha256.lower()
            ):
                raise ObservationConflictError(
                    "external asset ID already points to different image bytes"
                )
            self._fill_missing_asset_fields(asset_row, asset_input)

        previous = session.scalar(
            select(ObservationRow)
            .where(
                ObservationRow.asset_id == asset_row.asset_id,
                ObservationRow.current.is_(True),
            )
            .with_for_update()
        )
        version = 1 if previous is None else previous.version + 1
        if previous is not None:
            previous.current = False

        row = ObservationRow(
            user_id=user_id,
            asset_id=asset_row.asset_id,
            external_observation_id=observation.observation_id,
            schema_version=observation.schema_version,
            version=version,
            generated_at=observation.generated_at,
            producer_json=observation.producer.model_dump(mode="json"),
            raw_json=payload,
            content_hash=content_hash,
            current=True,
            supersedes_observation_id=(
                previous.observation_id if previous is not None else None
            ),
            user_presence=observation.scene.user_presence.value,
            safety_json=observation.safety.model_dump(mode="json"),
            is_sensitive=observation.safety.is_sensitive,
        )
        session.add(row)
        session.flush()

        for fact in observation.facts:
            blocked_fact = (
                fact.predicate in self.config.privacy.blocked_predicates
                or observation.safety.is_sensitive
            )
            evidence_type = fact.evidence[0].source.value
            fact_row = ObservationFact(
                user_id=user_id,
                observation_id=row.observation_id,
                external_fact_id=fact.fact_id,
                subject_ref=fact.subject_ref,
                predicate=fact.predicate,
                canonical_key=f"{fact.subject_ref}|{fact.predicate}",
                value_json=fact.value,
                evidence_type=evidence_type,
                evidence_json=[
                    item.model_dump(mode="json") for item in fact.evidence
                ],
                confidence=fact.confidence,
                uncertainty_note=fact.uncertainty,
                is_sensitive=blocked_fact,
                status="rejected" if blocked_fact else "active",
            )
            session.add(fact_row)

        job = self._new_job(user_id, request_key, row.observation_id, content_hash)
        session.add(job)
        asset_row.ingest_status = "queued"
        session.flush()

        return IngestResult(
            user_id=user_id,
            asset_id=asset_row.asset_id,
            observation_id=row.observation_id,
            observation_version=version,
            job_id=job.job_id,
            idempotent_replay=False,
        )

    @staticmethod
    def _new_job(
        user_id: uuid.UUID,
        key: str,
        observation_id: uuid.UUID,
        content_hash: str,
    ) -> ProcessingJob:
        return ProcessingJob(
            user_id=user_id,
            job_type="observation.process",
            idempotency_key=key,
            status=JobStatus.QUEUED,
            payload_json={
                "observation_id": str(observation_id),
                "content_hash": content_hash,
            },
            attempts=0,
        )

    @staticmethod
    def _asset_from_observation(observation: ImageObservation) -> AssetInput:
        metadata = observation.input_metadata
        captured_at = None
        location = None
        if metadata and metadata.capture_time and metadata.capture_time.value:
            try:
                captured_at = datetime.fromisoformat(
                    metadata.capture_time.value.replace("Z", "+00:00")
                )
            except ValueError:
                captured_at = None
        if metadata and metadata.coarse_location:
            from album_memory.contracts import CoarseAssetLocation

            location = CoarseAssetLocation(
                level=metadata.coarse_location.level,
                country=metadata.coarse_location.country,
                city=metadata.coarse_location.city,
                region=metadata.coarse_location.region,
                source="provided_metadata",
            )
        return AssetInput(
            external_asset_id=observation.asset_id,
            sha256=observation.producer.input_sha256,
            source_kind=observation.scene.source_kind,
            captured_at=captured_at,
            captured_time_confidence=0.8 if captured_at else 0.0,
            location_coarse=location,
        )

    @staticmethod
    def _fill_missing_asset_fields(row: MediaAsset, incoming: AssetInput) -> None:
        for name in ("storage_uri", "sha256", "byte_size", "captured_at", "width", "height"):
            if getattr(row, name) is None and getattr(incoming, name) is not None:
                setattr(row, name, getattr(incoming, name))
        if row.mime_type == "image/unknown" and incoming.mime_type != "image/unknown":
            row.mime_type = incoming.mime_type
        if row.source_kind == "unknown" and incoming.source_kind.value != "unknown":
            row.source_kind = incoming.source_kind.value
        if row.location_coarse is None and incoming.location_coarse is not None:
            row.location_coarse = incoming.location_coarse.model_dump(mode="json")
        row.captured_time_confidence = max(
            row.captured_time_confidence,
            incoming.captured_time_confidence,
        )
