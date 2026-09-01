from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def uuid_pk():
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[uuid.UUID] = uuid_pk()
    external_subject_key: Mapped[str] = mapped_column(String(255), unique=True)
    display_name: Mapped[str | None] = mapped_column(String(255))
    timezone: Mapped[str] = mapped_column(String(80), default="UTC")
    consent_state: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    profile_injection_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    privacy_tier: Mapped[int] = mapped_column(Integer, default=2)
    retention_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    erased_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint("privacy_tier BETWEEN 0 AND 3", name="ck_users_privacy_tier"),
    )


class MediaAsset(Base, TimestampMixin):
    __tablename__ = "media_assets"

    asset_id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), index=True
    )
    external_asset_id: Mapped[str] = mapped_column(String(160))
    storage_uri: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(String(64))
    mime_type: Mapped[str] = mapped_column(String(100), default="image/unknown")
    byte_size: Mapped[int | None] = mapped_column(BigInteger)
    source_kind: Mapped[str] = mapped_column(String(24), default="unknown", index=True)
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    captured_time_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    location_coarse: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    ingest_status: Mapped[str] = mapped_column(String(24), default="registered", index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    __table_args__ = (
        UniqueConstraint("user_id", "external_asset_id", name="uq_media_assets_external"),
        UniqueConstraint("user_id", "sha256", name="uq_media_assets_hash"),
        CheckConstraint(
            "captured_time_confidence BETWEEN 0 AND 1",
            name="ck_media_assets_time_confidence",
        ),
        CheckConstraint("byte_size IS NULL OR byte_size > 0", name="ck_media_assets_byte_size"),
    )


class ImageObservation(Base):
    __tablename__ = "image_observations"

    observation_id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), index=True
    )
    asset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("media_assets.asset_id", ondelete="CASCADE"), index=True
    )
    external_observation_id: Mapped[str] = mapped_column(String(120))
    schema_version: Mapped[str] = mapped_column(String(16))
    version: Mapped[int] = mapped_column(Integer, default=1)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    producer_json: Mapped[dict[str, Any]] = mapped_column(JSONB)
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSONB)
    content_hash: Mapped[str] = mapped_column(String(64))
    current: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    supersedes_observation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("image_observations.observation_id")
    )
    user_presence: Mapped[str] = mapped_column(String(16), default="unknown")
    safety_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    is_sensitive: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("user_id", "external_observation_id", name="uq_observation_external"),
        UniqueConstraint("asset_id", "version", name="uq_observation_asset_version"),
        Index(
            "uq_observation_current_asset",
            "asset_id",
            unique=True,
            postgresql_where=text("current = true"),
        ),
    )


class ObservationFact(Base):
    __tablename__ = "observation_facts"

    fact_id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), index=True
    )
    observation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("image_observations.observation_id", ondelete="CASCADE"),
        index=True,
    )
    external_fact_id: Mapped[str] = mapped_column(String(120))
    subject_ref: Mapped[str] = mapped_column(String(120))
    predicate: Mapped[str] = mapped_column(String(80), index=True)
    canonical_key: Mapped[str] = mapped_column(String(255), index=True)
    value_json: Mapped[Any] = mapped_column(JSONB)
    evidence_type: Mapped[str] = mapped_column(String(24), index=True)
    evidence_json: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    confidence: Mapped[float] = mapped_column(Float)
    uncertainty_note: Mapped[str] = mapped_column(Text)
    is_sensitive: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    status: Mapped[str] = mapped_column(String(16), default="active", index=True)
    supersedes_fact_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("observation_facts.fact_id")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("observation_id", "external_fact_id", name="uq_fact_observation_external"),
        CheckConstraint("confidence BETWEEN 0 AND 1", name="ck_fact_confidence"),
    )


class Event(Base, TimestampMixin):
    __tablename__ = "events"

    event_id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(64), default="unknown")
    title: Mapped[str] = mapped_column(String(200))
    summary: Mapped[str] = mapped_column(Text)
    start_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    end_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    time_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    location_coarse: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    cluster_method: Mapped[str] = mapped_column(String(120), default="metadata_gated_gmm_v1")
    cohesion_score: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(24), default="candidate", index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    supersedes_event_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("events.event_id")
    )

    __table_args__ = (
        CheckConstraint("time_confidence BETWEEN 0 AND 1", name="ck_event_time_confidence"),
        CheckConstraint("cohesion_score BETWEEN 0 AND 1", name="ck_event_cohesion"),
        CheckConstraint("end_at IS NULL OR start_at IS NULL OR end_at >= start_at", name="ck_event_time"),
        Index("ix_events_user_time", "user_id", text("start_at DESC")),
    )


class EventAsset(Base):
    __tablename__ = "event_assets"

    event_asset_id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), index=True
    )
    event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("events.event_id", ondelete="CASCADE"), index=True
    )
    asset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("media_assets.asset_id", ondelete="CASCADE"), index=True
    )
    relation_type: Mapped[str] = mapped_column(String(24), default="supporting")
    membership_score: Mapped[float] = mapped_column(Float, default=0.0)
    sequence_no: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("event_id", "asset_id", name="uq_event_asset"),
        CheckConstraint("membership_score BETWEEN 0 AND 1", name="ck_event_asset_score"),
    )


class MemoryItem(Base, TimestampMixin):
    __tablename__ = "memory_items"

    memory_id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), index=True
    )
    grain: Mapped[str] = mapped_column(String(4), index=True)
    asset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("media_assets.asset_id", ondelete="CASCADE"), index=True
    )
    event_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("events.event_id", ondelete="CASCADE"), index=True
    )
    observation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("image_observations.observation_id", ondelete="CASCADE")
    )
    parent_memory_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_items.memory_id")
    )
    text: Mapped[str] = mapped_column(Text)
    search_tokens: Mapped[list[str]] = mapped_column(ARRAY(String(80)), default=list)
    structured_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024))
    embedding_model_version: Mapped[str | None] = mapped_column(String(160))
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    source_reliability: Mapped[float] = mapped_column(Float, default=0.0)
    retrieval_count: Mapped[int] = mapped_column(Integer, default=0)
    injection_count: Mapped[int] = mapped_column(Integer, default=0)
    last_retrieved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_injected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_reinforced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    strength: Mapped[float] = mapped_column(Float, default=1.0)
    archived: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    archive_reason: Mapped[str | None] = mapped_column(String(255))
    version: Mapped[int] = mapped_column(Integer, default=1)
    supersedes_memory_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_items.memory_id")
    )

    __table_args__ = (
        CheckConstraint("grain IN ('L1','L2','L3')", name="ck_memory_grain"),
        CheckConstraint("source_reliability BETWEEN 0 AND 1", name="ck_memory_reliability"),
        CheckConstraint("strength BETWEEN 0 AND 1", name="ck_memory_strength"),
        CheckConstraint(
            "(grain IN ('L1','L2') AND asset_id IS NOT NULL) OR "
            "(grain = 'L3' AND event_id IS NOT NULL)",
            name="ck_memory_grain_owner",
        ),
        Index("ix_memory_user_grain_time", "user_id", "grain", text("captured_at DESC")),
        Index("ix_memory_tokens", "search_tokens", postgresql_using="gin"),
        Index(
            "ix_memory_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class MemoryEdge(Base, TimestampMixin):
    __tablename__ = "memory_edges"

    edge_id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), index=True
    )
    from_memory_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_items.memory_id", ondelete="CASCADE"), index=True
    )
    to_memory_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_items.memory_id", ondelete="CASCADE"), index=True
    )
    relation_type: Mapped[str] = mapped_column(String(32), index=True)
    weight: Mapped[float] = mapped_column(Float)
    evidence_json: Mapped[dict[str, Any]] = mapped_column(JSONB)
    algorithm_version: Mapped[str] = mapped_column(String(120))

    __table_args__ = (
        UniqueConstraint(
            "from_memory_id", "to_memory_id", "relation_type", name="uq_memory_edge"
        ),
        CheckConstraint("from_memory_id <> to_memory_id", name="ck_memory_edge_no_self"),
        CheckConstraint(
            "relation_type IN ('parent_of','same_asset','same_event',"
            "'temporal_adjacent','semantic_similar','supports','contradicts','supersedes')",
            name="ck_memory_edge_relation",
        ),
        CheckConstraint("weight BETWEEN 0 AND 1", name="ck_memory_edge_weight"),
    )


class ResearchRun(Base):
    __tablename__ = "research_runs"

    research_run_id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), index=True
    )
    run_type: Mapped[str] = mapped_column(String(40), index=True)
    status: Mapped[str] = mapped_column(String(24), default="running")
    model_id: Mapped[str | None] = mapped_column(String(160))
    prompt_version: Mapped[str | None] = mapped_column(String(80))
    rule_version: Mapped[str] = mapped_column(String(80))
    input_scope_json: Mapped[dict[str, Any]] = mapped_column(JSONB)
    output_summary_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ProfileClaim(Base, TimestampMixin):
    __tablename__ = "profile_claims"

    claim_id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), index=True
    )
    dimension_id: Mapped[str] = mapped_column(String(4), index=True)
    horizon: Mapped[str] = mapped_column(String(8), index=True)
    logical_key: Mapped[str] = mapped_column(String(255), index=True)
    status: Mapped[str] = mapped_column(String(16), default="candidate", index=True)
    statement: Mapped[str] = mapped_column(String(300))
    value_json: Mapped[dict[str, Any]] = mapped_column(JSONB)
    confidence: Mapped[float] = mapped_column(Float)
    source_type: Mapped[str] = mapped_column(String(16), default="album")
    gate_check_json: Mapped[dict[str, Any]] = mapped_column(JSONB)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    next_review_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    review_state: Mapped[str] = mapped_column(String(24), default="needs_review", index=True)
    resolution_reason: Mapped[str | None] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer, default=1)
    supersedes_claim_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profile_claims.claim_id")
    )
    research_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("research_runs.research_run_id")
    )
    archived: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    __table_args__ = (
        CheckConstraint("confidence BETWEEN 0 AND 1", name="ck_claim_confidence"),
        CheckConstraint(
            "dimension_id IN ('S1','S2','S3','S4','S5','S6','S7',"
            "'L1','L2','L3','L4','L5','L6','L7','L8','L9','L10')",
            name="ck_claim_dimension",
        ),
        CheckConstraint(
            "(dimension_id LIKE 'S%' AND horizon = 'short') OR "
            "(dimension_id LIKE 'L%' AND horizon = 'long')",
            name="ck_claim_horizon",
        ),
        Index(
            "uq_profile_active_logical",
            "user_id",
            "logical_key",
            unique=True,
            postgresql_where=text("status = 'active' AND archived = false"),
        ),
        Index(
            "ix_profile_active_lookup",
            "user_id",
            "dimension_id",
            "horizon",
            text("confidence DESC"),
            postgresql_where=text("status = 'active' AND archived = false"),
        ),
    )


class ClaimEvidence(Base):
    __tablename__ = "claim_evidence"

    claim_evidence_id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), index=True
    )
    claim_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profile_claims.claim_id", ondelete="CASCADE"), index=True
    )
    evidence_type: Mapped[str] = mapped_column(String(24))
    evidence_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    role: Mapped[str] = mapped_column(String(16))
    evidence_path: Mapped[str | None] = mapped_column(String(500))
    rationale: Mapped[str] = mapped_column(String(500))
    weight: Mapped[float] = mapped_column(Float)
    source_confidence_snapshot: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "claim_id", "evidence_type", "evidence_id", "role", name="uq_claim_evidence"
        ),
        CheckConstraint("role IN ('support','contradict')", name="ck_claim_evidence_role"),
        CheckConstraint("weight BETWEEN 0 AND 1", name="ck_claim_evidence_weight"),
    )


class RetrievalEvent(Base):
    __tablename__ = "retrieval_events"

    retrieval_id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), index=True
    )
    query: Mapped[str] = mapped_column(Text)
    intent: Mapped[str] = mapped_column(String(24))
    candidate_memory_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    selected_memory_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    selected_claim_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    injected_memory_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    injected_claim_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    score_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    feedback_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    injected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class UserConfirmation(Base):
    __tablename__ = "user_confirmations"

    confirmation_id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), index=True
    )
    target_type: Mapped[str] = mapped_column(String(24))
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    action: Mapped[str] = mapped_column(String(24))
    value_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProcessingJob(Base, TimestampMixin):
    __tablename__ = "processing_jobs"

    job_id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), index=True
    )
    job_type: Mapped[str] = mapped_column(String(40), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), unique=True)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONB)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint("attempts >= 0", name="ck_processing_job_attempts"),
    )
