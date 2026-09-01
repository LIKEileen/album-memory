from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from album_memory.enums import (
    ConsentState,
    EvidenceSource,
    Grain,
    Horizon,
    RecordStatus,
    RetrievalIntent,
    ReviewState,
    SourceKind,
    UserPresence,
)

StrictId = Annotated[str, StringConstraints(min_length=4, max_length=120, pattern=r"^[A-Za-z0-9_-]+$")]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Producer(StrictModel):
    model_id: str = Field(min_length=1, max_length=120)
    prompt_version: str = Field(min_length=1, max_length=80)
    input_sha256: str | None = Field(default=None, pattern=r"^[A-Fa-f0-9]{64}$")


class EvidenceLocator(StrictModel):
    source: EvidenceSource
    note: str = Field(min_length=1, max_length=300)
    bbox: list[float] | None = None
    ocr_text: str | None = Field(default=None, max_length=500)
    source_id: str | None = Field(default=None, max_length=120)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value):
        if value is None:
            return value
        if len(value) != 4 or any(x < 0 or x > 1 for x in value):
            raise ValueError("bbox must be [x1,y1,x2,y2] normalized to 0..1")
        if value[0] > value[2] or value[1] > value[3]:
            raise ValueError("bbox start coordinates must not exceed end coordinates")
        return value

    @model_validator(mode="after")
    def validate_source_fields(self):
        if self.ocr_text and self.source != EvidenceSource.OCR:
            raise ValueError("ocr_text is only valid for OCR evidence")
        return self


class CandidateLabel(StrictModel):
    label: str = Field(min_length=1, max_length=80)
    confidence: Confidence
    reason: str = Field(min_length=1, max_length=200)


class KeyEntity(StrictModel):
    entity_id: StrictId
    entity_type: Literal["person", "object", "text", "scene", "visible_action"]
    label: str = Field(min_length=1, max_length=80)
    count: int = Field(default=1, ge=1, le=1000)
    attributes: dict[str, str] = Field(default_factory=dict)
    candidate_labels: list[CandidateLabel] = Field(default_factory=list, max_length=5)
    evidence: list[EvidenceLocator] = Field(min_length=1, max_length=8)
    confidence: Confidence
    uncertainty: str = Field(min_length=1, max_length=300)


class GranularOutputs(StrictModel):
    key_entities: list[KeyEntity] = Field(default_factory=list, max_length=50)
    detailed_description: str = Field(min_length=1, max_length=1500)


class CoarseLocationObservation(StrictModel):
    level: Literal["none", "country", "city", "region"]
    text: str | None = Field(default=None, max_length=80)
    confidence: Confidence

    @model_validator(mode="after")
    def validate_empty_location(self):
        if self.level == "none" and (self.text is not None or self.confidence != 0):
            raise ValueError("unknown pixel location must be level=none, text=null, confidence=0")
        return self


class SceneObservation(StrictModel):
    visible_summary: str = Field(min_length=1, max_length=300)
    source_kind: SourceKind
    user_presence: UserPresence
    season_from_pixels: Literal["spring", "summer", "autumn", "winter", "unknown"]
    location_from_pixels: CoarseLocationObservation


class MetadataValue(StrictModel):
    value: str | None = Field(default=None, max_length=300)
    source: Literal["provided_metadata"] = "provided_metadata"
    note: str = Field(min_length=1, max_length=300)


class MetadataLocation(StrictModel):
    level: Literal["country", "city", "region", "unknown"]
    country: str | None = Field(default=None, max_length=80)
    city: str | None = Field(default=None, max_length=80)
    region: str | None = Field(default=None, max_length=80)
    source: Literal["provided_metadata"] = "provided_metadata"
    note: str = Field(min_length=1, max_length=300)


class MetadataLabel(StrictModel):
    value: str = Field(min_length=1, max_length=80)
    source: Literal["provided_metadata"] = "provided_metadata"
    usage_restriction: str = Field(min_length=1, max_length=200)


class InputMetadata(StrictModel):
    capture_time: MetadataValue | None = None
    coarse_location: MetadataLocation | None = None
    collection_source: MetadataValue | None = None
    labels: list[MetadataLabel] = Field(default_factory=list, max_length=30)


class ObservationFact(StrictModel):
    fact_id: StrictId
    subject_ref: str = Field(min_length=1, max_length=120)
    predicate: str = Field(min_length=1, max_length=80)
    value: JsonValue
    evidence: list[EvidenceLocator] = Field(min_length=1, max_length=8)
    confidence: Confidence
    uncertainty: str = Field(min_length=1, max_length=300)


class VisualStyle(StrictModel):
    composition: list[str] = Field(default_factory=list, max_length=20)
    lighting: list[str] = Field(default_factory=list, max_length=20)
    colors: list[str] = Field(default_factory=list, max_length=20)
    editing_signals: list[str] = Field(default_factory=list, max_length=20)
    uncertainty: str = Field(default="unknown", min_length=1, max_length=300)


class Limitations(StrictModel):
    unknown_identity: bool = True
    unknown_user_attribution: bool = True
    unknown_ownership: bool = True
    unknown_exact_location: bool = True
    notes: list[str] = Field(default_factory=list, max_length=30)


class SafetyAssessment(StrictModel):
    is_sensitive: bool = False
    contains_minor: bool | None = None
    blocked_from_profile: bool = False
    redactions: list[str] = Field(default_factory=list, max_length=30)
    reasons: list[str] = Field(default_factory=list, max_length=30)


class ImageObservation(StrictModel):
    schema_version: Literal["1.1"]
    observation_id: Annotated[str, StringConstraints(pattern=r"^obs_[A-Za-z0-9_-]{8,80}$")]
    asset_id: Annotated[str, StringConstraints(pattern=r"^asset_[A-Za-z0-9_-]{4,120}$")]
    generated_at: datetime
    producer: Producer
    granular_outputs: GranularOutputs
    scene: SceneObservation
    input_metadata: InputMetadata | None = None
    facts: list[ObservationFact] = Field(default_factory=list, max_length=100)
    visual_style: VisualStyle | None = None
    limitations: Limitations
    safety: SafetyAssessment


class CoarseAssetLocation(StrictModel):
    level: Literal["country", "city", "region", "unknown"] = "unknown"
    country: str | None = Field(default=None, max_length=80)
    city: str | None = Field(default=None, max_length=80)
    region: str | None = Field(default=None, max_length=80)
    source: Literal["exif", "provided_metadata", "unknown"] = "unknown"


class AssetInput(StrictModel):
    external_asset_id: str | None = Field(default=None, max_length=160)
    storage_uri: str | None = Field(default=None, max_length=1000)
    sha256: str | None = Field(default=None, pattern=r"^[A-Fa-f0-9]{64}$")
    mime_type: str = Field(default="image/unknown", max_length=100)
    byte_size: int | None = Field(default=None, gt=0)
    source_kind: SourceKind = SourceKind.UNKNOWN
    captured_at: datetime | None = None
    captured_time_confidence: Confidence = 0.0
    location_coarse: CoarseAssetLocation | None = None
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)


class IngestResult(StrictModel):
    user_id: UUID
    asset_id: UUID
    observation_id: UUID
    observation_version: int
    job_id: UUID
    idempotent_replay: bool = False


class ProcessingReport(StrictModel):
    claimed: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped_locked: int = 0
    job_ids: list[UUID] = Field(default_factory=list)


class MemoryHit(StrictModel):
    memory_id: UUID
    grain: Grain
    text: str
    score: float
    lexical_score: float = 0.0
    vector_score: float = 0.0
    decay_score: float = 1.0
    ppr_score: float = 0.0
    source_reliability: Confidence = 0.0
    captured_at: datetime | None = None
    event_id: UUID | None = None
    asset_id: UUID | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class ClaimView(StrictModel):
    claim_id: UUID
    dimension_id: str
    horizon: Horizon
    status: RecordStatus
    statement: str
    value: dict[str, Any]
    confidence: Confidence
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    expires_at: datetime | None = None
    next_review_at: datetime | None = None
    review_state: ReviewState
    supersedes_claim_id: UUID | None = None
    resolution_reason: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    counter_evidence_ids: list[str] = Field(default_factory=list)


class MemoryContext(StrictModel):
    retrieval_id: UUID
    user_id: UUID
    query: str
    intent: RetrievalIntent
    memories: list[MemoryHit] = Field(default_factory=list)
    claims: list[ClaimView] = Field(default_factory=list)
    answer_constraints: list[str] = Field(default_factory=list)


class ProfileSnapshot(StrictModel):
    user_id: UUID
    generated_at: datetime
    short_term: list[ClaimView] = Field(default_factory=list)
    long_term: list[ClaimView] = Field(default_factory=list)


class MaintenanceItem(StrictModel):
    object_type: Literal["memory", "claim", "user"]
    object_id: UUID
    action: str
    reason: str
    strength: float | None = None
    applied: bool = False


class MaintenanceReport(StrictModel):
    dry_run: bool
    scanned_memories: int = 0
    items: list[MaintenanceItem] = Field(default_factory=list)


class RegistrationResult(StrictModel):
    user_id: UUID
    consent_state: ConsentState
