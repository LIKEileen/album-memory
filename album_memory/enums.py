from enum import StrEnum


class ConsentState(StrEnum):
    GRANTED = "granted"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"


class SourceKind(StrEnum):
    CAMERA = "camera"
    SCREENSHOT = "screenshot"
    DOWNLOAD = "download"
    CHAT_RECEIVED = "chat_received"
    ALBUM = "album"
    IMPORTED = "imported"
    UNKNOWN = "unknown"


class EvidenceSource(StrEnum):
    PIXEL = "pixel"
    OCR = "ocr"
    EXIF = "exif"
    PROVIDED_METADATA = "provided_metadata"


class UserPresence(StrEnum):
    CONFIRMED = "confirmed"
    POSSIBLE = "possible"
    UNKNOWN = "unknown"
    ABSENT = "absent"


class Grain(StrEnum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


class MemoryRelation(StrEnum):
    PARENT_OF = "parent_of"
    SAME_ASSET = "same_asset"
    SAME_EVENT = "same_event"
    TEMPORAL_ADJACENT = "temporal_adjacent"
    SEMANTIC_SIMILAR = "semantic_similar"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"


class RecordStatus(StrEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    CONFLICTED = "conflicted"
    ENDED = "ended"
    REJECTED = "rejected"
    ARCHIVED = "archived"
    SPLIT = "split"
    MERGED = "merged"


class Horizon(StrEnum):
    SHORT = "short"
    LONG = "long"


class ReviewState(StrEnum):
    AUTO_PASSED = "auto_passed"
    NEEDS_REVIEW = "needs_review"
    HUMAN_CONFIRMED = "human_confirmed"
    HUMAN_REJECTED = "human_rejected"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RetrievalIntent(StrEnum):
    RECALL = "recall"
    RECOMMENDATION = "recommendation"
    PROFILE = "profile"
    OTHER = "other"
