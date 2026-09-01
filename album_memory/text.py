from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

import jieba

from album_memory.enums import EvidenceSource, SourceKind


def canonical_hash(payload: dict[str, Any]) -> str:
    data = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def tokenize(text: str) -> list[str]:
    words = []
    for word in jieba.lcut((text or "").lower()):
        value = re.sub(r"[^\w\u4e00-\u9fff-]+", "", word).strip("_-")
        if len(value) >= 1:
            words.append(value[:80])
    return sorted(set(words))


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def source_reliability(
    source_kind: str,
    evidence_sources: set[str],
    user_presence: str,
    imported: bool = False,
) -> float:
    if imported:
        base = 0.35
    elif EvidenceSource.EXIF in evidence_sources:
        base = 0.88
    elif EvidenceSource.OCR in evidence_sources:
        base = 0.78
    elif EvidenceSource.PIXEL in evidence_sources:
        base = 0.68
    else:
        base = 0.45
    if source_kind == SourceKind.CAMERA:
        base += 0.06
    elif source_kind in {SourceKind.DOWNLOAD, SourceKind.CHAT_RECEIVED}:
        base -= 0.12
    if user_presence == "confirmed":
        base += 0.06
    elif user_presence == "unknown":
        base -= 0.05
    return round(min(1.0, max(0.0, base)), 3)


def normalize_fact_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value).strip().lower()
