from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from album_memory.contracts import (
    AssetInput,
    CandidateLabel,
    CoarseAssetLocation,
    CoarseLocationObservation,
    EvidenceLocator,
    GranularOutputs,
    ImageObservation,
    InputMetadata,
    KeyEntity,
    Limitations,
    MetadataLabel,
    MetadataLocation,
    MetadataValue,
    ObservationFact,
    Producer,
    SafetyAssessment,
    SceneObservation,
)
from album_memory.enums import EvidenceSource, SourceKind, UserPresence


def adapt_legacy_record(record: dict[str, Any]) -> tuple[AssetInput, ImageObservation]:
    """Map legacy AlbumDoc without inventing pixel/OCR/identity evidence."""
    doc_id = str(record.get("doc_id") or "unknown")
    digest = hashlib.sha256(
        json.dumps(record, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    external_asset = f"asset_legacy_{_safe_id(doc_id)}"
    observation_id = f"obs_legacy_{digest[:16]}"
    captured = _parse_time(record.get("captured_time"))
    core_objects = [
        str(value) for value in (record.get("core_objects") or []) if str(value).strip()
    ]
    tags = [str(value) for value in (record.get("tags") or []) if str(value).strip()]
    evidence = EvidenceLocator(
        source=EvidenceSource.PROVIDED_METADATA,
        note="legacy flattened VLM field; original pixel locator is unavailable",
        source_id=doc_id,
    )
    entities = [
        KeyEntity(
            entity_id=f"ent_legacy_{index:03d}",
            entity_type="object",
            label=label[:80],
            count=1,
            attributes={},
            evidence=[evidence],
            confidence=0.35,
            uncertainty="legacy import; category and image grounding require reprocessing from the original image",
        )
        for index, label in enumerate(core_objects[:50], 1)
    ]
    facts = [
        ObservationFact(
            fact_id=f"fact_legacy_{index:03d}",
            subject_ref=f"ent_legacy_{index:03d}",
            predicate="contains_object",
            value={"label": label[:80]},
            evidence=[evidence],
            confidence=0.35,
            uncertainty="legacy imported label; user attribution and ownership are unknown",
        )
        for index, label in enumerate(core_objects[:100], 1)
    ]
    location = record.get("location") or {
        "country": record.get("country"),
        "city": record.get("city"),
        "region": record.get("detail"),
    }
    metadata_location = None
    asset_location = None
    if any(location.get(key) for key in ("country", "city", "region")):
        metadata_location = MetadataLocation(
            level="city" if location.get("city") else "country",
            country=location.get("country"),
            city=location.get("city"),
            region=location.get("region"),
            note="legacy metadata; does not prove user presence",
        )
        asset_location = CoarseAssetLocation(
            level=metadata_location.level,
            country=metadata_location.country,
            city=metadata_location.city,
            region=metadata_location.region,
            source="provided_metadata",
        )

    metadata = InputMetadata(
        capture_time=(
            MetadataValue(
                value=captured.isoformat(),
                note="legacy captured_time; trust level is reduced",
            )
            if captured
            else None
        ),
        coarse_location=metadata_location,
        collection_source=MetadataValue(
            value="legacy_albumdoc",
            note="imported from the existing AlbumDoc/docs.jsonl format",
        ),
        labels=[
            MetadataLabel(
                value=value[:80],
                usage_restriction="cannot independently activate a profile Claim",
            )
            for value in tags[:30]
        ],
    )
    observation = ImageObservation(
        schema_version="1.1",
        observation_id=observation_id,
        asset_id=external_asset,
        generated_at=captured or datetime.now(timezone.utc),
        producer=Producer(
            model_id="legacy-albumdoc-import",
            prompt_version="legacy-adapter-1.0",
            input_sha256=None,
        ),
        granular_outputs=GranularOutputs(
            key_entities=entities,
            detailed_description=(
                str(record.get("detailed_description") or "旧版记录没有详细描述。")[:1500]
            ),
        ),
        scene=SceneObservation(
            visible_summary=str(record.get("scene_name") or "旧版场景未知")[:300],
            source_kind=SourceKind.IMPORTED,
            user_presence=UserPresence.UNKNOWN,
            season_from_pixels="unknown",
            location_from_pixels=CoarseLocationObservation(
                level="none", text=None, confidence=0
            ),
        ),
        input_metadata=metadata,
        facts=facts,
        limitations=Limitations(
            unknown_identity=True,
            unknown_user_attribution=True,
            unknown_ownership=True,
            unknown_exact_location=True,
            notes=[
                "legacy record lacks the canonical image_observation evidence structure",
                "reprocess the original image before using it for an active profile",
            ],
        ),
        safety=SafetyAssessment(
            is_sensitive=False,
            blocked_from_profile=True,
            reasons=["legacy import must remain candidate until canonical evidence exists"],
        ),
    )
    asset = AssetInput(
        external_asset_id=external_asset,
        storage_uri=record.get("image_path"),
        source_kind=SourceKind.IMPORTED,
        captured_at=captured,
        captured_time_confidence=0.35 if captured else 0.0,
        location_coarse=asset_location,
    )
    return asset, observation


def iter_legacy_jsonl(path: str | Path) -> Iterable[tuple[AssetInput, ImageObservation]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield adapt_legacy_record(json.loads(line))


def _safe_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value)
    return (cleaned or "unknown")[:80]


def _parse_time(value) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
