import copy

import pytest
from pydantic import ValidationError

from album_memory.contracts import ImageObservation


def valid_observation() -> dict:
    return {
        "schema_version": "1.1",
        "observation_id": "obs_12345678",
        "asset_id": "asset_1234",
        "generated_at": "2026-01-01T00:00:00Z",
        "producer": {"model_id": "vlm", "prompt_version": "1"},
        "granular_outputs": {
            "key_entities": [
                {
                    "entity_id": "entity_1",
                    "entity_type": "object",
                    "label": "自行车",
                    "evidence": [{"source": "pixels", "note": "画面中可见"}],
                    "confidence": 0.9,
                    "uncertainty": "品牌未知",
                }
            ],
            "detailed_description": "画面中可见一辆自行车。",
        },
        "scene": {
            "visible_summary": "户外场景",
            "source_kind": "camera",
            "user_presence": "unknown",
            "season_from_pixels": "unknown",
            "location_from_pixels": {"level": "none", "text": None, "confidence": 0},
        },
        "facts": [
            {
                "fact_id": "fact_001",
                "subject_ref": "asset",
                "predicate": "contains_object",
                "value": "自行车",
                "evidence": [{"source": "pixels", "note": "画面中可见"}],
                "confidence": 0.9,
                "uncertainty": "型号未知",
            }
        ],
        "limitations": {},
        "safety": {},
    }


def test_accepts_canonical_v11_observation():
    parsed = ImageObservation.model_validate(valid_observation())
    assert parsed.schema_version == "1.1"


def test_rejects_extra_fields():
    payload = valid_observation()
    payload["invented"] = True
    with pytest.raises(ValidationError):
        ImageObservation.model_validate(payload)


def test_rejects_ocr_text_without_ocr_source():
    payload = copy.deepcopy(valid_observation())
    payload["facts"][0]["evidence"][0]["ocr_text"] = "ABC"
    with pytest.raises(ValidationError):
        ImageObservation.model_validate(payload)


def test_unknown_pixel_location_is_explicit():
    payload = valid_observation()
    payload["scene"]["location_from_pixels"] = {
        "level": "none",
        "text": "上海",
        "confidence": 0.8,
    }
    with pytest.raises(ValidationError):
        ImageObservation.model_validate(payload)
