from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from album_memory.errors import ConfigurationError


class DatabaseConfig(BaseModel):
    url: str | None = None
    url_env: str = "ALBUM_MEMORY_DATABASE_URL"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 5

    def resolved_url(self) -> str:
        value = self.url or os.getenv(self.url_env)
        if not value:
            raise ConfigurationError(f"database URL missing; set {self.url_env}")
        return value


class EmbeddingConfig(BaseModel):
    model_path: str = "/root/autodl-tmp/vlm/models/BAAI/bge-m3"
    model_version: str = "BAAI/bge-m3-local"
    dimension: int = 1024
    device: str = "cpu"
    batch_size: int = 2
    query_prefix: str = "为这个句子生成表示以用于检索："

    @model_validator(mode="after")
    def validate_dimension(self):
        if self.dimension != 1024:
            raise ValueError("the initial pgvector schema is fixed to 1024 dimensions")
        return self


class LLMConfig(BaseModel):
    enabled: bool = False
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    base_url_env: str = "ALBUM_MEMORY_LLM_BASE_URL"
    api_key_env: str = "ALBUM_MEMORY_LLM_API_KEY"
    model_env: str = "ALBUM_MEMORY_LLM_MODEL"
    timeout_seconds: int = 180

    def resolved(self) -> tuple[str, str, str] | None:
        if not self.enabled:
            return None
        base_url = self.base_url or os.getenv(self.base_url_env)
        api_key = self.api_key or os.getenv(self.api_key_env)
        model = self.model or os.getenv(self.model_env)
        if not all((base_url, api_key, model)):
            raise ConfigurationError("LLM is enabled but base URL, API key, or model is missing")
        return base_url, api_key, model


class ProcessingConfig(BaseModel):
    event_time_gap_hours: int = 72
    semantic_floor: float = 0.72
    semantic_strong_floor: float = 0.82
    min_event_source_reliability: float = 0.45
    require_subject_compatibility: bool = True
    max_event_candidates: int = 30
    max_semantic_edges: int = 6


class RetrievalConfig(BaseModel):
    vector_candidates_per_grain: int = 30
    lexical_candidates: int = 30
    temporal_candidates: int = 20
    graph_candidate_limit: int = 100
    ppr_damping: float = 0.85
    decay_lambda: float = 0.0019
    retrieval_decay_floor: float = 0.0
    frequency_eta: float = 0.15


class MaintenanceConfig(BaseModel):
    strength_threshold: float = 0.12
    idle_days: int = 365
    decay_lambda: float = 0.0019
    frequency_eta: float = 0.15


class PrivacyConfig(BaseModel):
    default_profile_injection_enabled: bool = False
    blocked_predicates: set[str] = Field(default_factory=lambda: {
        "mental_health_inference",
        "health_diagnosis",
        "precise_home_address",
        "real_time_location",
        "personality_inference",
    })


class MemoryConfig(BaseModel):
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    maintenance: MaintenanceConfig = Field(default_factory=MaintenanceConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MemoryConfig":
        with Path(path).open(encoding="utf-8") as handle:
            payload: dict[str, Any] = yaml.safe_load(handle) or {}
        return cls.model_validate(payload)
