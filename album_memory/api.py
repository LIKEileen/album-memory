from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import select

from album_memory.config import MemoryConfig
from album_memory.contracts import (
    AssetInput,
    ImageObservation,
    IngestResult,
    MaintenanceReport,
    MemoryContext,
    ProcessingReport,
    ProfileSnapshot,
    RegistrationResult,
)
from album_memory.db import Database
from album_memory.embedding import LocalBGEEmbedder
from album_memory.enums import ConsentState, Horizon, RetrievalIntent
from album_memory.ingestion import IngestionService
from album_memory.maintenance import MaintenanceService
from album_memory.models import User
from album_memory.processing import ProcessingService
from album_memory.profile import ProfileEngine
from album_memory.providers import build_chat_provider
from album_memory.rendering import render_profile_markdown
from album_memory.retrieval import RetrievalService


class AlbumMemory:
    """Importable facade used by a VLM agent or an explicit worker process."""

    def __init__(self, config: MemoryConfig | None = None):
        self.config = config or MemoryConfig()
        self._database: Database | None = None
        self._embedder: LocalBGEEmbedder | None = None
        self._provider = None
        self._ingestion = IngestionService(self.config)
        self._maintenance = MaintenanceService(self.config)

    @property
    def database(self) -> Database:
        if self._database is None:
            self._database = Database(self.config)
        return self._database

    @property
    def embedder(self) -> LocalBGEEmbedder:
        if self._embedder is None:
            self._embedder = LocalBGEEmbedder(self.config.embedding)
        return self._embedder

    @property
    def provider(self):
        if self._provider is None:
            self._provider = build_chat_provider(self.config.llm)
        return self._provider

    @property
    def profile_engine(self) -> ProfileEngine:
        return ProfileEngine(self.config, self.provider)

    def register_user(
        self,
        *,
        external_subject_key: str,
        display_name: str | None = None,
        timezone: str = "UTC",
        consent_state: ConsentState = ConsentState.PENDING,
        privacy_tier: int = 2,
        retention_until: datetime | None = None,
    ) -> RegistrationResult:
        external_subject_key = external_subject_key.strip()
        if not external_subject_key or len(external_subject_key) > 255:
            raise ValueError("external_subject_key must contain 1 to 255 characters")
        if not 0 <= privacy_tier <= 3:
            raise ValueError("privacy_tier must be between 0 and 3")
        with self.database.session() as session:
            return self._ingestion.register_user(
                session,
                external_subject_key=external_subject_key,
                display_name=display_name,
                timezone_name=timezone,
                consent_state=consent_state,
                privacy_tier=privacy_tier,
                retention_until=retention_until,
            )

    def set_profile_injection_enabled(
        self,
        user_id: uuid.UUID,
        enabled: bool,
    ) -> None:
        with self.database.session() as session:
            user = session.get(User, user_id)
            if user is None:
                raise ValueError("user not found")
            user.profile_injection_enabled = enabled

    def ingest_observation(
        self,
        user_id: uuid.UUID,
        observation: ImageObservation | dict[str, Any],
        *,
        asset: AssetInput | dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> IngestResult:
        observation_model = (
            observation
            if isinstance(observation, ImageObservation)
            else ImageObservation.model_validate(observation)
        )
        asset_model = (
            asset
            if isinstance(asset, AssetInput) or asset is None
            else AssetInput.model_validate(asset)
        )
        with self.database.session() as session:
            return self._ingestion.ingest(
                session,
                user_id=user_id,
                observation=observation_model,
                asset=asset_model,
                idempotency_key=idempotency_key,
            )

    def process_pending(self, *, limit: int = 20) -> ProcessingReport:
        if limit < 1:
            raise ValueError("limit must be positive")
        service = ProcessingService(
            self.config,
            self.database,
            self.embedder,
            self.provider,
        )
        return service.process_pending(limit=limit)

    def retrieve(
        self,
        user_id: uuid.UUID,
        query: str,
        *,
        intent: RetrievalIntent | str | None = None,
        top_k: int = 5,
    ) -> MemoryContext:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        normalized_intent = self._resolve_intent(query, intent)
        profile = self.profile_engine
        service = RetrievalService(self.config, self.embedder, profile)
        with self.database.session() as session:
            return service.retrieve(
                session,
                user_id=user_id,
                query=query,
                intent=normalized_intent,
                top_k=top_k,
            )

    def record_injection(
        self,
        retrieval_id: uuid.UUID,
        memory_ids: list[uuid.UUID],
        claim_ids: list[uuid.UUID],
    ) -> None:
        profile = self.profile_engine
        service = RetrievalService(self.config, self.embedder, profile)
        with self.database.session() as session:
            from album_memory.models import RetrievalEvent

            event = session.get(RetrievalEvent, retrieval_id)
            if event is None:
                raise ValueError("retrieval event not found")
            service.record_injection(
                session,
                user_id=event.user_id,
                retrieval_id=retrieval_id,
                memory_ids=memory_ids,
                claim_ids=claim_ids,
            )

    def record_feedback(
        self,
        retrieval_id: uuid.UUID,
        feedback: dict[str, Any],
    ) -> None:
        with self.database.session() as session:
            from album_memory.models import RetrievalEvent

            event = session.get(RetrievalEvent, retrieval_id)
            if event is None:
                raise ValueError("retrieval event not found")
            RetrievalService.record_feedback(
                session,
                user_id=event.user_id,
                retrieval_id=retrieval_id,
                feedback=feedback,
            )

    def get_profile(
        self,
        user_id: uuid.UUID,
        *,
        horizon: Horizon | None = None,
        dimensions: set[str] | None = None,
    ) -> ProfileSnapshot:
        with self.database.session() as session:
            if session.get(User, user_id) is None:
                raise ValueError("user not found")
            return self.profile_engine.get_snapshot(
                session,
                user_id=user_id,
                horizon=horizon,
                dimensions=dimensions,
            )

    def render_profile_markdown(
        self,
        user_id: uuid.UUID,
        *,
        horizon: Horizon | None = None,
        dimensions: set[str] | None = None,
    ) -> str:
        return render_profile_markdown(
            self.get_profile(
                user_id,
                horizon=horizon,
                dimensions=dimensions,
            )
        )

    def confirm_claim(
        self,
        user_id: uuid.UUID,
        claim_id: uuid.UUID,
        *,
        note: str | None = None,
    ):
        with self.database.session() as session:
            return self.profile_engine.confirm_claim(
                session,
                user_id=user_id,
                claim_id=claim_id,
                note=note,
            )

    def correct_claim(
        self,
        user_id: uuid.UUID,
        claim_id: uuid.UUID,
        *,
        statement: str,
        value: dict[str, Any],
        note: str | None = None,
    ):
        with self.database.session() as session:
            return self.profile_engine.correct_claim(
                session,
                user_id=user_id,
                claim_id=claim_id,
                statement=statement,
                value=value,
                note=note,
            )

    def run_maintenance(
        self,
        *,
        user_id: uuid.UUID | None = None,
        dry_run: bool = True,
    ) -> MaintenanceReport:
        with self.database.session() as session:
            return self._maintenance.run(
                session,
                user_id=user_id,
                dry_run=dry_run,
            )

    def close(self) -> None:
        if self._embedder is not None:
            self._embedder.close()
        if self._database is not None:
            self._database.dispose()

    @staticmethod
    def _resolve_intent(
        query: str,
        intent: RetrievalIntent | str | None,
    ) -> RetrievalIntent:
        if intent is not None:
            return intent if isinstance(intent, RetrievalIntent) else RetrievalIntent(intent)
        if any(marker in query for marker in ("推荐", "建议", "适合", "选哪个")):
            return RetrievalIntent.RECOMMENDATION
        if any(marker in query for marker in ("画像", "了解我", "我的习惯")):
            return RetrievalIntent.PROFILE
        return RetrievalIntent.RECALL
