from __future__ import annotations

import math
import traceback
import uuid
from datetime import timedelta
from typing import Any

from sqlalchemy import and_, func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from album_memory.clustering import adaptive_similarity_threshold
from album_memory.config import MemoryConfig
from album_memory.conflict import resolve_fact_conflicts
from album_memory.contracts import ImageObservation, ProcessingReport
from album_memory.db import Database
from album_memory.embedding import LocalBGEEmbedder
from album_memory.enums import Grain, JobStatus, MemoryRelation, RecordStatus
from album_memory.models import (
    Event,
    EventAsset,
    ImageObservation as ObservationRow,
    MediaAsset,
    MemoryEdge,
    MemoryItem,
    ObservationFact,
    ProcessingJob,
    ResearchRun,
)
from album_memory.profile import ProfileEngine
from album_memory.providers import ChatProvider
from album_memory.text import source_reliability, tokenize, utcnow


class ProcessingService:
    def __init__(
        self,
        config: MemoryConfig,
        database: Database,
        embedder: LocalBGEEmbedder,
        provider: ChatProvider,
    ):
        self.config = config
        self.database = database
        self.embedder = embedder
        self.provider = provider
        self.profile = ProfileEngine(config, provider)

    def process_pending(self, *, limit: int = 20) -> ProcessingReport:
        claimed_ids: list[uuid.UUID] = []
        with self.database.session() as session:
            jobs = list(
                session.scalars(
                    select(ProcessingJob)
                    .where(
                        ProcessingJob.status.in_([JobStatus.QUEUED, JobStatus.FAILED]),
                        ProcessingJob.attempts < 5,
                        ProcessingJob.available_at <= func.now(),
                    )
                    .order_by(ProcessingJob.created_at)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            )
            for job in jobs:
                job.status = JobStatus.RUNNING
                job.locked_at = utcnow()
                job.attempts += 1
                claimed_ids.append(job.job_id)

        report = ProcessingReport(claimed=len(claimed_ids), job_ids=claimed_ids)
        for job_id in claimed_ids:
            try:
                outcome = self._process_job(job_id)
                if outcome == "skipped_locked":
                    report.skipped_locked += 1
                else:
                    report.succeeded += 1
            except Exception as exc:
                report.failed += 1
                with self.database.session() as session:
                    job = session.get(ProcessingJob, job_id)
                    if job is not None:
                        job.status = JobStatus.FAILED
                        job.last_error = (
                            f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"
                        )[:8000]
                        job.available_at = utcnow() + timedelta(
                            minutes=min(60, 2 ** min(job.attempts, 5))
                        )
        return report

    def _process_job(self, job_id: uuid.UUID) -> str:
        with self.database.session() as session:
            job = session.get(ProcessingJob, job_id)
            if job is None or job.status != JobStatus.RUNNING:
                return "ignored"
            lock_key = str(job.user_id)
            locked = session.scalar(
                select(
                    func.pg_try_advisory_xact_lock(
                        func.hashtextextended(lock_key, 0)
                    )
                )
            )
            if not locked:
                job.status = JobStatus.QUEUED
                job.locked_at = None
                job.available_at = utcnow() + timedelta(seconds=10)
                return "skipped_locked"
            if job.job_type != "observation.process":
                raise ValueError(f"unsupported job type: {job.job_type}")
            observation_id = uuid.UUID(job.payload_json["observation_id"])
            self._process_observation(session, job.user_id, observation_id)
            job.status = JobStatus.SUCCEEDED
            job.last_error = None
            job.locked_at = None
        return "succeeded"

    def _process_observation(
        self,
        session: Session,
        user_id: uuid.UUID,
        observation_id: uuid.UUID,
    ) -> None:
        row = session.get(ObservationRow, observation_id)
        if row is None or row.user_id != user_id:
            raise ValueError("observation not found for user")
        asset = session.get(MediaAsset, row.asset_id)
        observation = ImageObservation.model_validate(row.raw_json)
        facts = list(
            session.scalars(
                select(ObservationFact).where(
                    ObservationFact.observation_id == observation_id
                )
            )
        )
        research = ResearchRun(
            user_id=user_id,
            run_type="observation_processing",
            status="running",
            model_id=self.config.embedding.model_version,
            prompt_version=observation.producer.prompt_version,
            rule_version="album-memory-rules-1.0",
            input_scope_json={
                "observation_id": str(observation_id),
                "asset_id": str(asset.asset_id),
            },
        )
        session.add(research)
        session.flush()

        conflict_outcomes = resolve_fact_conflicts(
            session,
            user_id=user_id,
            observation_id=observation_id,
            provider=self.provider,
        )
        evidence_sources = {
            item.source.value
            for entity in observation.granular_outputs.key_entities
            for item in entity.evidence
        }
        evidence_sources.update(fact.evidence_type for fact in facts)
        reliability = source_reliability(
            asset.source_kind,
            evidence_sources,
            row.user_presence,
            imported=asset.source_kind == "imported",
        )

        entity_labels = [
            entity.label for entity in observation.granular_outputs.key_entities
        ]
        fact_labels = [
            f"{fact.predicate}:{self._short_value(fact.value_json)}"
            for fact in facts
            if not fact.is_sensitive and fact.status != "rejected"
        ]
        l1_text = "；".join(entity_labels + fact_labels) or "无可安全引用的实体事实"
        l2_text = observation.granular_outputs.detailed_description
        vectors = self.embedder.encode([l1_text, l2_text])
        l1 = self._create_asset_memory(
            session,
            user_id=user_id,
            asset=asset,
            observation_id=observation_id,
            grain=Grain.L1,
            text=l1_text,
            vector=vectors[0].tolist(),
            payload={
                "entity_labels": entity_labels,
                "fact_ids": [str(f.fact_id) for f in facts],
                "limitations": observation.limitations.model_dump(mode="json"),
            },
            reliability=reliability,
        )
        session.flush()
        l2 = self._create_asset_memory(
            session,
            user_id=user_id,
            asset=asset,
            observation_id=observation_id,
            grain=Grain.L2,
            text=l2_text,
            vector=vectors[1].tolist(),
            payload={
                "fact_ids": [str(f.fact_id) for f in facts],
                "visible_summary": observation.scene.visible_summary,
                "user_presence": row.user_presence,
                "visual_style": (
                    observation.visual_style.model_dump(mode="json")
                    if observation.visual_style
                    else None
                ),
            },
            reliability=reliability,
        )
        session.flush()
        l1.parent_memory_id = l2.memory_id
        self._edge(
            session,
            user_id,
            l2.memory_id,
            l1.memory_id,
            MemoryRelation.PARENT_OF,
            1.0,
            {"asset_id": str(asset.asset_id)},
        )
        self._edge(
            session,
            user_id,
            l1.memory_id,
            l2.memory_id,
            MemoryRelation.SAME_ASSET,
            1.0,
            {"asset_id": str(asset.asset_id)},
        )

        event, membership, activated_event = self._assign_event(
            session, user_id, asset, l2
        )
        l3 = self._rebuild_event_memory(session, user_id, event)
        self._edge(
            session,
            user_id,
            l3.memory_id,
            l2.memory_id,
            MemoryRelation.PARENT_OF,
            membership,
            {"event_id": str(event.event_id)},
        )
        self._build_local_edges(session, user_id, event, l2)
        self.profile.apply_conflict_outcomes(
            session,
            user_id=user_id,
            outcomes=conflict_outcomes,
        )
        self.profile.refresh_for_observation(
            session,
            user_id=user_id,
            observation=row,
            event=event,
            research_run_id=research.research_run_id,
            refresh_long_term=activated_event or bool(conflict_outcomes),
        )

        research.status = "succeeded"
        research.output_summary_json = {
            "memory_ids": [str(l1.memory_id), str(l2.memory_id), str(l3.memory_id)],
            "event_id": str(event.event_id),
            "fact_conflicts": conflict_outcomes,
        }
        research.finished_at = utcnow()
        asset.ingest_status = "processed"

    def _create_asset_memory(
        self,
        session: Session,
        *,
        user_id: uuid.UUID,
        asset: MediaAsset,
        observation_id: uuid.UUID,
        grain: Grain,
        text: str,
        vector: list[float],
        payload: dict[str, Any],
        reliability: float,
        parent_memory_id: uuid.UUID | None = None,
    ) -> MemoryItem:
        prior = session.scalar(
            select(MemoryItem)
            .where(
                MemoryItem.user_id == user_id,
                MemoryItem.asset_id == asset.asset_id,
                MemoryItem.grain == grain.value,
                MemoryItem.archived.is_(False),
            )
            .order_by(MemoryItem.version.desc())
            .limit(1)
            .with_for_update()
        )
        version = 1
        supersedes = None
        if prior is not None:
            prior.archived = True
            prior.archive_reason = "superseded_by_new_observation"
            version = prior.version + 1
            supersedes = prior.memory_id
        memory = MemoryItem(
            user_id=user_id,
            grain=grain.value,
            asset_id=asset.asset_id,
            observation_id=observation_id,
            parent_memory_id=parent_memory_id,
            text=text,
            search_tokens=tokenize(text),
            structured_payload=payload,
            embedding=vector,
            embedding_model_version=self.config.embedding.model_version,
            captured_at=asset.captured_at,
            source_reliability=reliability,
            last_reinforced_at=utcnow(),
            strength=reliability,
            version=version,
            supersedes_memory_id=supersedes,
        )
        session.add(memory)
        return memory

    def _assign_event(
        self,
        session: Session,
        user_id: uuid.UUID,
        asset: MediaAsset,
        l2: MemoryItem,
    ) -> tuple[Event, float, bool]:
        existing_link = session.scalar(
            select(EventAsset).where(
                EventAsset.user_id == user_id,
                EventAsset.asset_id == asset.asset_id,
            )
        )
        if existing_link is not None:
            event = session.get(Event, existing_link.event_id)
            return event, existing_link.membership_score, False

        distance = MemoryItem.embedding.cosine_distance(l2.embedding).label("distance")
        candidates = list(
            session.execute(
                select(MemoryItem, distance)
                .where(
                    MemoryItem.user_id == user_id,
                    MemoryItem.grain == Grain.L3,
                    MemoryItem.archived.is_(False),
                    MemoryItem.embedding.is_not(None),
                )
                .order_by(distance)
                .limit(self.config.processing.max_event_candidates)
            )
        )
        gated: list[tuple[MemoryItem, Event, float]] = []
        for memory, dist in candidates:
            event = session.get(Event, memory.event_id)
            similarity = max(0.0, 1.0 - float(dist))
            if self._metadata_gate(
                asset,
                event,
                similarity,
                new_memory=l2,
                candidate_memory=memory,
            ):
                gated.append((memory, event, similarity))

        event = None
        membership = 1.0
        if gated:
            similarities = [item[2] for item in gated]
            threshold = adaptive_similarity_threshold(
                similarities,
                fallback=self.config.processing.semantic_floor,
                floor=self.config.processing.semantic_floor,
            )
            _, best_event, best_score = max(gated, key=lambda item: item[2])
            if best_score >= threshold:
                event = best_event
                membership = best_score

        if event is None:
            event = Event(
                user_id=user_id,
                event_type="unknown",
                title="候选相册事件",
                summary="单张观察形成的候选事件，尚不足以推出用户行为。",
                start_at=asset.captured_at,
                end_at=asset.captured_at,
                time_confidence=asset.captured_time_confidence,
                location_coarse=asset.location_coarse,
                cluster_method="metadata_gated_gmm_v1",
                cohesion_score=1.0,
                status=RecordStatus.CANDIDATE,
            )
            session.add(event)
            session.flush()

        link = EventAsset(
            user_id=user_id,
            event_id=event.event_id,
            asset_id=asset.asset_id,
            relation_type="primary" if event.status == "candidate" else "supporting",
            membership_score=membership,
        )
        session.add(link)
        session.flush()
        previous_status = event.status
        self._refresh_event_fields(session, event)
        activated_event = previous_status != "active" and event.status == "active"
        return event, membership, activated_event

    def _metadata_gate(
        self,
        asset: MediaAsset,
        event: Event,
        similarity: float,
        *,
        new_memory: MemoryItem,
        candidate_memory: MemoryItem,
    ) -> bool:
        cfg = self.config.processing
        if min(
            new_memory.source_reliability,
            candidate_memory.source_reliability,
        ) < cfg.min_event_source_reliability:
            return False
        if cfg.require_subject_compatibility:
            new_presence = (new_memory.structured_payload or {}).get(
                "user_presence", "unknown"
            )
            old_presences = set(
                (candidate_memory.structured_payload or {}).get(
                    "user_presence_values", []
                )
            )
            if (
                new_presence == "confirmed" and "absent" in old_presences
            ) or (
                new_presence == "absent" and "confirmed" in old_presences
            ):
                return False
        asset_loc = asset.location_coarse or {}
        event_loc = event.location_coarse or {}
        for key in ("country", "city"):
            if asset_loc.get(key) and event_loc.get(key) and asset_loc[key] != event_loc[key]:
                return False
        if asset.captured_at and event.start_at:
            nearest = min(
                abs((asset.captured_at - event.start_at).total_seconds()),
                abs((asset.captured_at - (event.end_at or event.start_at)).total_seconds()),
            )
            if nearest > cfg.event_time_gap_hours * 3600:
                return False
        elif similarity < cfg.semantic_strong_floor:
            return False
        return similarity >= cfg.semantic_floor

    def _refresh_event_fields(self, session: Session, event: Event) -> None:
        assets = list(
            session.scalars(
                select(MediaAsset)
                .join(EventAsset, EventAsset.asset_id == MediaAsset.asset_id)
                .where(EventAsset.event_id == event.event_id)
            )
        )
        times = sorted(asset.captured_at for asset in assets if asset.captured_at)
        if times:
            event.start_at, event.end_at = times[0], times[-1]
            event.time_confidence = sum(
                a.captured_time_confidence for a in assets if a.captured_at
            ) / len(times)
        if event.location_coarse is None:
            event.location_coarse = next(
                (a.location_coarse for a in assets if a.location_coarse), None
            )
        scores = list(
            session.scalars(
                select(EventAsset.membership_score).where(
                    EventAsset.event_id == event.event_id
                )
            )
        )
        event.cohesion_score = sum(scores) / max(1, len(scores))
        event.status = "active" if len(assets) >= 2 and event.cohesion_score >= 0.72 else "candidate"

    def _rebuild_event_memory(
        self,
        session: Session,
        user_id: uuid.UUID,
        event: Event,
    ) -> MemoryItem:
        l2_items = list(
            session.scalars(
                select(MemoryItem)
                .join(EventAsset, EventAsset.asset_id == MemoryItem.asset_id)
                .where(
                    EventAsset.event_id == event.event_id,
                    MemoryItem.grain == Grain.L2,
                    MemoryItem.archived.is_(False),
                )
                .order_by(MemoryItem.captured_at)
            )
        )
        evidence = "\n".join(
            f"- [{item.memory_id}] {item.captured_at or '时间未知'} {item.text[:500]}"
            for item in l2_items
        )
        generated = self.provider.summarize_event(evidence)
        summary = generated or self._deterministic_event_summary(l2_items)
        event.summary = summary
        event.title = summary[:80] or "候选相册事件"

        vector = self.embedder.encode([summary])[0].tolist()
        prior = session.scalar(
            select(MemoryItem)
            .where(
                MemoryItem.user_id == user_id,
                MemoryItem.event_id == event.event_id,
                MemoryItem.grain == Grain.L3,
                MemoryItem.archived.is_(False),
            )
            .order_by(MemoryItem.version.desc())
            .limit(1)
            .with_for_update()
        )
        version = 1
        supersedes = None
        if prior is not None:
            prior.archived = True
            prior.archive_reason = "event_summary_rebuilt"
            version = prior.version + 1
            supersedes = prior.memory_id
        reliability = (
            sum(item.source_reliability for item in l2_items) / max(1, len(l2_items))
        )
        memory = MemoryItem(
            user_id=user_id,
            grain=Grain.L3,
            event_id=event.event_id,
            text=summary,
            search_tokens=tokenize(summary),
            structured_payload={
                "member_memory_ids": [str(item.memory_id) for item in l2_items],
                "status": event.status,
                "user_presence_values": sorted(
                    {
                        (item.structured_payload or {}).get("user_presence", "unknown")
                        for item in l2_items
                    }
                ),
            },
            embedding=vector,
            embedding_model_version=self.config.embedding.model_version,
            captured_at=event.start_at,
            source_reliability=reliability,
            last_reinforced_at=utcnow(),
            strength=reliability,
            version=version,
            supersedes_memory_id=supersedes,
        )
        session.add(memory)
        session.flush()
        return memory

    def _build_local_edges(
        self,
        session: Session,
        user_id: uuid.UUID,
        event: Event,
        l2: MemoryItem,
    ) -> None:
        siblings = list(
            session.scalars(
                select(MemoryItem)
                .join(EventAsset, EventAsset.asset_id == MemoryItem.asset_id)
                .where(
                    EventAsset.event_id == event.event_id,
                    MemoryItem.grain == Grain.L2,
                    MemoryItem.archived.is_(False),
                    MemoryItem.memory_id != l2.memory_id,
                )
                .limit(20)
            )
        )
        for sibling in siblings:
            self._edge(
                session,
                user_id,
                l2.memory_id,
                sibling.memory_id,
                MemoryRelation.SAME_EVENT,
                event.cohesion_score,
                {"event_id": str(event.event_id)},
            )

        distance = MemoryItem.embedding.cosine_distance(l2.embedding).label("distance")
        neighbors = list(
            session.execute(
                select(MemoryItem, distance)
                .where(
                    MemoryItem.user_id == user_id,
                    MemoryItem.grain == Grain.L2,
                    MemoryItem.archived.is_(False),
                    MemoryItem.memory_id != l2.memory_id,
                    MemoryItem.embedding.is_not(None),
                )
                .order_by(distance)
                .limit(self.config.processing.max_semantic_edges)
            )
        )
        for neighbor, dist in neighbors:
            similarity = max(0.0, 1.0 - float(dist))
            if similarity >= self.config.processing.semantic_floor:
                self._edge(
                    session,
                    user_id,
                    l2.memory_id,
                    neighbor.memory_id,
                    MemoryRelation.SEMANTIC_SIMILAR,
                    similarity,
                    {"cosine_similarity": similarity},
                )
        temporal = next(
            iter(
                session.scalars(
                    select(MemoryItem)
                    .where(
                        MemoryItem.user_id == user_id,
                        MemoryItem.grain == Grain.L2,
                        MemoryItem.archived.is_(False),
                        MemoryItem.memory_id != l2.memory_id,
                        MemoryItem.captured_at.is_not(None),
                    )
                    .order_by(func.abs(func.extract("epoch", MemoryItem.captured_at - l2.captured_at)))
                    .limit(1)
                )
            ),
            None,
        ) if l2.captured_at else None
        if temporal is not None:
            self._edge(
                session,
                user_id,
                l2.memory_id,
                temporal.memory_id,
                MemoryRelation.TEMPORAL_ADJACENT,
                0.7,
                {"captured_at": str(l2.captured_at)},
            )

    @staticmethod
    def _edge(
        session: Session,
        user_id: uuid.UUID,
        from_id: uuid.UUID,
        to_id: uuid.UUID,
        relation: MemoryRelation,
        weight: float,
        evidence: dict[str, Any],
    ) -> None:
        if from_id == to_id:
            return
        statement = insert(MemoryEdge).values(
            user_id=user_id,
            from_memory_id=from_id,
            to_memory_id=to_id,
            relation_type=relation.value,
            weight=max(0.0, min(1.0, float(weight))),
            evidence_json=evidence,
            algorithm_version="album-memory-graph-1.0",
        )
        statement = statement.on_conflict_do_update(
            constraint="uq_memory_edge",
            set_={
                "weight": statement.excluded.weight,
                "evidence_json": statement.excluded.evidence_json,
                "algorithm_version": statement.excluded.algorithm_version,
                "updated_at": func.now(),
            },
        )
        session.execute(statement)

    @staticmethod
    def _deterministic_event_summary(items: list[MemoryItem]) -> str:
        if not items:
            return "候选事件暂无可安全引用的摘要。"
        fragments = [item.text.strip()[:220] for item in items[:5]]
        prefix = f"该候选事件包含{len(items)}条图片观察。"
        return (prefix + " ".join(fragments))[:1800]

    @staticmethod
    def _short_value(value: Any) -> str:
        text = str(value)
        return text if len(text) <= 80 else text[:77] + "..."
