from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any

import networkx as nx
from rank_bm25 import BM25Okapi
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from album_memory.config import MemoryConfig
from album_memory.contracts import ClaimView, MemoryContext, MemoryHit
from album_memory.embedding import LocalBGEEmbedder
from album_memory.enums import Grain, Horizon, RecordStatus, RetrievalIntent, ReviewState
from album_memory.models import (
    MemoryEdge,
    MemoryItem,
    ProfileClaim,
    RetrievalEvent,
    User,
)
from album_memory.profile import ProfileEngine
from album_memory.text import tokenize, utcnow


class RetrievalService:
    def __init__(
        self,
        config: MemoryConfig,
        embedder: LocalBGEEmbedder,
        profile: ProfileEngine,
    ):
        self.config = config
        self.embedder = embedder
        self.profile = profile

    def retrieve(
        self,
        session: Session,
        *,
        user_id: uuid.UUID,
        query: str,
        intent: RetrievalIntent = RetrievalIntent.RECALL,
        top_k: int = 5,
    ) -> MemoryContext:
        user = session.get(User, user_id)
        if user is None or user.erased_at is not None or user.consent_state != "granted":
            raise ValueError("user is unavailable for retrieval")
        q_tokens = tokenize(query)
        q_vector = self.embedder.encode([query], query=True)[0].tolist()
        cfg = self.config.retrieval
        vector_scores: dict[uuid.UUID, float] = {}
        candidates: dict[uuid.UUID, MemoryItem] = {}

        for grain in (Grain.L1, Grain.L2, Grain.L3):
            distance = MemoryItem.embedding.cosine_distance(q_vector).label("distance")
            rows = list(
                session.execute(
                    select(MemoryItem, distance)
                    .where(
                        MemoryItem.user_id == user_id,
                        MemoryItem.grain == grain.value,
                        MemoryItem.archived.is_(False),
                        MemoryItem.embedding.is_not(None),
                    )
                    .order_by(distance)
                    .limit(cfg.vector_candidates_per_grain)
                )
            )
            for memory, dist in rows:
                candidates[memory.memory_id] = memory
                vector_scores[memory.memory_id] = max(0.0, 1.0 - float(dist))

        if q_tokens:
            lexical = list(
                session.scalars(
                    select(MemoryItem)
                    .where(
                        MemoryItem.user_id == user_id,
                        MemoryItem.archived.is_(False),
                        MemoryItem.search_tokens.overlap(q_tokens),
                    )
                    .order_by(MemoryItem.source_reliability.desc())
                    .limit(cfg.lexical_candidates)
                )
            )
            for memory in lexical:
                candidates[memory.memory_id] = memory

        temporal = list(
            session.scalars(
                select(MemoryItem)
                .where(
                    MemoryItem.user_id == user_id,
                    MemoryItem.archived.is_(False),
                    MemoryItem.captured_at.is_not(None),
                )
                .order_by(MemoryItem.captured_at.desc())
                .limit(cfg.temporal_candidates)
            )
        )
        for memory in temporal:
            candidates[memory.memory_id] = memory

        candidate_list = list(candidates.values())[: cfg.graph_candidate_limit]
        if not candidate_list:
            event = RetrievalEvent(
                user_id=user_id,
                query=query,
                intent=intent.value,
                candidate_memory_ids=[],
                selected_memory_ids=[],
                selected_claim_ids=[],
                score_json={},
            )
            session.add(event)
            session.flush()
            return MemoryContext(
                retrieval_id=event.retrieval_id,
                user_id=user_id,
                query=query,
                intent=intent,
                answer_constraints=self._constraints(intent),
            )

        corpus = [memory.search_tokens or tokenize(memory.text) for memory in candidate_list]
        lexical_scores = self._bm25(corpus, q_tokens)
        preliminary: dict[uuid.UUID, float] = {}
        score_parts: dict[str, dict[str, float]] = {}
        now = utcnow()
        for memory, lexical_score in zip(candidate_list, lexical_scores):
            vector_score = vector_scores.get(memory.memory_id, 0.0)
            decay = self._retrieval_decay(memory, now)
            frequency = 1.0 + cfg.frequency_eta * math.log1p(memory.injection_count)
            grain_factor = (
                1.18
                if intent == RetrievalIntent.RECALL and memory.grain == Grain.L3.value
                else (0.95 if intent == RetrievalIntent.RECALL and memory.grain == Grain.L1.value else 1.0)
            )
            relevance = 0.55 * vector_score + 0.30 * lexical_score + 0.15 * memory.source_reliability
            score = relevance * decay * frequency * grain_factor
            preliminary[memory.memory_id] = score
            score_parts[str(memory.memory_id)] = {
                "vector": vector_score,
                "lexical": lexical_score,
                "decay": decay,
                "frequency": frequency,
                "grain_factor": grain_factor,
                "preliminary": score,
            }

        ppr = self._candidate_ppr(
            session,
            user_id=user_id,
            candidate_ids=set(preliminary),
            seeds=preliminary,
        )
        max_ppr = max(ppr.values(), default=1.0)
        final_scores = {
            memory_id: 0.75 * score
            + 0.25 * (ppr.get(memory_id, 0.0) / (max_ppr or 1.0))
            for memory_id, score in preliminary.items()
        }
        ranked = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        selected = [candidates[memory_id] for memory_id, _ in ranked]
        hits = []
        for memory_id, final in ranked:
            memory = candidates[memory_id]
            parts = score_parts[str(memory_id)]
            parts["ppr"] = ppr.get(memory_id, 0.0)
            parts["final"] = final
            evidence_ids = [
                str(item)
                for item in (memory.structured_payload or {}).get("fact_ids", [])
            ]
            hits.append(
                MemoryHit(
                    memory_id=memory.memory_id,
                    grain=Grain(memory.grain),
                    text=memory.text,
                    score=round(final, 6),
                    lexical_score=round(parts["lexical"], 6),
                    vector_score=round(parts["vector"], 6),
                    decay_score=round(parts["decay"], 6),
                    ppr_score=round(parts["ppr"], 6),
                    source_reliability=memory.source_reliability,
                    captured_at=memory.captured_at,
                    event_id=memory.event_id,
                    asset_id=memory.asset_id,
                    evidence_ids=evidence_ids,
                )
            )
            memory.retrieval_count += 1
            memory.last_retrieved_at = now

        claims = self._retrieve_claims(session, user, query, q_tokens, intent, top_k)
        retrieval = RetrievalEvent(
            user_id=user_id,
            query=query,
            intent=intent.value,
            candidate_memory_ids=[str(item.memory_id) for item in candidate_list],
            selected_memory_ids=[str(item.memory_id) for item in selected],
            selected_claim_ids=[str(item.claim_id) for item in claims],
            score_json=score_parts,
        )
        session.add(retrieval)
        session.flush()
        return MemoryContext(
            retrieval_id=retrieval.retrieval_id,
            user_id=user_id,
            query=query,
            intent=intent,
            memories=hits,
            claims=claims,
            answer_constraints=self._constraints(intent),
        )

    def record_injection(
        self,
        session: Session,
        *,
        user_id: uuid.UUID,
        retrieval_id: uuid.UUID,
        memory_ids: list[uuid.UUID],
        claim_ids: list[uuid.UUID],
    ) -> None:
        event = session.get(RetrievalEvent, retrieval_id)
        if event is None or event.user_id != user_id:
            raise ValueError("retrieval event not found")
        allowed_memory = set(event.selected_memory_ids or [])
        allowed_claim = set(event.selected_claim_ids or [])
        memory_values = [str(value) for value in memory_ids]
        claim_values = [str(value) for value in claim_ids]
        if not set(memory_values).issubset(allowed_memory):
            raise ValueError("only selected memories may be marked as injected")
        if not set(claim_values).issubset(allowed_claim):
            raise ValueError("only selected claims may be marked as injected")
        if event.injected_at is not None:
            if (
                event.injected_memory_ids == memory_values
                and event.injected_claim_ids == claim_values
            ):
                return
            raise ValueError("retrieval injection was already recorded with different IDs")
        now = utcnow()
        memories = list(
            session.scalars(
                select(MemoryItem).where(
                    MemoryItem.user_id == user_id,
                    MemoryItem.memory_id.in_(memory_ids),
                )
            )
        ) if memory_ids else []
        for memory in memories:
            memory.injection_count += 1
            memory.last_injected_at = now
            memory.last_reinforced_at = now
            memory.archived = False
            memory.archive_reason = None
            memory.strength = min(1.0, max(memory.strength, memory.source_reliability))
        event.injected_memory_ids = memory_values
        event.injected_claim_ids = claim_values
        event.injected_at = now

    @staticmethod
    def record_feedback(
        session: Session,
        *,
        user_id: uuid.UUID,
        retrieval_id: uuid.UUID,
        feedback: dict[str, Any],
    ) -> None:
        event = session.get(RetrievalEvent, retrieval_id)
        if event is None or event.user_id != user_id:
            raise ValueError("retrieval event not found")
        event.feedback_json = feedback

    @staticmethod
    def _bm25(corpus: list[list[str]], query_tokens: list[str]) -> list[float]:
        if not query_tokens or not corpus:
            return [0.0] * len(corpus)
        raw = BM25Okapi(corpus).get_scores(query_tokens)
        maximum = max(raw, default=0.0)
        if maximum <= 0:
            return [0.0] * len(corpus)
        return [float(score / maximum) for score in raw]

    def _retrieval_decay(self, memory: MemoryItem, now) -> float:
        if memory.captured_at is None:
            return 1.0
        captured = memory.captured_at
        if captured.tzinfo is None:
            captured = captured.replace(tzinfo=timezone.utc)
        days = max(0.0, (now - captured).total_seconds() / 86400)
        base = math.exp(-self.config.retrieval.decay_lambda * days)
        floor = self.config.retrieval.retrieval_decay_floor
        return floor + (1.0 - floor) * base

    def _candidate_ppr(
        self,
        session: Session,
        *,
        user_id: uuid.UUID,
        candidate_ids: set[uuid.UUID],
        seeds: dict[uuid.UUID, float],
    ) -> dict[uuid.UUID, float]:
        if not candidate_ids:
            return {}
        edges = list(
            session.scalars(
                select(MemoryEdge).where(
                    MemoryEdge.user_id == user_id,
                    MemoryEdge.from_memory_id.in_(candidate_ids),
                    MemoryEdge.to_memory_id.in_(candidate_ids),
                )
            )
        )
        graph = nx.Graph()
        graph.add_nodes_from(candidate_ids)
        graph.add_weighted_edges_from(
            (edge.from_memory_id, edge.to_memory_id, edge.weight)
            for edge in edges
        )
        total = sum(max(0.0, value) for value in seeds.values()) or 1.0
        personalization = {
            node: max(0.0, seeds.get(node, 0.0)) / total for node in graph.nodes
        }
        try:
            return nx.pagerank(
                graph,
                alpha=self.config.retrieval.ppr_damping,
                personalization=personalization,
                weight="weight",
            )
        except Exception:
            return seeds

    def _retrieve_claims(
        self,
        session: Session,
        user: User,
        query: str,
        q_tokens: list[str],
        intent: RetrievalIntent,
        top_k: int,
    ) -> list[ClaimView]:
        if not user.profile_injection_enabled:
            return []
        if intent not in {RetrievalIntent.RECOMMENDATION, RetrievalIntent.PROFILE}:
            return []
        dimensions = None
        if intent == RetrievalIntent.RECOMMENDATION:
            dimensions = {"L6", "L8", "L9"}
        query_stmt = select(ProfileClaim).where(
            ProfileClaim.user_id == user.user_id,
            ProfileClaim.status == "active",
            ProfileClaim.archived.is_(False),
            ProfileClaim.review_state.in_(["auto_passed", "human_confirmed"]),
            or_(ProfileClaim.expires_at.is_(None), ProfileClaim.expires_at > func.now()),
        )
        if dimensions:
            query_stmt = query_stmt.where(ProfileClaim.dimension_id.in_(dimensions))
        claims = list(session.scalars(query_stmt))
        scored = []
        token_set = set(q_tokens)
        for claim in claims:
            tokens = set(tokenize(claim.statement))
            lexical = len(token_set & tokens) / max(1, len(token_set | tokens))
            if intent == RetrievalIntent.PROFILE:
                lexical = max(lexical, 0.5)
            score = 0.65 * lexical + 0.35 * claim.confidence
            scored.append((score, claim))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [self.profile.claim_view(session, claim) for _, claim in scored[:top_k]]

    @staticmethod
    def _constraints(intent: RetrievalIntent) -> list[str]:
        values = [
            "仅把相关历史记忆表述为历史记录，不得包装成当前事实。",
            "候选、过期、低可信或未审核画像不得注入。",
            "不得从共现推出关系、从物体推出所有权、从重复行为推出喜欢或擅长。",
            "不得输出精确住宅、实时轨迹、心理健康、人格或敏感人口属性推断。",
        ]
        if intent == RetrievalIntent.RECOMMENDATION:
            values.append("建议使用“历史记录显示”“可能适合”等保守表达。")
        return values
