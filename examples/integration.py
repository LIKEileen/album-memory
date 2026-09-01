"""Minimal VLM integration example. It is documentation and is not executed."""

from album_memory import AlbumMemory, MemoryConfig
from album_memory.enums import ConsentState, RetrievalIntent


def accept_vlm_observation(observation_payload: dict, asset_payload: dict) -> None:
    memory = AlbumMemory(MemoryConfig.from_yaml("config.yaml"))
    try:
        registration = memory.register_user(
            external_subject_key="demo-tenant:demo-user",
            consent_state=ConsentState.GRANTED,
        )
        memory.ingest_observation(
            registration.user_id,
            observation_payload,
            asset=asset_payload,
            idempotency_key=f"vlm:{observation_payload['observation_id']}",
        )

        # Call this from an explicit worker, never from the VLM request path.
        memory.process_pending(limit=4)

        context = memory.retrieve(
            registration.user_id,
            "回顾最近一次户外活动",
            intent=RetrievalIntent.RECALL,
            top_k=5,
        )
        actually_injected = [item.memory_id for item in context.memories[:3]]
        memory.record_injection(
            context.retrieval_id,
            actually_injected,
            [],
        )
    finally:
        memory.close()
