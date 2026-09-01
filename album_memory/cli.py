from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from album_memory import AlbumMemory, AssetInput, ImageObservation, MemoryConfig
from album_memory.enums import ConsentState, Horizon, RetrievalIntent
from album_memory.legacy import iter_legacy_jsonl


def _module(args) -> AlbumMemory:
    config = MemoryConfig.from_yaml(args.config) if args.config else MemoryConfig()
    return AlbumMemory(config)


def _json_file(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(prog="album-memory")
    parser.add_argument("--config", help="YAML configuration path")
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("register-user")
    register.add_argument("--external-key", required=True)
    register.add_argument("--consent", choices=[x.value for x in ConsentState], default="pending")

    ingest = sub.add_parser("ingest")
    ingest.add_argument("--user-id", required=True)
    ingest.add_argument("--observation", required=True)
    ingest.add_argument("--asset")
    ingest.add_argument("--idempotency-key")

    process = sub.add_parser("process")
    process.add_argument("--limit", type=int, default=20)

    retrieve = sub.add_parser("retrieve")
    retrieve.add_argument("--user-id", required=True)
    retrieve.add_argument("--query", required=True)
    retrieve.add_argument(
        "--intent",
        choices=[x.value for x in RetrievalIntent],
        default=RetrievalIntent.RECALL,
    )
    retrieve.add_argument("--top-k", type=int, default=5)

    profile = sub.add_parser("profile")
    profile.add_argument("--user-id", required=True)
    profile.add_argument("--horizon", choices=[x.value for x in Horizon])
    profile.add_argument("--markdown", action="store_true")

    maintenance = sub.add_parser("maintenance")
    maintenance.add_argument("--user-id")
    maintenance.add_argument("--apply", action="store_true")

    legacy = sub.add_parser("import-legacy")
    legacy.add_argument("--user-id", required=True)
    legacy.add_argument("--jsonl", required=True)

    args = parser.parse_args()
    module = _module(args)
    try:
        if args.command == "register-user":
            result = module.register_user(
                external_subject_key=args.external_key,
                consent_state=ConsentState(args.consent),
            )
            print(result.model_dump_json(indent=2))
        elif args.command == "ingest":
            result = module.ingest_observation(
                uuid.UUID(args.user_id),
                ImageObservation.model_validate(_json_file(args.observation)),
                asset=AssetInput.model_validate(_json_file(args.asset)) if args.asset else None,
                idempotency_key=args.idempotency_key,
            )
            print(result.model_dump_json(indent=2))
        elif args.command == "process":
            print(module.process_pending(limit=args.limit).model_dump_json(indent=2))
        elif args.command == "retrieve":
            result = module.retrieve(
                uuid.UUID(args.user_id),
                args.query,
                intent=RetrievalIntent(args.intent),
                top_k=args.top_k,
            )
            print(result.model_dump_json(indent=2))
        elif args.command == "profile":
            user_id = uuid.UUID(args.user_id)
            horizon = Horizon(args.horizon) if args.horizon else None
            if args.markdown:
                print(module.render_profile_markdown(user_id, horizon=horizon))
            else:
                print(module.get_profile(user_id, horizon=horizon).model_dump_json(indent=2))
        elif args.command == "maintenance":
            result = module.run_maintenance(
                user_id=uuid.UUID(args.user_id) if args.user_id else None,
                dry_run=not args.apply,
            )
            print(result.model_dump_json(indent=2))
        elif args.command == "import-legacy":
            user_id = uuid.UUID(args.user_id)
            count = 0
            for asset, observation in iter_legacy_jsonl(args.jsonl):
                module.ingest_observation(
                    user_id,
                    observation,
                    asset=asset,
                    idempotency_key=f"legacy:{observation.observation_id}",
                )
                count += 1
            print(json.dumps({"queued": count}, ensure_ascii=False))
    finally:
        module.close()


if __name__ == "__main__":
    main()
