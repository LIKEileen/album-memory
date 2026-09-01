"""Convert legacy AlbumDoc/docs.jsonl to canonical, low-trust JSONL.

This script performs no database or model access. It is not executed as part of
this delivery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from album_memory.legacy import iter_legacy_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    with args.destination.open("w", encoding="utf-8") as output:
        for asset, observation in iter_legacy_jsonl(args.source):
            output.write(
                json.dumps(
                    {
                        "asset": asset.model_dump(mode="json"),
                        "observation": observation.model_dump(mode="json"),
                        "trust": "low",
                        "profile_activation_allowed": False,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
