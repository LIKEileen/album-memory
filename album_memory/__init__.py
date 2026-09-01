"""Public API for the album memory module.

Importing this package has no database or model side effects.
"""
from album_memory.api import AlbumMemory
from album_memory.config import MemoryConfig
from album_memory.contracts import (
    AssetInput,
    ImageObservation,
    IngestResult,
    MemoryContext,
    ProfileSnapshot,
)

__all__ = [
    "AlbumMemory",
    "MemoryConfig",
    "AssetInput",
    "ImageObservation",
    "IngestResult",
    "MemoryContext",
    "ProfileSnapshot",
]
