from __future__ import annotations

import gc

import numpy as np

from album_memory.config import EmbeddingConfig
from album_memory.errors import ProcessingError


class LocalBGEEmbedder:
    """Lazy local BGE-M3 adapter.

    Construction and import are cheap. The model is loaded only on the first
    encode call, which is intended to happen in process_pending or retrieve.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.config.model_path,
                device=self.config.device,
            )
        return self._model

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.config.dimension), dtype=np.float32)
        values = [
            f"{self.config.query_prefix}{text}" if query else text
            for text in texts
        ]
        vectors = self.model.encode(
            values,
            normalize_embeddings=True,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.config.dimension:
            raise ProcessingError(
                f"embedding dimension mismatch: expected {self.config.dimension}, got {array.shape}"
            )
        return array

    def close(self) -> None:
        self._model = None
        gc.collect()
