"""Embedding protocol + a deterministic FakeEncoder for offline tests.

The real `SentenceTransformerEncoder` lives behind the `embed` extra so
this module stays importable on Intel macOS (where torch dropped wheels).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Encoder(Protocol):
    """Text encoder producing dense float32 vectors. Deterministic + order-preserving."""

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class FakeEncoder:
    """blake2b-backed deterministic encoder so tests cover the full retrieval
    pipeline without downloading a model."""

    name: str = "fake-test-encoder"
    dim: int = 32

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.dim > 64:
            raise ValueError("FakeEncoder.dim is capped at 64")

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            digest = hashlib.blake2b(t.encode("utf-8"), digest_size=64).digest()
            raw = np.frombuffer(digest[: self.dim], dtype=np.uint8).astype(np.float32)
            vec = raw / 127.5 - 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            out[i] = vec
        return out


def load_sentence_transformer_encoder(model_name: str = "BAAI/bge-small-en-v1.5") -> Encoder:
    """Construct a real sentence-transformer encoder (requires the `embed` extra)."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - env-dependent
        raise ImportError(
            "sentence-transformers is required for load_sentence_transformer_encoder; "
            "install via `uv sync --extra embed` or use the bundled Docker image."
        ) from e

    @dataclass
    class _ST:
        name: str
        dim: int
        model: SentenceTransformer

        def encode(self, texts: list[str]) -> np.ndarray:
            if not texts:
                return np.zeros((0, self.dim), dtype=np.float32)
            return np.asarray(
                self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                dtype=np.float32,
            )

    model = SentenceTransformer(model_name)
    return _ST(name=model_name, dim=int(model.get_sentence_embedding_dimension()), model=model)
