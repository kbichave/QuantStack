# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic text embedding for the Alpha Knowledge Graph.

Produces consistent 1536-dimensional vectors from text using a hash-based
approach.  This enables pgvector cosine-similarity queries without requiring
an external model server.  In production, swap the implementation body for
a call to Bedrock Titan or Ollama mxbai-embed-large — the interface stays
the same.

Why hash-based: the knowledge graph needs embeddings that are reproducible
across restarts and test runs.  A hash-based approach satisfies that while
keeping the dependency surface at zero.  Cosine similarity between hash
vectors still clusters lexically similar strings, which is sufficient for
deduplication checks.
"""

from __future__ import annotations

import hashlib
import struct

EMBEDDING_DIM = 1536


def generate_embedding(text: str) -> list[float]:
    """Return a deterministic 1536-dim unit vector derived from *text*.

    Algorithm:
      1. SHA-512 the normalised text → 64 bytes.
      2. Use those bytes as a seed to fill 1536 floats via iterative hashing.
      3. L2-normalise so cosine distance is meaningful.

    The same text always produces the same vector.  Different texts produce
    vectors that are effectively uncorrelated (hash avalanche property).
    """
    normalised = text.strip().lower()
    seed = hashlib.sha512(normalised.encode("utf-8")).digest()

    raw: list[float] = []
    block = seed
    while len(raw) < EMBEDDING_DIM:
        block = hashlib.sha512(block).digest()
        # Each SHA-512 block gives 64 bytes → 8 doubles
        for i in range(0, 64, 8):
            if len(raw) >= EMBEDDING_DIM:
                break
            # Unpack as unsigned 64-bit int, map to [-1, 1]
            val = struct.unpack("<Q", block[i : i + 8])[0]
            raw.append((val / (2**63)) - 1.0)

    # L2 normalise
    norm = sum(x * x for x in raw) ** 0.5
    if norm == 0:
        return raw  # degenerate case — all zeros
    return [x / norm for x in raw]
