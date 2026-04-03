"""Embedding function backed by Ollama (mxbai-embed-large, 1024 dimensions)."""

import logging
import os

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 1024  # mxbai-embed-large output dimension


class OllamaEmbeddingFunction:
    """Embedding function using Ollama.

    Usage:
        ef = OllamaEmbeddingFunction()
        vectors = ef(["some text", "another text"])
    """

    def __init__(
        self,
        model_name: str = "mxbai-embed-large",
        base_url: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed a batch of texts via Ollama."""
        try:
            import ollama as ollama_lib
        except ImportError as exc:
            raise ImportError(
                "ollama package required for embeddings. "
                "Install with: pip install ollama"
            ) from exc

        client = ollama_lib.Client(host=self.base_url)
        try:
            response = client.embed(model=self.model_name, input=input)
            return response["embeddings"]
        except Exception as exc:
            raise ConnectionError(
                f"Failed to get embeddings from Ollama at {self.base_url}: {exc}"
            ) from exc
