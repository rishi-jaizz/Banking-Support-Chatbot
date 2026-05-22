"""
Embedding generation module.
Wraps sentence-transformers for generating text embeddings.
"""

import logging
import functools
from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Manages the embedding model for generating vector representations of text."""

    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.model = None

    def load_model(self):
        """Load the sentence-transformers embedding model."""
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Embedding model loaded. Dimension: {self.model.get_sentence_embedding_dimension()}")

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        if self.model is None:
            self.load_model()

        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

    @functools.lru_cache(maxsize=256)
    def generate_query_embedding(self, query: str) -> list[float]:
        """
        Generate embedding for a single query.

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector.
        """
        if self.model is None:
            self.load_model()

        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        if self.model is None:
            self.load_model()
        return self.model.get_sentence_embedding_dimension()


# Singleton instance
embedding_service = EmbeddingService()
