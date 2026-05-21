"""
Semantic retrieval module.
Performs similarity search against ChromaDB to find relevant document chunks.
"""

import logging
from app.config import settings
from app.rag.embeddings import embedding_service
from app.rag.ingestion import ingestion_pipeline
from app.models.schemas import SourceChunk

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    Retrieves the most relevant document chunks for a given query
    using cosine similarity search in ChromaDB.
    """

    def __init__(self):
        self.top_k = settings.TOP_K_RESULTS

    def retrieve(self, query: str, top_k: int = None) -> list[SourceChunk]:
        """
        Perform semantic search to find relevant document chunks.

        Args:
            query: The user's question or search query.
            top_k: Number of top results to return (overrides default).

        Returns:
            List of SourceChunk objects with content, source, and relevance score.
        """
        top_k = top_k or self.top_k

        if ingestion_pipeline.collection is None:
            logger.error("ChromaDB collection not initialized")
            return []

        # Generate query embedding
        query_embedding = embedding_service.generate_query_embedding(query)

        # Search ChromaDB
        results = ingestion_pipeline.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Parse results into SourceChunk objects
        chunks = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                # ChromaDB returns distances (lower = more similar for cosine)
                # Convert distance to similarity score (1 - distance for cosine)
                distance = results["distances"][0][i]
                similarity = 1 - distance  # cosine distance to similarity

                source = results["metadatas"][0][i].get("source", "unknown")

                chunks.append(SourceChunk(
                    content=doc,
                    source=source,
                    relevance_score=round(max(0, similarity), 4)
                ))

        logger.info(f"Retrieved {len(chunks)} chunks for query: '{query[:50]}...'")
        return chunks

    def retrieve_with_context(self, query: str, top_k: int = None) -> str:
        """
        Retrieve relevant chunks and format them as a context string
        for the LLM prompt.

        Args:
            query: The user's question.
            top_k: Number of results to retrieve.

        Returns:
            Formatted context string.
        """
        chunks = self.retrieve(query, top_k)

        if not chunks:
            return "No relevant information found in the knowledge base."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk.source} | Relevance: {chunk.relevance_score}]\n{chunk.content}"
            )

        return "\n\n---\n\n".join(context_parts)


# Singleton instance
retriever = SemanticRetriever()
