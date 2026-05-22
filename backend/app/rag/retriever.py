"""
Semantic retrieval module.
Performs similarity search against ChromaDB to find relevant document chunks.
"""

import logging
import re
import time
from pathlib import Path
from app.config import settings
from app.rag.embeddings import embedding_service
from app.rag.ingestion import ingestion_pipeline
from app.models.schemas import SourceChunk

logger = logging.getLogger(__name__)


def compute_lexical_score(query: str, doc_text: str) -> float:
    """
    Calculate a lexical relevance score based on token overlap and phrase matches.
    """
    stopwords = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
        "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
        "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
        "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
        "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
        "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not",
        "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
        "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
        "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
        "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
        "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
        "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
        "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
        "you've", "your", "yours", "yourself", "yourselves"
    }

    # Clean and normalize
    clean_query = re.sub(r"[^\w\s]", " ", query.lower())
    clean_doc = re.sub(r"[^\w\s]", " ", doc_text.lower())

    query_tokens = [w for w in clean_query.split() if w not in stopwords and len(w) > 1]

    if not query_tokens:
        return 0.0

    doc_tokens = set(clean_doc.split())

    # Calculate token overlap ratio
    matched = sum(1 for token in query_tokens if token in doc_tokens)
    token_overlap = matched / len(query_tokens)

    # Calculate phrase matching boost for multi-word concepts
    phrase_boost = 0.0
    normalized_doc = " ".join(clean_doc.split())

    if len(query_tokens) >= 2:
        joined_query = " ".join(query_tokens)
        if joined_query in normalized_doc:
            phrase_boost += 0.5
        else:
            # Check 2-word subsets
            for i in range(len(query_tokens) - 1):
                two_word = f"{query_tokens[i]} {query_tokens[i+1]}"
                if two_word in normalized_doc:
                    phrase_boost += 0.15

    return min(1.0, token_overlap + phrase_boost)

def expand_query_if_broad(query: str) -> str:
    """
    Appends key banking concepts to vague or extremely broad queries to improve retrieval.
    """
    q_clean = query.strip().lower().strip("?.!,")
    
    broad_queries = {
        "what topics are covered in the knowledge base": "topics banking services loans credit cards savings accounts digital banking upi kyc fixed deposits",
        "what documents are uploaded": "uploaded banking documentation documents guidelines policies",
        "what banking services are available": "services savings accounts home personal loans credit cards digital banking upi fixed deposits kyc policies",
        "topics covered": "topics banking services loans credit cards savings accounts digital banking upi kyc fixed deposits",
        "banking services": "services savings accounts home personal loans credit cards digital banking upi fixed deposits kyc policies",
        "uploaded documents": "uploaded banking documentation documents guidelines policies",
    }
    
    # Check for exact or close matches
    for broad_q, expansion in broad_queries.items():
        if broad_q in q_clean or q_clean in broad_q:
            return f"{query} {expansion}"
            
    # General check for terms like "topics", "services", "documents"
    if "topic" in q_clean or "covered" in q_clean:
        return f"{query} loans credit cards savings accounts digital banking upi kyc fixed deposits"
    if "service" in q_clean:
        return f"{query} loans credit cards savings accounts digital banking upi kyc fixed deposits"
    if "document" in q_clean or "upload" in q_clean:
        return f"{query} loans credit cards savings accounts digital banking upi kyc fixed deposits"
        
    return query


class SemanticRetriever:
    """
    Retrieves the most relevant document chunks for a given query
    using hybrid search (embeddings + BM25-like lexical scoring).
    """

    def __init__(self):
        self.top_k = settings.TOP_K_RESULTS

    def retrieve(self, query: str, top_k: int = None) -> list[SourceChunk]:
        """
        Perform similarity search and rerank results with lexical overlap and exact match boosting.

        Args:
            query: The user's question or search query.
            top_k: Number of top results to return (overrides default).

        Returns:
            List of SourceChunk objects with content, source, and relevance score.
        """
        start_time = time.time()
        top_k = top_k or self.top_k

        if ingestion_pipeline.collection is None:
            logger.error("ChromaDB collection not initialized")
            return []

        # Check if broad query and determine effective_top_k
        query_lower = query.lower()
        broad_terms = ["topic", "covered", "document", "service", "what can you do", "help", "capability", "menu", "support", "features"]
        is_broad = any(term in query_lower for term in broad_terms)
        effective_top_k = max(top_k, 8) if is_broad else top_k

        # Expand query if broad
        expanded_query = expand_query_if_broad(query)

        # Retrieve a larger initial candidate pool for reranking
        candidate_count = max(effective_top_k * 3, 15)

        # Generate query embedding
        t_embed_start = time.time()
        query_embedding = embedding_service.generate_query_embedding(expanded_query)
        embed_time = time.time() - t_embed_start

        # Search ChromaDB for semantic similarity
        t_db_start = time.time()
        results = ingestion_pipeline.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_count,
            include=["documents", "metadatas", "distances"]
        )
        db_time = time.time() - t_db_start

        # Parse and rerank candidates
        t_rerank_start = time.time()
        chunks = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                # Convert cosine distance to similarity (1.0 - distance)
                distance = results["distances"][0][i]
                similarity = max(0.0, 1.0 - distance)
                source = results["metadatas"][0][i].get("source", "unknown")

                # Compute lexical score (exact terms and phrase matches)
                lexical_val = compute_lexical_score(query, doc)

                # Category metadata alignment boost
                category_bonus = 0.0
                if "loan" in query_lower and "loan" in source.lower():
                    category_bonus = 0.15
                elif "card" in query_lower and "card" in source.lower():
                    category_bonus = 0.15
                elif "savings" in query_lower and "savings" in source.lower():
                    category_bonus = 0.15
                elif ("fd" in query_lower or "fixed deposit" in query_lower) and "savings" in source.lower():
                    category_bonus = 0.15

                # Blended combined score: 50% semantic, 40% lexical, 10% domain metadata alignment
                combined_score = 0.5 * similarity + 0.4 * lexical_val + category_bonus
                
                # Metadata-aware filename keyword boost
                source_clean = Path(source).stem.lower().replace("_", " ").replace("-", " ")
                source_words = set(source_clean.split())
                query_words = set(query_lower.split())
                overlap_words = source_words.intersection(query_words)
                if overlap_words:
                    # Boost score by 0.10 for each overlapping term, max 0.25
                    filename_bonus = min(0.25, len(overlap_words) * 0.10)
                    combined_score += filename_bonus

                combined_score = min(1.0, max(0.0, combined_score))

                chunks.append(SourceChunk(
                    content=doc,
                    source=source,
                    relevance_score=round(combined_score, 4)
                ))

            # Rerank: sort by combined relevance score
            chunks.sort(key=lambda x: x.relevance_score, reverse=True)

            # RAG Fallback check: if the best retrieved chunk doesn't pass the threshold,
            # we return an empty list. This flags the context as weak.
            if chunks and chunks[0].relevance_score < settings.MIN_RELEVANCE_THRESHOLD:
                logger.info(f"Top retrieval score {chunks[0].relevance_score} below threshold {settings.MIN_RELEVANCE_THRESHOLD}. Triggering weak context fallback.")
                chunks = []
            else:
                # Keep effective_top_k results
                chunks = chunks[:effective_top_k]

        rerank_time = time.time() - t_rerank_start
        total_time = time.time() - start_time
        
        logger.info(
            f"[RETRIEVE MONITOR] query='{query[:40]}...' | chunks_retrieved={len(chunks)} | "
            f"embed={embed_time:.3f}s | db={db_time:.3f}s | rerank={rerank_time:.3f}s | total={total_time:.3f}s"
        )
        return chunks


    def retrieve_with_context(self, query: str, top_k: int = None) -> str:
        """
        Retrieve relevant chunks and format them as a context string for the prompt.
        """
        chunks = self.retrieve(query, top_k)

        if not chunks:
            return "No relevant information found in the knowledge base."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk.source} | Relevance: {round(chunk.relevance_score * 100)}%]\n{chunk.content}"
            )

        return "\n\n---\n\n".join(context_parts)


# Singleton instance
retriever = SemanticRetriever()
