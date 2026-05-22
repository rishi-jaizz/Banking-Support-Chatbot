"""
Document ingestion module.
Handles loading, chunking, and indexing banking documents into ChromaDB.
"""

import os
import logging
import hashlib
import shutil
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.rag.embeddings import embedding_service

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """
    Handles the full document ingestion pipeline:
    1. Load documents from the data directory
    2. Split into chunks with overlap
    3. Generate embeddings
    4. Store in ChromaDB
    """

    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.documents_count = 0

    def copy_default_documents_if_needed(self):
        """
        In production with persistent disks, the target DATA_DIR might be empty.
        This copies default files from the package data folder to the target directory.
        """
        default_data_dir = Path(__file__).resolve().parent.parent / "data" / "banking_knowledge"
        target_data_dir = Path(settings.DATA_DIR)
        
        # Ensure target data directory exists
        os.makedirs(target_data_dir, exist_ok=True)
        
        # Check if default data directory exists and is different from the target
        if not default_data_dir.exists() or default_data_dir.resolve() == target_data_dir.resolve():
            return
            
        # Check if target directory is empty (no md, txt, or pdf files)
        target_files = (
            list(target_data_dir.glob("*.md")) +
            list(target_data_dir.glob("*.txt")) +
            list(target_data_dir.glob("*.pdf"))
        )
        
        if not target_files:
            logger.info(f"Target data directory '{target_data_dir}' is empty. Copying default documents from '{default_data_dir}'...")
            copied_count = 0
            for file_path in default_data_dir.iterdir():
                if file_path.is_file() and not file_path.name.startswith("."):
                    try:
                        shutil.copy2(file_path, target_data_dir)
                        logger.info(f"Copied default file: {file_path.name}")
                        copied_count += 1
                    except Exception as e:
                        logger.error(f"Failed to copy default file {file_path.name}: {e}")
            logger.info(f"Successfully copied {copied_count} default documents.")

    def initialize(self):
        """Initialize ChromaDB client and collection."""
        self.copy_default_documents_if_needed()
        logger.info(f"Initializing ChromaDB at: {settings.CHROMA_PERSIST_DIR}")

        # Create persist directory if it doesn't exist
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        self.documents_count = self.collection.count()
        logger.info(f"ChromaDB collection '{settings.COLLECTION_NAME}' initialized with {self.documents_count} documents")

    def load_documents(self) -> list[dict]:
        """
        Load all documents from the banking knowledge directory.

        Returns:
            List of dicts with 'content' and 'source' keys.
        """
        documents = []
        data_dir = Path(settings.DATA_DIR)

        if not data_dir.exists():
            logger.warning(f"Data directory not found: {data_dir}")
            return documents

        for file_path in sorted(data_dir.glob("*.md")):
            logger.info(f"Loading document: {file_path.name}")
            content = file_path.read_text(encoding="utf-8")
            documents.append({
                "content": content,
                "source": file_path.name
            })

        # Also load .txt files
        for file_path in sorted(data_dir.glob("*.txt")):
            logger.info(f"Loading document: {file_path.name}")
            content = file_path.read_text(encoding="utf-8")
            documents.append({
                "content": content,
                "source": file_path.name
            })

        # Also load .pdf files
        for file_path in sorted(data_dir.glob("*.pdf")):
            logger.info(f"Loading document: {file_path.name}")
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                content_parts = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        content_parts.append(text)
                content = "\n\n".join(content_parts)
                documents.append({
                    "content": content,
                    "source": file_path.name
                })
            except Exception as e:
                logger.error(f"Error loading PDF {file_path.name}: {e}")

        logger.info(f"Loaded {len(documents)} documents")
        return documents


    def chunk_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> list[str]:
        """
        Split text into overlapping chunks based on character count,
        respecting paragraph and sentence boundaries.

        Args:
            text: Full document text.
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Number of overlapping characters between chunks.

        Returns:
            List of text chunks.
        """
        chunk_size = chunk_size or settings.CHUNK_SIZE
        chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        # Split by paragraphs first (double newline)
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds chunk size, save current and start new
            if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep overlap from the end of the current chunk
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def generate_chunk_id(self, source: str, chunk_index: int, content: str) -> str:
        """Generate a deterministic ID for a chunk based on content hash."""
        hash_input = f"{source}:{chunk_index}:{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def ingest_documents(self):
        """
        Run the full ingestion pipeline:
        Load → Chunk → Embed → Store in ChromaDB.
        """
        if self.collection is None:
            self.initialize()

        # Check if documents are already indexed
        if self.collection.count() > 0:
            logger.info(f"Collection already has {self.collection.count()} chunks. Skipping ingestion.")
            self.documents_count = self.collection.count()
            return

        # Step 1: Load documents
        documents = self.load_documents()
        if not documents:
            logger.warning("No documents found to ingest")
            return

        all_chunks = []
        all_metadatas = []
        all_ids = []

        # Step 2: Chunk documents
        for doc in documents:
            chunks = self.chunk_text(doc["content"])
            for i, chunk in enumerate(chunks):
                chunk_id = self.generate_chunk_id(doc["source"], i, chunk)
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": doc["source"],
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
                all_ids.append(chunk_id)

        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")

        # Step 3: Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = embedding_service.generate_embeddings(all_chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")

        # Step 4: Store in ChromaDB (in batches to avoid memory issues)
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            end = min(i + batch_size, len(all_chunks))
            self.collection.add(
                ids=all_ids[i:end],
                documents=all_chunks[i:end],
                embeddings=embeddings[i:end],
                metadatas=all_metadatas[i:end]
            )
            logger.info(f"Indexed batch {i // batch_size + 1}: chunks {i} to {end}")

        self.documents_count = self.collection.count()
        logger.info(f"Ingestion complete. Total chunks in collection: {self.documents_count}")

    def reset_collection(self):
        """Delete and recreate the collection. Useful for re-ingestion."""
        if self.chroma_client:
            self.chroma_client.delete_collection(settings.COLLECTION_NAME)
            self.collection = self.chroma_client.create_collection(
                name=settings.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            self.documents_count = 0
            logger.info("Collection reset successfully")


# Singleton instance
ingestion_pipeline = DocumentIngestionPipeline()
