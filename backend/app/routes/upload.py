"""
Upload API routes.
Handles document upload, parsing (PDF/TXT/MD), chunking, embedding,
and dynamic ChromaDB indexing with production-grade validation and logging.
"""

import os
import logging
import json
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.config import settings
from app.models.schemas import UploadResponse
from app.rag.ingestion import ingestion_pipeline
from app.rag.embeddings import embedding_service
from app.utils.rate_limit import check_upload_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["upload"])

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_FILE_SIZE_MB = 10


class DocumentMetadataStore:
    """Persistent storage for document metadata such as chunk count and upload timestamp."""
    def __init__(self):
        self.metadata_path = Path(settings.METADATA_PATH)
        os.makedirs(self.metadata_path.parent, exist_ok=True)
        self._load()

    def _load(self):
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading document metadata: {e}")
                self.data = {}
        else:
            self.data = {}

    def save(self):
        try:
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving document metadata: {e}")

    def get_all(self) -> dict:
        self._load()
        return self.data

    def set(self, filename: str, info: dict):
        self.data[filename] = info
        self.save()

    def delete(self, filename: str):
        if filename in self.data:
            del self.data[filename]
            self.save()


metadata_store = DocumentMetadataStore()


@router.post("/upload", response_model=UploadResponse, dependencies=[Depends(check_upload_rate_limit)])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, TXT, or MD) and dynamically index it into ChromaDB.
    Validates file type and size, parses content, chunks, embeds, and stores.
    """
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    # Prevent duplicate uploads
    target_path = Path(settings.DATA_DIR) / filename
    if target_path.exists():
        logger.warning(f"[UPLOAD] Rejected duplicate file: {filename}")
        raise HTTPException(
            status_code=400,
            detail=f"A document named '{filename}' already exists in the knowledge base."
        )

    logger.info(f"[UPLOAD] Received file: {filename} (type={ext})")

    # Validate file extension
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"[UPLOAD] Rejected unsupported file type: {ext}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Read file content
    try:
        content_bytes = await file.read()
    except Exception as e:
        logger.error(f"[UPLOAD] Failed to read file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")

    # Validate file size
    file_size_mb = len(content_bytes) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({file_size_mb:.1f}MB). Maximum allowed: {MAX_FILE_SIZE_MB}MB."
        )

    # Ensure upload directory exists
    upload_dir = Path(settings.DATA_DIR)
    os.makedirs(upload_dir, exist_ok=True)
    target_path = upload_dir / filename

    # Save the file to the knowledge directory
    try:
        with open(target_path, "wb") as f:
            f.write(content_bytes)
        logger.info(f"[UPLOAD] Saved file to: {target_path} ({file_size_mb:.2f}MB)")
    except Exception as e:
        logger.error(f"[UPLOAD] Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Parse content based on file type
    try:
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(target_path)
            content_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content_parts.append(text)
            content = "\n\n".join(content_parts)
            logger.info(f"[UPLOAD] Parsed PDF: {len(reader.pages)} pages")
        else:
            content = content_bytes.decode("utf-8", errors="ignore")

        if not content.strip():
            if target_path.exists():
                os.remove(target_path)
            raise HTTPException(status_code=400, detail="The uploaded file contains no extractable text.")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[UPLOAD] Error parsing file: {e}")
        if target_path.exists():
            os.remove(target_path)
        raise HTTPException(status_code=500, detail=f"Error parsing file content: {str(e)}")

    # Chunk, embed, and index
    try:
        if ingestion_pipeline.collection is None:
            ingestion_pipeline.initialize()

        chunks = ingestion_pipeline.chunk_text(content)
        if not chunks:
            if target_path.exists():
                os.remove(target_path)
            raise HTTPException(status_code=400, detail="File could not be split into valid text chunks.")

        all_ids = []
        all_metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = ingestion_pipeline.generate_chunk_id(filename, i, chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "source": filename,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })

        logger.info(f"[UPLOAD] Generating embeddings for {len(chunks)} chunks...")
        embeddings = embedding_service.generate_embeddings(chunks)

        ingestion_pipeline.collection.add(
            ids=all_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=all_metadatas
        )

        ingestion_pipeline.documents_count = ingestion_pipeline.collection.count()
        logger.info(f"[UPLOAD] Successfully indexed {len(chunks)} chunks from {filename} | total={ingestion_pipeline.documents_count}")

        # Save to persistent metadata store
        doc_info = {
            "filename": filename,
            "size_kb": round(len(content_bytes) / 1024, 1),
            "type": ext.lstrip("."),
            "chunks": len(chunks),
            "uploaded_at": datetime.utcnow().isoformat() + "Z"
        }
        metadata_store.set(filename, doc_info)

        return UploadResponse(
            status="success",
            message=f"Successfully indexed {filename}",
            filename=filename,
            chunks_added=len(chunks),
            total_indexed_chunks=ingestion_pipeline.documents_count
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[UPLOAD] Error indexing file: {e}", exc_info=True)
        if target_path.exists():
            os.remove(target_path)
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")


@router.get("/documents")
async def list_documents():
    """List all uploaded documents and their indexing status."""
    if ingestion_pipeline.collection is None:
        ingestion_pipeline.initialize()

    metadata = metadata_store.get_all()
    base_docs = []
    data_dir = Path(settings.DATA_DIR)
    
    if data_dir.exists():
        for fp in sorted(data_dir.iterdir()):
            if fp.suffix.lower() in {".md", ".txt", ".pdf"}:
                filename = fp.name
                
                # Check if we have persistent metadata
                if filename in metadata:
                    doc_info = metadata[filename]
                else:
                    # Sync and discover metadata
                    size_kb = round(fp.stat().st_size / 1024, 1)
                    file_type = fp.suffix.lower().lstrip(".")
                    mtime = datetime.utcfromtimestamp(fp.stat().st_mtime).isoformat() + "Z"
                    
                    # Query ChromaDB for chunk count
                    try:
                        res = ingestion_pipeline.collection.get(where={"source": filename}, include=[])
                        chunks = len(res["ids"]) if res and "ids" in res else 0
                    except Exception:
                        chunks = 0
                    
                    doc_info = {
                        "filename": filename,
                        "size_kb": size_kb,
                        "type": file_type,
                        "chunks": chunks,
                        "uploaded_at": mtime
                    }
                    metadata_store.set(filename, doc_info)
                
                base_docs.append(doc_info)

    return {
        "documents": base_docs,
        "total_chunks": ingestion_pipeline.documents_count
    }


@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Delete a document from the knowledge base:
    1. Remove the file from disk.
    2. Delete corresponding chunks from ChromaDB.
    3. Update the global uploaded documents tracker and collection count.
    """
    logger.info(f"[DELETE] Request to delete document: {filename}")
    
    # Secure filename to prevent directory traversal
    safe_filename = Path(filename).name
    target_path = Path(settings.DATA_DIR) / safe_filename
    
    file_existed = False
    try:
        if target_path.exists():
            os.remove(target_path)
            file_existed = True
            logger.info(f"[DELETE] Removed file from disk: {target_path}")
    except Exception as e:
        logger.error(f"[DELETE] Failed to delete file {target_path} from disk: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file from disk: {str(e)}")

    # Delete from ChromaDB
    try:
        if ingestion_pipeline.collection is None:
            ingestion_pipeline.initialize()
            
        if ingestion_pipeline.collection:
            # Delete from ChromaDB where source matches filename
            ingestion_pipeline.collection.delete(where={"source": safe_filename})
            ingestion_pipeline.documents_count = ingestion_pipeline.collection.count()
            logger.info(f"[DELETE] Deleted chunks for {safe_filename} from ChromaDB. Remaining count: {ingestion_pipeline.documents_count}")
    except Exception as e:
        logger.error(f"[DELETE] Failed to delete chunks from ChromaDB: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove document chunks from search index: {str(e)}")

    # Remove from metadata store
    metadata_store.delete(safe_filename)

    if not file_existed:
        logger.warning(f"[DELETE] File {safe_filename} not found on disk, but ChromaDB cleanup attempted.")
        return {"status": "success", "message": f"Cleaned up document '{safe_filename}' search index."}

    return {"status": "success", "message": f"Successfully deleted document '{safe_filename}'."}
