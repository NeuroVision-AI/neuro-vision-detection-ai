from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import traceback

from api.services.rag_service import RAGService

router = APIRouter(prefix="/rag", tags=["RAG"])
rag_service = RAGService()


class ChatRequest(BaseModel):
    message: str
    provider: str = "medgemma"
    image_base64: Optional[str] = None


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Ingest a document into the ChromaDB knowledge base."""
    try:
        file_bytes = await file.read()
        result = rag_service.ingest_document(file_bytes, file.filename)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def chat(request: ChatRequest):
    """Chat with the RAG pipeline using the specified LLM provider."""
    try:
        res = rag_service.chat(request.message, request.provider, request.image_base64)
        return {
            "answer": res["answer"],
            "sources": res["sources"],
            "provider": request.provider
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """Return stats about the ChromaDB knowledge base."""
    return rag_service.get_collection_stats()
