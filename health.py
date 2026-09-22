from fastapi import APIRouter, Depends, HTTPException
import httpx
import chromadb
from config import get_settings

router = APIRouter(tags=["health"])


def _check_ollama(settings) -> bool:
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{settings.ollama_host}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


def _check_chroma(settings) -> bool:
    try:
        client = chromadb.PersistentClient(path=settings.db_path)
        client.heartbeat()
        return True
    except Exception:
        return False


@router.get("/health")
async def liveness():
    """Liveness probe - always returns ok if process running."""
    return {"status": "ok"}


@router.get("/ready")
async def readiness():
    """Readiness probe - checks Ollama and ChromaDB connectivity."""
    settings = get_settings()
    ollama_ok = _check_ollama(settings)
    chroma_ok = _check_chroma(settings)

    if not (ollama_ok and chroma_ok):
        raise HTTPException(status_code=503, detail={"ollama": ollama_ok, "chroma": chroma_ok})
    return {"status": "ready", "ollama": ollama_ok, "chroma": chroma_ok}