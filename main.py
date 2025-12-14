import os
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

from rag_engine import AwareRAGEngine

APP_NAME = "WHO Antibiotic Guide API"
PDF_PATH = os.getenv("PDF_PATH", "WHOAMR.pdf")

API_AUTH_KEY = os.getenv("APP_API_KEY", "")  # optional; you can leave it empty at first

class ChatRequest(BaseModel):
    question: str
    top_k: int = 5
    temperature: float = 0.0

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict]

app = FastAPI(title=APP_NAME)

engine: Optional[AwareRAGEngine] = None

@app.on_event("startup")
def startup() -> None:
    global engine
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is required.")
    if not os.path.exists(PDF_PATH):
        raise RuntimeError(f"PDF not found at {PDF_PATH}; keep WHOAMR.pdf in api folder or set PDF_PATH.")
    engine = AwareRAGEngine(
        pdf_path=PDF_PATH,
        openai_api_key=openai_key,
        chunk_size=int(os.getenv("CHUNK_SIZE", "1500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
    )

@app.get("/health")
def health() -> Dict:
    return {"status": "ok", "app": APP_NAME}

def _auth(x_api_key: Optional[str]) -> None:
    if not API_AUTH_KEY:
        return
    if not x_api_key or x_api_key != API_AUTH_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, x_api_key: Optional[str] = Header(default=None)) -> ChatResponse:
    _auth(x_api_key)
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    answer, sources = engine.answer(req.question, top_k=req.top_k, temperature=req.temperature)
    return ChatResponse(answer=answer, sources=sources)
