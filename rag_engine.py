import io
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import faiss
from pypdf import PdfReader
from openai import OpenAI

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

STEWARD_FOOTER = (
    "Stewardship note: use the narrowest effective antibiotic; reassess at 48 to 72 hours; "
    "follow local guidance and clinical judgment."
)

WHO_SYSTEM_PROMPT = """
You are WHO Antibiotic Guide; AWaRe (Access, Watch, Reserve) Clinical Assistant.

Purpose: support rational antibiotic use and antimicrobial stewardship using ONLY the provided WHO AWaRe book context.

Safety rules:
1: Use ONLY the provided WHO context; do not use outside knowledge.
2: If the answer is not explicitly supported by the context; say: "Not found in the WHO AWaRe book context provided."
3: Only recommend a "no antibiotic approach" if the retrieved WHO context explicitly states antibiotics are not needed; not recommended; should be avoided; or similar.
4: Do not diagnose; do not replace clinical judgment; do not replace local or national guidelines.

Formatting rules:
A: Answer; one short paragraph.
B: Dosing and duration; bullet points; include mg/kg; route; frequency; duration if present.
C: No antibiotic guidance; either:
   - "No antibiotic approach is appropriate" with WHO justification; OR
   - "Antibiotics are recommended" if WHO indicates antibiotics are needed; OR
   - "Not found in the WHO AWaRe book context provided."
D: Sources; page numbers; short excerpts.

Always end with this line:
Stewardship note: use the narrowest effective antibiotic; reassess at 48 to 72 hours; follow local guidance and clinical judgment.
""".strip()


def ensure_footer(text: str) -> str:
    if not text:
        return STEWARD_FOOTER
    if STEWARD_FOOTER.lower() in text.lower():
        return text
    return (text.rstrip() + "\n\n" + STEWARD_FOOTER).strip()


def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"\s*mg\s*/\s*kg\s*/\s*day", " mg/kg/day", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*mg\s*/\s*kg\s*/\s*dose", " mg/kg/dose", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*IV\s*/\s*IM", " IV/IM", s, flags=re.IGNORECASE)
    return s.strip()


def extract_openai_key(raw: Optional[str]) -> str:
    if not raw:
        return ""
    raw = raw.strip().strip('"').strip("'").strip()
    m = re.search(r"(sk-proj-[A-Za-z0-9_\-]{20,}|sk-[A-Za-z0-9_\-]{20,})", raw)
    if m:
        return m.group(1)
    ascii_only = raw.encode("ascii", errors="ignore").decode("ascii", errors="ignore")
    m2 = re.search(r"(sk-proj-[A-Za-z0-9_\-]{20,}|sk-[A-Za-z0-9_\-]{20,})", ascii_only)
    return m2.group(1) if m2 else ""


@dataclass
class SourceHit:
    score: float
    page: int
    text: str


class AwareRAGEngine:
    def __init__(self, pdf_path: str, openai_api_key: str, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.pdf_path = pdf_path
        self.openai_api_key = extract_openai_key(openai_api_key)
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY missing or invalid; use only sk-... without extra characters.")

        self.client = OpenAI(api_key=self.openai_api_key)
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)

        self.chunks: List[Dict] = []
        self.index: Optional[faiss.Index] = None

        self._load_and_index()

    def _read_pdf_bytes(self) -> bytes:
        with open(self.pdf_path, "rb") as f:
            return f.read()

    def _read_pages(self, pdf_bytes: bytes) -> List[Dict]:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages: List[Dict] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": clean_text(text)})
        return pages

    def _chunk_pages(self, pages: List[Dict]) -> List[Dict]:
        chunks: List[Dict] = []
        for p in pages:
            page_num = p["page"]
            text = p["text"]
            if not text:
                continue
            start = 0
            n = len(text)
            while start < n:
                end = min(start + self.chunk_size, n)
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append({"page": page_num, "text": chunk})
                if end >= n:
                    break
                start = max(0, end - self.chunk_overlap)
        return chunks

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        batch_size = 96
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embeddings.create(model=EMBED_MODEL, input=batch)
            vectors.extend([np.array(d.embedding, dtype=np.float32) for d in resp.data])
        return np.vstack(vectors).astype(np.float32)

    def _build_index(self, vectors: np.ndarray) -> faiss.Index:
        dim = vectors.shape[1]
        idx = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(vectors)
        idx.add(vectors)
        return idx

    def _load_and_index(self) -> None:
        pdf_bytes = self._read_pdf_bytes()
        pages = self._read_pages(pdf_bytes)
        chunks = self._chunk_pages(pages)
        if not chunks:
            raise RuntimeError("No text extracted from PDF; if scanned, use text based PDF or add OCR.")
        vectors = self._embed_texts([c["text"] for c in chunks])
        self.index = self._build_index(vectors)
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int) -> List[SourceHit]:
        if self.index is None:
            raise RuntimeError("Index not built.")
        qvec = self._embed_texts([query])
        faiss.normalize_L2(qvec)
        scores, ids = self.index.search(qvec, int(top_k))
        hits: List[SourceHit] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            c = self.chunks[int(idx)]
            hits.append(SourceHit(score=float(score), page=int(c["page"]), text=c["text"]))
        return hits

    def _context_block(self, hits: List[SourceHit], max_chars: int = 1200) -> str:
        blocks: List[str] = []
        for i, h in enumerate(hits, start=1):
            excerpt = h.text[:max_chars].rstrip()
            if len(h.text) > max_chars:
                excerpt += " ..."
            blocks.append(f"Source {i}; page {h.page}:\n{excerpt}")
        return "\n\n".join(blocks)

    def answer(self, question: str, top_k: int = 5, temperature: float = 0.0) -> Tuple[str, List[Dict]]:
        hits = self.retrieve(question, top_k=top_k)
        if not hits:
            return ensure_footer("Not found in the WHO AWaRe book context provided."), []

        context = self._context_block(hits)
        user_prompt = f"""
WHO AWaRe book context:
{context}

User question:
{question}

Write the answer following the required output format.
""".strip()

        resp = self.client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=float(temperature),
            messages=[
                {"role": "system", "content": WHO_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = ensure_footer(resp.choices[0].message.content or "")

        sources = [
            {"page": h.page, "score": h.score, "excerpt": h.text[:900] + (" ..." if len(h.text) > 900 else "")}
            for h in hits
        ]
        return text, sources
