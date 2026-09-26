"""FastAPI uygulaması: dosya yükleme, işleme ve soru-cevap endpoint'leri."""
import os
import shutil
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag.loaders import extract_text
from rag.chunking import chunk_text
from rag.embeddings import embed_texts
from rag.vector_store import VectorStore
from rag.qa import answer_question

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}

app = FastAPI(title="RAG Document Q&A")
store = VectorStore()


class AskRequest(BaseModel):
    question: str
    top_k: int = 4


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/documents")
def list_documents():
    return {"documents": store.list_documents()}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya türü: {ext}. İzin verilenler: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Aynı isimde çakışmayı önlemek için benzersiz bir ön ek ekle
    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    dest_path = UPLOAD_DIR / safe_name

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        raw_text = extract_text(str(dest_path))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Dosya okunamadı: {e}")

    if not raw_text.strip():
        raise HTTPException(status_code=422, detail="Dosyadan metin çıkarılamadı (boş içerik).")

    chunks = chunk_text(raw_text)
    if not chunks:
        raise HTTPException(status_code=422, detail="Dosya parçalara bölünemedi.")

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    store.add(doc_name=file.filename, texts=texts, embeddings=embeddings)

    return {
        "doc_name": file.filename,
        "chunks_added": len(chunks),
        "documents": store.list_documents(),
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz.")
    try:
        return answer_question(req.question, store, top_k=req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
