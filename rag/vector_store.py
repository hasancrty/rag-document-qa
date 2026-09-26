"""Basit, tek-işlemli (single-process) vektör deposu.

Üretimde Pinecone / Chroma / Qdrant gibi gerçek bir vektör veritabanı
kullanılmalı; bu implementasyon eğitim/demo amaçlıdır: tüm vektörler
belleğe yüklenir, cosine similarity ile brute-force arama yapılır ve
JSON dosyasına kalıcı hale getirilir.
"""
import json
import os
from dataclasses import dataclass, asdict

import numpy as np

STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "store.json")


@dataclass
class StoredChunk:
    id: str
    doc_name: str
    chunk_index: int
    text: str
    embedding: list[float]


class VectorStore:
    def __init__(self):
        self.chunks: list[StoredChunk] = []
        self._load()

    def _load(self):
        if os.path.exists(STORE_PATH):
            with open(STORE_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.chunks = [StoredChunk(**c) for c in raw]

    def _save(self):
        os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
        with open(STORE_PATH, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False)

    def add(self, doc_name: str, texts: list[str], embeddings: list[list[float]]):
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            self.chunks.append(
                StoredChunk(
                    id=f"{doc_name}-{i}",
                    doc_name=doc_name,
                    chunk_index=i,
                    text=text,
                    embedding=emb,
                )
            )
        self._save()

    def list_documents(self) -> list[str]:
        seen = []
        for c in self.chunks:
            if c.doc_name not in seen:
                seen.append(c.doc_name)
        return seen

    def search(self, query_embedding: list[float], top_k: int = 4) -> list[StoredChunk]:
        if not self.chunks:
            return []

        matrix = np.array([c.embedding for c in self.chunks])
        query = np.array(query_embedding)

        # Cosine similarity
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        norms[norms == 0] = 1e-10
        scores = (matrix @ query) / norms

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]
