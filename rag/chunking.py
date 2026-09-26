"""Uzun metinleri, embedding ve retrieval için örtüşen (overlapping) parçalara böler."""
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    index: int


def chunk_text(text: str, chunk_size: int = 220, overlap: int = 40) -> list[Chunk]:
    """
    Kelime bazlı, örtüşen chunk'lara böler.

    chunk_size: bir chunk'taki yaklaşık kelime sayısı
    overlap: ardışık chunk'lar arasında tekrar eden kelime sayısı
             (bağlamın chunk sınırında kopmaması için)
    """
    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = max(1, chunk_size - overlap)

    while start < len(words):
        piece = words[start : start + chunk_size]
        chunk_str = " ".join(piece).strip()
        if chunk_str:
            chunks.append(Chunk(text=chunk_str, index=index))
            index += 1
        start += step

    return chunks
