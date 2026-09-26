"""Retrieval-Augmented Generation çekirdek mantığı: en alakalı chunk'ları
bulur, bir prompt oluşturur ve seçilen LLM sağlayıcısından (OpenAI ya da
Anthropic/Claude) yalnızca verilen bağlama dayanan bir yanıt ister."""
import os

from .embeddings import embed_query
from .vector_store import VectorStore, StoredChunk

SYSTEM_PROMPT = (
    "Sen, kullanıcının yüklediği dokümanlar hakkında soru yanıtlayan bir "
    "asistansın. Yalnızca aşağıda verilen bağlamı (context) kullanarak yanıt "
    "ver. Bağlamda yanıt yoksa, bunu açıkça belirt; bilgi uydurma. Yanıtını "
    "Türkçe ver ve hangi kaynak parçadan ([Kaynak N]) yararlandığını belirt."
)


def _build_context(chunks: list[StoredChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[Kaynak {i} — {c.doc_name}, parça {c.chunk_index}]\n{c.text}")
    return "\n\n".join(parts)


def answer_question(question: str, store: VectorStore, top_k: int = 4) -> dict:
    query_embedding = embed_query(question)
    matches = store.search(query_embedding, top_k=top_k)

    if not matches:
        return {
            "answer": "Henüz hiçbir doküman yüklenmedi. Önce bir dosya yükleyin.",
            "sources": [],
        }

    context = _build_context(matches)
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()

    user_prompt = (
        f"Bağlam:\n{context}\n\n"
        f"Soru: {question}\n\n"
        "Yalnızca yukarıdaki bağlama dayanarak yanıt ver."
    )

    if provider == "anthropic":
        answer = _call_anthropic(user_prompt)
    else:
        answer = _call_openai(user_prompt)

    return {
        "answer": answer,
        "sources": [
            {"doc_name": c.doc_name, "chunk_index": c.chunk_index, "text": c.text}
            for c in matches
        ],
    }


def _call_openai(user_prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("CHAT_MODEL_OPENAI", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def _call_anthropic(user_prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.environ.get("CHAT_MODEL_ANTHROPIC", "claude-sonnet-4-6")

    response = client.messages.create(
        model=model,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
