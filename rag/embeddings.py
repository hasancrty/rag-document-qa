"""Embedding sağlayıcı soyutlaması. Şu an yalnızca OpenAI destekleniyor
(Anthropic genel kullanıma açık bir embedding API'si sunmuyor)."""
import os

from openai import OpenAI

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY tanımlı değil. Embedding üretimi için gereklidir "
                "(LLM_PROVIDER=anthropic seçili olsa bile)."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Birden fazla metni tek istekte embed eder."""
    if not texts:
        return []
    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    client = _get_client()
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
