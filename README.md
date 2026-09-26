# RAG Document Q&A

Kullanıcının yüklediği **PDF, TXT veya DOCX** dosyalarını okuyup, içerikle ilgili
soruları OpenAI veya Claude (Anthropic) API'sini kullanarak yanıtlayan bir
Retrieval-Augmented Generation (RAG) web uygulaması.

## Nasıl çalışır

1. **Yükleme & parçalama (chunking):** Dosya yüklenince metni çıkarılır
   (`pypdf`, `python-docx` veya düz metin okuma ile), ardından örtüşen
   parçalara (chunk) bölünür.
2. **Embedding:** Her parça, seçilen sağlayıcının embedding modeliyle
   (varsayılan: OpenAI `text-embedding-3-small`) vektöre çevrilir ve basit bir
   in-memory / diske kalıcı vektör deposuna (`numpy` tabanlı cosine similarity)
   eklenir.
3. **Soru sorma:** Kullanıcı bir soru sorduğunda, soru da embed edilir; vektör
   deposundan en alakalı `top_k` parça bulunur (retrieval).
4. **Yanıt üretme (generation):** Bulunan parçalar bağlam olarak prompt'a
   eklenir ve seçilen LLM'e (OpenAI GPT veya Claude) gönderilir; model yalnızca
   verilen bağlama dayanarak yanıt üretir ve hangi kaynaktan geldiğini belirtir.

```
Kullanıcı dosyası → metin çıkarımı → chunking → embedding → vektör deposu
                                                                   │
Kullanıcı sorusu → embedding ──────────────► en yakın chunk'lar ──┘
                                                    │
                                        LLM'e bağlam olarak eklenir
                                                    │
                                              Yanıt + kaynak chunk'lar
```

## Mimari

```
rag-document-qa/
├── app.py                 FastAPI uygulaması (upload + ask endpoint'leri, statik frontend)
├── rag/
│   ├── loaders.py         PDF / TXT / DOCX'den metin çıkarma
│   ├── chunking.py        Metni örtüşen parçalara bölme
│   ├── embeddings.py      OpenAI / Anthropic embedding sağlayıcı soyutlaması
│   ├── vector_store.py    numpy tabanlı basit vektör deposu (JSON'a kalıcı)
│   └── qa.py              Retrieval + prompt oluşturma + LLM çağrısı
├── static/index.html      Tek sayfalık yükleme + sohbet arayüzü
├── uploads/                Yüklenen ham dosyalar
├── data/                   Kalıcı vektör deposu (store.json)
└── requirements.txt
```

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # API anahtarlarını doldur
uvicorn app:app --reload       # http://localhost:8000
```

## Ortam değişkenleri (`.env`)

```
# En az biri gerekli — hangisi ayarlıysa o sağlayıcı kullanılır
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Hangi sağlayıcının kullanılacağı: "openai" veya "anthropic"
LLM_PROVIDER=openai

# Embedding her zaman OpenAI üzerinden yapılır (Anthropic embedding API sunmuyor);
# LLM_PROVIDER=anthropic seçiliyse bile embedding için OPENAI_API_KEY gereklidir.
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL_OPENAI=gpt-4o-mini
CHAT_MODEL_ANTHROPIC=claude-sonnet-4-6
```

> Not: Anthropic şu an genel kullanıma açık bir embedding API'si sunmuyor, bu
> yüzden retrieval adımı için embedding'ler daima OpenAI ile üretilir; yalnızca
> **cevap üretme (generation)** adımında `LLM_PROVIDER=anthropic` ile Claude
> kullanılabilir.

## Kullanım

1. Tarayıcıda `http://localhost:8000` adresini aç.
2. Bir PDF/TXT/DOCX dosyası yükle — işlenip vektör deposuna eklenir.
3. Dosyayla ilgili bir soru yaz; yanıt, hangi parçalardan (chunk) alıntı
   yapıldığı bilgisiyle birlikte döner.

## Sınırlamalar & genişletme fikirleri

- Vektör deposu basit ve tek işlemli (in-memory + JSON); üretimde
  Pinecone / Chroma / Qdrant gibi gerçek bir vektör veritabanına geçilmeli.
- Çok kullanıcılı/çok oturumlu izolasyon yok — her dosya global depoya eklenir.
- Büyük PDF'lerde OCR desteklenmiyor (taranmış/görsel PDF'ler için ek adım gerekir).
- Sohbet geçmişi (multi-turn context) şu an tutulmuyor, her soru bağımsız değerlendirilir.

## Lisans

MIT
