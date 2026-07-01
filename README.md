# RAG LLMOps

A Retrieval-Augmented Generation service: upload documents, ask questions, get
answers grounded in your own corpus with cited sources. Built to run entirely
on free tiers - local embeddings (no API key) + Google Gemini's free tier for
generation - so it costs nothing to run or deploy.

## Architecture

```
                 ┌─────────────────────┐
  file upload -> │  DocumentProcessor   │  load -> clean -> chunk (RecursiveCharacterTextSplitter)
                 └──────────┬───────────┘
                            v
                 ┌─────────────────────┐
                 │  ChromaVectorStore   │  local HuggingFace embeddings -> ChromaDB (persisted to disk)
                 └──────────┬───────────┘
                            v
  question    -> ┌─────────────────────┐
                 │      RAGChain        │  retriever | prompt | Gemini (gemini-2.5-flash) | parser
                 └──────────┬───────────┘
                            v
                    answer + cited sources

        Django REST API (backend/) exposes this as HTTP endpoints.
        evaluation/ scores answer quality with RAGAS, tracked in MLflow.
```

`rag/service.py` is the single facade tying these pieces together; Django
views, tests, and the evaluation script all depend on it rather than talking
to the pipeline directly.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # or: source venv/bin/activate on Linux/Mac
pip install -r requirements.txt

cp .env.example .env
# Add your free Gemini key (https://aistudio.google.com/apikey) to .env:
#   GOOGLE_API_KEY=...

python manage.py migrate
```

## Running

```bash
python manage.py runserver
```

```bash
# Health check
curl http://127.0.0.1:8000/api/health/

# Upload a document (.pdf, .docx, .txt, .md)
curl -F "file=@your_doc.pdf" http://127.0.0.1:8000/api/documents/

# Ask a question
curl -X POST -H "Content-Type: application/json" \
     -d '{"question": "What is this document about?"}' \
     http://127.0.0.1:8000/api/ask/
```

## Frontend

A small React + Vite UI lives in `frontend/` - upload docs, chat with the
index, see cited sources. Needs the Django server running too:

```bash
# terminal 1, repo root
python manage.py runserver

# terminal 2
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The dev server proxies `/api/*` to Django on
port 8000, so no CORS setup is needed locally. Nothing persists across a
page refresh (chat history and the "uploaded this session" list are just
React state) - matches the backend, which also doesn't keep uploaded files
around, only their indexed chunks.

## Tests

```bash
pytest
```

Unit tests use local embeddings and a fake LLM - no API key required. One
live Gemini integration test is skipped automatically unless `GOOGLE_API_KEY`
is set in the environment.

## Evaluation

```bash
python evaluation/run_eval.py
```

Indexes a sample document, runs a small Q&A set through the real pipeline,
and scores the answers with RAGAS (faithfulness, answer relevancy, context
precision, context recall) using Gemini as the judge model. Results are
logged to MLflow (`mlflow ui` to view) and written to
`evaluation/results/latest.json` (gitignored - it's a local run artifact,
regenerate it yourself rather than trusting a stale copy).

## Known limitations

- Uploaded files are not retained after ingestion - only their indexed chunks
  persist in ChromaDB.
- `data/chroma_db/` is local disk storage; on ephemeral free hosting tiers
  this won't survive a redeploy without an external volume.
- The local embedding model needs ~1GB RAM at runtime (torch + the model).
- **Free-tier Gemini quota is tight for evaluation.** A single
  `evaluation/run_eval.py` run makes ~25 Gemini calls (5 to generate answers +
  up to 20 RAGAS judge calls). Some free-tier API keys are capped at as few as
  20 `gemini-2.5-flash` requests/day, which one eval run alone can exhaust -
  you'll see `RESOURCE_EXHAUSTED` (429) errors and `NaN` scores if that
  happens. Check your quota at https://ai.dev/rate-limit, wait for it to
  reset, or reduce `evaluation/dataset.py`'s question count.

## Sprint history

1. ChromaDB + embeddings wiring
2. Document loading, cleaning, and chunking pipeline
3. RAG chain (retriever -> prompt -> Gemini) with cited sources
4. Django REST API (health / ask / document upload)
5. RAGAS evaluation + MLflow tracking
6. React + Vite frontend
