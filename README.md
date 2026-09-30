# 🎓 Campus Assistant - RAG Chatbot for University Documents

A chat app where students ask questions about university rules, admission details, hostel and
placement policies, and scheme/syllabus documents. Answers are generated **only from the uploaded
official PDFs**, with document and page-number citations. An admin can add or remove PDFs from a
password-protected page without touching the code.

> Built as a learning project to practise Retrieval-Augmented Generation (RAG), vector search and
> deploying an LLM app end to end.

**Live demo:** `<add your Streamlit link here>`
**Screenshots:** `<add 2-3 screenshots: chat with sources, admin page>`

---

## Features

**Student chat**
- Chat interface with conversation history and starter questions
- Answers grounded in the documents, with `[1] [2]` citations and an expandable **Sources** panel (file + page + text snippet)
- "I couldn't find this" fallback when nothing relevant is retrieved (similarity threshold)
- Understands Hinglish / short / follow-up questions (LLM-based query rewriting)
- Filter to search only selected documents
- Optional debug view: retrieval scores and the rewritten search query

**Admin panel** (`pages/1_Admin.py`)
- Password login
- Upload one or many PDFs; live progress bar while embedding
- Duplicate detection (same content under a different name is skipped)
- Uploading a file with the same name replaces the old version
- Delete documents (no re-embedding needed)
- "Rebuild index" option to re-process everything from the stored PDFs

---

## Tech stack

| Layer | Tool |
|---|---|
| Language / UI | Python, Streamlit (`st.chat_message`, multipage app) |
| PDF parsing | pypdf |
| Embeddings | FastEmbed - `BAAI/bge-small-en-v1.5` (384-dim, runs on CPU) |
| Vector search | FAISS (`IndexFlatIP`, cosine similarity on normalised vectors) |
| LLM | Groq API (`openai/gpt-oss-120b` for answers, `openai/gpt-oss-20b` for query rewriting) |
| Persistent storage | Supabase Storage (private bucket) |
| Hosting | Streamlit Community Cloud |

---

## How it works

```
                ADMIN UPLOAD                                     STUDENT QUESTION
                                                                        │
  PDF ──► extract text (per page) ──► clean ──► split into           rewrite as clear English
          overlapping chunks (800 chars)                              search query (small LLM)
                    │                                                    │
                    ▼                                                    ▼
         embed chunks (bge-small)                              embed query ──► FAISS search
                    │                                                    │
                    ▼                                            top-k chunks above threshold
   Supabase Storage: pdfs/, chunks.json,                                 │
   vectors.npy, manifest.json                                            ▼
                    │                                   LLM answers ONLY from these chunks
                    └──── loaded at app start ──►       + citations (file, page)
                          FAISS index in memory
```

**1. Ingestion.** Each PDF is read page by page, cleaned (hyphenated line breaks, extra whitespace)
and split into ~800-character overlapping chunks that try to end on sentence boundaries. Every chunk
keeps its `source` file name and `page` number so it can be cited later.

**2. Embedding and storage.** Chunks are embedded in batches and L2-normalised, so inner product
equals cosine similarity. Chunks, vectors, a manifest (hash, pages, chunk count, upload time) and the
original PDFs are saved to Supabase Storage. Adding a PDF embeds **only the new file** and appends to
the stored vectors; deleting a PDF just drops its rows.

**3. Retrieval.** At app start the vectors are loaded into an in-memory FAISS index. For each
question, a small LLM rewrites it into a standalone English search query (translating Hinglish,
expanding "maths" → "mathematics", resolving follow-ups). The query is embedded and the top-k
chunks are fetched; chunks below a similarity threshold are discarded.

**4. Generation.** The LLM gets the numbered chunks and a strict prompt: answer only from the
context, cite passages, say when the answer is not found, and mention which programme a fact belongs
to when several programmes appear.

---

## Project structure

```
campus-rag/
├── app.py               # student chat page
├── pages/1_Admin.py     # admin page (login, upload, delete, rebuild)
├── rag_core.py          # chunking, embeddings, add/remove/rebuild, retrieval
├── storage.py           # Supabase Storage backend (or local folder fallback)
├── requirements.txt
└── .streamlit/secrets.toml   # keys - NOT committed
```

---

## Setup

### 1. Run locally
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

### 2. Secrets (`.streamlit/secrets.toml`)
```toml
GROQ_API_KEY = "..."
ADMIN_PASSWORD = "..."
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "..."            # service_role / secret key - keep private
SUPABASE_BUCKET = "your-bucket-name"
GROQ_MODEL = "openai/gpt-oss-120b"        # optional
# GROQ_FAST_MODEL = "openai/gpt-oss-20b"  # optional, used for query rewriting
```
Without the Supabase lines the app uses a temporary local folder (`storage_local/`), which is fine
for testing but is wiped on cloud restarts.

### 3. Supabase
Create a project → Storage → new **private** bucket → put its name in `SUPABASE_BUCKET`.

### 4. Use it
Open the **Admin** page → log in → upload PDFs → go back to the chat page and ask questions.

### 5. Deploy (Streamlit Community Cloud)
Push the code (never `secrets.toml`) to GitHub → create the app with main file `app.py` → paste the
secrets under *Advanced settings → Secrets*.

---

## Challenges and how I handled them

| Challenge | What happened | Solution |
|---|---|---|
| **Cloud storage is temporary** | Files saved on Streamlit Cloud disappear on restart, so admin uploads would be lost | Stored PDFs and the index in Supabase Storage; the app loads them at startup |
| **Duplicate documents** | Several copies of the same PDF (`(1)`, `(2)`...) would return the same passage repeatedly and slow indexing | SHA-256 hash of each file; identical content is skipped, same name replaces the old version |
| **Slow, silent indexing** | Embedding ~1,000+ pages looked like the program had hung | Batched embedding with a progress bar; only new files are embedded, deletes need no re-embedding |
| **Hinglish and short queries missed the right passage** | "engineering maths ka syllabus batao" failed, while "Engineering Mathematics" worked (English embedding model, abbreviations) | Added an LLM query-rewriting step, larger default top-k, and a debug view to inspect the rewritten query |
| **Model availability changed** | The Groq model I first used returned "model not found" | Made models configurable through secrets and switched to currently available models |
| **Hallucination risk** | LLMs answer confidently even when the document has no answer | Similarity threshold + strict prompt + mandatory citations + "not found" fallback |
| **Storage-safe file names** | Names with spaces/brackets are awkward as storage keys | Sanitised file names before saving |

---

## Limitations

- **Scanned PDFs are not supported** (no OCR). PDFs must contain selectable text.
- **Tables and complex layouts** (marks schemes, credit tables) may be split awkwardly by the simple character-based chunker, so answers about tables can be incomplete.
- **Not formally evaluated yet.** I have tested it manually on sample questions but have no measured accuracy figure. A labelled test set (question → expected document/page) is still to do.
- **Similarity threshold is hand-tuned** and may need re-tuning as more documents are added.
- **Multi-programme confusion:** when several branches share similar wording, the bot can mix them up unless the question names the programme or the document filter is used.
- **English embedding model.** Hinglish works only because the query is rewritten to English first; retrieval quality depends on that rewrite.
- **Simple admin auth:** one shared password. Not suitable for sensitive documents or many admins.
- **Free-tier constraints:** Supabase free projects pause after a week of inactivity and have small storage and per-file limits; Groq has rate limits; Streamlit Community Cloud apps sleep when idle and have limited RAM.
- **No per-user chat history** is stored; history lives in the browser session only.
- **Answers can be wrong or outdated.** Always verify important information with the official university office.

---

## Future improvements

- Evaluation script: test questions with expected source page, report top-k hit rate
- Hybrid search (BM25 keywords + vectors) and a re-ranker for exact names like subject titles
- OCR for scanned PDFs
- Table-aware / structure-aware chunking for syllabus and scheme documents
- Multilingual embedding model
- Per-document "last updated" metadata and answer feedback (thumbs up/down)
- Proper authentication (Supabase Auth) and multiple admin roles

---

## Disclaimer

This is an educational project. It is not an official university service, and answers may be
incomplete or incorrect. Do not upload documents containing personal or confidential student data.
