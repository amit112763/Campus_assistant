"""
Core RAG logic (no Streamlit code here).

Stored in the store (Supabase or local):
    pdfs/<name>.pdf   original PDFs
    chunks.json       [{source, page, text}, ...]
    vectors.npy       one embedding row per chunk (normalised)
    manifest.json     {name: {sha256, pages, chunks, uploaded}}

Adding a PDF only embeds the NEW PDF. Deleting a PDF just drops its rows - no re-embedding.
"""
import hashlib
import io
import json
import re
from datetime import datetime
from functools import lru_cache

import faiss
import numpy as np
from pypdf import PdfReader

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384  # output size of bge-small
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# ------------------------------------------------------------------ text
def safe_name(filename: str) -> str:
    """Supabase keys dislike spaces/brackets, so keep only safe characters."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("_")
    return name if name.lower().endswith(".pdf") else name + ".pdf"


def clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind("; ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        piece = text[start:end].strip()
        if len(piece) > 50:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_chunks(data: bytes, name: str):
    reader = PdfReader(io.BytesIO(data))
    records, pages_with_text = [], 0
    for page_no, page in enumerate(reader.pages, start=1):
        text = clean(page.extract_text() or "")
        if not text:
            continue
        pages_with_text += 1
        for chunk in split_text(text):
            records.append({"source": name, "page": page_no, "text": chunk})
    return records, len(reader.pages), pages_with_text


# ------------------------------------------------------------- embeddings
@lru_cache(maxsize=1)
def get_embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(
        model_name=EMBED_MODEL,
        threads=1
    )


def _normalize(v: np.ndarray) -> np.ndarray:
    v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    return np.ascontiguousarray(v, dtype="float32")


def embed_texts(texts, progress=None, batch_size=32) -> np.ndarray:
    model, out = get_embedder(), []
    for i in range(0, len(texts), batch_size):
        out.extend(model.embed(texts[i:i + batch_size]))
        if progress:
            progress(min(i + batch_size, len(texts)), len(texts))
    return _normalize(np.array(out, dtype="float32"))


def embed_query(query: str) -> np.ndarray:
    return _normalize(np.array(list(get_embedder().query_embed([query])), dtype="float32"))


# ----------------------------------------------------------- stored state
def _empty_state():
    return {"manifest": {}, "chunks": [], "vectors": np.zeros((0, DIM), dtype="float32")}


def load_state(store):
    m, c, v = store.get("manifest.json"), store.get("chunks.json"), store.get("vectors.npy")
    if not (m and c and v):
        return _empty_state()
    return {
        "manifest": json.loads(m),
        "chunks": json.loads(c),
        "vectors": np.load(io.BytesIO(v)),
    }


def save_state(store, state):
    buf = io.BytesIO()
    np.save(buf, state["vectors"])
    store.put("vectors.npy", buf.getvalue())
    store.put("chunks.json", json.dumps(state["chunks"], ensure_ascii=False).encode("utf-8"),
              "application/json")
    store.put("manifest.json", json.dumps(state["manifest"], indent=1).encode("utf-8"),
              "application/json")


def _drop(state, name):
    keep = np.array([c["source"] != name for c in state["chunks"]], dtype=bool)
    state["chunks"] = [c for c, k in zip(state["chunks"], keep) if k]
    state["vectors"] = state["vectors"][keep] if len(keep) else state["vectors"]
    state["manifest"].pop(name, None)
    return state


# ------------------------------------------------------------ admin actions
def add_pdf(store, filename, data: bytes, progress=None):
    """Returns (ok: bool, message: str)."""
    name = safe_name(filename)
    digest = hashlib.sha256(data).hexdigest()
    state = load_state(store)

    for other, meta in state["manifest"].items():
        if meta["sha256"] == digest:
            if other == name:
                return False, f"'{name}' is already in the knowledge base (same file)."
            return False, f"'{filename}' is identical to '{other}' - skipped (duplicate)."

    replaced = name in state["manifest"]
    if replaced:
        state = _drop(state, name)

    records, n_pages, n_text = extract_chunks(data, name)
    if not records:
        return False, f"'{filename}': no text found (probably a scanned PDF - needs OCR)."

    new_vecs = embed_texts([r["text"] for r in records], progress)
    state["chunks"] += records
    state["vectors"] = np.vstack([state["vectors"], new_vecs]) if len(state["vectors"]) else new_vecs
    state["manifest"][name] = {
        "sha256": digest,
        "pages": n_pages,
        "chunks": len(records),
        "uploaded": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    store.put(f"pdfs/{name}", data, "application/pdf")
    save_state(store, state)
    verb = "Replaced" if replaced else "Added"
    return True, f"{verb} '{name}': {n_pages} pages ({n_text} with text), {len(records)} chunks."


def remove_pdf(store, name):
    state = _drop(load_state(store), name)
    save_state(store, state)
    store.remove(f"pdfs/{name}")


def rebuild_all(store, progress=None):
    """Re-read every stored PDF and re-embed from scratch (use if something looks broken)."""
    state = _empty_state()
    for name in store.list("pdfs"):
        data = store.get(f"pdfs/{name}")
        if not data:
            continue
        records, n_pages, _ = extract_chunks(data, name)
        if not records:
            continue
        vecs = embed_texts([r["text"] for r in records], progress)
        state["chunks"] += records
        state["vectors"] = np.vstack([state["vectors"], vecs]) if len(state["vectors"]) else vecs
        state["manifest"][name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "pages": n_pages,
            "chunks": len(records),
            "uploaded": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    save_state(store, state)
    return len(state["manifest"]), len(state["chunks"])


# ---------------------------------------------------------------- retrieval
def build_faiss(vectors: np.ndarray):

    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product == cosine (vectors normalised)

    index.add(np.ascontiguousarray(vectors, dtype="float32"))

    return index


def retrieve(kb, query, top_k, allowed_sources):
    chunks = kb["chunks"]
    all_selected = len(allowed_sources) >= len(kb["manifest"])
    k = min(len(chunks), top_k * 5) if all_selected else len(chunks)
    scores, ids = kb["index"].search(embed_query(query), k)
    results = []
    for score, i in zip(scores[0], ids[0]):
        if i == -1:
            continue
        c = chunks[i]
        if c["source"] in allowed_sources:
            results.append({**c, "score": float(score)})
            if len(results) == top_k:
                break
    return results
