"""
Where the knowledge base lives.

- If SUPABASE_URL + SUPABASE_KEY are set  -> Supabase Storage (permanent, works on Streamlit Cloud)
- Otherwise                               -> local folder `storage_local/` (fine for testing on your PC)

Both backends have the same 4 methods: put, get, list, remove.
"""
import os
from functools import lru_cache
from pathlib import Path

import streamlit as st


def get_secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)


class LocalStore:
    label = "Local folder (storage_local/) - temporary, for testing only"

    def __init__(self, root="storage_local"):
        self.root = Path(root)
        self.root.mkdir(exist_ok=True)

    def put(self, path, data: bytes, content_type="application/octet-stream"):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get(self, path):
        p = self.root / path
        return p.read_bytes() if p.exists() else None

    def list(self, prefix):
        d = self.root / prefix
        return sorted(f.name for f in d.iterdir() if f.is_file()) if d.exists() else []

    def remove(self, path):
        (self.root / path).unlink(missing_ok=True)


class SupabaseStore:
    def __init__(self, url, key, bucket):
        from supabase import create_client

        self._bucket = create_client(url, key).storage.from_(bucket)
        self.label = f"Supabase Storage (bucket: {bucket})"

    def put(self, path, data: bytes, content_type="application/octet-stream"):
        self._bucket.upload(path, data, {"content-type": content_type, "upsert": "true"})

    def get(self, path):
        try:
            return self._bucket.download(path)
        except Exception:
            return None

    def list(self, prefix):
        items = self._bucket.list(prefix, {"limit": 1000})
        return sorted(
            i["name"] for i in items
            if i.get("name") and i["name"] != ".emptyFolderPlaceholder"
        )

    def remove(self, path):
        self._bucket.remove([path])


@lru_cache(maxsize=1)
def get_store():
    url, key = get_secret("SUPABASE_URL"), get_secret("SUPABASE_KEY")
    if url and key:
        return SupabaseStore(url, key, get_secret("SUPABASE_BUCKET", "campus-rag"))
    return LocalStore()
