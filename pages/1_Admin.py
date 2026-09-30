import hmac
import sys
from pathlib import Path

import streamlit as st

# make imports from the project root work inside the pages/ folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_core import add_pdf, load_state, rebuild_all, remove_pdf  # noqa: E402
from storage import get_secret, get_store  # noqa: E402

st.set_page_config(page_title="Admin", page_icon="🔐", layout="centered")
st.title("🔐 Admin panel")

# ------------------------------------------------------------------ login
admin_password = get_secret("ADMIN_PASSWORD")
if not admin_password:
    st.error("ADMIN_PASSWORD is not set. Add it in `.streamlit/secrets.toml` or the cloud app's Secrets.")
    st.stop()

if not st.session_state.get("admin_ok"):
    entered = st.text_input("Admin password", type="password")
    if st.button("Login", type="primary"):
        if hmac.compare_digest(entered.encode(), str(admin_password).encode()):
            st.session_state.admin_ok = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()

store = get_store()
col_a, col_b = st.columns([3, 1])
col_a.caption(f"Storage: {store.label}")
if col_b.button("Logout"):
    st.session_state.admin_ok = False
    st.rerun()

if "Local folder" in store.label:
    st.warning("Using temporary local storage. On Streamlit Cloud this is wiped on restart - "
               "set SUPABASE_URL and SUPABASE_KEY in Secrets for permanent storage.")


def refresh_app_cache():
    st.cache_resource.clear()  # chat page will reload the updated knowledge base


# ----------------------------------------------------------------- upload
st.subheader("Upload PDFs")
files = st.file_uploader("Choose one or more PDF files", type="pdf", accept_multiple_files=True)

if st.button("Add to knowledge base", type="primary", disabled=not files):
    for f in files:
        bar = st.progress(0.0, text=f"Processing {f.name}...")

        def on_progress(done, total, _bar=bar, _name=f.name):
            _bar.progress(done / total, text=f"Embedding {_name}: {done}/{total} passages")

        try:
            ok, msg = add_pdf(store, f.name, f.getvalue(), on_progress)
        except Exception as e:  # show the real error instead of crashing
            ok, msg = False, f"'{f.name}' failed: {e}"
        bar.empty()
        (st.success if ok else st.warning)(msg)
    refresh_app_cache()

# ------------------------------------------------------------- documents
st.subheader("Documents in the knowledge base")
state = load_state(store)
manifest = state["manifest"]

if not manifest:
    st.info("No documents yet. Upload PDFs above.")
else:
    st.caption(f"{len(manifest)} documents, {len(state['chunks'])} passages")
    allow_delete = st.checkbox("I understand deleting a document is permanent")
    for name, meta in sorted(manifest.items()):
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{name}**  \n{meta['pages']} pages · {meta['chunks']} passages · "
                    f"uploaded {meta['uploaded']}")
        if c2.button("🗑 Delete", key=f"del_{name}", disabled=not allow_delete):
            remove_pdf(store, name)
            refresh_app_cache()
            st.rerun()

# ---------------------------------------------------------------- rebuild
with st.expander("Advanced: rebuild everything"):
    st.caption("Re-reads all stored PDFs and re-creates the index from scratch. "
               "Use only if answers look broken. Slow for many/large PDFs.")
    if st.button("Rebuild index"):
        bar = st.progress(0.0, text="Rebuilding...")
        n_docs, n_chunks = rebuild_all(
            store, lambda d, t: bar.progress(d / t, text=f"Embedding {d}/{t}")
        )
        bar.empty()
        refresh_app_cache()
        st.success(f"Rebuilt: {n_docs} documents, {n_chunks} passages.")
