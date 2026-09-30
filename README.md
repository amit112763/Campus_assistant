# Campus Assistant (RAG chatbot with admin panel)

Students ask questions in a chat; answers come only from the official PDFs, with page-level
source citations. An admin can upload / delete PDFs from a password-protected page.

**Stack:** Python, Streamlit, FastEmbed (BGE-small), FAISS, Groq (Llama), Supabase Storage

## Files
- `app.py` - student chat page
- `pages/1_Admin.py` - admin page (login, upload, delete, rebuild)
- `rag_core.py` - PDF -> chunks -> embeddings -> search
- `storage.py` - Supabase Storage (or a local folder when Supabase is not configured)

## Run locally
1. `pip install -r requirements.txt`
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`; fill GROQ_API_KEY and ADMIN_PASSWORD
   (skip the Supabase lines to test with a temporary local folder)
3. `streamlit run app.py`
4. Open the **Admin** page in the sidebar, log in, upload PDFs, then ask questions on the main page

## Supabase setup
1. Create a project at supabase.com
2. Storage -> New bucket -> name `campus-rag` -> keep it **private**
3. Project Settings -> API: copy the Project URL and the service_role (secret) key into secrets
4. Re-upload your PDFs from the Admin page (local test data does not transfer)

## Deploy (Streamlit Community Cloud)
Push the code (not secrets.toml) to GitHub, create the app with main file `app.py`, and paste
the same secrets under Settings -> Secrets.

## Limitations
- Scanned PDFs (images only) need OCR
- Simple password login, fine for a learning project, not for sensitive data
- Answers can be wrong; always verify with the university office
