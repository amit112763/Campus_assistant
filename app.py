import streamlit as st
from groq import Groq

from rag_core import build_faiss, load_state, retrieve
from storage import get_secret, get_store

DEFAULT_LLM = "openai/gpt-oss-120b"        # change via GROQ_MODEL secret if Groq retires it
FAST_LLM = "openai/gpt-oss-20b"           # small/fast model used only to rewrite search queries
MIN_SCORE = 0.40                          # below this we answer "not found" (tune it!)

# Change these to match the documents you upload
STARTER_QUESTIONS = [
    "What subjects are there in the first semester?",
    "What is the marks distribution for internal and external exams?",
    "How many credits are required to complete the programme?",
    "What are the attendance requirements?",
]

SYSTEM_PROMPT = """You are a Campus Assistant for university students.
Answer ONLY using the numbered context passages provided (university rules, schemes, syllabus).
- If the answer is not in the context, say: "I couldn't find this in the university documents. Please check with the official university office."
- Do not guess or add information that is not in the context.
- The documents may cover several programmes/branches. If the context shows different details for different programmes, say which programme each fact belongs to.
- Cite the passages you used like [1], [2].
- Keep answers clear and short. Use bullet points for lists.
- If passages conflict or look outdated, mention it."""

st.set_page_config(page_title="Campus Assistant", page_icon="🎓", layout="centered")

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_kb():
    state = load_state(get_store())
    if not state["chunks"]:
        return None

    vectors = state["vectors"]
    index = build_faiss(vectors)

    del vectors
    del state["vectors"]

    return {
        "index": index,
        "chunks": state["chunks"],
        "manifest": state["manifest"],
    }


def build_search_query(question, history):
    """Short follow-ups ('and for hostel?') need the previous question for context."""
    if len(question.split()) < 8:
        prev = [m["content"] for m in history if m["role"] == "user"]
        if prev:
            return f"{prev[-1]} {question}"
    return question


def rewrite_query(question, history, api_key, model_name):
    """Turn Hinglish / short / follow-up questions into a clear English search query.
    Falls back to the simple heuristic if the call fails."""
    fallback = build_search_query(question, history)
    recent = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
    prompt = (
        "Rewrite the user's latest question as ONE standalone English search query for searching "
        "university documents (syllabus, schemes, rules, admissions, hostel, placement).\n"
        "- Translate Hindi/Hinglish to English.\n"
        "- Expand abbreviations (maths -> mathematics, sem -> semester, syllabus -> syllabus units topics).\n"
        "- Use the chat history to resolve follow-ups like 'and for AIML?'.\n"
        "- Output ONLY the query, no explanation.\n\n"
        f"Chat history:\n{recent or '(none)'}\n\nLatest question: {question}"
    )
    try:
        resp = Groq(api_key=api_key).chat.completions.create(
            model=get_secret("GROQ_FAST_MODEL", FAST_LLM),
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=600,
        )
        q = (resp.choices[0].message.content or "").strip().strip('"')
        return q if 3 <= len(q) <= 300 else fallback
    except Exception:
        return fallback


def ask_llm(question, contexts, history, api_key, model_name):
    context_text = "\n\n".join(
        f"[{i}] (Source: {c['source']}, page {c['page']})\n{c['text']}"
        for i, c in enumerate(contexts, start=1)
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    messages.append({"role": "user",
                     "content": f"Context passages:\n{context_text}\n\nQuestion: {question}"})
    resp = Groq(api_key=api_key).chat.completions.create(
        model=model_name, messages=messages, temperature=0.1, max_tokens=2000
    )
    return resp.choices[0].message.content


def show_sources(sources, show_debug):
    with st.expander("📄 Sources"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {s['source']} - page {s['page']}**")
            st.caption(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))
            if show_debug:
                st.caption(f"similarity: {s['score']:.3f}")


# ---------------------------------------------------------------------- UI
st.title("🎓 Campus Assistant")
st.caption("Ask about university rules, schemes and syllabus. Answers come from the official documents.")

kb = load_kb()
if kb is None:
    st.info("The knowledge base is empty. An admin needs to upload PDFs on the **Admin** page (left sidebar).")
    st.stop()

api_key = get_secret("GROQ_API_KEY")
model_name = get_secret("GROQ_MODEL", DEFAULT_LLM)
if not api_key:
    st.error("GROQ_API_KEY is missing. Add it in `.streamlit/secrets.toml` or the cloud app's Secrets.")
    st.stop()

all_sources = sorted(kb["manifest"].keys())

with st.sidebar:
    st.header("Settings")
    selected = st.multiselect("Search in documents", all_sources, default=all_sources)
    top_k = st.slider("Passages to retrieve", 2, 10, 6)
    show_debug = st.checkbox("Show retrieval scores")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption(f"📚 {len(all_sources)} documents, {len(kb['chunks'])} passages indexed")
    st.caption("⚠️ This bot can make mistakes. Rules change - always verify important "
               "information with the official university office.")

if "messages" not in st.session_state:
    st.session_state.messages = []

starter = None
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(2)
    for n, ex in enumerate(STARTER_QUESTIONS):
        if cols[n % 2].button(ex, use_container_width=True):
            starter = ex

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"], show_debug)

question = st.chat_input("Ask a question about the university...") or starter

if question:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        sources = []
        if not selected:
            answer = "Please select at least one document in the sidebar."
        else:
            with st.spinner("Searching documents..."):
                search_query = rewrite_query(question, history, api_key, model_name)
                results = retrieve(kb, search_query, top_k, set(selected))
            if show_debug:
                st.caption(f"🔎 Search query used: {search_query}")
            relevant = [r for r in results if r["score"] >= MIN_SCORE]
            if not relevant:
                answer = ("I couldn't find this in the university documents. "
                          "Please check with the official university office.")
            else:
                try:
                    with st.spinner("Writing answer..."):
                        answer = ask_llm(question, relevant, history, api_key, model_name)
                    sources = relevant
                except Exception as e:
                    answer = f"Sorry, the language model request failed: `{e}`"

        st.markdown(answer)
        if sources:
            show_sources(sources, show_debug)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})