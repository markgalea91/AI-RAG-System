import requests
import streamlit as st

st.set_page_config(page_title="RAG Demo", layout="wide")

API_BASE = st.sidebar.text_input("API base URL", value="http://localhost:8000")

st.title("RAG Demo")

with st.sidebar:
    st.subheader("Settings")
    top_k = st.slider("Top-K", 1, 20, 3)
    max_chars = st.slider("Max chars per chunk", 200, 3000, 1200, step=100)
    include_debug = st.checkbox("Show debug context (stuffed_context)", value=False)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            st.caption("Sources: " + " | ".join(m["sources"]))

prompt = st.chat_input("Ask a question…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            r = requests.post(
                f"{API_BASE}/rag/query",
                json={
                    "query": prompt,
                    "top_k": top_k,
                    "max_chars_per_chunk": max_chars,
                    "include_debug": include_debug,
                },
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()

        st.markdown(data["answer"])

        if data.get("sources"):
            st.markdown("---")
            st.markdown("### Sources")
            for s in data["sources"]:
                st.markdown(f"- {s}")

        # Evidence panel
        with st.expander("Evidence (retrieved chunks)", expanded=True):
            chunks = data.get("chunks", [])
            if not chunks:
                st.info("No chunks retrieved.")
            else:
                for i, c in enumerate(chunks, start=1):
                    title = c.get("source", f"Chunk {i}")
                    score = c.get("score")
                    st.markdown(f"**{i}. {title}**" + (f" (score={score:.4f})" if isinstance(score, (int, float)) else ""))
                    st.write(c.get("text", ""))

        # Debug context
        if include_debug and data.get("stuffed_context"):
            with st.expander("Stuffed context", expanded=False):
                st.code(data["stuffed_context"])

        sources = data.get("sources", [])
        st.session_state.messages.append(
            {"role": "assistant", "content": data["answer"], "sources": sources}
        )