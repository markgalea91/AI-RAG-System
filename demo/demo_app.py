import requests
import streamlit as st
from rag_platform.config.settings import get_settings
from uuid import uuid4

st.set_page_config(page_title="MTCA", layout="wide")

# --- Config (no sidebar) ---
s = get_settings()

API_BASE =  "http://192.168.100.130:12001"
DEFAULT_LLM_MODEL = s.generation_model
DEFAULT_PROVIDER = s.provider
DEFAULT_REASONING = s.reasoning
DEFAULT_DOTNET_IS_CHAT = s.is_chat
DEFAULT_DOTNET_CLIENT_GUID = s.api_client_guid

st.title("MTCA")

# Keep a stable GUID per user session
if "dotnet_client_guid" not in st.session_state:
    st.session_state.dotnet_client_guid = DEFAULT_DOTNET_CLIENT_GUID

# Each item is a "turn": {"query": str, "answer": str, "sources": [str], "retrieval": {...}}
if "turns" not in st.session_state:
    st.session_state.turns = []

# Render history (query -> answer -> sources)
for t in st.session_state.turns:
    st.markdown(f"**You:** {t['query']}")
    st.markdown(t["answer"])

    sources = t.get("sources", [])
    if sources:
        st.caption("Sources: " + " | ".join(sources))

    # Optional: show evidence
    retrieval = t.get("retrieval")
    if retrieval and retrieval.get("chunks"):
        with st.expander("Evidence (retrieved chunks)", expanded=False):
            for i, c in enumerate(retrieval["chunks"], start=1):
                src = c.get("source_name", "unknown_source")
                page = c.get("page_number")
                score = c.get("score")

                meta = src
                if page is not None:
                    meta += f", p.{page}"
                if isinstance(score, (int, float)):
                    meta += f", score={score:.4f}"

                st.markdown(f"**{i}. {meta}**")
                st.write(c.get("text", ""))

    st.markdown("---")

prompt = st.chat_input("Ask a question…")
if prompt:
    # Display user's query immediately
    st.markdown(f"**You:** {prompt}")

    with st.spinner("Thinking…"):
        r = requests.post(
            f"{API_BASE}/rag/query",
            json={
                "query": prompt,
                "llm_model": DEFAULT_LLM_MODEL,
                "reasoning": DEFAULT_REASONING,
                "provider": DEFAULT_PROVIDER,
                "dotnet_is_chat": DEFAULT_DOTNET_IS_CHAT,
                "dotnet_client_guid": st.session_state.dotnet_client_guid,
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()

    # Display answer directly below query
    # --- DEBUG: inspect what the API actually returned ---
    with st.expander("Raw API response", expanded=False):
        st.json(data)

    # --- Robust answer extraction ---
    answer = (
            data.get("answer")
            or data.get("Answer")
            or data.get("final_answer")
            or data.get("result", {}).get("answer")
            or ""
    )

    if not answer:
        st.error("API response did not include a non-empty 'answer' field.")
    else:
        st.markdown(answer)

    # answer = data.get("answer", "")
    # st.markdown(answer)

    # Display sources directly under the answer
    sources = data.get("sources", [])
    if sources:
        st.caption("Sources: " + " | ".join(sources))

    st.markdown("---")

    # Save turn for persistence on rerun
    st.session_state.turns.append(
        {
            "query": data.get("query", prompt),
            "answer": answer,
            "sources": sources,
            "retrieval": data.get("retrieval", {}),
        }
    )