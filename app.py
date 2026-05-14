import streamlit as st
from dotenv import load_dotenv
from main import run_pipeline
from core.rag_engine import ask_question

load_dotenv()

st.set_page_config(page_title="Video Assistant", page_icon="🎬", layout="centered")

st.title("🎬 Video Assistant")
st.caption("Paste a YouTube URL to transcribe, summarize, and chat with any video.")

# ── Input ──────────────────────────────────────────────────────────────────
url = st.text_input("YouTube URL", placeholder="https://youtu.be/...")

col1, col2 = st.columns([3, 1])
with col1:
    analyze = st.button("Analyze", type="primary", disabled=not url)
with col2:
    if st.button("Clear", disabled="result" not in st.session_state):
        del st.session_state["result"]
        st.session_state["messages"] = []
        st.rerun()

if analyze:
    with st.spinner("Processing video... this may take several minutes."):
        try:
            result = run_pipeline(url)
            st.session_state["result"] = result
            st.session_state["messages"] = []
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(f"Something went wrong: {e}")

# ── Results ────────────────────────────────────────────────────────────────
if "result" in st.session_state:
    r = st.session_state["result"]

    st.markdown(f"## {r['title']}")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Key Topics", "Key Quotes", "Takeaways"])

    with tab1:
        st.write(r["summary"])

    with tab2:
        st.write(r["topics"])

    with tab3:
        st.write(r["quotes"])

    with tab4:
        st.write(r["takeaways"])

    with st.expander("View full transcript"):
        st.text(r["transcript"])

    st.divider()

    # ── Chat ───────────────────────────────────────────────────────────────
    st.subheader("💬 Chat with your video")

    for msg in st.session_state.get("messages", []):
        st.chat_message(msg["role"]).write(msg["content"])

    question = st.chat_input("Ask anything about the video...")
    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_question(r["rag_chain"], question)
            st.write(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})
