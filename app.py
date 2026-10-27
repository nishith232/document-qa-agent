
import streamlit as st
import tempfile
import os
import hashlib

from rag_pipeline import load_document, create_index, ask_question

st.set_page_config(
    page_title="Document Q&A Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

st.markdown("""
<style>
    .block-container {
        max-width: 950px;
        padding-top: 2rem;
        padding-bottom: 7rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }

    .welcome {
        text-align: center;
        padding-top: 12vh;
        padding-bottom: 2rem;
    }

    .welcome h1 {
        font-size: 2.2rem;
        font-weight: 600;
    }

    .welcome p {
        font-size: 1.05rem;
        opacity: 0.75;
    }

    .stChatMessage {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("📄 Document Q&A")
    st.caption("Your personal document assistant")

    st.divider()

    if st.button("＋ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.subheader("Your document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if st.session_state.document_name:
        st.caption(f"Current file: {st.session_state.document_name}")

    st.divider()

    st.markdown("**How it works**")
    st.caption("1. Upload a PDF")
    st.caption("2. Ask questions")
    st.caption("3. Get answers with source pages")

    st.divider()

    st.caption("Built with Python, Streamlit, FAISS and Gemini.")

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    document_id = hashlib.sha256(file_bytes).hexdigest()

    if document_id != st.session_state.document_id:
        file_path = None

        try:
            with st.spinner("Reading and processing your document..."):
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:
                    temp_file.write(file_bytes)
                    file_path = temp_file.name

                chunks = load_document(file_path)
                index = create_index(chunks)

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.document_id = document_id
            st.session_state.document_name = uploaded_file.name
            st.session_state.messages = []

            st.toast("Document processed successfully!")

        except Exception as e:
            st.error(f"Could not process the document: {e}")

        finally:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)

st.title("Document Q&A Agent")

if st.session_state.index is None:
    st.markdown("""
    <div class="welcome">
        <h1>What would you like to know?</h1>
        <p>Upload a PDF using the sidebar and start a conversation.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.caption(f"Chatting with: {st.session_state.document_name}")

    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome">
            <h1>Ask anything about your document.</h1>
            <p>Your answers are generated using relevant information from your PDF.</p>
        </div>
        """, unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("pages"):
                st.caption(
                    "Source pages: "
                    + ", ".join(map(str, message["pages"]))
                )

    question = st.chat_input(
        "Message your document..."
    )

    if question and question.strip():
        history = [
            {
                "role": message["role"],
                "content": message["content"]
            }
            for message in st.session_state.messages
        ]

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    answer, pages = ask_question(
                        question,
                        st.session_state.chunks,
                        st.session_state.index,
                        history
                    )

                st.markdown(answer)

                st.caption(
                    "Source pages: "
                    + ", ".join(map(str, pages))
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "pages": pages
                })

            except Exception as e:
                error_message = f"Sorry, I couldn't generate an answer: {e}"
                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
