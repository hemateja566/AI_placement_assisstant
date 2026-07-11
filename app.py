#gradio==5.49.1
import os
import time
import uuid
import threading
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 1. Load configuration
# ============================================================
# Local system:
#   Create a .env file containing HF_TOKEN=hf_your_token
#
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is missing. Add it to a local .env file or to "
        "Hugging Face Space Settings -> Secrets."
    )


# ============================================================
# 2. Create clients/models once when the app starts
# ============================================================
# The hosted LLM generates the final answer.
client = InferenceClient(
    provider="auto",
    token=HF_TOKEN,
)

# This local embedding model converts text and questions into vectors.
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ============================================================
# 3. Store each user's retriever separately
# ============================================================
# Gradio users can open the app at the same time. A session ID prevents
# one user's uploaded PDFs from being mixed with another user's PDFs.
SESSION_RETRIEVERS = {}
SESSION_LOCK = threading.Lock()


def get_or_create_session_id(session_id):
    """Return an existing session ID or create a new one."""
    return session_id or str(uuid.uuid4())


# ============================================================
# 4. Build the vector database from uploaded PDF files
# ============================================================
def build_knowledge_base(files, session_id, progress=gr.Progress()):
    """
    Load PDFs -> split pages into chunks -> create embeddings ->
    build a FAISS vector database -> create a retriever.
    """
    session_id = get_or_create_session_id(session_id)

    if not files:
        return (
            session_id,
            "Please upload at least one PDF file.",
            [],
        )

    try:
        progress(0.05, desc="Loading PDF files")

        documents = []

        for index, file_path in enumerate(files, start=1):
            path = Path(file_path)

            if path.suffix.lower() != ".pdf":
                continue

            loader = PyPDFLoader(str(path))
            loaded_pages = loader.load()

            # Save the readable filename in metadata for source display.
            for page in loaded_pages:
                page.metadata["source_name"] = path.name

            documents.extend(loaded_pages)
            progress(
                min(0.35, 0.05 + (index / max(len(files), 1)) * 0.30),
                desc=f"Loaded {path.name}",
            )

        if not documents:
            return (
                session_id,
                "No readable PDF content was found.",
                [],
            )

        progress(0.45, desc="Splitting documents into chunks")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120,
            length_function=len,
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            return (
                session_id,
                "The PDFs were loaded, but no text chunks were created.",
                [],
            )

        progress(0.60, desc="Creating embeddings")

        # FAISS.from_documents automatically:
        # 1. creates an embedding for every chunk,
        # 2. builds the FAISS index,
        # 3. links vectors back to their original chunks.
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model,
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        with SESSION_LOCK:
            SESSION_RETRIEVERS[session_id] = retriever

        file_names = [Path(file_path).name for file_path in files]
        progress(1.0, desc="Knowledge base ready")

        status = (
            f"Knowledge base created successfully.\n\n"
            f"PDF files: {len(file_names)}\n"
            f"Pages loaded: {len(documents)}\n"
            f"Chunks created: {len(chunks)}"
        )

        return session_id, status, []

    except Exception as error:
        return (
            session_id,
            f"Could not create the knowledge base.\n\nError: {error}",
            [],
        )


# ============================================================
# 5. Generate an answer using the retrieved PDF context
# ============================================================
def generate_answer(question, context, attempts=3):
    """
    Send the question and retrieved context to the hosted LLM.
    Retry temporary API failures before returning an error message.
    """
    system_prompt = (
        "You are an AI Placement Assistant. "
        "Answer only from the supplied document context. "
        "Do not use outside knowledge. "
        "If the answer is not present in the context, say exactly: "
        "\"I don't know based on the uploaded documents.\" "
        "Keep the answer clear and student-friendly."
    )

    user_prompt = f"""
DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=350,
                temperature=0.2,
            )

            answer = response.choices[0].message.content

            if not answer:
                raise ValueError("The model returned an empty response.")

            return answer.strip()

        except Exception as error:
            if attempt < attempts:
                time.sleep(2)
            else:
                return (
                    "The AI service is temporarily unavailable. "
                    f"Please try again.\n\nTechnical details: {error}"
                )


# ============================================================
# 6. Handle each question from the Gradio interface
# ============================================================
def ask_question(question, history, session_id):
    """
    Retrieve relevant chunks, generate an answer, and update chat history.
    """
    question = (question or "").strip()
    history = history or []

    if not question:
        return history, "", "Please enter a question."

    if not session_id:
        return (
            history,
            "",
            "Upload PDFs and click 'Build Knowledge Base' first.",
        )

    with SESSION_LOCK:
        retriever = SESSION_RETRIEVERS.get(session_id)

    if retriever is None:
        return (
            history,
            "",
            "Your knowledge base is not available. Please build it again.",
        )

    try:
        relevant_docs = retriever.invoke(question)

        if not relevant_docs:
            answer = "I don't know based on the uploaded documents."
            sources_text = "No relevant source chunks were found."
        else:
            context_parts = []
            source_lines = []

            for number, document in enumerate(relevant_docs, start=1):
                source_name = document.metadata.get(
                    "source_name",
                    Path(document.metadata.get("source", "Unknown PDF")).name,
                )

                # PyPDFLoader stores zero-based page numbers.
                page_number = document.metadata.get("page")
                readable_page = page_number + 1 if isinstance(page_number, int) else "Unknown"

                context_parts.append(
                    f"[Source {number}: {source_name}, page {readable_page}]\n"
                    f"{document.page_content}"
                )
                source_lines.append(
                    f"{number}. {source_name} — page {readable_page}"
                )

            context = "\n\n".join(context_parts)
            answer = generate_answer(question, context)
            sources_text = "\n".join(source_lines)

        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

        return history, "", sources_text

    except Exception as error:
        history = history + [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": (
                    "I could not process that question. "
                    f"Please try again.\n\nError: {error}"
                ),
            },
        ]

        return history, "", "Retrieval failed."


def clear_session(session_id):
    """Remove a user's vector database and clear the interface."""
    if session_id:
        with SESSION_LOCK:
            SESSION_RETRIEVERS.pop(session_id, None)

    return None, [], "Upload PDF files to begin.", ""


# ============================================================
# 7. Build the Gradio web interface
# ============================================================
with gr.Blocks(title="AI Placement Assistant") as demo:
    session_state = gr.State(value=None)

    gr.Markdown(
        """
        # AI Placement Assistant
        Upload one or more placement-related PDFs, create the knowledge base,
        and ask questions grounded in those documents.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_files = gr.File(
                label="Upload PDF files",
                file_types=[".pdf"],
                file_count="multiple",
                type="filepath",
            )

            build_button = gr.Button(
                "Build Knowledge Base",
                variant="primary",
            )

            clear_button = gr.Button("Clear PDFs and Chat")

            status_box = gr.Textbox(
                label="Knowledge Base Status",
                value="Upload PDF files to begin.",
                lines=5,
                interactive=False,
            )

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Document Chat",
                type="messages",
                height=460,
            )

            question_box = gr.Textbox(
                label="Ask a question",
                placeholder="Example: Explain about Programming from the uploaded notes.",
                lines=2,
            )

            ask_button = gr.Button("Ask", variant="primary")

            sources_box = gr.Textbox(
                label="Retrieved Sources",
                lines=5,
                interactive=False,
            )

    build_button.click(
        fn=build_knowledge_base,
        inputs=[pdf_files, session_state],
        outputs=[session_state, status_box, chatbot],
    )

    ask_button.click(
        fn=ask_question,
        inputs=[question_box, chatbot, session_state],
        outputs=[chatbot, question_box, sources_box],
    )

    question_box.submit(
        fn=ask_question,
        inputs=[question_box, chatbot, session_state],
        outputs=[chatbot, question_box, sources_box],
    )

    clear_button.click(
        fn=clear_session,
        inputs=[session_state],
        outputs=[session_state, chatbot, status_box, sources_box],
    )


# ============================================================
# 8. Start the application
# ============================================================
if __name__ == "__main__":
    demo.queue().launch()
