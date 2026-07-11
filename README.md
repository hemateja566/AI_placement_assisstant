# AI Placement Assistant

A Retrieval-Augmented Generation application that:

Accepts PDF uploads
Splits documents into chunks
Creates embeddings
Stores vectors in FAISS
Retrieves relevant document sections
Generates grounded answers using a Hugging Face model
Displays PDF source pages


User Uploads PDFs --> PyPDFLoader reads pages --> Text is divided into chunks --> Embedding model converts chunks into $
-->FAISS creates the vector database --> User asks a question --> Retriever finds the top relevant chunks
-->Retrieved context is sent to Qwen --> Answer and source pages are displayed...
