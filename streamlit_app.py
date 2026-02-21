"""
Document Q&A System - Streamlit Version
A clean, working implementation for PDF document analysis and Q&A.
"""

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import io
import pytesseract
from PIL import Image


# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load secrets (works for both local .streamlit/secrets.toml and Streamlit Cloud)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ChunkMetadata:
    chunk_id: str
    doc_type: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    embedding: Optional[np.ndarray] = None


# =============================================================================
# Caching - Models loaded once and reused
# =============================================================================

@st.cache_resource
def load_embedding_model():
    """Load embedding model once and cache it."""
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_gemini_model():
    """Load Gemini model once and cache it."""
    if not GEMINI_API_KEY:
        return None
    google.generativeai.configure(api_key=GEMINI_API_KEY)
    return google.generativeai.GenerativeModel("gemini-2.0-flash")


# =============================================================================
# Document Processing Functions
# =============================================================================

def extract_text_from_pdf(pdf_bytes: bytes) -> List[Dict]:
    """Extract text from PDF with OCR fallback for scanned pages."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    
    progress_bar = st.progress(0, text="Extracting text from PDF...")
    
    for i, page in enumerate(doc):
        text = page.get_text()
        
        # OCR fallback for pages with no extractable text
        if not text.strip():
            try:
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                text = pytesseract.image_to_string(img)
            except Exception as e:
                st.warning(f"OCR failed on page {i+1}: {e}")
                text = ""
        
        pages.append({
            "page_num": i + 1,
            "text": text
        })
        progress_bar.progress((i + 1) / len(doc), text=f"Extracted page {i+1}/{len(doc)}")
    
    doc.close()
    progress_bar.empty()
    return pages


def classify_document(text: str, gemini_model) -> str:
    """Classify document type using Gemini."""
    if not gemini_model:
        return "Document"
    
    doc_types = [
        "Resume", "Contract", "Invoice", "Bank Statement", "Tax Document",
        "Insurance", "Report", "Letter", "Form", "ID Document", "Medical", "Other"
    ]
    
    prompt = f"""Classify this document into ONE category: {', '.join(doc_types)}
    
Document sample (first 1500 chars):
{text[:1500]}

Respond with ONLY the category name, nothing else."""

    try:
        response = gemini_model.generate_content(prompt)
        result = response.text.strip()
        
        for doc_type in doc_types:
            if result.lower() == doc_type.lower():
                return doc_type
        return "Other"
    except Exception as e:
        st.warning(f"Classification failed: {e}")
        return "Document"


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks for better retrieval."""
    if not text:
        return []
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


def create_embeddings(chunks: List[str], embed_model) -> np.ndarray:
    """Create embeddings for text chunks."""
    if not chunks:
        return np.array([])
    return embed_model.encode(chunks, show_progress_bar=True)


def build_faiss_index(embeddings: np.ndarray):
    """Build FAISS index from embeddings."""
    if embeddings.size == 0:
        return None
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))
    return index


# =============================================================================
# RAG Query Functions
# =============================================================================

def retrieve_chunks(query: str, index, chunks: List[str], embed_model, k: int = 4) -> List[str]:
    """Retrieve most relevant chunks for a query."""
    if index is None or not chunks:
        return []
    
    query_embedding = embed_model.encode([query])
    D, I = index.search(query_embedding.astype('float32'), min(k, len(chunks)))
    
    return [chunks[i] for i in I[0] if i < len(chunks)]


def generate_answer(query: str, context: str, gemini_model, doc_type: str = "Document") -> str:
    """Generate answer using Gemini with retrieved context."""
    if not gemini_model:
        return "⚠️ Gemini API key not configured. Please add GEMINI_API_KEY to your secrets."
    
    prompt = f"""You are a helpful assistant answering questions about a {doc_type}.
Use the provided context to answer the question accurately.
If the answer is not in the context, say so clearly.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating answer: {e}"


# =============================================================================
# Main App
# =============================================================================

def main():
    st.title("📄 Document Q&A System")
    st.markdown("Upload a PDF and ask questions about its contents.")
    
    # Initialize session state
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "index" not in st.session_state:
        st.session_state.index = None
    if "doc_type" not in st.session_state:
        st.session_state.doc_type = ""
    if "pages" not in st.session_state:
        st.session_state.pages = []
    if "processing_stats" not in st.session_state:
        st.session_state.processing_stats = {}
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Load models
    embed_model = load_embedding_model()
    gemini_model = load_gemini_model()
    
    if not GEMINI_API_KEY:
        st.sidebar.warning("⚠️ Add GEMINI_API_KEY to secrets for full functionality")
    
    # Sidebar - File Upload
    with st.sidebar:
        st.header("📤 Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file:
            st.info(f"📁 {uploaded_file.name}")
        
        # Document info
        if st.session_state.processing_stats:
            st.divider()
            st.subheader("📊 Document Info")
            for key, value in st.session_state.processing_stats.items():
                st.write(f"**{key}:** {value}")
        
        # Clear button
        if st.session_state.document_processed:
            st.divider()
            if st.button("🗑️ Clear Document"):
                st.session_state.document_processed = False
                st.session_state.chunks = []
                st.session_state.embeddings = None
                st.session_state.index = None
                st.session_state.doc_type = ""
                st.session_state.pages = []
                st.session_state.processing_stats = {}
                st.session_state.messages = []
                st.rerun()
    
    # Main content area
    if uploaded_file and not st.session_state.document_processed:
        if st.button("🔄 Process Document", type="primary"):
            with st.spinner("Processing document..."):
                start_time = datetime.now()
                
                # Extract text
                extraction_start = datetime.now()
                pdf_bytes = uploaded_file.read()
                pages = extract_text_from_pdf(pdf_bytes)
                extraction_time = (datetime.now() - extraction_start).total_seconds()
                
                # Combine text
                full_text = "\n\n".join([p["text"] for p in pages if p["text"].strip()])
                
                if not full_text.strip():
                    st.error("❌ No text could be extracted from this PDF.")
                    return
                
                # Classify document
                doc_type = classify_document(full_text, gemini_model)
                
                # Chunk text
                chunking_start = datetime.now()
                chunks = chunk_text(full_text)
                chunking_time = (datetime.now() - chunking_start).total_seconds()
                
                # Create embeddings and index
                indexing_start = datetime.now()
                embeddings = create_embeddings(chunks, embed_model)
                index = build_faiss_index(embeddings)
                indexing_time = (datetime.now() - indexing_start).total_seconds()
                
                total_time = (datetime.now() - start_time).total_seconds()
                
                # Save to session state
                st.session_state.pages = pages
                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.index = index
                st.session_state.doc_type = doc_type
                st.session_state.document_processed = True
                st.session_state.processing_stats = {
                    "Pages": len(pages),
                    "Document Type": doc_type,
                    "Chunks": len(chunks),
                    "Total Time": f"{total_time:.1f}s",
                    "Extraction": f"{extraction_time:.1f}s",
                    "Chunking": f"{chunking_time:.1f}s",
                    "Indexing": f"{indexing_time:.1f}s"
                }
                st.session_state.messages = []
                
                st.success(f"✅ Processed! Document type: {doc_type}")
                st.rerun()
    
    # Q&A Section
    if st.session_state.document_processed:
        st.divider()
        st.subheader(f"💬 Ask Questions about your {st.session_state.doc_type}")
        
        # Chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your document..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Retrieve relevant chunks
                    relevant_chunks = retrieve_chunks(
                        prompt,
                        st.session_state.index,
                        st.session_state.chunks,
                        embed_model
                    )
                    
                    if relevant_chunks:
                        context = "\n\n---\n\n".join(relevant_chunks)
                        response = generate_answer(
                            prompt,
                            context,
                            gemini_model,
                            st.session_state.doc_type
                        )
                    else:
                        response = "I couldn't find relevant information in the document."
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Instructions when no document
    if not uploaded_file:
        st.info("👆 Upload a PDF file to get started!")
        st.markdown("""
        ### How it works:
        1. **Upload** a PDF document
        2. **Process** - Text extraction, classification, and indexing
        3. **Ask questions** - Get AI-powered answers with context from your document
        
        ### Features:
        - 📄 PDF text extraction with OCR fallback
        - 🏷️ Automatic document classification
        - 🔍 Semantic search with vector embeddings
        - 💬 Q&A with source attribution
        """)


if __name__ == "__main__":
    main()
