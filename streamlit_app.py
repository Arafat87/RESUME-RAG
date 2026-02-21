"""
Resume RAG Chatbot - Streamlit Version
BGE-base embeddings + FAISS + Llama 3 (Groq) + Guardrails

Instead of reading bullet points, recruiters can query experience directly.
"""

import streamlit as st
import os
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import httpx

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Resume RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get API key from secrets or environment
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Chunk:
    id: int
    text: str
    source: str
    embedding: Optional[np.ndarray] = None

# =============================================================================
# EMBEDDING GENERATION
# =============================================================================

def chunk_resume(resume_text: str) -> List[Dict]:
    """Chunk resume by sections and sliding windows."""
    chunks = []
    lines = resume_text.split("\n")
    current_section = "Header"
    current_text = []
    
    for line in lines:
        if line.startswith("#"):
            if current_text:
                chunk_text = "\n".join(current_text).strip()
                if chunk_text:
                    chunks.append({
                        "id": len(chunks),
                        "text": chunk_text,
                        "source": current_section
                    })
            current_section = line.lstrip("#").strip()
            current_text = [line]
        else:
            current_text.append(line)
    
    # Add last section
    if current_text:
        chunk_text = "\n".join(current_text).strip()
        if chunk_text:
            chunks.append({
                "id": len(chunks),
                "text": chunk_text,
                "source": current_section
            })
    
    # Also create sliding window chunks
    all_text = resume_text.split()
    chunk_size = 200
    overlap = 50
    
    for i in range(0, len(all_text), chunk_size - overlap):
        chunk_text = " ".join(all_text[i:i + chunk_size])
        if chunk_text.strip():
            chunks.append({
                "id": len(chunks),
                "text": chunk_text,
                "source": f"Content chunk {len(chunks)}"
            })
    
    return chunks

def generate_embeddings_from_resume(embed_model) -> Tuple[List[Dict], faiss.Index, np.ndarray]:
    """Generate embeddings from resume.md."""
    resume_path = os.path.join(os.getcwd(), "resume.md")
    
    if not os.path.exists(resume_path):
        return None, None, None
    
    with open(resume_path, "r") as f:
        resume_text = f.read()
    
    if not resume_text.strip():
        return None, None, None
    
    # Chunk the resume
    chunks = chunk_resume(resume_text)
    
    if not chunks:
        return None, None, None
    
    # Generate embeddings
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    
    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))
    
    return chunks, index, embeddings

# =============================================================================
# MODEL LOADING (CACHED)
# =============================================================================

@st.cache_resource
def load_embedding_model():
    """Load BGE-base-en-v1.5 embedding model."""
    return SentenceTransformer("BAAI/bge-base-en-v1.5")

@st.cache_resource
def load_resume_data(_embed_model):
    """Load or generate resume embeddings."""
    base_path = os.getcwd()
    
    # Try loading existing files first
    chunks_path = os.path.join(base_path, "chunks.pkl")
    index_path = os.path.join(base_path, "resume.index")
    
    if os.path.exists(chunks_path) and os.path.exists(index_path):
        with open(chunks_path, "rb") as f:
            chunks_data = pickle.load(f)
        index = faiss.read_index(index_path)
        embeddings_path = os.path.join(base_path, "embeddings.npy")
        embeddings = np.load(embeddings_path) if os.path.exists(embeddings_path) else None
        return chunks_data, index, embeddings
    
    # Generate from resume.md
    return generate_embeddings_from_resume(_embed_model)

# =============================================================================
# RAG FUNCTIONS
# =============================================================================

def retrieve_chunks(
    query: str,
    embed_model,
    index,
    chunks: List[Dict],
    k: int = 5
) -> List[Tuple[Dict, float]]:
    """Retrieve top-k relevant chunks using FAISS."""
    if index is None or not chunks:
        return []
    
    query_embedding = embed_model.encode([query], normalize_embeddings=True)
    D, I = index.search(query_embedding.astype('float32'), k)
    
    results = []
    for dist, i in zip(D[0], I[0]):
        if i < len(chunks):
            similarity = 1 / (1 + dist)
            results.append((chunks[i], similarity))
    
    return results

def generate_answer_groq(
    query: str,
    retrieved_chunks: List[Tuple[Dict, float]],
    guardrails: bool = True
) -> Dict:
    """Generate answer using Llama 3 via Groq with guardrails."""
    
    if not GROQ_API_KEY:
        return {
            "answer": "⚠️ **Groq API key not configured.**\n\nAdd your GROQ_API_KEY in Streamlit secrets.",
            "sources": [],
            "confidence": 0.0
        }
    
    if not retrieved_chunks:
        return {
            "answer": "I couldn't find relevant information in the resume to answer this question.",
            "sources": [],
            "confidence": 0.0
        }
    
    context_parts = []
    sources = []
    
    for chunk, score in retrieved_chunks:
        context_parts.append(f"[{chunk['source']}]\n{chunk['text']}")
        sources.append({
            "section": chunk['source'],
            "relevance": f"{score:.1%}",
            "preview": chunk['text'][:150] + "..." if len(chunk['text']) > 150 else chunk['text']
        })
    
    context = "\n\n---\n\n".join(context_parts)
    
    system_prompt = """You are a helpful assistant that answers questions about a candidate's resume.

CRITICAL RULES:
1. Answer ONLY based on the provided context from the resume
2. If the context doesn't contain enough information, say "Based on the resume provided, I don't have specific information about that."
3. Be honest - don't exaggerate or invent experiences
4. Quote specific details from the context when relevant
5. Keep answers concise but complete
6. If asked about something not in the resume, politely redirect to what IS available""" if guardrails else "You are a helpful assistant that answers questions about a candidate's resume accurately and concisely."
    
    user_prompt = f"""Context from resume:
{context}

Question: {query}

Answer based on the context above:"""
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500
                }
            )
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
        
        avg_similarity = sum(s for _, s in retrieved_chunks) / len(retrieved_chunks)
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": avg_similarity
        }
        
    except Exception as e:
        return {
            "answer": f"❌ Error: {str(e)}",
            "sources": sources,
            "confidence": 0.0
        }

# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #888; margin-bottom: 2rem; }
    .tech-badge { display: inline-block; padding: 0.25rem 0.75rem; background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); border-radius: 9999px; font-size: 0.75rem; margin: 0.25rem; color: white; }
    .source-card { background: #1a1a2e; border-radius: 0.5rem; padding: 1rem; margin: 0.5rem 0; border-left: 3px solid #4f46e5; }
    .confidence-bar { height: 4px; background: #333; border-radius: 2px; margin: 0.5rem 0; }
    .confidence-fill { height: 100%; background: linear-gradient(90deg, #4f46e5, #7c3aed); border-radius: 2px; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">📄 Resume RAG Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Query my experience directly instead of scanning a PDF</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <span class="tech-badge">BGE-base embeddings</span>
        <span class="tech-badge">FAISS vector index</span>
        <span class="tech-badge">Llama 3 (Groq)</span>
        <span class="tech-badge">Guardrails</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models and data
    embed_model = load_embedding_model()
    chunks, index, embeddings = load_resume_data(embed_model)
    
    if chunks is None:
        st.error("⚠️ No resume found. Please add a `resume.md` file to the project directory.")
        st.stop()
    
    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "k_chunks" not in st.session_state:
        st.session_state.k_chunks = 5
    if "guardrails" not in st.session_state:
        st.session_state.guardrails = True
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.session_state.k_chunks = st.slider("Chunks to retrieve (k)", 1, 10, st.session_state.k_chunks)
        st.session_state.guardrails = st.checkbox("Enable guardrails", st.session_state.guardrails)
        
        st.divider()
        st.header("📊 Resume Stats")
        st.metric("Total Chunks", len(chunks))
        st.metric("Embedding Dim", index.d if index else "N/A")
        
        st.divider()
        st.header("🔑 API Status")
        if GROQ_API_KEY:
            st.success("✅ Groq API configured")
        else:
            st.error("❌ Groq API key missing")
            st.markdown("Add to `.streamlit/secrets.toml`:\n```toml\nGROQ_API_KEY = \"gsk_xxxxx\"\n```")
        
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 💡 Example Questions")
        example_questions = [
            "What experience do you have with distributed systems?",
            "Have you built production RAG pipelines?",
            "How have you improved inference throughput?",
            "What's your experience with LLMs?",
            "Tell me about your ML background"
        ]
        
        cols = st.columns(len(example_questions))
        for i, q in enumerate(example_questions):
            if cols[i].button(q, key=f"example_{i}"):
                st.session_state.messages.append({"role": "user", "content": q})
        
        st.divider()
        st.markdown("### 💬 Chat")
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                    with st.expander("📍 Sources"):
                        for src in msg["sources"]:
                            st.markdown(f"<div class='source-card'><strong>{src['section']}</strong> <span style='color:#888;font-size:0.8rem;'>Relevance: {src['relevance']}</span><br><br>{src['preview']}</div>", unsafe_allow_html=True)
        
        # Auto-generate response for pending user messages (from example buttons)
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            prompt = st.session_state.messages[-1]["content"]
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    retrieved = retrieve_chunks(prompt, embed_model, index, chunks, st.session_state.k_chunks)
                    result = generate_answer_groq(prompt, retrieved, st.session_state.guardrails)
                
                st.markdown(result["answer"])
                
                if result["confidence"] > 0:
                    st.markdown(f"<div style='margin-top:1rem;'><small>Confidence: {result['confidence']:.0%}</small><div class='confidence-bar'><div class='confidence-fill' style='width:{result['confidence']*100}%'></div></div></div>", unsafe_allow_html=True)
                
                if result["sources"]:
                    with st.expander("📍 Sources"):
                        for src in result["sources"]:
                            st.markdown(f"<div class='source-card'><strong>{src['section']}</strong> <span style='color:#888;font-size:0.8rem;'>Relevance: {src['relevance']}</span><br><br>{src['preview']}</div>", unsafe_allow_html=True)
            
            st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
            st.rerun()
        
        if prompt := st.chat_input("Ask about my experience..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    retrieved = retrieve_chunks(prompt, embed_model, index, chunks, st.session_state.k_chunks)
                    result = generate_answer_groq(prompt, retrieved, st.session_state.guardrails)
                
                st.markdown(result["answer"])
                
                if result["confidence"] > 0:
                    st.markdown(f"<div style='margin-top:1rem;'><small>Confidence: {result['confidence']:.0%}</small><div class='confidence-bar'><div class='confidence-fill' style='width:{result['confidence']*100}%'></div></div></div>", unsafe_allow_html=True)
                
                if result["sources"]:
                    with st.expander("📍 Sources"):
                        for src in result["sources"]:
                            st.markdown(f"<div class='source-card'><strong>{src['section']}</strong> <span style='color:#888;font-size:0.8rem;'>Relevance: {src['relevance']}</span><br><br>{src['preview']}</div>", unsafe_allow_html=True)
                
                st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
    
    with col2:
        st.markdown("### 🔧 Architecture")
        st.markdown("""```\nQuery\n  ↓\n┌─────────────┐\n│ BGE-base    │\n│ Embeddings  │\n└─────────────┘\n  ↓\n┌─────────────┐\n│ FAISS       │\n│ Vector      │\n│ Search      │\n└─────────────┘\n  ↓\n┌─────────────┐\n│ Top-k       │\n│ Context     │\n└─────────────┘\n  ↓\n┌─────────────┐\n│ Llama 3     │\n│ (Groq)      │\n│ + Guardrails│\n└─────────────┘\n  ↓\nAnswer\n```""")

if __name__ == "__main__":
    main()
