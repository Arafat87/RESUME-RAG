"""
Resume RAG Chatbot - Streamlit Version
BGE-base embeddings + FAISS + Llama 3 (Groq) + Guardrails
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

st.set_page_config(
    page_title="Resume RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")

@dataclass
class Chunk:
    id: int
    text: str
    source: str
    embedding: Optional[np.ndarray] = None

def chunk_resume(resume_text: str) -> List[Dict]:
    chunks = []
    lines = resume_text.split("\n")
    current_section = "Header"
    current_text = []
    
    for line in lines:
        if line.startswith("#"):
            if current_text:
                chunk_text = "\n".join(current_text).strip()
                if chunk_text:
                    chunks.append({"id": len(chunks), "text": chunk_text, "source": current_section})
            current_section = line.lstrip("#").strip()
            current_text = [line]
        else:
            current_text.append(line)
    
    if current_text:
        chunk_text = "\n".join(current_text).strip()
        if chunk_text:
            chunks.append({"id": len(chunks), "text": chunk_text, "source": current_section})
    
    all_text = resume_text.split()
    chunk_size, overlap = 200, 50
    for i in range(0, len(all_text), chunk_size - overlap):
        chunk_text = " ".join(all_text[i:i + chunk_size])
        if chunk_text.strip():
            chunks.append({"id": len(chunks), "text": chunk_text, "source": f"Content chunk {len(chunks)}"})
    
    return chunks

def generate_embeddings_from_resume(embed_model) -> Tuple[List[Dict], faiss.Index, np.ndarray]:
    resume_path = os.path.join(os.getcwd(), "resume.md")
    if not os.path.exists(resume_path):
        return None, None, None
    
    with open(resume_path, "r") as f:
        resume_text = f.read()
    
    if not resume_text.strip():
        return None, None, None
    
    chunks = chunk_resume(resume_text)
    if not chunks:
        return None, None, None
    
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))
    
    return chunks, index, embeddings

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-base-en-v1.5")

@st.cache_resource
def load_resume_data(_embed_model):
    base_path = os.getcwd()
    chunks_path = os.path.join(base_path, "chunks.pkl")
    index_path = os.path.join(base_path, "resume.index")
    
    if os.path.exists(chunks_path) and os.path.exists(index_path):
        with open(chunks_path, "rb") as f:
            chunks_data = pickle.load(f)
        index = faiss.read_index(index_path)
        embeddings_path = os.path.join(base_path, "embeddings.npy")
        embeddings = np.load(embeddings_path) if os.path.exists(embeddings_path) else None
        return chunks_data, index, embeddings
    
    return generate_embeddings_from_resume(_embed_model)

def retrieve_chunks(query: str, embed_model, index, chunks: List[Dict], k: int = 5) -> List[Tuple[Dict, float]]:
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

def generate_answer_groq(query: str, retrieved_chunks: List[Tuple[Dict, float]], guardrails: bool = True) -> Dict:
    if not GROQ_API_KEY:
        return {"answer": "⚠️ **Groq API key not configured.** Add your GROQ_API_KEY in Streamlit secrets.", "sources": [], "confidence": 0.0}
    
    if not retrieved_chunks:
        return {"answer": "I couldn't find relevant information in the resume.", "sources": [], "confidence": 0.0}
    
    context_parts = []
    sources = []
    for chunk, score in retrieved_chunks:
        context_parts.append(f"[{chunk['source']}]\n{chunk['text']}")
        sources.append({"section": chunk['source'], "relevance": f"{score:.1%}", "preview": chunk['text'][:150] + "..." if len(chunk['text']) > 150 else chunk['text']})
    
    context = "\n\n---\n\n".join(context_parts)
    
    system_prompt = """You are a helpful assistant answering questions about a resume.
CRITICAL RULES:
1. Answer ONLY based on the provided context
2. If information isn't in context, say so
3. Be concise and honest
4. Quote specific details when relevant""" if guardrails else "You are a helpful assistant that answers questions about a resume."
    
    user_prompt = f"Context from resume:\n{context}\n\nQuestion: {query}\n\nAnswer based on the context above:"
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    "temperature": 0.3,
                    "max_tokens": 500
                }
            )
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
        
        avg_similarity = sum(s for _, s in retrieved_chunks) / len(retrieved_chunks)
        return {"answer": answer, "sources": sources, "confidence": avg_similarity}
    except Exception as e:
        return {"answer": f"❌ Error: {str(e)}", "sources": sources, "confidence": 0.0}

def main():
    st.markdown("""<style>
    .main-header { font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #888; margin-bottom: 2rem; }
    .tech-badge { display: inline-block; padding: 0.25rem 0.75rem; background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); border-radius: 9999px; font-size: 0.75rem; margin: 0.25rem; color: white; }
    </style>""", unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">📄 Resume RAG Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Query my experience directly instead of scanning a PDF</p>', unsafe_allow_html=True)
    st.markdown('<div><span class="tech-badge">BGE-base embeddings</span> <span class="tech-badge">FAISS</span> <span class="tech-badge">Llama 3 (Groq)</span> <span class="tech-badge">Guardrails</span></div>', unsafe_allow_html=True)
    
    embed_model = load_embedding_model()
    chunks, index, embeddings = load_resume_data(embed_model)
    
    if chunks is None:
        st.error("⚠️ No resume found. Please add a `resume.md` file to the project.")
        st.stop()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "k_chunks" not in st.session_state:
        st.session_state.k_chunks = 5
    if "guardrails" not in st.session_state:
        st.session_state.guardrails = True
    
    with st.sidebar:
        st.header("⚙️ Settings")
        st.session_state.k_chunks = st.slider("Chunks to retrieve", 1, 10, st.session_state.k_chunks)
        st.session_state.guardrails = st.checkbox("Enable guardrails", st.session_state.guardrails)
        st.divider()
        st.metric("Total Chunks", len(chunks))
        st.divider()
        if GROQ_API_KEY:
            st.success("✅ Groq API configured")
        else:
            st.error("❌ Add GROQ_API_KEY to secrets")
        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    st.markdown("### 💡 Example Questions")
    cols = st.columns(5)
    for i, q in enumerate(["Distributed systems experience?", "RAG pipeline experience?", "Inference optimization?", "LLM experience?", "ML background?"]):
        if cols[i].button(q, key=f"ex_{i}"):
            st.session_state.messages.append({"role": "user", "content": q})
    
    st.divider()
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask about my experience..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retrieved = retrieve_chunks(prompt, embed_model, index, chunks, st.session_state.k_chunks)
                result = generate_answer_groq(prompt, retrieved, st.session_state.guardrails)
            st.markdown(result["answer"])
            st.caption(f"Confidence: {result['confidence']:.0%}")
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

if __name__ == "__main__":
    main()
