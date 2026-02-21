"""
Resume RAG System
BGE-base embeddings + FAISS + Llama 3 (Groq) + Guardrails
"""

import os
import json
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import httpx

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Chunk:
    id: str
    text: str
    source: str  # "experience", "skills", "projects", "faq", "education"
    metadata: Dict

class ResumeRAG:
    def __init__(self, groq_api_key: Optional[str] = None):
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        
        # Load BGE-base embeddings model
        self.embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        self.embedding_dim = 768
        
        # FAISS index
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Chunk] = []
        self.chunk_id_to_idx: Dict[str, int] = {}
        
        # Initialize
        self._initialized = False
    
    def _generate_chunk_id(self, text: str, source: str) -> str:
        """Generate unique chunk ID."""
        hash_input = f"{source}:{text}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _chunk_resume(self, resume_text: str) -> List[Chunk]:
        """
        Structured chunking of resume with metadata tagging.
        Splits by sections and creates semantically meaningful chunks.
        """
        chunks = []
        current_section = "general"
        current_text = []
        
        lines = resume_text.split('\n')
        
        # Section headers that indicate new sections
        section_markers = {
            "experience": ["## Experience", "## Professional Experience"],
            "skills": ["## Technical Skills", "## Skills"],
            "projects": ["## Projects"],
            "education": ["## Education"],
            "faq": ["## FAQ"],
            "summary": ["## Summary"],
            "contact": ["## Contact"],
        }
        
        def add_chunk(text: str, section: str, metadata: Dict):
            text = text.strip()
            if len(text) < 20:
                return
            
            # Split large chunks
            if len(text) > 500:
                words = text.split()
                for i in range(0, len(words), 100):
                    chunk_text = ' '.join(words[i:i+120])
                    if len(chunk_text.strip()) > 20:
                        chunks.append(Chunk(
                            id=self._generate_chunk_id(chunk_text, section),
                            text=chunk_text,
                            source=section,
                            metadata={**metadata, "part": i // 100}
                        ))
            else:
                chunks.append(Chunk(
                    id=self._generate_chunk_id(text, section),
                    text=text,
                    source=section,
                    metadata=metadata
                ))
        
        current_metadata = {}
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check for section headers
            new_section = None
            for section, markers in section_markers.items():
                if any(marker in line for marker in markers):
                    # Save previous chunk
                    if current_text:
                        add_chunk('\n'.join(current_text), current_section, current_metadata)
                        current_text = []
                    
                    new_section = section
                    current_section = section
                    current_metadata = {"section_header": line_stripped}
                    break
            
            if new_section:
                continue
            
            # Check for job entries (company names are typically ### headers or bold)
            if line_stripped.startswith('### '):
                if current_text:
                    add_chunk('\n'.join(current_text), current_section, current_metadata)
                    current_text = []
                current_metadata = {"role": line_stripped.replace('### ', '')}
            
            # Check for FAQ questions
            elif line_stripped.startswith('### ') and current_section == "faq":
                if current_text:
                    add_chunk('\n'.join(current_text), current_section, current_metadata)
                    current_text = []
                current_metadata = {"question": line_stripped.replace('### ', '')}
            
            # Accumulate text
            if line_stripped:
                current_text.append(line_stripped)
        
        # Don't forget the last chunk
        if current_text:
            add_chunk('\n'.join(current_text), current_section, current_metadata)
        
        return chunks
    
    def build_index(self, resume_text: str):
        """Build FAISS index from resume text."""
        # Chunk the resume
        self.chunks = self._chunk_resume(resume_text)
        
        if not self.chunks:
            raise ValueError("No chunks created from resume")
        
        # Generate embeddings
        texts = [chunk.text for chunk in self.chunks]
        embeddings = self.embed_model.encode(texts, normalize_embeddings=True)
        
        # Build FAISS index (Inner Product for cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        
        # Build ID mapping
        self.chunk_id_to_idx = {chunk.id: i for i, chunk in enumerate(self.chunks)}
        
        self._initialized = True
        
        return {
            "total_chunks": len(self.chunks),
            "sections": list(set(c.source for c in self.chunks)),
            "initialized": True
        }
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        """Retrieve top-k relevant chunks."""
        if not self._initialized:
            raise RuntimeError("Index not initialized. Call build_index first.")
        
        # Encode query
        query_embedding = self.embed_model.encode([query], normalize_embeddings=True)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def _build_guardrails_prompt(self, query: str, context: str) -> str:
        """Build prompt with guardrails for grounded responses."""
        return f"""You are a helpful assistant answering questions about a candidate's resume and professional experience.

IMPORTANT RULES:
1. ONLY answer based on the provided context from the resume
2. If the context doesn't contain information to answer the question, say "I don't have specific information about that in the resume"
3. Be specific and cite relevant experience when possible
4. Don't make up or infer information not explicitly stated
5. Keep responses concise but informative

CONTEXT FROM RESUME:
{context}

QUESTION: {query}

Answer based ONLY on the context above:"""
    
    def generate_answer(self, query: str, k: int = 5) -> Dict:
        """Generate answer using Groq (Llama 3) with retrieved context."""
        if not self.groq_api_key:
            return {
                "answer": "GROQ_API_KEY not configured. Please add it to your secrets.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Retrieve relevant chunks
        retrieved = self.retrieve(query, k=k)
        
        if not retrieved:
            return {
                "answer": "I couldn't find relevant information in the resume to answer your question.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Build context
        context_parts = []
        sources = []
        
        for chunk, score in retrieved:
            context_parts.append(f"[{chunk.source.upper()}]\n{chunk.text}")
            sources.append({
                "section": chunk.source,
                "text_preview": chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text,
                "relevance": round(score, 3)
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Call Groq API (Llama 3)
        prompt = self._build_guardrails_prompt(query, context)
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.groq_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant that answers questions about a candidate's resume accurately and honestly."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 500
                    }
                )
                response.raise_for_status()
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": sources,
                "confidence": 0.0
            }
        
        # Calculate confidence as average relevance
        avg_relevance = sum(s["relevance"] for s in sources) / len(sources)
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": round(avg_relevance, 3),
            "chunks_used": len(retrieved)
        }
    
    def get_sample_questions(self) -> List[str]:
        """Get sample questions for the resume."""
        return [
            "What experience do you have with distributed systems?",
            "Have you built production RAG pipelines?",
            "How have you improved inference throughput?",
            "What's your approach to LLM safety and guardrails?",
            "What vector databases have you worked with?",
            "Tell me about your ML engineering experience.",
            "What companies have you worked for?",
            "What certifications do you have?"
        ]


# Singleton instance
_rag_instance: Optional[ResumeRAG] = None

def get_rag_instance() -> ResumeRAG:
    """Get or create the RAG instance."""
    global _rag_instance
    
    if _rag_instance is None:
        _rag_instance = ResumeRAG()
        
        # Load resume
        resume_path = "/home/workspace/resume-rag/resume.md"
        if os.path.exists(resume_path):
            with open(resume_path, 'r') as f:
                resume_text = f.read()
            _rag_instance.build_index(resume_text)
    
    return _rag_instance


if __name__ == "__main__":
    # Test the RAG system
    rag = ResumeRAG()
    
    with open("/home/workspace/resume-rag/resume.md", 'r') as f:
        resume_text = f.read()
    
    stats = rag.build_index(resume_text)
    print(f"Built index with {stats['total_chunks']} chunks from sections: {stats['sections']}")
    
    # Test query
    result = rag.generate_answer("What experience do you have with distributed systems?")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nConfidence: {result['confidence']}")
    print(f"\nSources: {len(result['sources'])}")
