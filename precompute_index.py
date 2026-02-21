"""
Pre-compute embeddings and FAISS index for fast API loading.
Run this when the resume changes.
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Load resume
with open("/home/workspace/resume-rag/resume.md", 'r') as f:
    resume_text = f.read()

# Load BGE-base model
print("Loading BGE-base-en-v1.5 model...")
embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Chunk the resume with metadata
def chunk_resume(text: str) -> list:
    """Structured chunking with metadata."""
    chunks = []
    current_section = "general"
    current_text = []
    
    lines = text.split('\n')
    
    section_markers = {
        "experience": ["## Experience", "## Professional Experience"],
        "skills": ["## Technical Skills", "## Skills"],
        "projects": ["## Projects"],
        "education": ["## Education"],
        "faq": ["## FAQ"],
        "summary": ["## Summary"],
        "contact": ["## Contact"],
    }
    
    current_metadata = {}
    
    def add_chunk(text: str, section: str, metadata: dict):
        text = text.strip()
        if len(text) < 20:
            return
        
        # Split large chunks
        if len(text) > 500:
            words = text.split()
            for i in range(0, len(words), 100):
                chunk_text = ' '.join(words[i:i+120])
                if len(chunk_text.strip()) > 20:
                    chunks.append({
                        "text": chunk_text,
                        "source": section,
                        "metadata": {**metadata, "part": i // 100}
                    })
        else:
            chunks.append({
                "text": text,
                "source": section,
                "metadata": metadata
            })
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check for section headers
        new_section = None
        for section, markers in section_markers.items():
            if any(marker in line for marker in markers):
                if current_text:
                    add_chunk('\n'.join(current_text), current_section, current_metadata)
                    current_text = []
                
                new_section = section
                current_section = section
                current_metadata = {"section_header": line_stripped}
                break
        
        if new_section:
            continue
        
        # Check for job entries
        if line_stripped.startswith('### '):
            if current_text:
                add_chunk('\n'.join(current_text), current_section, current_metadata)
                current_text = []
            current_metadata = {"role": line_stripped.replace('### ', '')}
        
        # Accumulate text
        if line_stripped:
            current_text.append(line_stripped)
    
    # Last chunk
    if current_text:
        add_chunk('\n'.join(current_text), current_section, current_metadata)
    
    return chunks

print("Chunking resume...")
chunks = chunk_resume(resume_text)
print(f"Created {len(chunks)} chunks from sections: {set(c['source'] for c in chunks)}")

# Generate embeddings
print("Generating embeddings...")
texts = [chunk["text"] for chunk in chunks]
embeddings = embed_model.encode(texts, normalize_embeddings=True)

# Build FAISS index
print("Building FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings.astype('float32'))

# Save everything
print("Saving index and chunks...")
faiss.write_index(index, "/home/workspace/resume-rag/resume.index")

with open("/home/workspace/resume-rag/chunks.pkl", 'wb') as f:
    pickle.dump(chunks, f)

# Also save embeddings for reference
np.save("/home/workspace/resume-rag/embeddings.npy", embeddings)

print("✅ Pre-computation complete!")
print(f"   - Index saved to: resume.index")
print(f"   - Chunks saved to: chunks.pkl")
print(f"   - Embeddings saved to: embeddings.npy")
print(f"   - Total chunks: {len(chunks)}")
print(f"   - Embedding dimension: {dim}")
