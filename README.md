# Resume RAG Chatbot

A retrieval-augmented generation (RAG) system that lets recruiters query your resume conversationally instead of scanning bullet points.

**Live Demo:** https://resume-rag-sani.zocomputer.io

## Features

- **Conversational Q&A** - Ask questions like "What experience do you have with distributed systems?"
- **Semantic Search** - BGE-base-en-v1.5 embeddings for accurate retrieval
- **Fast Vector Index** - FAISS for efficient similarity search
- **LLM Powered** - Llama 3.3 70B via Groq API
- **Source Attribution** - Shows exactly which resume sections informed each answer
- **Guardrails** - Ensures answers stay grounded in your actual experience

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Embeddings | BGE-base-en-v1.5 (Sentence Transformers) |
| Vector Store | FAISS |
| LLM | Llama 3.3 70B via Groq |
| Language | Python 3.12+ |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export GROQ_API_KEY=your_groq_api_key
```

Get a free Groq API key at https://console.groq.com/keys

### 3. Add Your Resume

Edit `resume.md` with your actual resume content, then pre-compute the embeddings:

```bash
python precompute_index.py
```

### 4. Run the App

```bash
streamlit run streamlit_app.py
```

## How It Works

```
User Query
    ↓
┌─────────────────────────────────────────┐
│         Embed Query (BGE-base)          │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│     FAISS Vector Search (Top-K)         │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│   Context Injection + Guardrails        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│      Llama 3 Inference (Groq)           │
└─────────────────────────────────────────┘
    ↓
   Answer
```

## File Structure

```
resume-rag/
├── streamlit_app.py      # Main Streamlit application
├── rag_engine.py         # RAG pipeline (embeddings, FAISS, Groq)
├── precompute_index.py   # Pre-compute embeddings for faster startup
├── requirements.txt      # Python dependencies
├── resume.md             # Your resume content
├── README.md             # This file
└── .gitignore            # Git ignore rules
```

## Example Questions

- "What experience do you have with distributed systems?"
- "Have you built production RAG pipelines?"
- "How have you improved inference throughput?"
- "What's your experience with vector databases?"
- "Tell me about your ML projects"

## Customization

### Change the Embedding Model

Edit `rag_engine.py`:

```python
self.embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")  # Larger model
```

### Change the LLM

Edit `rag_engine.py`:

```python
self.llm_model = "llama-3.1-8b-instant"  # Faster, smaller model
```

### Adjust Chunk Size

Edit `precompute_index.py`:

```python
chunks = chunk_text(full_text, chunk_size=300, overlap=50)
```

## License

MIT

