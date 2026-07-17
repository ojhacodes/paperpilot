# PaperPilot — Multi-Paper Research Assistant

PaperPilot is a lightweight, local, and grounded research assistant that lets you search and compare multiple scientific PDFs at once. It uses a custom text ingestion pipeline, a local vector database, and an LLM-orchestrated loop to retrieve page-and-section-aware source text, synthesize answers, and proactively prevent hallucinations.

## Key Features
- **Page-and-Section Aware Parsing**: Parses PDFs via PyMuPDF, dynamically extracts page numbers, and utilizes layout hierarchy to attach section headers to every text chunk.
- **Local Embeddings**: Runs `sentence-transformers/all-MiniLM-L6-v2` locally (free, zero API cost) to embed chunks and query vectors.
- **Multi-Paper Tag-Based Orchestrator**: Uses a single LLM chat loop with an XML tag tool-calling interface (`<search paper_id="..." query="..." />`) to allow the LLM to search multiple papers, inspect details, and synthesize cross-document comparisons.
- **Quantitative Evaluation Harness**: Built-in verification script testing retrieval accuracy against a fixed set of 15 Q&A reference pairs on standard papers (*Attention Is All You Need*, *BERT*, and *Retrieval-Augmented Generation*).

---

## Quantitative Evaluation Results

As outlined in the PRD, retrieval quality was measured quantitatively using a custom evaluation dataset. The benchmark assesses the **Hit Rate** (whether the ground-truth reference page appears in the top-K retrieved chunks) across different chunking configurations:

| Config | Hit Rate @ top-3 | Hit Rate @ top-5 |
| :--- | :---: | :---: |
| **Chunk 256, No Overlap** | 53.3% | 73.3% |
| **Chunk 512, 50-token Overlap** | 46.7% | 60.0% |
| **Chunk 512, 200-token Overlap** (Optimized) | **80.0%** | **86.7%** |

### Key Takeaways
1. **Overlap is Critical**: Increasing overlap from 50 tokens to 200 tokens for 512-token chunks improved the Top-3 Hit Rate by **33.3%**. This occurs because scientific papers have dense section boundaries, and larger overlaps prevent content from being split mid-paragraph.
2. **Larger Context Window Helps**: A chunk size of 512 tokens with 200-token overlap captures complete semantic arguments (e.g., equations + explanations), outperforming the smaller 256-token chunk.

---

## Technical Stack & Architecture

```
User → React/Vite Frontend (Dark Glassmorphic UI)
            │
            ▼
    FastAPI Backend
            │
    ┌───────┴───────┐
    ▼               ▼
LLM Orchestrator  Vector DB (ChromaDB)
(Gemini/Claude)         │
    │               ▼
    │         Local Embeddings
    │         (all-MiniLM-L6-v2)
    │               ▲
    ▼               │
PDF Parser (PyMuPDF) ──► Word/Token Overlapping Chunker
```

- **Backend**: FastAPI, PyMuPDF (fitz), ChromaDB, Sentence-Transformers, Uvicorn.
- **Frontend**: Vite, React, Vanilla CSS.
- **LLM Support**: Gemini (via `google-genai`), OpenAI (`openai`), Anthropic (`anthropic`).

---

## Setup & Running Guide

### 1. Environment Configuration
Create a `.env` file in the project root (using the template `.env.example`):
```bash
cp .env.example .env
```
Populate at least one LLM provider key in `.env`:
```env
GEMINI_API_KEY=AIzaSy...
DEFAULT_PROVIDER=gemini
```

### 2. Install Dependencies & Run Backend
Start the FastAPI server:
```bash
# Activate virtual environment
source venv/bin/activate

# Download sample papers (Attention, BERT, RAG)
PYTHONPATH=. python backend/scripts/download_papers.py

# Ingest them and initialize the vector store database
PYTHONPATH=. python backend/scripts/run_eval.py

# Launch FastAPI
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Run Frontend
Open a separate terminal, install node packages, and run the Vite dev server:
```bash
cd frontend
npm install
npm run dev
```

Open your browser at `http://localhost:5173`. You can chat, examine the citations details drawer, upload new research papers, and trigger retrieval evaluations dynamically from the dashboard!
