import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz # PyMuPDF

from backend.config import settings
from backend.services.pdf_parser import PDFParser
from backend.services.vector_store import VectorStore
from backend.services.llm_service import LLMService
from backend.services.orchestrator import Orchestrator

app = FastAPI(title="PaperPilot API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REGISTRY_PATH = Path(settings.CHROMA_DB_PATH).parent / "papers_registry.json"

# Services instances
pdf_parser = PDFParser()
vector_store = VectorStore()

def load_registry() -> Dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {}
    with open(REGISTRY_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_registry(registry: Dict[str, Any]):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

def generate_paper_id(filename: str) -> str:
    # Clean filename to generate paper id
    base = Path(filename).stem.lower()
    clean = re.sub(r'[^a-z0-9\_]', '', base.replace(' ', '_').replace('-', '_'))
    return clean or "unnamed_paper"

import re

class ChatRequest(BaseModel):
    question: str
    provider: Optional[str] = None # gemini, openai, anthropic

@app.get("/api/papers")
def list_papers():
    registry = load_registry()
    return list(registry.values())

@app.post("/api/papers/upload")
async def upload_paper(
    file: UploadFile = File(...),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    c_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
    c_overlap = chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP
    
    # Save file to papers directory
    safe_filename = file.filename.replace(" ", "_")
    dest_path = Path(settings.PAPERS_DIR) / safe_filename
    
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Open PDF to get details
        doc = fitz.open(str(dest_path))
        page_count = len(doc)
        
        # Determine title
        title = doc.metadata.get("title")
        if not title or len(title.strip()) < 3:
            # Try to read the first few lines of text
            first_page_text = doc[0].get_text().strip()
            first_lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]
            title = first_lines[0] if first_lines else dest_path.stem.replace("_", " ")
        doc.close()
        
        paper_id = generate_paper_id(safe_filename)
        
        # Check if already in registry and remove old chunks
        registry = load_registry()
        if paper_id in registry:
            vector_store.delete_paper(paper_id)
            
        # Parse and chunk PDF
        pages_data = pdf_parser.parse_pdf(str(dest_path))
        chunks = pdf_parser.chunk_pages(pages_data, chunk_size_tokens=c_size, overlap_tokens=c_overlap)
        
        # Embed and insert chunks
        vector_store.add_paper_chunks(paper_id, title, chunks)
        
        # Update registry
        registry[paper_id] = {
            "id": paper_id,
            "title": title,
            "filename": safe_filename,
            "page_count": page_count,
            "chunk_count": len(chunks),
            "chunk_size": c_size,
            "chunk_overlap": c_overlap
        }
        save_registry(registry)
        
        return registry[paper_id]
        
    except Exception as e:
        # Cleanup file if processing failed
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.delete("/api/papers/{paper_id}")
def delete_paper(paper_id: str):
    registry = load_registry()
    if paper_id not in registry:
        raise HTTPException(status_code=404, detail="Paper not found.")
        
    info = registry[paper_id]
    pdf_path = Path(settings.PAPERS_DIR) / info["filename"]
    
    # Delete from vector store
    vector_store.delete_paper(paper_id)
    
    # Delete file
    if pdf_path.exists():
        pdf_path.unlink()
        
    # Remove from registry
    del registry[paper_id]
    save_registry(registry)
    
    return {"status": "success", "detail": f"Paper '{paper_id}' deleted."}

@app.post("/api/chat")
def chat(request: ChatRequest):
    registry = load_registry()
    available_papers = list(registry.values())
    
    if not available_papers:
        raise HTTPException(
            status_code=400, 
            detail="No papers uploaded yet. Please upload papers before asking questions."
        )
        
    # Check if API Key exists for requested provider
    provider = request.provider or settings.DEFAULT_PROVIDER
    provider = provider.lower()
    
    # Dynamically verify if target API key exists
    key_exists = False
    if provider == "gemini":
        key_exists = bool(settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY"))
    elif provider == "openai":
        key_exists = bool(settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY"))
    elif provider == "anthropic":
        key_exists = bool(settings.ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY"))
        
    if not key_exists:
        raise HTTPException(
            status_code=400,
            detail=f"API key for provider '{provider}' is not configured. Please add it to your .env file."
        )
        
    try:
        llm_service = LLMService(provider=provider)
        orchestrator = Orchestrator(vector_store, llm_service)
        
        result = orchestrator.query(request.question, available_papers)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/eval")
def get_eval_results():
    eval_results_path = Path(__file__).resolve().parent / "eval" / "eval_results.json"
    if not eval_results_path.exists():
        return []
    with open(eval_results_path, "r") as f:
        return json.load(f)

@app.post("/api/eval/run")
def trigger_eval():
    try:
        from backend.scripts.run_eval import run_evaluation
        results = run_evaluation()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health_check():
    # Simple check to see if key models are present or if API keys are set
    providers = {
        "gemini": bool(settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")),
        "openai": bool(settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")),
        "anthropic": bool(settings.ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY"))
    }
    return {
        "status": "healthy",
        "configured_providers": providers,
        "default_provider": settings.DEFAULT_PROVIDER
    }
