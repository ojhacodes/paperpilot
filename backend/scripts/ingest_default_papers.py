import os
import sys
import json
from pathlib import Path
import fitz  # PyMuPDF

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from backend.config import settings
from backend.services.pdf_parser import PDFParser
from backend.services.vector_store import VectorStore
from backend.main import REGISTRY_PATH, load_registry, save_registry, generate_paper_id

def ingest_default_papers():
    parser = PDFParser()
    vector_store = VectorStore()
    
    papers_dir = Path(settings.PAPERS_DIR)
    default_papers = {
        "attention_is_all_you_need.pdf": "Attention Is All You Need",
        "bert.pdf": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "retrieval_augmented_generation.pdf": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    }
    
    registry = load_registry()
    c_size = settings.DEFAULT_CHUNK_SIZE
    c_overlap = 200  # Use the optimized config from eval!
    
    print("Ingesting default papers into the main production vector database...")
    for filename, default_title in default_papers.items():
        pdf_path = papers_dir / filename
        if not pdf_path.exists():
            print(f"File {pdf_path} does not exist. Skipping.")
            continue
            
        paper_id = generate_paper_id(filename)
        
        # Open PDF to get details
        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        
        # Try to extract actual title, default to the known title
        title = doc.metadata.get("title") or default_title
        if not title or len(title.strip()) < 3:
            title = default_title
        doc.close()
        
        print(f"Ingesting '{title}' ({filename})...")
        
        # Clean existing chunks
        vector_store.delete_paper(paper_id)
        
        # Parse and chunk PDF
        pages_data = parser.parse_pdf(str(pdf_path))
        chunks = parser.chunk_pages(pages_data, chunk_size_tokens=c_size, overlap_tokens=c_overlap)
        
        # Embed and insert chunks
        vector_store.add_paper_chunks(paper_id, title, chunks)
        
        # Update registry
        registry[paper_id] = {
            "id": paper_id,
            "title": title,
            "filename": filename,
            "page_count": page_count,
            "chunk_count": len(chunks),
            "chunk_size": c_size,
            "chunk_overlap": c_overlap
        }
        
    save_registry(registry)
    print("Ingestion of default papers complete! Registry updated.")

if __name__ == "__main__":
    ingest_default_papers()
