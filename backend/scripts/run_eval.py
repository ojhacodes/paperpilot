import os
import sys
import json
from pathlib import Path

# Add project root to sys.path so we can import backend packages
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from backend.config import settings
from backend.services.pdf_parser import PDFParser
from backend.services.vector_store import VectorStore

CONFIGS = [
    {"name": "Chunk 256, No Overlap", "size": 256, "overlap": 0},
    {"name": "Chunk 512, 50-token Overlap", "size": 512, "overlap": 50},
    {"name": "Chunk 512, 200-token Overlap", "size": 512, "overlap": 200}
]

PAPERS = {
    "attention_is_all_you_need": {
        "filename": "attention_is_all_you_need.pdf",
        "title": "Attention Is All You Need"
    },
    "bert": {
        "filename": "bert.pdf",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    },
    "retrieval_augmented_generation": {
        "filename": "retrieval_augmented_generation.pdf",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    }
}

def load_eval_dataset():
    eval_path = Path(__file__).resolve().parent.parent / "eval" / "eval_dataset.json"
    with open(eval_path, "r") as f:
        return json.load(f)

def run_evaluation():
    parser = PDFParser()
    vector_store = VectorStore()
    dataset = load_eval_dataset()
    
    # Check if PDFs are downloaded
    papers_dir = Path(settings.PAPERS_DIR)
    for p_id, info in PAPERS.items():
        pdf_path = papers_dir / info["filename"]
        if not pdf_path.exists() or pdf_path.stat().st_size < 10000:
            print(f"Error: PDF for paper '{info['title']}' not found at {pdf_path}.")
            print("Please run backend/scripts/download_papers.py first.")
            sys.exit(1)
            
    eval_results = []
    
    for config in CONFIGS:
        print(f"\nEvaluating configuration: {config['name']} (Size={config['size']}, Overlap={config['overlap']})...")
        
        # Reset Chroma DB and write to a test-specific collection name
        collection_name = f"test_{config['size']}_{config['overlap']}"
        try:
            vector_store.client.delete_collection(collection_name)
        except Exception:
            pass
        
        test_collection = vector_store.client.create_collection(collection_name)
        
        # Parse and ingest all papers for this configuration
        for p_id, info in PAPERS.items():
            pdf_path = papers_dir / info["filename"]
            pages_data = parser.parse_pdf(str(pdf_path))
            chunks = parser.chunk_pages(pages_data, config["size"], config["overlap"])
            
            # Embed and insert chunks
            texts = [c["text"] for c in chunks]
            embeddings = vector_store.model.encode(texts).tolist()
            ids = [f"{p_id}_chunk_{c['chunk_index']}" for c in chunks]
            metadatas = [{
                "paper_id": p_id,
                "paper_title": info["title"],
                "chunk_index": c["chunk_index"],
                "pages_covered": ",".join(map(str, c["pages_covered"])),
                "primary_page": c["primary_page"]
            } for c in chunks]
            
            test_collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            print(f"  Ingested '{info['title']}' -> {len(chunks)} chunks.")
            
        # Run queries
        hit_top3 = 0
        hit_top5 = 0
        total_queries = len(dataset)
        
        for qa in dataset:
            q_embedding = vector_store.model.encode(qa["question"]).tolist()
            
            # Search in the active collection, filtered by the target paper
            results = test_collection.query(
                query_embeddings=[q_embedding],
                n_results=5,
                where={"paper_id": qa["paper_id"]}
            )
            
            # Extract pages covered in results
            pages_in_results = []
            if results and "metadatas" in results and results["metadatas"]:
                for meta in results["metadatas"][0]:
                    pages_covered_str = meta.get("pages_covered", "")
                    pages = [int(p) for p in pages_covered_str.split(",") if p.isdigit()]
                    pages_in_results.append(pages)
            
            # A hit means the expected page is in the pages covered by the chunk
            expected = qa["expected_page"]
            
            # Top-3 hit
            is_hit_top3 = False
            for pages in pages_in_results[:3]:
                if expected in pages:
                    is_hit_top3 = True
                    break
            if is_hit_top3:
                hit_top3 += 1
                
            # Top-5 hit
            is_hit_top5 = False
            for pages in pages_in_results[:5]:
                if expected in pages:
                    is_hit_top5 = True
                    break
            if is_hit_top5:
                hit_top5 += 1
                
        rate_top3 = (hit_top3 / total_queries) * 100
        rate_top5 = (hit_top5 / total_queries) * 100
        
        print(f"Results for {config['name']}:")
        print(f"  Hit Rate @ top-3: {rate_top3:.2f}% ({hit_top3}/{total_queries})")
        print(f"  Hit Rate @ top-5: {rate_top5:.2f}% ({hit_top5}/{total_queries})")
        
        eval_results.append({
            "config": config["name"],
            "chunk_size": config["size"],
            "overlap": config["overlap"],
            "hit_rate_top3": round(rate_top3, 2),
            "hit_rate_top5": round(rate_top5, 2)
        })
        
        # Clean up collection
        vector_store.client.delete_collection(collection_name)
        
    # Output markdown table
    print("\n" + "="*50)
    print("FINAL COMPARISON TABLE")
    print("="*50)
    print("| Config | Hit rate @ top-3 | Hit rate @ top-5 |")
    print("| --- | --- | --- |")
    for res in eval_results:
        print(f"| {res['config']} | {res['hit_rate_top3']:.1f}% | {res['hit_rate_top5']:.1f}% |")
    print("="*50)
    
    # Save results to json for FastAPI to serve
    results_path = Path(__file__).resolve().parent.parent / "eval" / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)
        
    return eval_results

if __name__ == "__main__":
    run_evaluation()
