import os
import time
import requests
from pathlib import Path
from backend.config import settings

PAPERS = {
    "attention_is_all_you_need": {
        "id": "1706.03762",
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762.pdf"
    },
    "bert": {
        "id": "1810.04805",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "url": "https://arxiv.org/pdf/1810.04805.pdf"
    },
    "retrieval_augmented_generation": {
        "id": "2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "url": "https://arxiv.org/pdf/2005.11401.pdf"
    }
}

def download_papers():
    dest_dir = Path(settings.PAPERS_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting paper download script...")
    for filename, info in PAPERS.items():
        dest_path = dest_dir / f"{filename}.pdf"
        if dest_path.exists() and dest_path.stat().st_size > 10000:
            print(f"Paper '{info['title']}' already exists at {dest_path}. Skipping.")
            continue
            
        print(f"Downloading '{info['title']}' from {info['url']}...")
        headers = {
            "User-Agent": "PaperPilot/1.0 (academic research assistant; contact: academic@example.com)"
        }
        
        # arXiv polite request limits: 1 request every 3 seconds
        time.sleep(3.0)
        
        try:
            response = requests.get(info["url"], headers=headers, stream=True)
            response.raise_for_status()
            
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Successfully downloaded and saved to {dest_path}")
        except Exception as e:
            print(f"Failed to download '{info['title']}': {e}")
            if dest_path.exists():
                dest_path.unlink() # clean up corrupted files

if __name__ == "__main__":
    download_papers()
