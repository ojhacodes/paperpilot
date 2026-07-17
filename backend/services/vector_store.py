import os
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from backend.config import settings

class VectorStore:
    def __init__(self):
        # Initialize chroma client
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection("research_papers")
        
        # Load local embedding model
        # Note: loading is deferred to first embedding call or manual initialization to keep startup fast
        self._model = None

    @property
    def model(self):
        if self._model is None:
            print("Loading sentence-transformers/all-MiniLM-L6-v2 embedding model...")
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            print("Embedding model loaded.")
        return self._model

    def add_paper_chunks(self, paper_id: str, paper_title: str, chunks: List[Dict[str, Any]]):
        """
        Embeds chunks and adds them to ChromaDB.
        """
        if not chunks:
            return
            
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(texts).tolist()
        
        ids = [f"{paper_id}_chunk_{chunk['chunk_index']}" for chunk in chunks]
        
        metadatas = []
        for chunk in chunks:
            metadatas.append({
                "paper_id": paper_id,
                "paper_title": paper_title,
                "chunk_index": chunk["chunk_index"],
                "primary_page": chunk["primary_page"],
                "primary_section": chunk["primary_section"],
                "pages_covered": ",".join(map(str, chunk["pages_covered"])),
                "sections_covered": ",".join(chunk["sections_covered"])
            })
            
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        print(f"Added {len(chunks)} chunks for paper '{paper_title}' to ChromaDB.")

    def search_chunks(
        self, 
        query: str, 
        paper_id: Optional[str] = None, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Searches ChromaDB for chunks matching the query.
        Optionally filters by paper_id.
        """
        # Embed query
        query_embedding = self.model.encode(query).tolist()
        
        # Build filter dict
        where_filter = None
        if paper_id:
            where_filter = {"paper_id": paper_id}
            
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_filter
        )
        
        formatted_results = []
        if results and "documents" in results and results["documents"]:
            # Chroma returns a list of lists since we passed a list of query embeddings
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            ids = results["ids"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            
            for doc, meta, cid, dist in zip(docs, metas, ids, distances):
                # Parse comma separated covered pages back to list
                pages_covered_str = meta.get("pages_covered", "")
                pages_covered = [int(p) for p in pages_covered_str.split(",") if p.isdigit()]
                
                formatted_results.append({
                    "id": cid,
                    "text": doc,
                    "score": float(dist), # lower distance means higher similarity
                    "paper_id": meta.get("paper_id"),
                    "paper_title": meta.get("paper_title"),
                    "chunk_index": meta.get("chunk_index"),
                    "primary_page": meta.get("primary_page"),
                    "primary_section": meta.get("primary_section"),
                    "pages_covered": pages_covered,
                    "sections_covered": meta.get("sections_covered", "").split(",")
                })
                
        return formatted_results

    def delete_paper(self, paper_id: str):
        """
        Deletes all chunks associated with a paper.
        """
        self.collection.delete(where={"paper_id": paper_id})
        print(f"Deleted all chunks for paper '{paper_id}' from ChromaDB.")

    def reset_db(self):
        """
        Deletes all documents in the collection.
        """
        self.client.delete_collection("research_papers")
        self.collection = self.client.get_or_create_collection("research_papers")
        print("Vector database collection reset.")
