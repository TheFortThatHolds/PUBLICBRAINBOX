"""
Semantic Search for BrainBox
============================

Uses the embedding model to search files by MEANING not just keywords.
Perfect for finding that document where you talked about "that thing" 
but can't remember the exact words.
"""

import requests
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import pickle

class SemanticSearcher:
    """Simple semantic search using local embedding model"""
    
    def __init__(self):
        self.api_url = "http://10.14.0.2:1234"
        self.embed_model = "text-embedding-nomic-embed-text-v1.5"
        self.index_path = Path("brainbox_data/semantic_index.pkl")
        self.embeddings = {}  # file_path -> vector
        self.texts = {}  # file_path -> text content
        
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text"""
        try:
            response = requests.post(
                f"{self.api_url}/v1/embeddings",
                json={
                    "model": self.embed_model,
                    "input": text
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['data'][0]['embedding']
            else:
                print(f"[SEMANTIC] Error getting embedding: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"[SEMANTIC] Exception: {e}")
            return None
    
    def index_file(self, file_path: Path):
        """Index a single file"""
        try:
            text = file_path.read_text(encoding='utf-8')
            
            # Chunk text (simple version - by paragraphs)
            chunks = text.split('\n\n')
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 50:  # Skip tiny chunks
                    chunk_key = f"{file_path}:chunk_{i}"
                    
                    # Get embedding
                    embedding = self.get_embedding(chunk[:2000])  # Limit chunk size
                    
                    if embedding:
                        self.embeddings[chunk_key] = np.array(embedding)
                        self.texts[chunk_key] = chunk
                        print(f"[SEMANTIC] Indexed {chunk_key}")
                        
        except Exception as e:
            print(f"[SEMANTIC] Error indexing {file_path}: {e}")
    
    def index_directory(self, directory: Path, extensions: List[str] = ['.md', '.txt', '.py']):
        """Index all files in directory"""
        print(f"[SEMANTIC] Indexing {directory}...")
        
        for ext in extensions:
            for file_path in directory.rglob(f"*{ext}"):
                print(f"[SEMANTIC] Processing {file_path}")
                self.index_file(file_path)
        
        # Save index
        self.save_index()
        print(f"[SEMANTIC] Indexed {len(self.embeddings)} chunks")
    
    def save_index(self):
        """Save index to disk"""
        self.index_path.parent.mkdir(exist_ok=True)
        
        with open(self.index_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'texts': self.texts
            }, f)
        
        print(f"[SEMANTIC] Index saved to {self.index_path}")
    
    def load_index(self):
        """Load index from disk"""
        if self.index_path.exists():
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
                self.texts = data['texts']
            print(f"[SEMANTIC] Loaded {len(self.embeddings)} chunks from index")
            return True
        return False
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Search for similar content"""
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        query_vec = np.array(query_embedding)
        
        # Calculate similarities
        results = []
        for chunk_key, chunk_vec in self.embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_vec, chunk_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)
            )
            results.append((chunk_key, self.texts[chunk_key], similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results[:top_k]
    
    def search_and_display(self, query: str, top_k: int = 3):
        """Search and display results nicely"""
        print(f"\n[SEMANTIC SEARCH] Query: {query}")
        print("=" * 60)
        
        results = self.search(query, top_k)
        
        if not results:
            print("No results found!")
            return
        
        for i, (chunk_key, text, similarity) in enumerate(results, 1):
            file_path = chunk_key.split(':chunk_')[0]
            print(f"\n{i}. {file_path} (similarity: {similarity:.3f})")
            print("-" * 40)
            # Show preview (first 200 chars)
            preview = text[:200].replace('\n', ' ')
            print(f"{preview}...")
            print()

# Example usage
if __name__ == "__main__":
    searcher = SemanticSearcher()
    
    # Try to load existing index
    if not searcher.load_index():
        print("No index found. Indexing current directory...")
        # Index the current BrainBox directory
        searcher.index_directory(Path("."))
    
    # Test searches
    test_queries = [
        "emotional routing system",
        "business ethics and decision making",
        "voice family collaboration",
        "madugu quadrant system",
        "breakfast chain logging"
    ]
    
    for query in test_queries:
        searcher.search_and_display(query)
        print("\n" + "="*60 + "\n")