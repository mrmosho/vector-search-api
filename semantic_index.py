"""
Semantic Index Module
Responsible for: Creating and managing semantic embeddings and FAISS index
"""
import numpy as np
import faiss


class SemanticIndex:
    """Handles semantic embeddings and FAISS index operations"""
    
    def __init__(self, model_manager, file_manager):
        self.model_manager = model_manager
        self.file_manager = file_manager
        self.embeddings = None
        self.index = None
        self.embeddings_file = "embeddings.npy"
        self.index_file = "faiss.index"
    
    def load_or_create_index(self, texts):
        """Load existing index or create new one"""
        if self._load_existing_index():
            print("✅ Loaded existing semantic index")
            return True
        
        if not self.model_manager.is_loaded():
            print("⚠️  No model available for semantic indexing")
            return False
        
        return self._create_new_index(texts)
    
    def _load_existing_index(self):
        """Load existing embeddings and index from files"""
        if (self.file_manager.file_exists(self.embeddings_file) and 
            self.file_manager.file_exists(self.index_file)):
            try:
                self.embeddings = self.file_manager.load_numpy_array(self.embeddings_file)
                self.index = self.file_manager.load_faiss_index(self.index_file)
                return True
            except Exception as e:
                print(f"⚠️  Failed to load existing semantic index: {e}")
                return False
        return False
    
    def _create_new_index(self, texts):
        """Create new embeddings and FAISS index"""
        try:
            print("⚡ Creating semantic embeddings...")
            self.embeddings = self.model_manager.encode(
                texts,
                show_progress_bar=True,
                normalize_embeddings=True,
                batch_size=32
            ).astype("float32")
            
            print("⚡ Creating FAISS index...")
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)
            
            # Save to files
            self.file_manager.save_numpy_array(self.embeddings_file, self.embeddings)
            self.file_manager.save_faiss_index(self.index_file, self.index)
            
            print(f"✅ Semantic index created: {self.embeddings.shape[0]} documents")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create semantic index: {e}")
            return False
    
    def search(self, query, top_k=50):
        """Search using semantic similarity"""
        if not self.is_ready():
            return []
        
        try:
            query_embedding = self.model_manager.encode(
                [query], 
                normalize_embeddings=True
            ).astype("float32")
            
            distances, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for idx, score in zip(indices[0], distances[0]):
                if idx != -1:
                    results.append({
                        'idx': idx,
                        'score': score,
                        'type': 'semantic'
                    })
            return results
            
        except Exception as e:
            print(f"⚠️  Semantic search failed: {e}")
            return []
    
    def is_ready(self):
        """Check if semantic index is ready for searching"""
        return (self.embeddings is not None and 
                self.index is not None and 
                self.model_manager.is_loaded())