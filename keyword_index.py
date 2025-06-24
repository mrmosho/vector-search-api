"""
Keyword Index Module
Responsible for: Creating and managing TF-IDF keyword search index
"""
from sklearn.feature_extraction.text import TfidfVectorizer


class KeywordIndex:
    """Handles TF-IDF vectorization and keyword search operations"""
    
    def __init__(self, file_manager):
        self.file_manager = file_manager
        self.vectorizer = None
        self.tfidf_matrix = None
        self.matrix_file = "tfidf_matrix.npz"
        self.vocab_file = "tfidf_vocab.npy"
    
    def load_or_create_index(self, texts):
        """Load existing TF-IDF index or create new one"""
        if self._load_existing_index():
            print("✅ Loaded existing TF-IDF index")
            return True
        
        return self._create_new_index(texts)
    
    def _load_existing_index(self):
        """Load existing TF-IDF matrix and vocabulary"""
        if (self.file_manager.file_exists(self.matrix_file) and 
            self.file_manager.file_exists(self.vocab_file)):
            try:
                self.tfidf_matrix = self.file_manager.load_sparse_matrix(self.matrix_file)
                vocab = self.file_manager.load_dictionary(self.vocab_file)
                
                # Create a new vectorizer and manually set required attributes
                self.vectorizer = TfidfVectorizer(
                    analyzer='word',
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=10000,
                    lowercase=True,
                    max_df=0.95,
                    stop_words='english'
                )
                
                # Manually set the attributes that make it "fitted"
                self.vectorizer.vocabulary_ = vocab
                # Create reverse vocabulary mapping
                self.vectorizer._tfidf = self.vectorizer._get_hasher().fit([])  # Dummy fit for hasher
                # The easier approach: just recreate if loading fails
                return True
                
            except Exception as e:
                print(f"⚠️  TF-IDF loading failed, will recreate: {e}")
                return False
        return False
    
    def _create_new_index(self, texts):
        """Create new TF-IDF matrix and vocabulary"""
        try:
            print("⚡ Creating TF-IDF index (memory-efficient)...")
            
            # Memory-efficient TF-IDF configuration
            self.vectorizer = TfidfVectorizer(
                analyzer='word',
                ngram_range=(1, 2),
                min_df=2,
                max_features=10000,
                lowercase=True,
                max_df=0.95,
                stop_words='english'
            )
            
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Save matrix and vocabulary
            self.file_manager.save_sparse_matrix(self.matrix_file, self.tfidf_matrix)
            self.file_manager.save_dictionary(self.vocab_file, self.vectorizer.vocabulary_)
            
            print(f"✅ TF-IDF index created: {self.tfidf_matrix.shape} matrix")
            print(f"📊 Memory usage: ~{self.tfidf_matrix.data.nbytes / 1024 / 1024:.1f} MB")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create TF-IDF index: {e}")
            return False
    
    def search(self, query, top_k=50):
        """Search using keyword similarity"""
        if not self.is_ready():
            return []
        
        try:
            # Check if vectorizer is properly fitted
            if not hasattr(self.vectorizer, 'vocabulary_') or not self.vectorizer.vocabulary_:
                print("⚠️  Vectorizer not properly loaded, rebuilding...")
                return []
            
            query_tfidf = self.vectorizer.transform([query])
            # Use sparse matrix multiplication for efficiency
            similarities = (query_tfidf * self.tfidf_matrix.T).toarray().flatten()
            
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:
                    results.append({
                        'idx': idx,
                        'score': similarities[idx],
                        'type': 'keyword'
                    })
            return results
            
        except Exception as e:
            print(f"⚠️  Keyword search failed: {e}")
            print("💡 Try deleting tfidf_matrix.npz and tfidf_vocab.npy files to rebuild the index")
            return []
    
    def is_ready(self):
        """Check if keyword index is ready for searching"""
        return self.vectorizer is not None and self.tfidf_matrix is not None