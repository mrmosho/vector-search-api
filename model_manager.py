"""
Model Manager Module
Responsible for: Loading and managing sentence transformer models
"""
from sentence_transformers import SentenceTransformer


class ModelManager:
    """Manages sentence transformer model loading with fallbacks"""
    
    MODEL_OPTIONS = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2", 
        "sentence-transformers/all-mpnet-base-v2"
    ]
    
    def __init__(self):
        self.model = None
        self.model_name = None
    
    def load_model(self):
        """Load model with fallback options"""
        for model_name in self.MODEL_OPTIONS:
            try:
                print(f"🔄 Attempting to load model: {model_name}")
                self.model = SentenceTransformer(model_name)
                self.model_name = model_name
                print(f"✅ Successfully loaded: {model_name}")
                return True
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {str(e)[:100]}...")
                continue
        
        print("🚨 All model downloads failed! Using keyword-only mode.")
        return False
    
    def encode(self, texts, **kwargs):
        """Encode texts to embeddings"""
        if self.model is None:
            raise Exception("No model loaded")
        return self.model.encode(texts, **kwargs)
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None
    
    def get_model_name(self):
        """Get the name of the loaded model"""
        return self.model_name