from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import json
import logging
from datetime import datetime
import time
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vector Search API",
    description="A vector search API using FAISS and sentence transformers",
    version="1.0.0"
)

# Add CORS middleware for future frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS FOR API RESPONSES
# ============================================================================

class SearchResult(BaseModel):
    euclidean_distance: float
    data: Dict[str, Any]

class SearchResponse(BaseModel):
    method: str
    status: str
    text: str
    results: List[SearchResult]
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    search_engine_initialized: bool
    total_vectors: Optional[int] = None

class ProgressUpdate(BaseModel):
    stage: str
    progress: float
    message: str
    current: Optional[int] = None
    total: Optional[int] = None

# ============================================================================
# CLASSES FROM YOUR CLI (ADAPTED FOR FASTAPI)
# ============================================================================

class TextEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embedder with a sentence transformer model"""
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def embed_csv(self, csv_path: str, text_column: str, 
                  batch_size: int = 100, progress_callback=None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Convert CSV text column to embeddings with progress tracking"""
        print(f"\n📁 Reading CSV: {csv_path}")
        
        # Update progress - Reading CSV
        if progress_callback:
            progress_callback("reading_csv", 0.0, "Reading CSV file...")
        
        df = pd.read_csv(csv_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV. Available: {list(df.columns)}")
        
        # Clean text data
        texts = df[text_column].fillna('').astype(str).tolist()
        print(f"📊 Found {len(texts):,} text entries")
        
        # Update progress - Starting embeddings
        if progress_callback:
            progress_callback("generating_embeddings", 0.0, f"Generating embeddings for {len(texts)} texts...", 0, len(texts))
        
        # Generate embeddings in batches with progress bar
        print(f"\n🧠 Generating embeddings in batches of {batch_size}...")
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Create progress bar
        with tqdm(total=len(texts), desc="Processing texts", unit="texts") as pbar:
            for i, batch_start in enumerate(range(0, len(texts), batch_size)):
                batch_texts = texts[batch_start:batch_start + batch_size]
                batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
                embeddings.append(batch_embeddings)
                
                # Update progress
                current_progress = (i + 1) / total_batches
                current_items = min(batch_start + batch_size, len(texts))
                
                # Update tqdm progress bar
                pbar.update(len(batch_texts))
                pbar.set_postfix({
                    'Batch': f"{i+1}/{total_batches}",
                    'Progress': f"{current_progress:.1%}"
                })
                
                if progress_callback:
                    progress_callback(
                        "generating_embeddings", 
                        current_progress, 
                        f"Generated embeddings for {current_items}/{len(texts)} texts (batch {i+1}/{total_batches})",
                        current_items,
                        len(texts)
                    )
                
                # Print progress every 10 batches or at significant milestones
                if (i + 1) % 10 == 0 or current_progress in [0.25, 0.5, 0.75]:
                    print(f"✅ Completed batch {i+1}/{total_batches} ({current_progress:.1%}) - {current_items:,}/{len(texts):,} texts processed")
        
        embeddings_array = np.vstack(embeddings)
        print(f"✨ Successfully generated {embeddings_array.shape[0]:,} embeddings of dimension {embeddings_array.shape[1]}")
        
        # Update progress - Preparing metadata
        if progress_callback:
            progress_callback("preparing_metadata", 0.9, "Preparing metadata...")
        
        # Prepare metadata (all columns)
        metadata = df.to_dict('records')
        
        return embeddings_array, metadata

class FAISSIndexer:
    def __init__(self):
        self.index = None
        self.dimension = None
    
    def build_ivf_index(self, embeddings: np.ndarray, 
                       n_clusters: Optional[int] = None,
                       index_path: str = "index.ivf",
                       progress_callback=None) -> None:
        """Build IVF FAISS index from embeddings with progress tracking"""
        self.dimension = embeddings.shape[1]
        n_vectors = embeddings.shape[0]
        
        # Auto-determine cluster count if not provided
        if n_clusters is None:
            n_clusters = min(max(int(np.sqrt(n_vectors) * 2), 50), 1000)  # Increased clusters for better diversity
        
        print(f"\n🏗️  Building IVF index with {n_clusters} clusters for {n_vectors:,} vectors")
        
        # Update progress - Creating index
        if progress_callback:
            progress_callback("creating_index", 0.0, f"Creating IVF index with {n_clusters} clusters...")
        
        # Create IVF index with better configuration
        print("📐 Creating index structure...")
        quantizer = faiss.IndexFlatL2(self.dimension)
        
        # Use IVFFlat for exact distances within clusters
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, n_clusters)
        
        # Update progress - Training
        if progress_callback:
            progress_callback("training_index", 0.3, "Training FAISS index...")
        
        # Train the index
        print("🎯 Training FAISS index...")
        self.index.train(embeddings.astype(np.float32))
        print("✅ Index training completed!")
        
        # Update progress - Adding vectors
        if progress_callback:
            progress_callback("adding_vectors", 0.7, f"Adding {n_vectors} vectors to index...")
        
        # Add vectors to index
        print(f"📥 Adding {n_vectors:,} vectors to index...")
        self.index.add(embeddings.astype(np.float32))
        print("✅ All vectors added to index!")
        
        # Configure search parameters for better recall
        self.index.nprobe = min(n_clusters // 4, 50)  # Search more clusters by default
        
        # Update progress - Saving
        if progress_callback:
            progress_callback("saving_index", 0.9, "Saving index to disk...")
        
        # Save index
        print(f"💾 Saving index to {index_path}...")
        faiss.write_index(self.index, index_path)
        print(f"✅ Index saved successfully!")
        print(f"📊 Final index contains {self.index.ntotal:,} vectors")
        print(f"🎯 Default nprobe set to: {self.index.nprobe}")
        
        # Update progress - Complete
        if progress_callback:
            progress_callback("complete", 1.0, f"Index building complete! {self.index.ntotal} vectors indexed.")
    
    def load_index(self, index_path: str = "index.ivf") -> None:
        """Load existing FAISS index"""
        print(f"📂 Loading index from {index_path}")
        self.index = faiss.read_index(index_path)
        self.dimension = self.index.d
        print(f"✅ Loaded index with {self.index.ntotal:,} vectors, dimension {self.dimension}")
        
        # Set better default search parameters
        if hasattr(self.index, 'nprobe'):
            if hasattr(self.index, 'nlist'):
                self.index.nprobe = min(self.index.nlist // 4, 50)
                print(f"🎯 Set nprobe to: {self.index.nprobe} (out of {self.index.nlist} clusters)")

class VectorSearcher:
    def __init__(self, indexer, metadata: List[Dict[str, Any]], 
                 model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize searcher with loaded index and metadata"""
        self.indexer = indexer
        self.metadata = metadata
        self.model = SentenceTransformer(model_name)
        
        if len(metadata) != indexer.index.ntotal:
            logger.warning(f"Metadata length ({len(metadata)}) doesn't match index size ({indexer.index.ntotal})")
    
    def search(self, query: str, k: int = 5, nprobe: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors with duplicate filtering"""
        start_time = time.time()
        
        try:
            # Encode query
            query_vector = self.model.encode([query])
            
            # Set search parameters - increase nprobe for better recall
            self.indexer.index.nprobe = min(nprobe * 2, 50)  # Increase search scope
            
            # Search for more results than requested to filter duplicates
            search_k = min(k * 3, 50)  # Get more results to filter from
            
            # Perform search
            distances, indices = self.indexer.index.search(
                query_vector.astype(np.float32), search_k
            )
            
            # Format results with comprehensive validation and duplicate filtering
            results = []
            seen_titles = set()  # Track seen titles to avoid duplicates
            seen_descriptions = set()  # Track seen description content
            
            for distance, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.metadata) and len(results) < k:
                    try:
                        # Handle invalid distance values
                        if isinstance(distance, np.floating):
                            distance = float(distance)
                        
                        if not isinstance(distance, (int, float)) or math.isnan(distance) or math.isinf(distance):
                            distance = 999999.0
                        
                        # Get metadata
                        metadata = self.metadata[idx]
                        
                        # Create content fingerprints to detect duplicates
                        title = str(metadata.get('TITLE', '')).strip()
                        description = str(metadata.get('DESCRIPTION', '')).strip()
                        
                        # Create a content hash for duplicate detection
                        title_clean = title.lower().replace('&#8206;', '').replace('…', '').strip()
                        desc_first_100 = description[:100].lower().strip()
                        
                        # Skip if we've seen very similar content
                        if title_clean in seen_titles or desc_first_100 in seen_descriptions:
                            logger.debug(f"Skipping duplicate result: {title[:50]}...")
                            continue
                        
                        # Add to seen sets
                        seen_titles.add(title_clean)
                        seen_descriptions.add(desc_first_100)
                        
                        # Sanitize metadata to ensure all values are JSON-serializable
                        sanitized_metadata = sanitize_for_json(metadata)
                        
                        result = {
                            'euclidean_distance': float(distance),
                            'data': sanitized_metadata
                        }
                        results.append(result)
                        
                    except Exception as e:
                        logger.warning(f"Error processing result at index {idx}: {e}")
                        continue
            
            # If we still don't have enough unique results, log a warning
            if len(results) < k:
                logger.info(f"Only found {len(results)} unique results out of {k} requested")
            
            search_time = time.time() - start_time
            logger.info(f"Search completed in {search_time:.3f}s, found {len(results)} unique results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

# Global searcher instance
searcher = None
initialization_status = {"initialized": False, "error": None, "total_vectors": 0}
build_progress = {"stage": "idle", "progress": 0.0, "message": "Ready to build index"}

import math

def sanitize_for_json(obj):
    """Recursively sanitize an object to make it JSON-serializable"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 999999.0 if obj > 0 else -999999.0
        return obj
    elif isinstance(obj, np.floating):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return 999999.0 if val > 0 else -999999.0
        return val
    elif isinstance(obj, np.integer):
        return int(obj)
    elif obj is None:
        return None
    else:
        return obj

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_metadata(metadata: List[Dict[str, Any]], filepath: str = "metadata.json"):
    """Save metadata to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved {len(metadata)} metadata entries to {filepath}")

def load_metadata(filepath: str = "metadata.json") -> List[Dict[str, Any]]:
    """Load metadata from JSON file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metadata file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        metadata = json.load(f)
    logger.info(f"Loaded {len(metadata)} metadata entries from {filepath}")
    return metadata

def check_files_exist(*filepaths):
    """Check if required files exist"""
    missing = [f for f in filepaths if not os.path.exists(f)]
    if missing:
        logger.error(f"Missing files: {', '.join(missing)}")
        return False
    return True
    """Save metadata to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved {len(metadata)} metadata entries to {filepath}")

def load_metadata(filepath: str = "metadata.json") -> List[Dict[str, Any]]:
    """Load metadata from JSON file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metadata file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        metadata = json.load(f)
    logger.info(f"Loaded {len(metadata)} metadata entries from {filepath}")
    return metadata

def check_files_exist(*filepaths):
    """Check if required files exist"""
    missing = [f for f in filepaths if not os.path.exists(f)]
    if missing:
        logger.error(f"Missing files: {', '.join(missing)}")
        return False
    return True

def update_build_progress(stage: str, progress: float, message: str, current: int = None, total: int = None):
    """Update the global build progress"""
    global build_progress
    build_progress = {
        "stage": stage,
        "progress": progress,
        "message": message,
        "current": current,
        "total": total,
        "timestamp": datetime.now().isoformat()
    }
    # Print progress to terminal with emojis and formatting
    progress_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
    print(f"\n🔄 [{progress_bar}] {progress:.1%} - {stage.replace('_', ' ').title()}")
    print(f"   {message}")
    if current and total:
        print(f"   Progress: {current:,} / {total:,}")
    print("-" * 60)

def initialize_search_engine():
    """Initialize the search engine on startup"""
    global searcher, initialization_status
    
    try:
        print("\n" + "="*60)
        print("🚀 INITIALIZING VECTOR SEARCH ENGINE")
        print("="*60)
        
        update_build_progress("initializing", 0.0, "Starting search engine initialization...")
        
        # Check if index exists, if not create it
        if not check_files_exist("index.ivf", "metadata.json"):
            print("📋 Index files not found - will create from data.csv")
            update_build_progress("checking_files", 0.1, "Index not found, creating from data.csv...")
            
            if not os.path.exists("data.csv"):
                error_msg = "❌ data.csv file not found in current directory"
                print(error_msg)
                initialization_status["error"] = "data.csv file not found"
                update_build_progress("error", 0.0, "data.csv file not found")
                return
            
            print("✅ Found data.csv - proceeding with index creation")
            
            # Create index from CSV with progress tracking
            print("\n🤖 Loading sentence transformer model...")
            embedder = TextEmbedder()
            
            embeddings, metadata = embedder.embed_csv(
                "data.csv", 
                "DESCRIPTION", 
                batch_size=100,
                progress_callback=update_build_progress
            )
            
            # Build index with progress tracking
            indexer = FAISSIndexer()
            indexer.build_ivf_index(
                embeddings,
                progress_callback=update_build_progress
            )
            
            # Save metadata
            print("\n💾 Saving metadata...")
            update_build_progress("saving_metadata", 0.95, "Saving metadata...")
            save_metadata(metadata)
            print("✅ Metadata saved successfully!")
        else:
            print("📂 Found existing index files - loading...")
            update_build_progress("loading_existing", 0.5, "Loading existing index...")
        
        # Load index and metadata
        print("\n📥 Loading index and metadata...")
        indexer = FAISSIndexer()
        indexer.load_index()
        metadata = load_metadata()
        
        # Initialize searcher
        print("🔍 Initializing search engine...")
        update_build_progress("finalizing", 0.98, "Initializing search engine...")
        searcher = VectorSearcher(indexer, metadata)
        initialization_status["initialized"] = True
        initialization_status["total_vectors"] = indexer.index.ntotal
        
        # Final success message
        print("\n" + "="*60)
        print("🎉 SEARCH ENGINE READY!")
        print(f"📊 Total vectors indexed: {indexer.index.ntotal:,}")
        print(f"📐 Vector dimension: {indexer.index.d}")
        print(f"🔍 Ready to serve search requests!")
        print("="*60)
        
        update_build_progress("complete", 1.0, f"Search engine ready! {indexer.index.ntotal:,} vectors loaded.")
        
    except Exception as e:
        error_msg = f"❌ Failed to initialize search engine: {str(e)}"
        print(f"\n{error_msg}")
        print("="*60)
        logger.error(error_msg)
        initialization_status["error"] = error_msg
        update_build_progress("error", 0.0, error_msg)

# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize search engine when FastAPI starts"""
    initialize_search_engine()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": "Vector Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        search_engine_initialized=initialization_status["initialized"],
        total_vectors=initialization_status.get("total_vectors")
    )

@app.get("/build-progress", response_model=ProgressUpdate, tags=["Progress"])
async def get_build_progress():
    """Get the current progress of index building"""
    global build_progress
    return ProgressUpdate(**build_progress)

@app.get("/build-progress/stream", tags=["Progress"])
async def stream_build_progress():
    """Stream build progress updates via Server-Sent Events"""
    async def generate():
        last_progress = -1
        while True:
            current_progress = build_progress.get("progress", 0)
            
            # Send update if progress changed or if we're not complete
            if current_progress != last_progress or current_progress < 1.0:
                yield f"data: {json.dumps(build_progress)}\n\n"
                last_progress = current_progress
            
            # Stop streaming when complete or error
            if build_progress.get("stage") in ["complete", "error"]:
                break
                
            await asyncio.sleep(1)  # Update every second
    
    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search(
    q: str = Query(..., description="Search query text", min_length=1),
    k: int = Query(5, description="Number of results to return", ge=1, le=50),
    nprobe: int = Query(20, description="Search precision (higher = better recall)", ge=1, le=100)
):
    """
    Search for similar vectors based on text query
    
    - **q**: The search query text
    - **k**: Number of results to return (default: 5, max: 50)
    - **nprobe**: Search precision - higher values search more clusters (default: 20)
    
    Returns results with euclidean distance scores and metadata.
    """
    try:
        # Check if search engine is initialized
        if not initialization_status["initialized"]:
            error_msg = initialization_status.get("error", "Search engine not initialized")
            return SearchResponse(
                method="get",
                status="error",
                text=q,
                results=[],
                error=error_msg
            )
        
        # Validate query
        if not q or q.strip() == "":
            return SearchResponse(
                method="get",
                status="error",  
                text=q,
                results=[],
                error="Query cannot be empty"
            )
        
        # Perform search with specified nprobe
        logger.info(f"Searching for: '{q}' with k={k}, nprobe={nprobe}")
        search_results = searcher.search(q.strip(), k=k, nprobe=nprobe)
        
        if not search_results:
            return SearchResponse(
                method="get",
                status="success",
                text=q,
                results=[],
                error="No results found"
            )
        
        # Convert to response format with comprehensive validation
        results = []
        for result in search_results:
            try:
                # Ensure all data is properly sanitized
                sanitized_result = sanitize_for_json(result)
                
                search_result = SearchResult(
                    euclidean_distance=sanitized_result["euclidean_distance"],
                    data=sanitized_result["data"]
                )
                results.append(search_result)
                
            except Exception as e:
                logger.warning(f"Skipping invalid result: {e}")
                continue
        
        response = SearchResponse(
            method="get",
            status="success",
            text=q,
            results=results
        )
        
        return response
        
    except Exception as e:
        error_msg = f"Search error: {str(e)}"
        logger.error(error_msg)
        return SearchResponse(
            method="get",
            status="error",
            text=q,
            results=[],
            error=error_msg
        )

@app.get("/debug-search", tags=["Debug"])
async def debug_search(
    q: str = Query(..., description="Search query text", min_length=1),
    k: int = Query(1, description="Number of results to return", ge=1, le=5)
):
    """
    Debug version of search that returns raw data to identify serialization issues
    """
    try:
        if not initialization_status["initialized"]:
            return {"error": "Search engine not initialized"}
        
        # Perform search
        query_vector = searcher.model.encode([q])
        searcher.indexer.index.nprobe = 10
        distances, indices = searcher.indexer.index.search(query_vector.astype(np.float32), k)
        
        debug_results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(searcher.metadata):
                debug_info = {
                    "result_index": i,
                    "faiss_index": int(idx),
                    "raw_distance": str(distance),
                    "distance_type": str(type(distance)),
                    "distance_isnan": bool(np.isnan(distance)),
                    "distance_isinf": bool(np.isinf(distance)),
                    "metadata_keys": list(searcher.metadata[idx].keys()),
                    "sample_metadata": {}
                }
                
                # Sample some metadata values
                for key, value in list(searcher.metadata[idx].items())[:3]:
                    debug_info["sample_metadata"][key] = {
                        "value": str(value)[:100],
                        "type": str(type(value)),
                        "is_nan": str(value) == 'nan' if isinstance(value, (int, float)) else False
                    }
                
                debug_results.append(debug_info)
        
        return {"debug_results": debug_results}
        
    except Exception as e:
        return {"error": f"Debug search failed: {str(e)}"}
async def rebuild_index():
    """
    Rebuild the search index from data.csv with progress tracking
    Use this if you've updated your data.csv file
    """
    try:
        if not os.path.exists("data.csv"):
            raise HTTPException(status_code=404, detail="data.csv file not found")
        
        logger.info("Rebuilding index from data.csv...")
        update_build_progress("rebuild_start", 0.0, "Starting index rebuild...")
        
        # Run the rebuild in a separate thread to avoid blocking
        def rebuild_task():
            global searcher, initialization_status
            
            try:
                # Create new index with progress tracking
                embedder = TextEmbedder()
                embeddings, metadata = embedder.embed_csv(
                    "data.csv", 
                    "DESCRIPTION", 
                    batch_size=100,
                    progress_callback=update_build_progress
                )
                
                # Build index with progress tracking
                indexer = FAISSIndexer()
                indexer.build_ivf_index(
                    embeddings,
                    progress_callback=update_build_progress
                )
                
                # Save metadata
                update_build_progress("saving_metadata", 0.95, "Saving metadata...")
                save_metadata(metadata)
                
                # Reinitialize searcher
                update_build_progress("finalizing", 0.98, "Finalizing new index...")
                searcher = VectorSearcher(indexer, metadata)
                initialization_status["initialized"] = True
                initialization_status["total_vectors"] = indexer.index.ntotal
                initialization_status["error"] = None
                
                update_build_progress("complete", 1.0, f"Index rebuilt successfully! {indexer.index.ntotal} vectors indexed.")
                
            except Exception as e:
                error_msg = f"Rebuild failed: {str(e)}"
                logger.error(error_msg)
                update_build_progress("error", 0.0, error_msg)
                initialization_status["error"] = error_msg
        
        # Start rebuild in background
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(rebuild_task)
        
        return {
            "status": "started",
            "message": "Index rebuild started. Check /build-progress for updates.",
            "progress_endpoint": "/build-progress",
            "stream_endpoint": "/build-progress/stream"
        }
        
    except Exception as e:
        logger.error(f"Index rebuild error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)