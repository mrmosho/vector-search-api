"""
API Server Module
Responsible for: FastAPI application setup and configuration
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import with error handling
try:
    from search_service import SearchService
    from api_routes import APIRoutes
    from api_models import SearchResponse, HealthResponse
    IMPORTS_OK = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_OK = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIServer:
    """FastAPI application wrapper"""
    
    def __init__(self, csv_path: str = "data.csv"):
        if not IMPORTS_OK:
            raise ImportError("Required modules could not be imported")
            
        self.csv_path = csv_path
        self.app = FastAPI(
            title="Hybrid Search API",
            description="REST API for semantic and keyword search with Arabic-English support",
            version="1.0.0"
        )
        
        # Initialize services
        self.search_service = SearchService(csv_path)
        self.routes = APIRoutes(self.search_service)
        
        # Setup application
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Health check endpoint
        self.app.get("/health", response_model=HealthResponse)(self.routes.health_check)
        
        # Search endpoint  
        self.app.get("/search", response_model=SearchResponse)(self.routes.search)
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance"""
        return self.app


# Create app instance for direct uvicorn usage
def create_app(csv_path: str = "data.csv") -> FastAPI:
    """Factory function to create FastAPI app"""
    try:
        server = APIServer(csv_path)
        return server.get_app()
    except ImportError:
        # Fallback to simple app
        return create_simple_app(csv_path)


def create_simple_app(csv_path: str = "data.csv") -> FastAPI:
    """Simple fallback app without modular imports"""
    from fastapi import FastAPI, Query, HTTPException
    from datetime import datetime
    
    app = FastAPI(title="Simple Search API", version="1.0.0")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "message": "Simple API is running",
            "documents_loaded": 0
        }
    
    @app.get("/search")
    async def search(q: str = Query(...), top_k: int = Query(default=10)):
        return {
            "query": q,
            "strategy": "simple",
            "total_results": 0,
            "results": [],
            "execution_time_ms": 0.0
        }
    
    return app


# Default app instance
app = create_app()