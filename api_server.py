"""
API Server Module
Responsible for: FastAPI application setup and configuration
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
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


class SearchRequest(BaseModel):
    q: str
    share_search: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    top_k: Optional[int] = 10


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
        
        # Search endpoint - POST with request body (standard approach)
        @self.app.post("/search", response_model=SearchResponse)
        async def search_endpoint(request: SearchRequest):
            try:
                # Extract parameters from request body
                query = request.q
                top_k = request.top_k or 10
                start_date = request.date_from
                end_date = request.date_to
                symbol = request.share_search
                
                # Call your search service with the parameters
                results = self.search_service.search(
                    query=query,
                    top_k=top_k,
                    start_date=start_date,
                    end_date=end_date,
                    symbol=symbol
                )
                
                return {
                    "query": query,
                    "strategy": results.get("strategy", "simple"),
                    "total_results": len(results.get("results", [])),
                    "results": results.get("results", []),
                    "execution_time_ms": 0.0
                }
            except Exception as e:
                logger.error(f"Search error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
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
    from fastapi import FastAPI, HTTPException
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
    
    @app.post("/search")
    async def search(request: SearchRequest):
        try:
            # Extract parameters from request body
            query = request.q
            top_k = request.top_k or 10
            start_date = request.date_from
            end_date = request.date_to
            symbol = request.share_search
            
            # This is the simple fallback - you can add actual search logic here
            return {
                "query": query,
                "strategy": "simple",
                "total_results": 0,
                "results": [],
                "execution_time_ms": 0.0
            }
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# Default app instance
app = create_app()
