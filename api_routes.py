"""
API Routes Module
Responsible for: Defining FastAPI route handlers
"""
from fastapi import HTTPException, Query
from datetime import datetime
import logging

from api_models import SearchResponse, HealthResponse
from search_service import SearchService

logger = logging.getLogger(__name__)


class APIRoutes:
    """Class containing all API route handlers"""
    
    def __init__(self, search_service: SearchService):
        self.search_service = search_service
    
    async def health_check(self) -> HealthResponse:
        """Health check endpoint"""
        try:
            health_status = self.search_service.get_health_status()
            
            return HealthResponse(
                status=health_status["status"],
                message=health_status["message"],
                documents_loaded=health_status["documents_loaded"],
                search_capabilities=health_status["search_capabilities"]
            )
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
    
    async def search(
        self,
        q: str = Query(..., description="Search query"),
        top_k: int = Query(default=10, ge=1, le=50, description="Number of results to return")
    ) -> SearchResponse:
        """Search endpoint - returns JSON similar to CLI output"""
        start_time = datetime.now()
        
        try:
            # Perform search using the search service
            search_results = self.search_service.search(query=q, top_k=top_k)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return SearchResponse(
                query=q,
                strategy=search_results["strategy"],
                total_results=len(search_results["results"]),
                results=search_results["results"],
                execution_time_ms=round(execution_time, 2)
            )
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")