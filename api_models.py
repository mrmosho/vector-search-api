"""
API Models Module
Responsible for: Defining Pydantic models for API requests and responses
"""
from pydantic import BaseModel
from typing import List, Dict, Any


class SearchResult(BaseModel):
    """Single search result matching CLI output format"""
    result_number: int
    title: str
    date: str
    source: str
    summary: str
    score: float


class SearchResponse(BaseModel):
    """Response model matching CLI output format"""
    query: str
    strategy: str
    total_results: int
    results: List[SearchResult]
    execution_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    documents_loaded: int
    search_capabilities: Dict[str, Any]