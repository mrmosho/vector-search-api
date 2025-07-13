"""
Search Service Module
Responsible for: Business logic for search operations
"""
import logging
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
from datetime import datetime

import pandas as pd

from data_loader import DataLoader
from text_processor import TextProcessor
from model_manager import ModelManager
from file_manager import FileManager
from semantic_index import SemanticIndex
from keyword_index import KeywordIndex
from query_analyzer import QueryAnalyzer
from result_formatter import ResultFormatter
from hybrid_searcher import HybridSearcher
from api_models import SearchResult

logger = logging.getLogger(__name__)


class SearchService:
    """Service class handling search business logic"""
    
    def __init__(self, csv_path: str = "data.csv"):
        self.csv_path = csv_path
        self.hybrid_searcher = None
        self.df = None
        self.is_initialized = False
        self.initialization_error = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the search system"""
        try:
            logger.info("🚀 Initializing search service...")
            
            # Initialize components
            data_loader = DataLoader(self.csv_path)
            text_processor = TextProcessor()
            model_manager = ModelManager()
            file_manager = FileManager()
            
            # Load and validate data
            self.df = data_loader.load_data()
            data_loader.validate_columns(["TITLE", "DESCRIPTION"])
            
            # Process text data
            processed_texts = text_processor.process_documents(self.df)
            
            # Initialize components
            result_formatter = ResultFormatter(self.df)
            semantic_available = model_manager.load_model()
            
            # Initialize search indices
            semantic_index = SemanticIndex(model_manager, file_manager)
            keyword_index = KeywordIndex(file_manager)
            
            # Create/load indices
            if semantic_available:
                semantic_index.load_or_create_index(processed_texts)
            keyword_index.load_or_create_index(processed_texts)
            
            # Initialize hybrid searcher
            query_analyzer = QueryAnalyzer()
            self.hybrid_searcher = HybridSearcher(
                semantic_index,
                keyword_index,
                query_analyzer,
                result_formatter
            )
            
            self.is_initialized = True
            logger.info("✅ Search service initialized successfully")
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"❌ Search service initialization failed: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the search service"""
        if self.is_initialized and self.hybrid_searcher:
            capabilities = self.hybrid_searcher.get_search_capabilities()
            return {
                "status": "healthy",
                "message": "Search system is running",
                "documents_loaded": len(self.df) if self.df is not None else 0,
                "search_capabilities": capabilities
            }
        else:
            return {
                "status": "unhealthy",
                "message": f"Search system initialization failed: {self.initialization_error}",
                "documents_loaded": 0,
                "search_capabilities": {}
            }

    # Update method signature:
    def search(self, query: str, top_k: int = 10,
           start_date: Optional[datetime] = None,
           end_date: Optional[datetime] = None,
           symbol: Optional[str] = None) -> Dict[str, Any]:

        if not self.is_initialized:
            raise Exception(f"Search service not initialized: {self.initialization_error}")

        df_filtered = self.df.copy()
        
        # Debug: Log original data info
        logger.info(f"Original data shape: {df_filtered.shape}")
        logger.info(f"Sample MOD_DATE values: {df_filtered['MOD_DATE'].head()}")
        logger.info(f"Sample SYMBOLS values: {df_filtered['SYMBOLS'].head()}")
        
        # Debug: Check how many KABO records exist
        if symbol:
            kabo_count = len(df_filtered[df_filtered['SYMBOLS'] == symbol])
            logger.info(f"Total records with SYMBOLS='{symbol}': {kabo_count}")

        # 🛠 Ensure MOD_DATE is datetime for safe comparison
        # The date format is "4/17/2025 1:45:37.000000 PM" - handle this properly
        try:
            # Try the actual format from your data
            df_filtered["MOD_DATE"] = pd.to_datetime(df_filtered["MOD_DATE"], format="%m/%d/%Y %I:%M:%S.%f %p", errors="coerce")
        except:
            try:
                # Try without microseconds
                df_filtered["MOD_DATE"] = pd.to_datetime(df_filtered["MOD_DATE"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
            except:
                try:
                    # Try standard formats
                    df_filtered["MOD_DATE"] = pd.to_datetime(df_filtered["MOD_DATE"], format="%Y-%m-%d", errors="coerce")
                except:
                    # Last resort - let pandas figure it out
                    df_filtered["MOD_DATE"] = pd.to_datetime(df_filtered["MOD_DATE"], errors="coerce")
        
        # Debug: Log after datetime conversion
        logger.info(f"After datetime conversion: {df_filtered['MOD_DATE'].head()}")
        logger.info(f"Date range in data: {df_filtered['MOD_DATE'].min()} to {df_filtered['MOD_DATE'].max()}")
        
        # Apply symbol filter first (before date filtering)
        if symbol:
            logger.info(f"Filtering by symbol: {symbol}")
            df_filtered = df_filtered[df_filtered['SYMBOLS'] == symbol]
            logger.info(f"After symbol filter: {df_filtered.shape}")

        # Remove rows with invalid dates for date filtering
        if start_date or end_date:
            df_filtered = df_filtered[df_filtered["MOD_DATE"].notna()]
            logger.info(f"After removing invalid dates: {df_filtered.shape}")

        # Fix date comparisons by ensuring both sides are pandas datetime objects
        if start_date:
            start_date_pd = pd.to_datetime(start_date).date()
            df_filtered["MOD_DATE_DATE"] = df_filtered["MOD_DATE"].dt.date
            logger.info(f"Filtering by start_date: {start_date_pd}")
            df_filtered = df_filtered[df_filtered["MOD_DATE_DATE"] >= start_date_pd]
            logger.info(f"After start_date filter: {df_filtered.shape}")

        if end_date:
            end_date_pd = pd.to_datetime(end_date).date()
            if "MOD_DATE_DATE" not in df_filtered.columns:
                df_filtered["MOD_DATE_DATE"] = df_filtered["MOD_DATE"].dt.date
            logger.info(f"Filtering by end_date: {end_date_pd}")
            df_filtered = df_filtered[df_filtered["MOD_DATE_DATE"] <= end_date_pd]
            logger.info(f"After end_date filter: {df_filtered.shape}")

        if df_filtered.empty:
           logger.warning("No results after filtering")
           return {"results": [], "strategy": "No results (filtered)"}

        # Log final filtered data
        logger.info(f"Final filtered data shape: {df_filtered.shape}")

        self.hybrid_searcher.result_formatter.df = df_filtered
        query_analysis = self.hybrid_searcher.query_analyzer.analyze_query(query)

        semantic_results = self.hybrid_searcher._get_semantic_results(query, query_analysis)
        keyword_results = self.hybrid_searcher._get_keyword_results(query, query_analysis)
        combined_scores = self.hybrid_searcher._combine_results(semantic_results, keyword_results, query_analysis)

        semantic_weight = query_analysis['semantic_weight']
        keyword_weight = query_analysis['keyword_weight']

        strategy = f"{semantic_weight:.0%} semantic + {keyword_weight:.0%} keyword ({query_analysis['strategy']})" if semantic_weight > 0 else "Keyword matching only"

        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Debug: Log search results
        logger.info(f"Search found {len(sorted_results)} potential results")

        results = []
        seen_combinations = set()  # Track title + date combinations instead of just titles
        displayed = 0
    
        valid_indices = set(df_filtered.index)

        for idx, combined_score in sorted_results:
            if idx not in valid_indices:
                continue  # Skip rows filtered out by date/symbol

            if displayed >= top_k:
                break

            row = df_filtered.loc[idx]
            title = row["TITLE"].strip()

            # Format date for comparison
            date = row.get("MOD_DATE", "N/A")
            if pd.isna(date):
                date_str = "N/A"
            elif isinstance(date, pd.Timestamp):
                date_str = date.strftime("%Y-%m-%d")
            else:
                date_str = str(date)

            # Create unique combination of title + date
            combination = f"{title}|{date_str}"
            
            # Skip if we've seen this exact title + date combination
            if combination in seen_combinations:
                logger.info(f"Skipping duplicate combination: {title} on {date_str}")
                continue
            
            seen_combinations.add(combination)

            # Format date properly for frontend (reuse the date_str from above)
            date = date_str
                
            # Debug: Log the date being returned
            logger.info(f"Result {displayed + 1}: Date = {date}, Original = {row.get('MOD_DATE', 'N/A')}")
            
            source = row.get("SOURCE_NAME", "N/A")
            description = BeautifulSoup(row.get("DESCRIPTION", ""), "html.parser").get_text(separator=" ")
            short_summary = description.strip().replace("\n", " ")[:300]
            if len(short_summary) == 300 and len(description) > 300:
                short_summary += "..."

            result = SearchResult(
               result_number=displayed + 1,
               title=title,
               date=str(date),
               source=source,
               summary=short_summary,
               score=float(combined_score)
           )
            results.append(result)
            displayed += 1

        logger.info(f"Returning {len(results)} results")

        return {
           "results": results,
           "strategy": strategy
        }
