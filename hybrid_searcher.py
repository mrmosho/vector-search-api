"""
Hybrid Searcher Module
Responsible for: Coordinating semantic and keyword search, combining results
"""


class HybridSearcher:
    """Coordinates semantic and keyword search to provide hybrid results"""
    
    def __init__(self, semantic_index, keyword_index, query_analyzer, result_formatter):
        self.semantic_index = semantic_index
        self.keyword_index = keyword_index
        self.query_analyzer = query_analyzer
        self.result_formatter = result_formatter
    
    def search(self, query, top_k=10):
        """Perform hybrid search combining semantic and keyword approaches"""
        # Analyze query to determine strategy
        query_analysis = self.query_analyzer.analyze_query(query)
        
        # Get results from both search methods
        semantic_results = self._get_semantic_results(query, query_analysis)
        keyword_results = self._get_keyword_results(query, query_analysis)
        
        # Combine results with appropriate weighting
        combined_scores = self._combine_results(
            semantic_results, 
            keyword_results, 
            query_analysis
        )
        
        # Format and display results
        self.result_formatter.format_and_display_results(
            query, 
            combined_scores, 
            query_analysis, 
            top_k
        )
    
    def _get_semantic_results(self, query, query_analysis):
        """Get semantic search results if available"""
        if not self.semantic_index.is_ready():
            return []
        
        # Adjust result count based on query type
        result_count = 30 if query_analysis['is_short'] else 40
        return self.semantic_index.search(query, top_k=result_count)
    
    def _get_keyword_results(self, query, query_analysis):
        """Get keyword search results"""
        if not self.keyword_index.is_ready():
            return []
        
        # Adjust result count based on query type
        result_count = 30 if query_analysis['is_short'] else 20
        return self.keyword_index.search(query, top_k=result_count)
    
    def _combine_results(self, semantic_results, keyword_results, query_analysis):
        """Combine semantic and keyword results with appropriate weighting"""
        combined_scores = {}
        
        semantic_weight = query_analysis['semantic_weight']
        keyword_weight = query_analysis['keyword_weight']
        
        # Add semantic results
        for result in semantic_results:
            idx = result['idx']
            combined_scores[idx] = combined_scores.get(idx, 0) + (result['score'] * semantic_weight)
        
        # Add keyword results
        for result in keyword_results:
            idx = result['idx']
            combined_scores[idx] = combined_scores.get(idx, 0) + (result['score'] * keyword_weight)
        
        return combined_scores
    
    def get_search_capabilities(self):
        """Return information about available search capabilities"""
        capabilities = {
            'semantic_available': self.semantic_index.is_ready(),
            'keyword_available': self.keyword_index.is_ready(),
            'hybrid_mode': self.semantic_index.is_ready() and self.keyword_index.is_ready()
        }
        
        if capabilities['hybrid_mode']:
            capabilities['mode'] = 'hybrid'
        elif capabilities['keyword_available']:
            capabilities['mode'] = 'keyword_only'
        else:
            capabilities['mode'] = 'none'
        
        return capabilities