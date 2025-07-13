"""
Query Analyzer Module
Responsible for: Analyzing queries and determining search strategy
"""


class QueryAnalyzer:
    """Analyzes queries to determine optimal search strategy"""
    
    @staticmethod
    def is_short_query(query, max_length=6):
        """Determine if query is short (likely acronym or specific term)"""
        cleaned_query = query.strip()
        return len(cleaned_query) <= max_length and cleaned_query.isalnum()
    
    @staticmethod
    def get_search_weights(query):
        """Determine weights for semantic vs keyword search based on query"""
        if QueryAnalyzer.is_short_query(query):
            
            return {
                'semantic_weight': 0.9,
                'keyword_weight': 0.1,
                'strategy': 'keyword-focused'
            }
        else:
            
            return {
                'semantic_weight': 0.3,
                'keyword_weight': 0.7,
                'strategy': 'semantic-focused'
            }
    
    @staticmethod
    def analyze_query(query):
        """Complete query analysis"""
        weights = QueryAnalyzer.get_search_weights(query)
        is_short = QueryAnalyzer.is_short_query(query)
        
        return {
            'query': query.strip(),
            'is_short': is_short,
            'length': len(query.strip()),
            'semantic_weight': weights['semantic_weight'],
            'keyword_weight': weights['keyword_weight'],
            'strategy': weights['strategy']
        }
