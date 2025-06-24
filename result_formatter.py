"""
Result Formatter Module
Responsible for: Formatting and displaying search results
"""
from bs4 import BeautifulSoup


class ResultFormatter:
    """Handles formatting and display of search results"""
    
    def __init__(self, df):
        self.df = df
    
    def format_and_display_results(self, query, combined_scores, query_analysis, top_k=10):
        """Format and display search results"""
        if not combined_scores:
            self._display_no_results(query)
            return
        
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Display header
        self._display_header(query, query_analysis)
        
        # Display results
        seen_titles = set()
        displayed = 0
        
        for idx, combined_score in sorted_results:
            if displayed >= top_k:
                break
                
            title = self.df.iloc[idx]["TITLE"].strip()
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            self._display_single_result(idx, combined_score, displayed + 1)
            displayed += 1
        
        if displayed == 0:
            self._display_no_results(query)
    
    def _display_header(self, query, query_analysis):
        """Display search header with strategy info"""
        print(f"\n🔍 Search results for: \"{query}\"")
        
        semantic_weight = query_analysis['semantic_weight']
        keyword_weight = query_analysis['keyword_weight']
        strategy = query_analysis['strategy']
        
        if semantic_weight > 0:
            print(f"📊 Strategy: {semantic_weight:.0%} semantic + {keyword_weight:.0%} keyword ({strategy})")
        else:
            print("📊 Strategy: Keyword matching only")
        
        print("=" * 60)
    
    def _display_single_result(self, idx, score, result_number):
        """Display a single search result"""
        row = self.df.iloc[idx]
        
        title = row["TITLE"].strip()
        date = row.get("MOD_DATE", "N/A")
        source = row.get("SOURCE_NAME", "N/A")
        
        # Clean and truncate description
        description = BeautifulSoup(row.get("DESCRIPTION", ""), "html.parser").get_text(separator=" ")
        short_summary = description.strip().replace("\n", " ")[:300]
        
        print(f"📄 Result #{result_number} (Score: {score:.4f})")
        print(f"   Title   : {title}")
        print(f"   Date    : {date} | Source: {source}")
        print(f"   Summary : {short_summary}...")
        print("-" * 60)
    
    def _display_no_results(self, query):
        """Display message when no results found"""
        print("❌ No results found")
        
        # Show direct string matches for debugging
        query_lower = query.lower()
        title_matches = self.df[self.df['TITLE'].str.lower().str.contains(query_lower, na=False)]
        desc_matches = self.df[self.df['DESCRIPTION'].str.lower().str.contains(query_lower, na=False)]
        
        if len(title_matches) > 0 or len(desc_matches) > 0:
            print(f"\n🔧 Found {len(title_matches)} direct matches in TITLE, {len(desc_matches)} in DESCRIPTION")
            print("This suggests the search algorithm needs adjustment.")