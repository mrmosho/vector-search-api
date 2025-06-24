"""
Main Application Module
Responsible for: Coordinating all components and providing user interface
"""
import os
from data_loader import DataLoader
from text_processor import TextProcessor
from model_manager import ModelManager
from file_manager import FileManager
from semantic_index import SemanticIndex
from keyword_index import KeywordIndex
from query_analyzer import QueryAnalyzer
from result_formatter import ResultFormatter
from hybrid_searcher import HybridSearcher

# Disable symlink warnings
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


class SearchApplication:
    """Main application class that coordinates all components"""
    
    def __init__(self, csv_path="data.csv"):
        self.csv_path = csv_path
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all application components"""
        print("🚀 Initializing Search System...")
        
        # Core components
        self.data_loader = DataLoader(self.csv_path)
        self.text_processor = TextProcessor()
        self.model_manager = ModelManager()
        self.file_manager = FileManager()
        
        # Search components (will be initialized after data loading)
        self.semantic_index = None
        self.keyword_index = None
        self.query_analyzer = QueryAnalyzer()
        self.result_formatter = None
        self.hybrid_searcher = None
    
    def setup(self):
        """Setup the search system - load data and create indices"""
        try:
            # Load and validate data
            df = self.data_loader.load_data()
            self.data_loader.validate_columns(["TITLE", "DESCRIPTION"])
            
            # Process text data
            processed_texts = self.text_processor.process_documents(df)
            
            # Initialize result formatter with dataframe
            self.result_formatter = ResultFormatter(df)
            
            # Attempt to load semantic model
            semantic_available = self.model_manager.load_model()
            
            # Initialize search indices
            self.semantic_index = SemanticIndex(self.model_manager, self.file_manager)
            self.keyword_index = KeywordIndex(self.file_manager)
            
            # Create/load indices
            if semantic_available:
                self.semantic_index.load_or_create_index(processed_texts)
            
            self.keyword_index.load_or_create_index(processed_texts)
            
            # Initialize hybrid searcher
            self.hybrid_searcher = HybridSearcher(
                self.semantic_index,
                self.keyword_index,
                self.query_analyzer,
                self.result_formatter
            )
            
            # Report capabilities
            self._report_capabilities()
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def _report_capabilities(self):
        """Report what search capabilities are available"""
        capabilities = self.hybrid_searcher.get_search_capabilities()
        
        print(f"\n📋 Search System Status:")
        print(f"   Semantic Search: {'✅' if capabilities['semantic_available'] else '❌'}")
        print(f"   Keyword Search:  {'✅' if capabilities['keyword_available'] else '❌'}")
        print(f"   Mode: {capabilities['mode'].replace('_', ' ').title()}")
        
        if capabilities['hybrid_mode']:
            print("🎯 Hybrid search enabled - optimal for both short and long queries")
        elif capabilities['mode'] == 'keyword_only':
            print("🔤 Keyword-only search - still effective for exact matches like 'COMI'")
        else:
            print("⚠️  Limited search capabilities")
    
    def search(self, query, top_k=10):
        """Perform search with the given query"""
        if not self.hybrid_searcher:
            print("❌ Search system not initialized. Run setup() first.")
            return
        
        self.hybrid_searcher.search(query, top_k)
    
    def run_interactive(self):
        """Run interactive search loop"""
        if not self.hybrid_searcher:
            print("❌ Search system not initialized. Run setup() first.")
            return
        
        print("\n" + "="*60)
        print("🔍 Interactive Search Ready!")
        print("Commands:")
        print("  - Enter any search query")
        print("  - Type 'quit' or 'exit' to stop")
        print("="*60)
        
        while True:
            try:
                user_query = input("\n💬 Search: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_query:
                    self.search(user_query)
                else:
                    print("Please enter a search query.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error during search: {e}")


def main():
    """Main entry point"""
    app = SearchApplication("data.csv")
    
    if app.setup():
        app.run_interactive()
    else:
        print("Failed to initialize search system.")


if __name__ == "__main__":
    main()