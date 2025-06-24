"""
Text Processor Module
Responsible for: Cleaning and preprocessing text data
"""
from bs4 import BeautifulSoup


class TextProcessor:
    """Handles text cleaning and preprocessing"""
    
    @staticmethod
    def clean_html(text):
        """Remove HTML tags and clean text"""
        if not text:
            return ""
        return BeautifulSoup(text, "html.parser").get_text(separator=" ").strip()
    
    @staticmethod
    def normalize_whitespace(text):
        """Normalize whitespace while preserving structure"""
        return ' '.join(str(text).split())
    
    def process_document(self, title, description):
        """Process a single document (title + description)"""
        clean_title = self.clean_html(title)
        clean_description = self.clean_html(description)
        
        # Weight title more heavily for better short query matching
        combined_text = f"{clean_title} {clean_title} {clean_description}"
        return self.normalize_whitespace(combined_text)
    
    def process_documents(self, df, title_col="TITLE", desc_col="DESCRIPTION"):
        """Process all documents in the dataframe"""
        print("📝 Processing text data...")
        processed_texts = []
        
        for _, row in df.iterrows():
            processed_text = self.process_document(row[title_col], row[desc_col])
            processed_texts.append(processed_text)
        
        print(f"✅ Processed {len(processed_texts):,} documents")
        return processed_texts