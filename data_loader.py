"""
Data Loader Module
Responsible for: Loading and validating CSV data
"""
import pandas as pd


class DataLoader:
    """Handles CSV data loading and basic validation"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
    
    def load_data(self):
        """Load CSV data with error handling"""
        try:
            self.df = pd.read_csv(self.csv_path, dtype=str)
            self.df.fillna("", inplace=True)
            print(f"📊 Loaded {len(self.df):,} records from {self.csv_path}")
            return self.df
        except Exception as e:
            raise Exception(f"Failed to load CSV: {e}")
    
    def validate_columns(self, required_columns):
        """Validate that required columns exist"""
        if self.df is None:
            raise Exception("Data not loaded. Call load_data() first.")
        
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}. Available: {list(self.df.columns)}")
    
    def get_data(self):
        """Get the loaded dataframe"""
        if self.df is None:
            raise Exception("Data not loaded. Call load_data() first.")
        return self.df