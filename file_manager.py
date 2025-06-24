"""
File Manager Module
Responsible for: Saving and loading cached files (embeddings, indices, etc.)
"""
import os
import numpy as np
import faiss
from scipy import sparse


class FileManager:
    """Handles file operations for caching embeddings and indices"""
    
    def __init__(self, base_path=""):
        self.base_path = base_path
    
    def _get_path(self, filename):
        """Get full path for a file"""
        return os.path.join(self.base_path, filename)
    
    def file_exists(self, filename):
        """Check if file exists"""
        return os.path.exists(self._get_path(filename))
    
    def save_numpy_array(self, filename, array):
        """Save numpy array"""
        np.save(self._get_path(filename), array)
    
    def load_numpy_array(self, filename):
        """Load numpy array"""
        return np.load(self._get_path(filename), allow_pickle=True)
    
    def save_sparse_matrix(self, filename, matrix):
        """Save sparse matrix"""
        sparse.save_npz(self._get_path(filename), matrix)
    
    def load_sparse_matrix(self, filename):
        """Load sparse matrix"""
        return sparse.load_npz(self._get_path(filename))
    
    def save_faiss_index(self, filename, index):
        """Save FAISS index"""
        faiss.write_index(index, self._get_path(filename))
    
    def load_faiss_index(self, filename):
        """Load FAISS index"""
        return faiss.read_index(self._get_path(filename))
    
    def save_dictionary(self, filename, dictionary):
        """Save dictionary as numpy file"""
        np.save(self._get_path(filename), dictionary)
    
    def load_dictionary(self, filename):
        """Load dictionary from numpy file"""
        return np.load(self._get_path(filename), allow_pickle=True).item()