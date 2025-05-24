# ===========================================================================
# RETRIEVAL ENGINE - core/retrieval_engine.py
# ===========================================================================

import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
import pickle
import streamlit as st
from pathlib import Path

class RetrievalEngine:
    """
    Manages vector similarity search and retrieval
    Uses FAISS for efficient similarity search across visual embeddings
    """
    
    def __init__(self):
        """
        Initialize FAISS index and retrieval configuration
        """
        from config.settings import AppConfig
        self.config = AppConfig()
        
        self.index = None
        self.embeddings_metadata = []
        self.is_trained = False
        
        self._initialize_faiss_index()
    
    # ---------------------------------------------------------------------------
    # FAISS Index Management
    # ---------------------------------------------------------------------------
    
    def _initialize_faiss_index(self):
        """
        Initialize FAISS index for vector similarity search
        """
        try:
            # Create FAISS index for cosine similarity
            self.index = faiss.IndexFlatIP(self.config.VECTOR_DIMENSION)
            
            # Try to load existing index
            self._load_existing_index()
            
        except Exception as e:
            st.warning(f"Failed to initialize FAISS index: {str(e)}")
    
    def _load_existing_index(self):
        """
        Load previously saved FAISS index and metadata
        """
        index_path = self.config.VECTOR_STORE_DIR / "faiss_index.bin"
        metadata_path = self.config.VECTOR_STORE_DIR / "metadata.pkl"
        
        if index_path.exists() and metadata_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                
                with open(metadata_path, 'rb') as f:
                    self.embeddings_metadata = pickle.load(f)
                
                self.is_trained = True
                st.success(f"Loaded existing index with {len(self.embeddings_metadata)} documents")
                
            except Exception as e:
                st.warning(f"Failed to load existing index: {str(e)}")
    
    def _save_index(self):
        """
        Save FAISS index and metadata to disk
        """
        try:
            index_path = self.config.VECTOR_STORE_DIR / "faiss_index.bin"
            metadata_path = self.config.VECTOR_STORE_DIR / "metadata.pkl"
            
            faiss.write_index(self.index, str(index_path))
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.embeddings_metadata, f)
                
        except Exception as e:
            st.error(f"Failed to save index: {str(e)}")
    
    # ---------------------------------------------------------------------------
    # Document Indexing
    # ---------------------------------------------------------------------------
    
    def add_documents(self, embeddings_data: List[Dict[str, Any]]):
        """
        Add new documents to the vector index
        
        Args:
            embeddings_data: List of embedding dictionaries from embedding engine
        """
        if not embeddings_data:
            st.warning("No embeddings data to add")
            return
        
        try:
            # Extract embeddings and normalize for cosine similarity
            embeddings = np.array([item["embedding"] for item in embeddings_data])
            embeddings = self._normalize_embeddings(embeddings)
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Store metadata
            self.embeddings_metadata.extend(embeddings_data)
            
            self.is_trained = True
            
            # Save updated index
            self._save_index()
            
            st.success(f"Added {len(embeddings_data)} documents to index")
            
        except Exception as e:
            st.error(f"Error adding documents to index: {str(e)}")
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings for cosine similarity search
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
    
    # ---------------------------------------------------------------------------
    # Similarity Search
    # ---------------------------------------------------------------------------
    
    def search_similar_documents(self, query_embedding: np.ndarray, top_k: int = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for most similar documents to query
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of tuples containing (document_data, similarity_score)
        """
        if not self.is_trained:
            st.warning("No documents indexed yet")
            return []
        
        if top_k is None:
            top_k = self.config.TOP_K_RESULTS
        
        try:
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1)
            query_embedding = self._normalize_embeddings(query_embedding)
            
            # Search FAISS index
            similarities, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.embeddings_metadata) and similarity > self.config.SIMILARITY_THRESHOLD:
                    document_data = self.embeddings_metadata[idx]
                    results.append((document_data, float(similarity)))
            
            return results
            
        except Exception as e:
            st.error(f"Error during similarity search: {str(e)}")
            return []
    
    # ---------------------------------------------------------------------------
    # Index Management
    # ---------------------------------------------------------------------------
    
    def clear_index(self):
        """
        Clear all documents from the index
        """
        self._initialize_faiss_index()
        self.embeddings_metadata = []
        self.is_trained = False
        
        # Remove saved files
        index_path = self.config.VECTOR_STORE_DIR / "faiss_index.bin"
        metadata_path = self.config.VECTOR_STORE_DIR / "metadata.pkl"
        
        if index_path.exists():
            index_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
        
        st.success("Index cleared successfully")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index
        """
        return {
            "total_documents": len(self.embeddings_metadata),
            "is_trained": self.is_trained,
            "vector_dimension": self.config.VECTOR_DIMENSION,
            "similarity_threshold": self.config.SIMILARITY_THRESHOLD
        }

# This is the basic structure. The remaining files (generation_engine.py, 
# ui_helpers.py, image_helpers.py, vector_helpers.py) follow the same 
# pattern with detailed comments and modular design.