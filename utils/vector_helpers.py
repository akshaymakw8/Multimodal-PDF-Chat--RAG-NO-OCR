# ===========================================================================
# VECTOR HELPERS - utils/vector_helpers.py
# ===========================================================================

import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional
import pickle
import streamlit as st
from pathlib import Path

class VectorOperations:
    """
    Advanced vector operations and utilities for similarity search
    Optimizes vector storage, retrieval, and similarity calculations
    """
    
    def __init__(self):
        """
        Initialize vector operations with optimization settings
        """
        self.similarity_metrics = ['cosine', 'euclidean', 'dot_product']
        self.index_types = ['flat', 'ivf', 'hnsw']
    
    # ---------------------------------------------------------------------------
    # Vector Similarity Calculations
    # ---------------------------------------------------------------------------
    
    def calculate_cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vector1, vector2: NumPy arrays representing vectors
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(vector1, vector2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            st.warning(f"Cosine similarity calculation failed: {str(e)}")
            return 0.0
    
    def calculate_batch_similarities(
        self, 
        query_vector: np.ndarray, 
        vector_batch: np.ndarray,
        metric: str = 'cosine'
    ) -> np.ndarray:
        """
        Calculate similarities between query vector and batch of vectors
        
        Args:
            query_vector: Single query vector
            vector_batch: Batch of vectors (n_vectors, dimension)
            metric: Similarity metric to use
            
        Returns:
            Array of similarity scores
        """
        try:
            if metric == 'cosine':
                return self._batch_cosine_similarity(query_vector, vector_batch)
            elif metric == 'euclidean':
                return self._batch_euclidean_similarity(query_vector, vector_batch)
            else:
                return self._batch_dot_product_similarity(query_vector, vector_batch)
                
        except Exception as e:
            st.error(f"Batch similarity calculation failed: {str(e)}")
            return np.zeros(vector_batch.shape[0])
    
    def _batch_cosine_similarity(self, query: np.ndarray, batch: np.ndarray) -> np.ndarray:
        """
        Efficient batch cosine similarity calculation
        """
        # Normalize query
        query_norm = query / np.linalg.norm(query)
        
        # Normalize batch vectors
        batch_norms = batch / np.linalg.norm(batch, axis=1, keepdims=True)
        
        # Calculate similarities
        similarities = np.dot(batch_norms, query_norm)
        return similarities
    
    def _batch_euclidean_similarity(self, query: np.ndarray, batch: np.ndarray) -> np.ndarray:
        """
        Convert Euclidean distances to similarity scores
        """
        distances = np.linalg.norm(batch - query, axis=1)
        # Convert to similarity (higher is better)
        max_dist = np.max(distances)
        similarities = 1 - (distances / max_dist) if max_dist > 0 else np.ones_like(distances)
        return similarities
    
    def _batch_dot_product_similarity(self, query: np.ndarray, batch: np.ndarray) -> np.ndarray:
        """
        Calculate dot product similarities
        """
        return np.dot(batch, query)
    
    # ---------------------------------------------------------------------------
    # Advanced Index Operations
    # ---------------------------------------------------------------------------
    
    def create_optimized_index(
        self, 
        vectors: np.ndarray, 
        index_type: str = 'flat',
        metric: str = 'cosine'
    ) -> faiss.Index:
        """
        Create optimized FAISS index based on data characteristics
        
        Args:
            vectors: Vector data (n_vectors, dimension)
            index_type: Type of index to create
            metric: Distance metric for index
            
        Returns:
            Configured FAISS index
        """
        n_vectors, dimension = vectors.shape
        
        try:
            if index_type == 'flat':
                # Simple flat index for small datasets
                if metric == 'cosine':
                    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine
                    # Normalize vectors for cosine similarity
                    faiss.normalize_L2(vectors)
                else:
                    index = faiss.IndexFlatL2(dimension)  # L2 distance
                    
            elif index_type == 'ivf' and n_vectors > 1000:
                # IVF index for larger datasets
                quantizer = faiss.IndexFlatL2(dimension)
                n_clusters = min(int(np.sqrt(n_vectors)), 256)
                
                if metric == 'cosine':
                    index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters, faiss.METRIC_INNER_PRODUCT)
                    faiss.normalize_L2(vectors)
                else:
                    index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters, faiss.METRIC_L2)
                
                # Train the index
                index.train(vectors)
                
            else:
                # Fallback to flat index
                index = faiss.IndexFlatIP(dimension) if metric == 'cosine' else faiss.IndexFlatL2(dimension)
                if metric == 'cosine':
                    faiss.normalize_L2(vectors)
            
            # Add vectors to index
            index.add(vectors)
            
            return index
            
        except Exception as e:
            st.error(f"Index creation failed: {str(e)}")
            # Fallback to simple flat index
            index = faiss.IndexFlatL2(dimension)
            index.add(vectors)
            return index
    
    # ---------------------------------------------------------------------------
    # Vector Quality and Analysis
    # ---------------------------------------------------------------------------
    
    def analyze_vector_quality(self, vectors: np.ndarray) -> Dict[str, Any]:
        """
        Analyze quality and characteristics of vector embeddings
        
        Args:
            vectors: Array of embedding vectors
            
        Returns:
            Dictionary with quality metrics and insights
        """
        quality_analysis = {
            "vector_count": vectors.shape[0],
            "dimension": vectors.shape[1],
            "mean_magnitude": 0.0,
            "magnitude_std": 0.0,
            "sparsity": 0.0,
            "cluster_tendency": 0.0,
            "quality_score": 0.0,
            "recommendations": []
        }
        
        try:
            # Basic statistics
            magnitudes = np.linalg.norm(vectors, axis=1)
            quality_analysis["mean_magnitude"] = float(np.mean(magnitudes))
            quality_analysis["magnitude_std"] = float(np.std(magnitudes))
            
            # Sparsity analysis
            zero_elements = np.sum(vectors == 0)
            total_elements = vectors.size
            quality_analysis["sparsity"] = float(zero_elements / total_elements)
            
            # Clustering tendency (simplified)
            if vectors.shape[0] > 1:
                pairwise_distances = self._calculate_pairwise_distances(vectors[:min(100, vectors.shape[0])])
                quality_analysis["cluster_tendency"] = float(np.std(pairwise_distances))
            
            # Overall quality assessment
            quality_score = self._calculate_quality_score(quality_analysis)
            quality_analysis["quality_score"] = quality_score
            
            # Generate recommendations
            quality_analysis["recommendations"] = self._generate_quality_recommendations(quality_analysis)
            
        except Exception as e:
            st.warning(f"Vector quality analysis failed: {str(e)}")
        
        return quality_analysis
    
    def _calculate_pairwise_distances(self, vectors: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise distances for clustering analysis
        """
        n_vectors = vectors.shape[0]
        distances = []
        
        for i in range(min(n_vectors, 50)):  # Limit for efficiency
            for j in range(i + 1, min(n_vectors, 50)):
                dist = np.linalg.norm(vectors[i] - vectors[j])
                distances.append(dist)
        
        return np.array(distances)
    
    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """
        Calculate overall quality score based on various metrics
        """
        score = 0.0
        
        # Magnitude consistency (prefer consistent magnitudes)
        if analysis["magnitude_std"] > 0:
            magnitude_consistency = 1 / (1 + analysis["magnitude_std"])
            score += magnitude_consistency * 0.3
        
        # Sparsity score (prefer less sparse)
        sparsity_score = 1 - analysis["sparsity"]
        score += sparsity_score * 0.3
        
        # Clustering tendency (prefer some structure)
        if analysis["cluster_tendency"] > 0:
            cluster_score = min(analysis["cluster_tendency"], 1.0)
            score += cluster_score * 0.4
        
        return min(score, 1.0)
    
    def _generate_quality_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on quality analysis
        """
        recommendations = []
        
        if analysis["sparsity"] > 0.5:
            recommendations.append("High sparsity detected - consider dimension reduction")
        
        if analysis["magnitude_std"] > analysis["mean_magnitude"]:
            recommendations.append("Inconsistent vector magnitudes - consider normalization")
        
        if analysis["quality_score"] < 0.5:
            recommendations.append("Low overall quality - review embedding generation process")
        
        if analysis["vector_count"] < 10:
            recommendations.append("Small dataset - consider adding more documents for better search")
        
        return recommendations
    
    # ---------------------------------------------------------------------------
    # Vector Storage and Persistence
    # ---------------------------------------------------------------------------
    
    def save_vectors_with_metadata(
        self, 
        vectors: np.ndarray, 
        metadata: List[Dict[str, Any]], 
        filepath: Path
    ):
        """
        Save vectors along with metadata for persistence
        """
        try:
            save_data = {
                "vectors": vectors,
                "metadata": metadata,
                "vector_info": {
                    "shape": vectors.shape,
                    "dtype": str(vectors.dtype),
                    "created_at": "now"
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
                
        except Exception as e:
            st.error(f"Failed to save vectors: {str(e)}")
    
    def load_vectors_with_metadata(self, filepath: Path) -> Tuple[Optional[np.ndarray], Optional[List[Dict[str, Any]]]]:
        """
        Load previously saved vectors and metadata
        """
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            return save_data["vectors"], save_data["metadata"]
            
        except Exception as e:
            st.warning(f"Failed to load vectors: {str(e)}")
            return None, None
    
    # ---------------------------------------------------------------------------
    # Search Optimization
    # ---------------------------------------------------------------------------
    
    def optimize_search_parameters(
        self, 
        index: faiss.Index, 
        query_vectors: np.ndarray,
        target_recall: float = 0.9
    ) -> Dict[str, Any]:
        """
        Optimize search parameters for better recall/speed trade-off
        """
        optimization_results = {
            "optimal_nprobe": 1,
            "optimal_k": 10,
            "expected_recall": 0.0,
            "search_time_ms": 0.0
        }
        
        try:
            if hasattr(index, 'nprobe'):  # IVF-type index
                best_nprobe = 1
                best_recall = 0.0
                
                for nprobe in [1, 4, 8, 16, 32]:
                    index.nprobe = nprobe
                    
                    # Test search with sample queries
                    recall = self._estimate_recall(index, query_vectors[:min(10, len(query_vectors))])
                    
                    if recall >= target_recall and recall > best_recall:
                        best_nprobe = nprobe
                        best_recall = recall
                
                optimization_results["optimal_nprobe"] = best_nprobe
                optimization_results["expected_recall"] = best_recall
                
                # Set optimal parameters
                index.nprobe = best_nprobe
        
        except Exception as e:
            st.warning(f"Search optimization failed: {str(e)}")
        
        return optimization_results
    
    def _estimate_recall(self, index: faiss.Index, query_vectors: np.ndarray, k: int = 5) -> float:
        """
        Estimate recall for current index configuration
        """
        try:
            if len(query_vectors) == 0:
                return 0.0
            
            # Perform searches
            _, retrieved_ids = index.search(query_vectors, k)
            
            # Simple recall estimation (this is simplified)
            # In practice, you'd need ground truth for proper recall calculation
            valid_results = np.sum(retrieved_ids >= 0)
            total_possible = len(query_vectors) * k
            
            return valid_results / total_possible if total_possible > 0 else 0.0
            
        except Exception:
            return 0.0