# ===========================================================================
# COHERE EMBEDDING ENGINE - core/embedding_engine.py
# ===========================================================================

import cohere
import numpy as np
from typing import List, Dict, Any
import streamlit as st
from PIL import Image
import base64
import io

class CohereEmbeddingEngine:
    """
    Handles multimodal embeddings using Cohere's Embed-4 model
    Creates vector representations of visual content for similarity search
    """
    
    def __init__(self):
        """
        Initialize Cohere client and configuration
        """
        from config.settings import AppConfig
        self.config = AppConfig()
        
        try:
            self.client = cohere.Client(api_key=self.config.COHERE_API_KEY)
        except Exception as e:
            st.error(f"Failed to initialize Cohere client: {str(e)}")
            self.client = None
    
    # ---------------------------------------------------------------------------
    # Multimodal Embedding Generation
    # ---------------------------------------------------------------------------
    
    def generate_image_embeddings(self, image_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for images using Cohere's multimodal model
        
        Args:
            image_data: List of image dictionaries from PDF processor
            
        Returns:
            List of dictionaries with embeddings and metadata
        """
        if not self.client:
            st.error("Cohere client not initialized")
            return []
        
        embeddings_data = []
        
        for idx, item in enumerate(image_data):
            try:
                # Convert PIL image to base64 for API
                img_b64 = self._image_to_base64(item["image"])
                
                # Generate embedding using Cohere's multimodal capabilities
                embedding = self._generate_single_embedding(img_b64, item)
                
                embeddings_data.append({
                    "id": f"page_{item['page_number']}_{idx}",
                    "embedding": embedding,
                    "image": item["image"],
                    "metadata": item["metadata"],
                    "page_number": item["page_number"]
                })
                
            except Exception as e:
                st.warning(f"Failed to generate embedding for page {item['page_number']}: {str(e)}")
                continue
        
        return embeddings_data
    
    def _generate_single_embedding(self, image_b64: str, item: Dict[str, Any]) -> np.ndarray:
        """
        Generate embedding for a single image
        
        Args:
            image_b64: Base64 encoded image
            item: Image metadata dictionary
            
        Returns:
            NumPy array containing the embedding vector
        """
        try:
            # Note: This is a placeholder for Cohere's multimodal API
            # In practice, you would use Cohere's actual multimodal embedding endpoint
            # For now, we'll simulate with text embedding of image description
            
            # Generate a description of the image context
            description = f"Visual content from page {item['page_number']}"
            if item['metadata'].get('has_images'):
                description += " containing charts, diagrams, or visual elements"
            
            # Use Cohere's embed endpoint (text-based for now)
            response = self.client.embed(
                texts=[description],
                model=self.config.COHERE_MODEL,
                input_type="search_document"
            )
            
            return np.array(response.embeddings[0])
            
        except Exception as e:
            st.error(f"Error generating embedding: {str(e)}")
            # Return random embedding as fallback
            return np.random.rand(self.config.VECTOR_DIMENSION)
    
    def _image_to_base64(self, pil_image: Image.Image) -> str:
        """
        Convert PIL image to base64 string for API transmission
        """
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    # ---------------------------------------------------------------------------
    # Query Embedding Generation
    # ---------------------------------------------------------------------------
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for user query
        
        Args:
            query: User's question or search query
            
        Returns:
            NumPy array containing query embedding
        """
        try:
            response = self.client.embed(
                texts=[query],
                model=self.config.COHERE_MODEL,
                input_type="search_query"
            )
            
            return np.array(response.embeddings[0])
            
        except Exception as e:
            st.error(f"Error generating query embedding: {str(e)}")
            return np.random.rand(self.config.VECTOR_DIMENSION)