# ===========================================================================
# IMAGE HELPERS - utils/image_helpers.py
# ===========================================================================

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
from typing import Tuple, List, Dict, Any, Optional
import streamlit as st

class ImageProcessor:
    """
    Advanced image processing utilities for optimal visual analysis
    Enhances images for better AI understanding without OCR
    """
    
    def __init__(self):
        """
        Initialize image processor with default settings
        """
        self.max_dimension = 1024
        self.quality_threshold = 0.7
        self.supported_formats = ['PNG', 'JPEG', 'JPG', 'WEBP']
    
    # ---------------------------------------------------------------------------
    # Image Enhancement for AI Analysis
    # ---------------------------------------------------------------------------
    
    def enhance_for_analysis(self, image: Image.Image) -> Image.Image:
        """
        Enhance image quality for better AI visual understanding
        
        Args:
            image: PIL Image object
            
        Returns:
            Enhanced PIL Image optimized for AI analysis
        """
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply enhancement pipeline
            enhanced = self._apply_enhancement_pipeline(image)
            
            # Optimize dimensions
            enhanced = self._optimize_dimensions(enhanced)
            
            return enhanced
            
        except Exception as e:
            st.warning(f"Image enhancement failed: {str(e)}")
            return image
    
    def _apply_enhancement_pipeline(self, image: Image.Image) -> Image.Image:
        """
        Apply series of enhancements to improve visual clarity
        """
        # 1. Brightness and contrast optimization
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)  # Slight contrast boost
        
        # 2. Sharpness enhancement for text and diagrams
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)  # Subtle sharpening
        
        # 3. Color saturation for better chart readability
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.05)  # Minimal color boost
        
        return image
    
    def _optimize_dimensions(self, image: Image.Image) -> Image.Image:
        """
        Optimize image dimensions while preserving aspect ratio
        """
        width, height = image.size
        
        # Calculate optimal dimensions
        if max(width, height) > self.max_dimension:
            if width > height:
                new_width = self.max_dimension
                new_height = int((height * self.max_dimension) / width)
            else:
                new_height = self.max_dimension
                new_width = int((width * self.max_dimension) / height)
            
            # Use high-quality resampling
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    # ---------------------------------------------------------------------------
    # Image Quality Assessment
    # ---------------------------------------------------------------------------
    
    def assess_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        """
        Assess image quality for visual analysis suitability
        
        Args:
            image: PIL Image to assess
            
        Returns:
            Dictionary with quality metrics and recommendations
        """
        quality_metrics = {
            "resolution_score": 0.0,
            "clarity_score": 0.0,
            "overall_quality": "poor",
            "recommendations": []
        }
        
        try:
            # Resolution assessment
            width, height = image.size
            pixel_count = width * height
            
            if pixel_count > 500000:  # > 0.5MP
                quality_metrics["resolution_score"] = 1.0
            elif pixel_count > 200000:  # > 0.2MP
                quality_metrics["resolution_score"] = 0.7
            else:
                quality_metrics["resolution_score"] = 0.3
                quality_metrics["recommendations"].append("Consider higher resolution image")
            
            # Clarity assessment using edge detection
            clarity_score = self._assess_image_clarity(image)
            quality_metrics["clarity_score"] = clarity_score
            
            if clarity_score < 0.3:
                quality_metrics["recommendations"].append("Image may be blurry - consider enhancement")
            
            # Overall quality determination
            overall = (quality_metrics["resolution_score"] + clarity_score) / 2
            
            if overall > 0.8:
                quality_metrics["overall_quality"] = "excellent"
            elif overall > 0.6:
                quality_metrics["overall_quality"] = "good"
            elif overall > 0.4:
                quality_metrics["overall_quality"] = "fair"
            else:
                quality_metrics["overall_quality"] = "poor"
                quality_metrics["recommendations"].append("Consider using a different image")
            
        except Exception as e:
            st.warning(f"Quality assessment failed: {str(e)}")
        
        return quality_metrics
    
    def _assess_image_clarity(self, image: Image.Image) -> float:
        """
        Assess image clarity using edge detection
        """
        try:
            # Convert to grayscale for edge detection
            gray = image.convert('L')
            
            # Apply edge detection filter
            edges = gray.filter(ImageFilter.FIND_EDGES)
            
            # Calculate edge intensity as clarity metric
            edge_array = np.array(edges)
            edge_intensity = np.std(edge_array) / 255.0
            
            # Normalize to 0-1 range
            return min(edge_intensity * 2, 1.0)
            
        except Exception:
            return 0.5  # Default neutral score
    
    # ---------------------------------------------------------------------------
    # Image Format Conversion and Optimization
    # ---------------------------------------------------------------------------
    
    def convert_to_optimal_format(self, image: Image.Image, target_format: str = "PNG") -> bytes:
        """
        Convert image to optimal format for AI processing
        
        Args:
            image: PIL Image object
            target_format: Target format (PNG, JPEG, etc.)
            
        Returns:
            Optimized image as bytes
        """
        buffer = io.BytesIO()
        
        try:
            if target_format.upper() == "JPEG":
                # Optimize JPEG with high quality
                image.save(buffer, format="JPEG", quality=95, optimize=True)
            else:
                # Use PNG for lossless quality
                image.save(buffer, format="PNG", optimize=True)
            
            return buffer.getvalue()
            
        except Exception as e:
            st.error(f"Format conversion failed: {str(e)}")
            return b""
    
    def create_thumbnail(self, image: Image.Image, size: Tuple[int, int] = (200, 200)) -> Image.Image:
        """
        Create thumbnail for UI display
        """
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        return thumbnail
    
    # ---------------------------------------------------------------------------
    # Specialized Processing for Document Types
    # ---------------------------------------------------------------------------
    
    def enhance_financial_charts(self, image: Image.Image) -> Image.Image:
        """
        Specialized enhancement for financial charts and graphs
        """
        try:
            # Increase contrast for better line/bar visibility
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.3)
            
            # Sharpen for crisp text and lines
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)
            
            # Boost color saturation for better chart element distinction
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.1)
            
            return image
            
        except Exception as e:
            st.warning(f"Financial chart enhancement failed: {str(e)}")
            return image
    
    def enhance_technical_diagrams(self, image: Image.Image) -> Image.Image:
        """
        Specialized enhancement for technical diagrams and schematics
        """
        try:
            # High contrast for clear component boundaries
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.4)
            
            # Maximum sharpness for technical details
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.3)
            
            # Reduce color saturation for focus on structure
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.9)
            
            return image
            
        except Exception as e:
            st.warning(f"Technical diagram enhancement failed: {str(e)}")
            return image
    
    # ---------------------------------------------------------------------------
    # Batch Processing Utilities
    # ---------------------------------------------------------------------------
    
    def process_image_batch(self, images: List[Image.Image], enhancement_type: str = "general") -> List[Image.Image]:
        """
        Process multiple images with specified enhancement type
        
        Args:
            images: List of PIL Images
            enhancement_type: Type of enhancement ("general", "financial", "technical")
            
        Returns:
            List of processed images
        """
        processed_images = []
        
        for i, image in enumerate(images):
            try:
                if enhancement_type == "financial":
                    processed = self.enhance_financial_charts(image)
                elif enhancement_type == "technical":
                    processed = self.enhance_technical_diagrams(image)
                else:
                    processed = self.enhance_for_analysis(image)
                
                processed_images.append(processed)
                
            except Exception as e:
                st.warning(f"Failed to process image {i+1}: {str(e)}")
                processed_images.append(image)  # Use original on failure
        
        return processed_images
    
    # ---------------------------------------------------------------------------
    # Image Analysis Utilities
    # ---------------------------------------------------------------------------
    
    def detect_image_content_type(self, image: Image.Image) -> Dict[str, Any]:
        """
        Detect the type of content in the image (chart, diagram, text, etc.)
        """
        content_analysis = {
            "primary_type": "unknown",
            "confidence": 0.0,
            "characteristics": [],
            "suggestions": []
        }
        
        try:
            # Convert to numpy array for analysis
            img_array = np.array(image.convert('RGB'))
            
            # Color distribution analysis
            color_diversity = self._analyze_color_distribution(img_array)
            
            # Edge density analysis
            edge_density = self._analyze_edge_density(image)
            
            # Determine content type based on characteristics
            if color_diversity > 0.7 and edge_density > 0.6:
                content_analysis["primary_type"] = "chart_or_graph"
                content_analysis["confidence"] = 0.8
                content_analysis["characteristics"].append("High color diversity")
                content_analysis["characteristics"].append("Complex edge patterns")
                content_analysis["suggestions"].append("Use financial chart enhancement")
                
            elif edge_density > 0.8:
                content_analysis["primary_type"] = "technical_diagram"
                content_analysis["confidence"] = 0.75
                content_analysis["characteristics"].append("High edge density")
                content_analysis["suggestions"].append("Use technical diagram enhancement")
                
            elif color_diversity < 0.3:
                content_analysis["primary_type"] = "text_document"
                content_analysis["confidence"] = 0.7
                content_analysis["characteristics"].append("Low color diversity")
                content_analysis["suggestions"].append("Use text-optimized processing")
            
            else:
                content_analysis["primary_type"] = "mixed_content"
                content_analysis["confidence"] = 0.6
                content_analysis["suggestions"].append("Use general enhancement")
            
        except Exception as e:
            st.warning(f"Content type detection failed: {str(e)}")
        
        return content_analysis
    
    def _analyze_color_distribution(self, img_array: np.ndarray) -> float:
        """
        Analyze color distribution to detect chart-like content
        """
        try:
            # Calculate color histogram
            colors = img_array.reshape(-1, 3)
            unique_colors = len(np.unique(colors.view(np.dtype((np.void, colors.dtype.itemsize * colors.shape[1])))))
            total_pixels = colors.shape[0]
            
            # Normalize color diversity
            color_diversity = min(unique_colors / (total_pixels * 0.1), 1.0)
            return color_diversity
            
        except Exception:
            return 0.5
    
    def _analyze_edge_density(self, image: Image.Image) -> float:
        """
        Analyze edge density to detect structured content
        """
        try:
            # Convert to grayscale and apply edge detection
            gray = image.convert('L')
            edges = gray.filter(ImageFilter.FIND_EDGES)
            
            # Calculate edge pixel ratio
            edge_array = np.array(edges)
            edge_pixels = np.sum(edge_array > 50)  # Threshold for edge detection
            total_pixels = edge_array.size
            
            edge_density = edge_pixels / total_pixels
            return min(edge_density * 10, 1.0)  # Normalize and amplify
            
        except Exception:
            return 0.5