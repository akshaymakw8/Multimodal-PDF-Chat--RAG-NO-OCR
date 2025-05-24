# ===========================================================================
# PDF PROCESSOR - core/pdf_processor.py
# ===========================================================================

import fitz  # PyMuPDF
from PIL import Image
import io
from typing import List, Tuple, Dict, Any
import streamlit as st
from pathlib import Path

class PDFProcessor:
    """
    Handles PDF to image conversion without OCR
    Focuses on preserving visual information for multimodal analysis
    """
    
    def __init__(self):
        """
        Initialize PDF processor with configuration
        """
        from config.settings import AppConfig
        self.config = AppConfig()
    
    # ---------------------------------------------------------------------------
    # PDF to Image Conversion
    # ---------------------------------------------------------------------------
    
    def convert_pdf_to_images(self, pdf_file) -> List[Dict[str, Any]]:
        """
        Convert PDF pages to high-quality images for visual analysis
        
        Args:
            pdf_file: Uploaded PDF file object
            
        Returns:
            List of dictionaries containing image data and metadata
        """
        images = []
        
        try:
            # Open PDF document
            pdf_document = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")
            
            for page_num in range(len(pdf_document)):
                page_data = self._process_pdf_page(pdf_document, page_num)
                images.append(page_data)
                
            pdf_document.close()
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return []
        
        return images
    
    def _process_pdf_page(self, pdf_document, page_num: int) -> Dict[str, Any]:
        """
        Process individual PDF page to extract visual information
        
        Args:
            pdf_document: PyMuPDF document object
            page_num: Page number to process
            
        Returns:
            Dictionary containing page image and metadata
        """
        page = pdf_document[page_num]
        
        # Convert page to high-resolution image
        mat = fitz.Matrix(self.config.PDF_DPI / 72, self.config.PDF_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        pil_image = Image.open(io.BytesIO(img_data))
        
        # Resize if necessary
        pil_image = self._resize_image(pil_image)
        
        # Extract page metadata
        metadata = self._extract_page_metadata(page, page_num)
        
        return {
            "image": pil_image,
            "page_number": page_num + 1,
            "metadata": metadata,
            "image_bytes": img_data
        }
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image while maintaining aspect ratio
        """
        if image.size[0] > self.config.MAX_IMAGE_SIZE[0] or image.size[1] > self.config.MAX_IMAGE_SIZE[1]:
            image.thumbnail(self.config.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
        return image
    
    def _extract_page_metadata(self, page, page_num: int) -> Dict[str, Any]:
        """
        Extract metadata from PDF page for enhanced context
        """
        return {
            "page_number": page_num + 1,
            "page_size": page.rect,
            "rotation": page.rotation,
            "has_images": len(page.get_images()) > 0,
            "has_text": bool(page.get_text().strip())
        }
    
    # ---------------------------------------------------------------------------
    # Image Processing for Individual Uploads
    # ---------------------------------------------------------------------------
    
    def process_uploaded_image(self, image_file) -> Dict[str, Any]:
        """
        Process individually uploaded images
        
        Args:
            image_file: Uploaded image file
            
        Returns:
            Dictionary containing processed image data
        """
        try:
            pil_image = Image.open(image_file)
            pil_image = self._resize_image(pil_image)
            
            # Convert to bytes for storage
            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
            
            return {
                "image": pil_image,
                "page_number": 1,
                "metadata": {
                    "filename": image_file.name,
                    "format": pil_image.format,
                    "size": pil_image.size,
                    "mode": pil_image.mode
                },
                "image_bytes": img_bytes
            }
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None