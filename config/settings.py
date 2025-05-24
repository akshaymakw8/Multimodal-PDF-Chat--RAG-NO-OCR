import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

@dataclass
class AppConfig:
    """
    Application configuration and environment settings
    """
    # ---------------------------------------------------------------------------
    # API Configuration
    # ---------------------------------------------------------------------------
    COHERE_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    
    # ---------------------------------------------------------------------------
    # Model Configuration
    # ---------------------------------------------------------------------------
    COHERE_MODEL: str = "embed-english-v3.0"  # Will use embed-4 when available
    GEMINI_MODEL: str = "gemini-2.0-flash-exp"  # Latest Gemini model
    
    # ---------------------------------------------------------------------------
    # Processing Configuration
    # ---------------------------------------------------------------------------
    PDF_DPI: int = 200  # Image resolution for PDF conversion
    MAX_IMAGE_SIZE: tuple = (1024, 1024)  # Maximum image dimensions
    CHUNK_SIZE: int = 1000  # Text chunking size
    
    # ---------------------------------------------------------------------------
    # Vector Store Configuration
    # ---------------------------------------------------------------------------
    VECTOR_DIMENSION: int = 1024  # Cohere embedding dimension
    SIMILARITY_THRESHOLD: float = 0.7  # Minimum similarity for retrieval
    TOP_K_RESULTS: int = 5  # Number of results to retrieve
    
    # ---------------------------------------------------------------------------
    # File Paths
    # ---------------------------------------------------------------------------
    DATA_DIR: Path = field(default_factory=lambda: Path("data"))
    VECTOR_STORE_DIR: Path = field(default_factory=lambda: Path("data") / "vector_store")
    SAMPLE_PDFS_DIR: Path = field(default_factory=lambda: Path("data") / "sample_pdfs")
    
    # ---------------------------------------------------------------------------
    # UI Configuration - FIXED: Using default_factory for mutable defaults
    # ---------------------------------------------------------------------------
    MAX_FILE_SIZE_MB: int = 50  # Maximum upload file size
    SUPPORTED_FORMATS: List[str] = field(default_factory=lambda: [".pdf", ".png", ".jpg", ".jpeg"])
    
    def __post_init__(self):
        """
        Validate configuration and create necessary directories
        """
        # Load environment variables if not set
        if self.COHERE_API_KEY is None:
            self.COHERE_API_KEY = os.getenv("COHERE_API_KEY")
        if self.GOOGLE_API_KEY is None:
            self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        
        # Validate API keys (optional for testing)
        # Uncomment these lines if you want strict validation
        # self.validate_api_keys()
        
        # Create directories
        self.create_directories()
    
    def validate_api_keys(self):
        """
        Validate that required API keys are present
        """
        missing_keys = []
        
        if not self.COHERE_API_KEY or self.COHERE_API_KEY == "your_cohere_api_key_here":
            missing_keys.append("COHERE_API_KEY")
            
        if not self.GOOGLE_API_KEY or self.GOOGLE_API_KEY == "your_google_api_key_here":
            missing_keys.append("GOOGLE_API_KEY")
        
        if missing_keys:
            print(f"⚠️ Warning: Missing API keys: {', '.join(missing_keys)}")
            print("💡 The app will run in demo mode. Set API keys in .env for full functionality.")
    
    def create_directories(self):
        """
        Create necessary data directories
        """
        try:
            self.DATA_DIR.mkdir(exist_ok=True)
            self.VECTOR_STORE_DIR.mkdir(exist_ok=True)
            self.SAMPLE_PDFS_DIR.mkdir(exist_ok=True)
        except Exception as e:
            print(f"⚠️ Warning: Could not create directories: {e}")
    
    def is_api_configured(self) -> bool:
        """
        Check if API keys are properly configured
        """
        return (
            self.COHERE_API_KEY and 
            self.COHERE_API_KEY != "your_cohere_api_key_here" and
            self.GOOGLE_API_KEY and 
            self.GOOGLE_API_KEY != "your_google_api_key_here"
        )
    
    def get_status(self) -> dict:
        """
        Get configuration status for debugging
        """
        return {
            "api_configured": self.is_api_configured(),
            "cohere_key_set": bool(self.COHERE_API_KEY and self.COHERE_API_KEY != "your_cohere_api_key_here"),
            "google_key_set": bool(self.GOOGLE_API_KEY and self.GOOGLE_API_KEY != "your_google_api_key_here"),
            "data_dir_exists": self.DATA_DIR.exists(),
            "vector_store_exists": self.VECTOR_STORE_DIR.exists(),
        }