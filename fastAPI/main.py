import os
import tempfile
import shutil
import asyncio
from pathlib import Path
from typing import Optional
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import easyocr
import pytesseract
import cv2
import requests
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response validation
class OCRResponse(BaseModel):
    engine: Optional[str]
    success: bool
    text: str
    confidence: Optional[float] = None
    processing_time: Optional[float] = None

class URLRequest(BaseModel):
    image_url: HttpUrl

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# App configuration
app = FastAPI(
    title="OCR API",
    description="Production-ready OCR API with WebP support using EasyOCR and Tesseract",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize EasyOCR once (expensive operation)
try:
    easy = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have CUDA
    logger.info("EasyOCR initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize EasyOCR: {e}")
    easy = None

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_CONTENT_TYPES = [
    "image/webp", "image/jpeg", "image/png", "image/gif", "image/bmp"
]
REQUEST_TIMEOUT = 30  # seconds

class OCRProcessor:
    @staticmethod
    def validate_image_file(file: UploadFile) -> None:
        """Validate uploaded image file"""
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type. Allowed types: {', '.join(ALLOWED_CONTENT_TYPES)}"
            )
        
        # Check file size (FastAPI doesn't provide this directly, so we estimate)
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )

    @staticmethod
    def preprocess_image(img_path: str) -> np.ndarray:
        """Preprocess image for better OCR results"""
        try:
            # Use PIL to handle various formats including WebP
            pil_img = Image.open(img_path)
            
            # Convert to RGB if needed
            if pil_img.mode in ('RGBA', 'P'):
                pil_img = pil_img.convert('RGB')
            
            # Convert PIL image to OpenCV format
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold for better text recognition
            _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return threshold
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to process image"
            )

    @staticmethod
    async def extract_text(img_path: str) -> OCRResponse:
        """Extract text using EasyOCR and Tesseract fallback"""
        import time
        start_time = time.time()
        
        # Try EasyOCR first
        if easy:
            try:
                result = easy.readtext(img_path, detail=True)
                if result:
                    text_parts = []
                    total_confidence = 0
                    
                    for (bbox, text, confidence) in result:
                        if confidence > 0.3:  # Filter low-confidence results
                            text_parts.append(text)
                            total_confidence += confidence
                    
                    if text_parts:
                        avg_confidence = total_confidence / len(result)
                        processing_time = time.time() - start_time
                        
                        return OCRResponse(
                            engine="EasyOCR",
                            success=True,
                            text=" ".join(text_parts),
                            confidence=round(avg_confidence, 3),
                            processing_time=round(processing_time, 3)
                        )
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}")

        # Fallback to Tesseract
        try:
            pil_img = Image.open(img_path)
            text = pytesseract.image_to_string(pil_img, config='--psm 6')
            
            if text.strip():
                processing_time = time.time() - start_time
                return OCRResponse(
                    engine="Tesseract",
                    success=True,
                    text=text.strip(),
                    processing_time=round(processing_time, 3)
                )
        except Exception as e:
            logger.error(f"Tesseract failed: {e}")

        # Both engines failed
        processing_time = time.time() - start_time
        return OCRResponse(
            engine=None,
            success=False,
            text="",
            processing_time=round(processing_time, 3)
        )

# Initialize processor
ocr_processor = OCRProcessor()

# API Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "easyocr_available": easy is not None,
        "tesseract_available": True  # Assuming it's installed
    }

@app.post("/ocr/file", response_model=OCRResponse)
async def ocr_from_file(file: UploadFile = File(...)):
    """Extract text from uploaded image file"""
    
    # Validate file
    ocr_processor.validate_image_file(file)
    
    # Create temporary file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        logger.info(f"Processing file: {file.filename} ({file.content_type})")
        
        # Process image
        result = await ocr_processor.extract_text(tmp_path)
        
        logger.info(f"OCR completed: {result.engine}, success: {result.success}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during file processing"
        )
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.debug(f"Cleaned up temporary file: {tmp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {tmp_path}: {e}")

@app.post("/ocr/url", response_model=OCRResponse)
async def ocr_from_url(request: URLRequest):
    """Extract text from image URL"""
    
    tmp_path = None
    try:
        # Download image with timeout
        logger.info(f"Downloading image from: {request.image_url}")
        
        response = requests.get(
            str(request.image_url),
            stream=True,
            timeout=REQUEST_TIMEOUT,
            headers={'User-Agent': 'OCR-API/1.0'}
        )
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if not any(allowed in content_type for allowed in ALLOWED_CONTENT_TYPES):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported content type: {content_type}"
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        
        # Process image
        result = await ocr_processor.extract_text(tmp_path)
        
        logger.info(f"OCR completed: {result.engine}, success: {result.success}")
        return result
        
    except requests.RequestException as e:
        logger.error(f"Failed to download image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to download image from URL"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during URL processing"
        )
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.debug(f"Cleaned up temporary file: {tmp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {tmp_path}: {e}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "OCR API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "ocr_file": "/ocr/file",
            "ocr_url": "/ocr/url",
            "docs": "/docs"
        }
    }

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# Run server
if __name__ == "__main__":
    import uvicorn
    
    # Production configuration
    uvicorn.run(
        "main:app",  # Change this to your actual module name
        host="0.0.0.0",
        port=8000,
        workers=1,  # Increase for production, but be mindful of memory usage with OCR models
        log_level="info",
        access_log=True
    )