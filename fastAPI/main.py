from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import easyocr
import pytesseract
import cv2
import tempfile
import shutil
import requests

app = FastAPI()

# Initialize EasyOCR once
easy = easyocr.Reader(['en'])

# ===== Preprocessing =====
def preprocess(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return gray

# ===== Bulletproof OCR (EasyOCR + Tesseract) =====
def bulletproof_ocr(img_path):
    img = preprocess(img_path)

    # EasyOCR
    try:
        result = easy.readtext(img_path, detail=1)
        if result:
            text = " ".join([line[1] for line in result])
            return {"engine": "EasyOCR", "success": True, "text": text}
    except:
        pass

    # Tesseract fallback
    try:
        text = pytesseract.image_to_string(img)
        if text.strip():
            return {"engine": "Tesseract", "success": True, "text": text.strip()}
    except:
        pass

    return {"engine": None, "success": False, "text": ""}

# ===== API Endpoints =====
@app.post("/ocr/file")
async def ocr_from_file(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    result = bulletproof_ocr(tmp_path)
    return JSONResponse(content=result)

@app.post("/ocr/url")
async def ocr_from_url(image_url: str = Form(...)):
    response = requests.get(image_url, stream=True)
    if response.status_code != 200:
        return JSONResponse(content={"error": "Failed to fetch image"}, status_code=400)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        shutil.copyfileobj(response.raw, tmp_file)
        tmp_path = tmp_file.name

    result = bulletproof_ocr(tmp_path)
    return JSONResponse(content=result)

# ===== Run Server =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
