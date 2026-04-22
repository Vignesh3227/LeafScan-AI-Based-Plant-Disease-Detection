import io
import base64
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from PIL import Image
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
from inference import predict, load_model
from recommendations import get_recommendation
from marketplace import get_products_for_disease, get_all_products

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup logic ---
    load_model()
    print("Model loaded and ready.")
    
    yield  # This tells FastAPI to run the app now
    
    # --- Shutdown logic (optional) ---
    # You can leave this blank or add cleanup code here later
app = FastAPI(
    title="Plant Disease Detection API",
    description="AI-powered plant disease detection using EfficientNet-B0",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ADD THIS NEW CODE



@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Plant Disease Detection API</h1><p>Visit /docs for API documentation.</p>")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/predict")
async def predict_disease(
    file: UploadFile = File(...),
    gradcam: bool = Query(default=False, description="Generate Grad-CAM visualization"),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Max 20MB.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process image file.")

    try:
        result = predict(image, use_gradcam=gradcam)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    recommendation = get_recommendation(result["raw_class"])
    products = get_products_for_disease(result["raw_class"])

    original_b64 = None
    buffered = io.BytesIO()
    resized = image.copy()
    resized.thumbnail((600, 600))
    resized.save(buffered, format="JPEG", quality=85)
    original_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"

    return {
        "prediction": {
            "plant": result["plant"],
            "disease": result["disease"],
            "is_healthy": result["is_healthy"],
            "confidence": result["confidence"],
            "raw_class": result["raw_class"],
            "top5": result["top5"],
            "demo_mode": result.get("demo_mode", False),
        },
        "recommendation": recommendation,
        "products": products,
        "images": {
            "original": original_b64,
            "gradcam": result.get("gradcam_image"),
        },
    }


@app.get("/api/products")
async def list_products(category: Optional[str] = None):
    products = get_all_products()
    if category:
        products = [p for p in products if p["category"] == category]
    return {"products": products, "total": len(products)}


@app.get("/api/products/{product_id}")
async def get_product(product_id: str):
    products = get_all_products()
    product = next((p for p in products if p["id"] == product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found.")
    return product


@app.get("/api/classes")
async def list_classes():
    import json
    class_path = os.path.join(os.path.dirname(__file__), "..", "model", "class_names.json")
    if os.path.exists(class_path):
        with open(class_path) as f:
            classes = json.load(f)
        return {"classes": classes, "total": len(classes)}
    return {"classes": [], "total": 0}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)