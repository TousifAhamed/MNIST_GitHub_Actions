from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from ..core.security import SecurityHandler
from ..ml.model import MNISTNet
from .augmentation_endpoints import router as augmentation_router

app = FastAPI(title="MNIST Classifier API")

# Include augmentation router
app.include_router(augmentation_router, prefix="/augmentation", tags=["augmentation"])

# Security
security = HTTPBearer()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = MNISTNet()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "MNIST Classifier API",
        "endpoints": {
            "predict": "/predict",
            "augmentation_ui": "/augmentation/augmentation-ui",
            "augment": "/augmentation/augment"
        }
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    token: str = Depends(SecurityHandler.verify_token)
):
    # Verify file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read and transform image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('L')
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Make prediction
    with torch.no_grad():
        tensor = transform(image).unsqueeze(0)
        prediction = model(tensor)
        digit = prediction.argmax(dim=1).item()
    
    return {"predicted_digit": digit} 