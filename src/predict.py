import os

import mlflow
import mlflow.pytorch
import pandas as pd 
from fastapi import APIRouter, FastAPI, File, UploadFile, HTTPException

from configs.config import *
from src.schemas.response import RetinalDiseaseClassificationResponse

MLFLOW_TRACKING_URI = MLFLOW_TRACKING_URI
print("MLFLOW_TRACKING_URI:", MLFLOW_TRACKING_URI)
mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)

model_name = MODEL_NAME
model_version = MODEL_VERSION
alias = "production"

model_uri = f"models:/{model_name}/{model_version}"

_model = None
retinal_router = APIRouter(prefix="/retinal")

def get_model():
    global _model
    print('model_uri:', model_uri)
    if _model is None:
        _model = mlflow.pytorch.load_model(model_uri)
    return _model

model = get_model()

print('load xong model')

@retinal_router.post("/predict", response_model=RetinalDiseaseClassificationResponse)
async def func_predict(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="file qua lon")
    img_type = imghdr.what(None, h=contents)
    if img_type is None or img_type.lower() not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"khong ho tro dinh dang")
    try:
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_transformation = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
        img = img_transformation(img)
        with torch.no_grad():
            img = img.unsqueeze(0)
            img = img.to(device)
            pred = model(img)
            pred = ((pred>0.5)*1).squeeze().tolist()
            print('pred:', pred)
            result = [labels[i] for i, val in enumerate(pred) if val == 1]
    except Exception as err:
        raise HTTPException(status_code=400, detail=err)
    return RetinalDiseaseClassificationResponse(retinal_classification=result)
