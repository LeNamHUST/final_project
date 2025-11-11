from enum import Enum 
import uvicorn
import imghdr
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
from pydantic import BaseModel 
import mlflow
from mlflow.tracking import MlflowClient
from src.models.resnet_model import RetinalResnetModel
from configs.config import *

app = FastAPI()

ALLOWED_TYPES = ALLOWED_TYPES
MAX_BYTES = MAX_BYTES
IMAGE_SIZE = IMAGE_SIZE
IMAGENET_MEAN = IMAGENET_MEAN
IMAGENET_STD = IMAGENET_STD
labels = LABELS

device = "cuda" if torch.cuda.is_available() else "cpu"
PATH_TRAINED_MODEL="/home/namdao/projects/final_project/retina_best_score.pth"
model = RetinalResnetModel(num_classes=7).to(device)
model.load_state_dict(torch.load(PATH_TRAINED_MODEL, weights_only=False, map_location=device)['model'])
model.eval()

# mlflow load model
def model_mlflow(model_name):
    try:
        if model_name is None:
            model_name = os.environ.get("MLFLOW_MODEL_NAME")
        if not model_name:
            print("MLFLOW_MODEL_NAME not set; skipping MLflow load")
            return None, None, None
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        client = MlflowClient(tracking_uri) if tracking_uri else MlflowClient()

        prod_versions = client.get_latest_versions(name=model_name, stages=["Production"])
        if not prod_versions:
            print(f"No Production versions found for model {model_name}")
            return None, None, None

        ver = prod_versions[0].version
        model_uri = f"models: /{model_name}/Production"
        print(f"Loading Mlflow model from {model_uri}")
        loaded = mlflow.pytorch.load_model(model_uri, map_location=device)
        loaded.to(device)
        loaded.eval()
        return loaded, f"mlflow://{model_name}/Production", str(ver)
    except Exception as err:
        pritn("failed to load model from mlflow:", err)
        return None, None, None

@app.post("/retinal_diesase")
async def retinal_diesase(file:UploadFile = File(...)):
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
    return {"message": result}

if __name__ == "__main__":
    uvicorn.run("api:app", host=HOST, port=PORT, reload=True)