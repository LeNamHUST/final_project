import logging
import os
import pandas as pd
import mlflow
import torch.nn as nn

from pathlib import Path 
from torch import optim
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


from src.models.resnet_model import RetinalResnetModel
from src.data.pre_processing import RetinalDataset
from src.utils import *
from configs.config import *


MAX_ITER = MAX_ITER
IMAGE_SIZE = IMAGE_SIZE
IMAGENET_MEAN = IMAGENET_MEAN
IMAGENET_STD = IMAGENET_STD
BATCH_SIZE = BATCH_SIZE
LEARNING_RATE = LEARNING_RATE
LEARNING_RATE_SCHEDULE_FACTOR = LEARNING_RATE_SCHEDULE_FACTOR
LEARNING_RATE_SCHEDULE_PATIENCE = LEARNING_RATE_SCHEDULE_PATIENCE
MAX_EPOCHS = MAX_EPOCHS
TRAINING_TIME_OUT = TRAINING_TIME_OUT
RANDOM_STATE = RANDOM_STATE
MODEL_PATH = MODEL_PATH
model_name = MODEL_NAME


mlflow.set_tracking_uri(uri="http://localhost:5000")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logger = logging.getLogger("retinal_disease")

def train():
    mlflow.set_experiment("retinal_disease_classification")

    PROJECT_ROOT = Path(os.getcwd())
    DATA_ROOT_PATH = PROJECT_ROOT / "data"
    DATA_PATH = DATA_ROOT_PATH / "train.csv"
    DATA_TRAIN_PATH = DATA_ROOT_PATH / "train/train"
    DATA_TEST_PATH = DATA_ROOT_PATH / "test/test"
    ARTIFACT_DIR = PROJECT_ROOT / "artifact_dir"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = PROJECT_ROOT / "models"

    logger.info(f"Data path: {DATA_PATH}")
    logger.info(f"Artifact dir: {ARTIFACT_DIR}")

    logging.info("Loading dataset ...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = pd.read_csv(DATA_PATH)
    LABELS = data.columns[1:]
    train_data, val_data = train_test_split(data, test_size=0.15, random_state=2025)
    train_dataset = RetinalDataset(DATA_TRAIN_PATH, 'train', train_data, (IMAGE_SIZE,IMAGE_SIZE), True)
    val_dataset = RetinalDataset(DATA_TRAIN_PATH, 'val', val_data, (IMAGE_SIZE,IMAGE_SIZE), True)
    
    sampler = sample_data(train_data)
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, sampler=sampler)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=None, num_workers=4, pin_memory=True)

    model = RetinalResnetModel(num_classes=len(LABELS))

    loss_criteria = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = LEARNING_RATE_SCHEDULE_FACTOR, patience = LEARNING_RATE_SCHEDULE_PATIENCE, mode = 'max')

    logger.info("Training Model ...")
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("max_iter", MAX_ITER)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("random_state", RANDOM_STATE)

        best_score = 0.0
        no_improve = 0
        for epoch in range(MAX_EPOCHS):
            train_loss, f1_train = train_pharse(model, train_dataloader, device, loss_criteria, optimizer)
            val_loss, f1_val = evaluation_pharse(model, val_dataloader, device, loss_criteria, optimizer)
            logger.info("Training set metric:")
            logger.info(f"f1 train: {f1_train:.4f}")
            
            logger.info("Evaluating set metric:")
            logger.info(f"f1 val: {f1_val:.4f}")
            if best_score < f1_val:
                best_score = f1_val
                mlflow.log_metric("f1_train", f1_train)
                mlflow.log_metric("f1_val", f1_val)
                mlflow.log_artifact(MODEL_PATH, "artifacts")
                mlflow.pytorch.log_model(model, "model")
                model_uri = f"runs:/{run.info.run_id}/model"
                registered = mlflow.register_model(
                    model_uri=model_uri,
                    name=model_name
                )
            else:
                no_improve += 1
            if no_improve > 10:
                break

if __name__ == "__main__":
    train()