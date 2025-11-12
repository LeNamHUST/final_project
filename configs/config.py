import os
from pathlib import Path 




HOST = "0.0.0.0"
PORT = 3000

ALLOWED_TYPES = {"jpeg", "png", "jpg"}
MAX_BYTES = 10*1024*1024
IMAGE_SIZE = 32
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MAX_ITER = 5000
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
LEARNING_RATE_SCHEDULE_FACTOR = 0.1
LEARNING_RATE_SCHEDULE_PATIENCE = 5
MAX_EPOCHS = 100
TRAINING_TIME_OUT = 3600*10
RANDOM_STATE = 20
MODEL_PATH = "retinal_best_score.pth"

LABELS = [
    "opacity",
    "diabetic retinopathy",
    "glaucoma",
    "macular edema",
    "macular degeneration",
    "retinal vascular occlusion",
    "normal"
]

DATA_DIR = Path("/home/namdao/projects/final_project/data")
DATA_TRAIN_PATH = "/home/namdao/projects/final_project/data/train/train"
DATA_TEST_PATH = "/home/namdao/projects/final_project/data/test/test"
TRAIN_PATH_CSV = "/home/namdao/projects/final_project/data/train.csv"