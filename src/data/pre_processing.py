import os
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

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


class RetinalDataset(Dataset):
    def __init__(self, folder_dir, type_data, dataframe, image_size, normalization=True):
        self.image_paths = []
        self.image_labels = []
        self.type_data = type_data

        train_transformation = [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomAffine(degrees=30, translate=(0.2, 0.2))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]

        val_transformation = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]

        if normalization:
            if self.type_data == "train":
                self.image_transformation = transforms.Compose(train_transformation)
            else:
                self.image_transformation = transforms.Compose(val_transformation)

        for index, row in dataframe.iterrows():
            images_path = os.path.join(folder_dir, row.filename)
            self.image_paths.append(images_path)
            labels = []
            for col in row[1:]:
                if col == 1:
                    labels.append(1)
                else:
                    labels.append(0)
            self.image_labels.append(labels)

    def __len__(self):
        return len(self.image_paths)
    
    def get_labels(self):
        label = self.image_labels
        return label

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image_data = Image.open(image_path).convert("RGB")
        image_data = self.image_transformation(image_data)
        return image_data, torch.FloatTensor(self.image_labels[index]), image_path