# import torch
# import pytest
# import pandas as pd
# from PIL import Image
# from pathlib import Path
# from configs.config import *
# from src.data.pre_processing import RetinalDataset

# IMAGE_SIZE = IMAGE_SIZE
# DATA_DIR = DATA_DIR
# DATA_TRAIN_PATH = DATA_TRAIN_PATH
# TRAIN_PATH_CSV = TRAIN_PATH_CSV
# train_data = pd.read_csv(TRAIN_PATH_CSV)
# filename = train_data["filename"]
# train_dataset = RetinalDataset(DATA_TRAIN_PATH, 'test', train_data, (IMAGE_SIZE,IMAGE_SIZE), True)

# @pytest.mark.parametrize("result", train_dataset)
# def test_image(result):
#     image, label, image_path = result
#     try:
#         print("label:", image_path)
#         assert isinstance(image, torch.Tensor), "image not tensor"
#         assert isinstance(label, torch.Tensor), "label not tensor"
#         assert label.ndim == 1, "label not ndim 1"
#         assert label.shape[0] == 7, "label not shape 7"
#         assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE), "image error size"
#     except:
#         pytest.fail(f"Cannot identify image file {image_path}")