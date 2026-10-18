import os
import io
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"--> Saiswaroop Benchmark Suite using: {device}")

# 1. 10 Imagenette Synsets mapped to ImageNet-1K indices
IMAGENETTE_CLASSES = {
    "n01440764": {"idx": 0,   "name": "Tench (fish)"},
    "n02102040": {"idx": 217, "name": "English springer"},
    "n02979186": {"idx": 482, "name": "Cassette player"},
    "n03000684": {"idx": 491, "name": "Chain saw"},
    "n03028079": {"idx": 497, "name": "Church"},
    "n03394916": {"idx": 566, "name": "French horn"},
    "n03425413": {"idx": 569, "name": "Garbage truck"},
    "n03444034": {"idx": 571, "name": "Gas pump"},
    "n03445777": {"idx": 574, "name": "Golf ball"},
    "n03888257": {"idx": 701, "name": "Parachute"}
}

target_indices = [info["idx"] for info in IMAGENETTE_CLASSES.values()]
target_tensor = torch.tensor(target_indices, device=device)

def cartoonize_opencv_clean(pil_img):
    img_np = np.array(pil_img.convert('RGB'))
    color = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 7)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(cv2.bitwise_and(color, edges_color))

def compress_jpeg(pil_img, quality):
    if quality == "uncompressed":
        return pil_img
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf)
