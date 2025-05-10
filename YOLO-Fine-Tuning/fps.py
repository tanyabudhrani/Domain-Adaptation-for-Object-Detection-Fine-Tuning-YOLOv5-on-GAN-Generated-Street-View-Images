import time
import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# Set path to your trained model and validation images
weights_path = 'runs/train/cpu_fast_train/weights/best.pt'  # or 'last.pt'
source = 'fine_tune_data/images/val'  # folder with validation images

# Set device (CPU or GPU if available)
device = select_device('')
model = DetectMultiBackend(weights_path, device=device)
stride, imgsz = model.stride, 640
model.warmup(imgsz=(1, 3, imgsz, imgsz))  # warm-up

# Load images
dataset = LoadImages(source, img_size=imgsz, stride=stride)

# Inference + timing loop
total_time = 0
num_images = 0

for path, img, im0s, vid_cap, s in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    start = time.time()
    pred = model(img)
    pred = non_max_suppression(pred, 0.25, 0.45)
    total_time += time.time() - start
    num_images += 1

# Output FPS
fps = num_images / total_time
print(f"Inference Speed (trained model): {fps:.2f} FPS")
