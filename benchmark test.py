import os
from ultralytics import YOLO

model = YOLO (r"best.pt")
model.benchmark (imgsz = 480, half = False, device = 0)