import os
import time
from ultralytics import YOLO
import glob

def test_yolo_fps(model_path, image_folder, batch_size=32):

    model = YOLO(model_path)
    
    images = glob.glob(f"{image_folder}/*.bmp")

    model.predict(images[0])

    start_time = time.time()
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        results = model(batch, verbose=False)
    
    total_time = time.time() - start_time
    total_images = len(images)
    fps = total_images / total_time
    
    print(f"Total images processed: {total_images}")
    print(f"FPS: {fps:.2f}")
    print(f"Time per image: {total_time/total_images:.3f} seconds")

test_yolo_fps(r"model-path", r"dataset-path", batch_size=32)

