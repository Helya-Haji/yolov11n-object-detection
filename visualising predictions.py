import os
os.makedirs(DST_DIR, exist_ok=True)
import cv2
from ultralytics import YOLO

SRC_DIR = r""   #path of raw images
DST_DIR = r""   #path to save results
MODEL_PATH = r""   #path of model

model = YOLO(MODEL_PATH)

def process_jpg(file_path: str, parent_folder: str):
    """Run YOLO inference on JPG and save annotated image only."""
    img = cv2.imread(file_path)
    if img is None:
        print(f"⚠️ Could not read {file_path}, skipping.")
        return

    results = model(img)[0]

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box.tolist())
        cls_id = int(cls.item())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(cls_id), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    base_name = f"{parent_folder}--{os.path.basename(file_path)}"
    annotated_path = os.path.join(DST_DIR, base_name)
    cv2.imwrite(annotated_path, img)


for root, _, files in os.walk(SRC_DIR):
    folder_name = os.path.basename(root)
    for file_name in files:
        if file_name.lower().endswith(".bmp"):
            process_jpg(os.path.join(root, file_name), folder_name)

print("✅ JPG inference complete. Annotated images saved (no txt files).")