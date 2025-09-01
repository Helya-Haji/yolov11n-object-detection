from ultralytics import YOLO

model = YOLO(r"yolov8n.pt")

image_path = r""
results = model(image_path)

results[0].show()
