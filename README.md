# yolov11n-object-detection
This repo contains scripts for training yolo11n for object detection, inference, benchmarking, and visualization.

### Install all dependencies:

pip install -r requirements.txt

### Scripts 

* train.py → Train YOLO11n model on dataset (data.yaml).
* inference_jpg.py → Run inference on .bmp images, save annotated images.
* test_fps.py → Measure FPS on a folder of .bmp images.
* benchmark.py → Benchmark YOLO model speed & accuracy.
* draw_labels.py → Visualize YOLO .txt annotations on .bmp images.
