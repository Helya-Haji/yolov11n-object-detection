from ultralytics import YOLO
import torch

def main():
    model = YOLO(r"yolov11n.pt")

    model.train(
        data=r"data.yaml",
        task='detect',
        mode='train',
        epochs=30,
        imgsz=[640, 640],           
        batch=32,
        project='runs',
        name='vehicle detection',
        pretrained=True,
        rect=True,
        flipud=0.25,
        workers=2,
        scale=0.1,
        mosaic=0.1,
        deterministic=True,
        patience=0,
        translate=0.0,
        amp=True,
        single_cls=True, 
        cos_lr=True,
        save=True,
        device=0,
        lr0=0.001
    )

    metrics = model.val()

    #test_image = r"image.bmp"
    #results[0].show()

if __name__ == '__main__':
    main()
