from ultralytics import YOLO

model = YOLO("yolo26n.yaml")

model.train(
    data="haris.yaml",
    epochs=150,
    imgsz=640,  # pixel size (resnet 224x224)
    freeze=0,
    batch=64,
)
