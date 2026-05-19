from ultralytics import YOLO

model = YOLO("yolo26n.yaml")

model.train(
    data="haris.yaml",
    epochs=50,
    imgsz=640,  # pixel size (resnet 224x224)
    freeze=10
    batch=32,
)