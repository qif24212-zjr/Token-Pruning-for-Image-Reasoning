from ultralytics import YOLO

model = YOLO(
    "/root/autodl-tmp/LLaVA/yolov8s-world.pt"
)

results = model.predict(
    "/root/autodl-tmp/LLaVA/images/cat.png",
    verbose=False
)

r = results[0]

print(r.boxes.xyxy)