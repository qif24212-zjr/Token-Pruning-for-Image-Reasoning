from ultralytics import YOLO

model = YOLO(
    "/root/autodl-tmp/LLaVA/yolov8s-world.pt"
)

results = model.predict(
    "/root/autodl-tmp/LLaVA/images/cat.png",
    verbose=False
)

labels = []

for r in results:
    for cls in r.boxes.cls:
        labels.append(
            r.names[int(cls)]
        )

print(labels)