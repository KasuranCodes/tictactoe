from ultralytics import YOLO
model = YOLO("yolov8s.pt")
model.train(data="/home/strawberry/Documents/Programming/Python/yolo/data.yaml", epochs=50, batch=16, imgsz=640)
results = model("/home/strawberry/Documents/Programming/Python/yolo/test/images/hand1.jpg", save=True)
metrics = model.val()