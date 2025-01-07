from ultralytics import YOLO
model = YOLO("yolo11m.pt")
#Export the model to ONNX format
export_path = model.export(format="onnx")