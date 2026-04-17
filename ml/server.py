from flask import Flask, request, jsonify
from ultralytics import YOLO
import base64
import numpy as np
import cv2

app = Flask(__name__)

model = YOLO("yolov8n.pt") 

@app.route("/detect", methods=["POST"])
def detect():
    if "frame" not in request.files:
        return jsonify({"error": "No frame"}), 400

    file = request.files["frame"]

    img_bytes = file.read()

    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    results = model(img)[0]

    boxes = []

    for box in results.boxes:
        cls = int(box.cls[0])

        if cls == 0:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            boxes.append({
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1)
            })

    return jsonify({"boxes": boxes})

if __name__ == "__main__":
    app.run(port=5000)