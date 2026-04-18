from flask import Flask, request, jsonify
import cv2
import json
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Официальные модели Ultralytics.
# Если первая загрузка займёт время — это нормально.
pose_model = YOLO("yolo26n-pose.pt")
seg_model = YOLO("yolo26n-seg.pt")

KP = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


def clamp(v, lo, hi):
    return max(lo, min(int(v), hi))


def box_from_keypoints(kpts, confs, ids, w, h, pad=30, min_size=24):
    pts = []
    for idx in ids:
        if confs is None or confs[idx] >= 0.25:
            x, y = kpts[idx]
            pts.append((float(x), float(y)))

    if not pts:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    x1 = clamp(min(xs) - pad, 0, w - 1)
    y1 = clamp(min(ys) - pad, 0, h - 1)
    x2 = clamp(max(xs) + pad, 0, w - 1)
    y2 = clamp(max(ys) + pad, 0, h - 1)

    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1

    if bw < min_size:
        cx = (x1 + x2) // 2
        x1 = clamp(cx - min_size // 2, 0, w - 1)
        x2 = clamp(cx + min_size // 2, 0, w - 1)
        bw = x2 - x1

    if bh < min_size:
        cy = (y1 + y2) // 2
        y1 = clamp(cy - min_size // 2, 0, h - 1)
        y2 = clamp(cy + min_size // 2, 0, h - 1)
        bh = y2 - y1

    return {"x": int(x1), "y": int(y1), "w": int(bw), "h": int(bh)}


def parse_settings():
    raw = request.form.get("settings", "{}")
    try:
        return json.loads(raw)
    except Exception:
        return {}


@app.route("/detect", methods=["POST"])
def detect():
    if "frame" not in request.files:
        return jsonify({"boxes": [], "polygons": []})

    settings = parse_settings()

    file = request.files["frame"]
    img_bytes = file.read()

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"boxes": [], "polygons": []})

    h, w, _ = frame.shape

    need_pose = any(
        settings.get(k, False)
        for k in ["face", "torso", "arms", "legs"]
    )
    need_seg = settings.get("silhouette", False)

    boxes = []
    polygons = []

    pose_result = None
    seg_result = None

    if need_pose:
        pose_result = pose_model.predict(
            frame,
            imgsz=640,
            conf=0.25,
            verbose=False
        )[0]

    if need_seg:
        seg_result = seg_model.predict(
            frame,
            imgsz=640,
            conf=0.25,
            classes=[0],   # person only
            verbose=False
        )[0]

    # ===== POSE =====
    if pose_result is not None and pose_result.keypoints is not None:
        kpts_xy = pose_result.keypoints.xy.cpu().numpy()
        kpts_conf = None
        if pose_result.keypoints.conf is not None:
            kpts_conf = pose_result.keypoints.conf.cpu().numpy()

        for person_i, kpts in enumerate(kpts_xy):
            confs = kpts_conf[person_i] if kpts_conf is not None else None

            if settings.get("face", True):
                face_box = box_from_keypoints(
                    kpts, confs,
                    [KP["nose"], KP["left_eye"], KP["right_eye"], KP["left_ear"], KP["right_ear"]],
                    w, h,
                    pad=40,
                    min_size=50
                )
                if face_box:
                    face_box["part"] = "face"
                    boxes.append(face_box)

            if settings.get("torso", True):
                torso_box = box_from_keypoints(
                    kpts, confs,
                    [KP["left_shoulder"], KP["right_shoulder"], KP["left_hip"], KP["right_hip"]],
                    w, h,
                    pad=50,
                    min_size=80
                )
                if torso_box:
                    torso_box["part"] = "torso"
                    boxes.append(torso_box)

            if settings.get("arms", False):
                left_arm = box_from_keypoints(
                    kpts, confs,
                    [KP["left_shoulder"], KP["left_elbow"], KP["left_wrist"]],
                    w, h,
                    pad=45,
                    min_size=60
                )
                if left_arm:
                    left_arm["part"] = "arms"
                    boxes.append(left_arm)

                right_arm = box_from_keypoints(
                    kpts, confs,
                    [KP["right_shoulder"], KP["right_elbow"], KP["right_wrist"]],
                    w, h,
                    pad=45,
                    min_size=60
                )
                if right_arm:
                    right_arm["part"] = "arms"
                    boxes.append(right_arm)

            if settings.get("legs", False):
                left_leg = box_from_keypoints(
                    kpts, confs,
                    [KP["left_hip"], KP["left_knee"], KP["left_ankle"]],
                    w, h,
                    pad=45,
                    min_size=70
                )
                if left_leg:
                    left_leg["part"] = "legs"
                    boxes.append(left_leg)

                right_leg = box_from_keypoints(
                    kpts, confs,
                    [KP["right_hip"], KP["right_knee"], KP["right_ankle"]],
                    w, h,
                    pad=45,
                    min_size=70
                )
                if right_leg:
                    right_leg["part"] = "legs"
                    boxes.append(right_leg)

    # ===== SEGMENTATION =====
    if seg_result is not None and seg_result.masks is not None:
        for poly in seg_result.masks.xy:
            if poly is None or len(poly) < 3:
                continue

            points = [{"x": int(x), "y": int(y)} for x, y in poly]
            if len(points) >= 3:
                polygons.append({
                    "part": "silhouette",
                    "points": points
                })

    return jsonify({
        "boxes": boxes,
        "polygons": polygons
    })


if __name__ == "__main__":
    app.run(port=5000)