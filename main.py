import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    drawing_utils,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

try:
    import joblib
except ImportError:  # pragma: no cover - optional until the user installs extras
    joblib = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
DOWNLOAD_TIMEOUT_SECONDS = 30

DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
LOGS_DIR = BASE_DIR / "logs"
DATASET_PATH = DATA_DIR / "gesture_samples.csv"
CLASSIFIER_PATH = ARTIFACTS_DIR / "gesture_classifier.joblib"
LABELS_PATH = ARTIFACTS_DIR / "gesture_labels.json"

PANEL_WIDTH = 300
SMOOTHING_WINDOW = 7
STABLE_THRESHOLD = 0.6
SESSION_LOG_COOLDOWN = 1.0
PUBLIC_STABILITY_WINDOW = 7
PUBLIC_STABILITY_THRESHOLD = 0.6
PUBLIC_STABILITY_MIN_COUNT = 4
PRIVATE_SIGNAL_WINDOW = 9
PRIVATE_SIGNAL_THRESHOLD = 0.88
PRIVATE_SIGNAL_MIN_COUNT = 8
PRIVATE_SIGNAL_LABEL = "__signal_variant_01__"
PRIVATE_SIGNAL_DISPLAY = "SECRET"
PRIVATE_SIGNAL_COLOR = (170, 230, 170)
LANDMARK_INDEX_COLOR = (245, 245, 245)
LANDMARK_INDEX_BG_COLOR = (35, 35, 35)
HAND_LANDMARK_COLOR = (235, 235, 235)
HAND_CONNECTION_COLOR = (120, 120, 120)
FEATURE_VERSION = 2

TIP_IDS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

PIP_IDS = {
    "thumb": 3,
    "index": 6,
    "middle": 10,
    "ring": 14,
    "pinky": 18,
}

COLLECTION_KEYS = {
    ord("1"): "ok",
    ord("2"): "no",
    ord("3"): "thumbs_up",
    ord("4"): "peace",
    ord("5"): "stop",
    ord("6"): "fist",
}

GESTURE_INFO = {
    "ok": {
        "display": "OK",
        "color": (0, 200, 0),
        "description": "Polegar e indicador unidos",
    },
    "no": {
        "display": "NO",
        "color": (0, 0, 255),
        "description": "Apenas o indicador levantado",
    },
    "thumbs_up": {
        "display": "JOINHA",
        "color": (40, 180, 255),
        "description": "Polegar para cima",
    },
    "peace": {
        "display": "PAZ",
        "color": (0, 255, 255),
        "description": "Indicador e medio levantados",
    },
    "stop": {
        "display": "PARE",
        "color": (255, 170, 0),
        "description": "Palma aberta",
    },
    "fist": {
        "display": "PUNHO",
        "color": (180, 180, 180),
        "description": "Mao fechada",
    },
}

CHALLENGE_GESTURES = list(GESTURE_INFO)
TIP_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

HAND_LANDMARK_SPEC = drawing_utils.DrawingSpec(
    color=HAND_LANDMARK_COLOR,
    thickness=2,
    circle_radius=3,
)
HAND_CONNECTION_SPEC = drawing_utils.DrawingSpec(
    color=HAND_CONNECTION_COLOR,
    thickness=2,
    circle_radius=1,
)


@dataclass
class GesturePrediction:
    label: str
    confidence: float
    source: str


def is_private_signal(label):
    return label == PRIVATE_SIGNAL_LABEL


def sanitize_public_prediction(prediction):
    if not prediction or not is_private_signal(prediction.label):
        return prediction

    return GesturePrediction("", 0.0, "private")


def is_collectible_prediction(prediction):
    return not prediction or not is_private_signal(prediction.label)


def stability_policy(label):
    if is_private_signal(label):
        return PRIVATE_SIGNAL_WINDOW, PRIVATE_SIGNAL_THRESHOLD, PRIVATE_SIGNAL_MIN_COUNT
    return PUBLIC_STABILITY_WINDOW, PUBLIC_STABILITY_THRESHOLD, PUBLIC_STABILITY_MIN_COUNT


@dataclass
class GestureSmoother:
    history: deque = field(
        default_factory=lambda: deque(
            maxlen=max(PUBLIC_STABILITY_WINDOW, PRIVATE_SIGNAL_WINDOW)
        )
    )

    def update(self, prediction):
        self.history.append(prediction)
        valid_labels = {item.label for item in self.history if item and item.label}
        if not valid_labels:
            return GesturePrediction("", 0.0, "none"), 0.0

        best_prediction = GesturePrediction("", 0.0, "unstable")
        best_ratio = 0.0

        for label in valid_labels:
            window_size, threshold, min_count = stability_policy(label)
            tail = list(self.history)[-window_size:]
            count = sum(1 for item in tail if item and item.label == label)
            ratio = count / len(tail)

            if ratio > best_ratio:
                best_ratio = ratio

            if count < min_count or ratio < threshold:
                continue

            confidences = [item.confidence for item in tail if item and item.label == label]
            confidence = sum(confidences) / len(confidences)
            source = next(
                (item.source for item in reversed(tail) if item and item.label == label),
                "rules",
            )

            if (
                ratio > best_ratio
                or not best_prediction.label
                or is_private_signal(label)
            ):
                best_prediction = GesturePrediction(label, confidence, source)
                best_ratio = ratio

        if best_prediction.label:
            return best_prediction, best_ratio

        return GesturePrediction("", 0.0, "unstable"), best_ratio


@dataclass
class ChallengeState:
    active: bool = False
    target_label: str = ""
    score: int = 0
    status_text: str = "Pressione G para iniciar o desafio"
    last_success_at: float = 0.0

    def toggle(self):
        self.active = not self.active
        self.score = 0
        self.last_success_at = 0.0
        if self.active:
            self.pick_target()
            self.status_text = "Repita o gesto mostrado"
        else:
            self.target_label = ""
            self.status_text = "Desafio pausado"

    def pick_target(self):
        self.target_label = random.choice(CHALLENGE_GESTURES)

    def update(self, stable_prediction):
        if not self.active or not stable_prediction.label:
            return

        now = time.time()
        if stable_prediction.label == self.target_label and now - self.last_success_at > 1.2:
            self.score += 1
            self.last_success_at = now
            self.status_text = "Acertou! Novo gesto sorteado"
            self.pick_target()


class SessionLogger:
    def __init__(self):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = LOGS_DIR / f"session_{timestamp}.csv"
        self.last_write = {}
        self._write_header()

    def _write_header(self):
        with self.path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(["timestamp", "label", "confidence", "source"])

    def log(self, prediction):
        if not prediction.label or is_private_signal(prediction.label):
            return

        now = time.time()
        last = self.last_write.get(prediction.label, 0.0)
        if now - last < SESSION_LOG_COOLDOWN:
            return

        with self.path.open("a", newline="", encoding="utf-8") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    prediction.label,
                    f"{prediction.confidence:.3f}",
                    prediction.source,
                ]
            )
        self.last_write[prediction.label] = now


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconhecimento de gestos com webcam, coleta e desafio."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    return parser.parse_args()


def pixel_coords(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)


def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def landmark_distance_3d(point_a, point_b):
    return math.sqrt(
        (point_a.x - point_b.x) ** 2
        + (point_a.y - point_b.y) ** 2
        + (point_a.z - point_b.z) ** 2
    )


def landmark_angle(point_a, point_b, point_c):
    vector_ab = (
        point_a.x - point_b.x,
        point_a.y - point_b.y,
        point_a.z - point_b.z,
    )
    vector_cb = (
        point_c.x - point_b.x,
        point_c.y - point_b.y,
        point_c.z - point_b.z,
    )
    magnitude_ab = math.sqrt(sum(value * value for value in vector_ab))
    magnitude_cb = math.sqrt(sum(value * value for value in vector_cb))
    if magnitude_ab == 0 or magnitude_cb == 0:
        return 0.0

    cosine = sum(a_value * c_value for a_value, c_value in zip(vector_ab, vector_cb))
    cosine /= magnitude_ab * magnitude_cb
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def normalized_value(value, lower, upper):
    if upper <= lower:
        return 0.0
    return clamp((value - lower) / (upper - lower))


def feature_columns():
    columns = ["feature_version", "handedness_right"]
    for index in range(21):
        columns.extend([f"lm_{index}_x", f"lm_{index}_y", f"lm_{index}_z"])
    for finger_name in TIP_ORDER:
        columns.append(f"{finger_name}_tip_wrist_distance")
    for left_index, left_name in enumerate(TIP_ORDER):
        for right_name in TIP_ORDER[left_index + 1 :]:
            columns.append(f"{left_name}_{right_name}_tip_distance")
    return columns


def open_camera(camera_index=0):
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]

    for backend in backends:
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        success, _ = cap.read()
        if success:
            return cap

        cap.release()

    return None


def file_sha256(file_path):
    digest = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(download_path):
    with urllib.request.urlopen(MODEL_URL, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        with download_path.open("wb") as file_obj:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file_obj.write(chunk)


def ensure_model():
    if MODEL_PATH.exists() and file_sha256(MODEL_PATH) == MODEL_SHA256:
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Baixando modelo da MediaPipe...")
    temp_path = MODEL_PATH.with_suffix(".tmp")

    try:
        download_model(temp_path)
        if file_sha256(temp_path) != MODEL_SHA256:
            raise RuntimeError("Falha na verificacao de integridade do modelo.")
        temp_path.replace(MODEL_PATH)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    print("Modelo salvo em:", MODEL_PATH)


def create_hand_landmarker():
    ensure_model()

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return HandLandmarker.create_from_options(options)


def load_classifier():
    if joblib is None or not CLASSIFIER_PATH.exists() or not LABELS_PATH.exists():
        return None

    model = joblib.load(CLASSIFIER_PATH)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return {"model": model, "labels": labels}


def is_thumb_extended(landmarks, handedness_label):
    wrist = landmarks[0]
    thumb_cmc = landmarks[1]
    thumb_mcp = landmarks[2]
    thumb_tip = landmarks[TIP_IDS["thumb"]]
    thumb_ip = landmarks[PIP_IDS["thumb"]]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]

    palm_size = max(landmark_distance_3d(wrist, middle_mcp), 1e-6)
    thumb_spread = landmark_distance_3d(thumb_tip, index_mcp) / palm_size
    thumb_reach = landmark_distance_3d(thumb_tip, wrist) / palm_size
    thumb_open_angle = landmark_angle(thumb_mcp, thumb_ip, thumb_tip)
    thumb_base_angle = landmark_angle(thumb_cmc, thumb_mcp, thumb_ip)

    if handedness_label == "Right":
        lateral_direction = thumb_tip.x < thumb_ip.x
    else:
        lateral_direction = thumb_tip.x > thumb_ip.x

    other_fingers_extended = sum(
        [
            is_finger_extended(landmarks, "index"),
            is_finger_extended(landmarks, "middle"),
            is_finger_extended(landmarks, "ring"),
            is_finger_extended(landmarks, "pinky"),
        ]
    )

    strong_thumb_extension = (
        thumb_spread > 0.44
        and thumb_reach > 0.82
        and thumb_open_angle > 145
        and thumb_base_angle > 135
        and lateral_direction
    )

    open_palm_thumb_extension = (
        other_fingers_extended >= 3
        and thumb_spread > 0.36
        and thumb_reach > 0.74
        and thumb_open_angle > 135
    )

    return strong_thumb_extension or open_palm_thumb_extension


def is_finger_extended(landmarks, finger_name):
    tip = landmarks[TIP_IDS[finger_name]]
    pip = landmarks[PIP_IDS[finger_name]]
    return tip.y < pip.y


def finger_states(landmarks, handedness_label):
    return {
        "thumb": is_thumb_extended(landmarks, handedness_label),
        "index": is_finger_extended(landmarks, "index"),
        "middle": is_finger_extended(landmarks, "middle"),
        "ring": is_finger_extended(landmarks, "ring"),
        "pinky": is_finger_extended(landmarks, "pinky"),
    }


def count_raised_fingers(states):
    return sum(1 for state in states.values() if state)


def compute_hand_metrics(landmarks, handedness_label, width, height):
    thumb_tip = pixel_coords(landmarks[TIP_IDS["thumb"]], width, height)
    index_tip = pixel_coords(landmarks[TIP_IDS["index"]], width, height)
    middle_tip = pixel_coords(landmarks[TIP_IDS["middle"]], width, height)
    ring_tip = pixel_coords(landmarks[TIP_IDS["ring"]], width, height)
    pinky_tip = pixel_coords(landmarks[TIP_IDS["pinky"]], width, height)
    wrist = pixel_coords(landmarks[0], width, height)
    middle_mcp = pixel_coords(landmarks[9], width, height)
    palm_size = max(distance(wrist, middle_mcp), 1)
    states = finger_states(landmarks, handedness_label)
    extension_ratios = {}
    tip_above_wrist_ratios = {}

    for finger_name in TIP_IDS:
        tip = landmarks[TIP_IDS[finger_name]]
        pip = landmarks[PIP_IDS[finger_name]]
        extension_ratios[finger_name] = ((pip.y - tip.y) * height) / palm_size
        tip_above_wrist_ratios[finger_name] = ((landmarks[0].y - tip.y) * height) / palm_size

    return {
        "thumb_tip": thumb_tip,
        "index_tip": index_tip,
        "middle_tip": middle_tip,
        "ring_tip": ring_tip,
        "pinky_tip": pinky_tip,
        "wrist": wrist,
        "palm_size": palm_size,
        "ok_distance_ratio": distance(thumb_tip, index_tip) / palm_size,
        "thumb_above_wrist": landmarks[TIP_IDS["thumb"]].y < landmarks[0].y,
        "states": states,
        "finger_count": count_raised_fingers(states),
        "extension_ratios": extension_ratios,
        "tip_above_wrist_ratios": tip_above_wrist_ratios,
        "tip_gap_ratios": {
            "index_to_middle": (index_tip[1] - middle_tip[1]) / palm_size,
            "ring_to_middle": (ring_tip[1] - middle_tip[1]) / palm_size,
            "pinky_to_middle": (pinky_tip[1] - middle_tip[1]) / palm_size,
        },
    }


def canonical_landmark_triplets(landmarks, handedness_label):
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    palm_size = max(
        math.sqrt(
            (middle_mcp.x - wrist.x) ** 2
            + (middle_mcp.y - wrist.y) ** 2
            + (middle_mcp.z - wrist.z) ** 2
        ),
        1e-6,
    )
    mirror = -1.0 if handedness_label == "Left" else 1.0

    triplets = []
    for landmark in landmarks:
        triplets.append(
            (
                ((landmark.x - wrist.x) / palm_size) * mirror,
                (landmark.y - wrist.y) / palm_size,
                (landmark.z - wrist.z) / palm_size,
            )
        )
    return triplets


def detect_private_signal(metrics):
    states = metrics["states"]
    extension_ratios = metrics["extension_ratios"]
    tip_above_wrist_ratios = metrics["tip_above_wrist_ratios"]
    tip_gap_ratios = metrics["tip_gap_ratios"]

    is_strict_middle_only = all(
        [
            states["middle"],
            not states["thumb"],
            not states["index"],
            not states["ring"],
            not states["pinky"],
            metrics["finger_count"] == 1,
            metrics["ok_distance_ratio"] > 0.65,
            extension_ratios["middle"] > 0.32,
            extension_ratios["index"] < 0.18,
            extension_ratios["ring"] < 0.18,
            extension_ratios["pinky"] < 0.16,
            tip_above_wrist_ratios["middle"] > 0.58,
            tip_above_wrist_ratios["index"] < 0.50,
            tip_above_wrist_ratios["ring"] < 0.50,
            tip_above_wrist_ratios["pinky"] < 0.46,
            tip_gap_ratios["index_to_middle"] > 0.22,
            tip_gap_ratios["ring_to_middle"] > 0.22,
            tip_gap_ratios["pinky_to_middle"] > 0.28,
        ]
    )

    if not is_strict_middle_only:
        return GesturePrediction("", 0.0, "rules-private")

    confidence = sum(
        [
            normalized_value(extension_ratios["middle"], 0.32, 0.62),
            1.0 - normalized_value(extension_ratios["index"], 0.08, 0.18),
            1.0 - normalized_value(extension_ratios["ring"], 0.08, 0.18),
            1.0 - normalized_value(extension_ratios["pinky"], 0.06, 0.16),
            normalized_value(tip_gap_ratios["index_to_middle"], 0.22, 0.42),
            normalized_value(tip_gap_ratios["ring_to_middle"], 0.22, 0.42),
        ]
    ) / 6.0

    return GesturePrediction(
        PRIVATE_SIGNAL_LABEL,
        0.90 + 0.09 * clamp(confidence),
        "rules-private",
    )


def rule_based_prediction(metrics):
    private_prediction = detect_private_signal(metrics)
    if private_prediction.label:
        return private_prediction, {}

    states = metrics["states"]
    scores = {}

    ok_closeness = clamp(1.0 - normalized_value(metrics["ok_distance_ratio"], 0.25, 0.55))
    scores["ok"] = (
        0.55 * ok_closeness
        + 0.15 * float(states["middle"])
        + 0.15 * float(states["ring"])
        + 0.15 * float(states["pinky"])
    )

    no_close_penalty = clamp(normalized_value(metrics["ok_distance_ratio"], 0.45, 0.85))
    no_folded = sum(
        [not states["thumb"], not states["middle"], not states["ring"], not states["pinky"]]
    ) / 4.0
    scores["no"] = 0.45 * float(states["index"]) + 0.35 * no_folded + 0.20 * no_close_penalty

    thumbs_up_folded = sum(
        [not states["index"], not states["middle"], not states["ring"], not states["pinky"]]
    ) / 4.0
    scores["thumbs_up"] = (
        0.45 * float(states["thumb"])
        + 0.30 * thumbs_up_folded
        + 0.25 * float(metrics["thumb_above_wrist"])
    )

    peace_folded = sum([not states["ring"], not states["pinky"]]) / 2.0
    scores["peace"] = (
        0.35 * float(states["index"])
        + 0.35 * float(states["middle"])
        + 0.20 * peace_folded
        + 0.10 * float(not states["ring"] and not states["pinky"])
    )

    scores["stop"] = metrics["finger_count"] / 5.0
    scores["fist"] = sum([not state for state in states.values()]) / 5.0

    best_label, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score < 0.72:
        return GesturePrediction("", best_score, "rules"), scores

    return GesturePrediction(best_label, best_score, "rules"), scores


def flatten_landmarks(landmarks, handedness_label):
    triplets = canonical_landmark_triplets(landmarks, handedness_label)

    features = [float(FEATURE_VERSION), 1.0 if handedness_label == "Right" else 0.0]
    for x_value, y_value, z_value in triplets:
        features.extend([x_value, y_value, z_value])

    wrist_triplet = triplets[0]
    tip_triplets = {finger_name: triplets[TIP_IDS[finger_name]] for finger_name in TIP_ORDER}
    for finger_name in TIP_ORDER:
        tip = tip_triplets[finger_name]
        features.append(
            math.sqrt(
                (tip[0] - wrist_triplet[0]) ** 2
                + (tip[1] - wrist_triplet[1]) ** 2
                + (tip[2] - wrist_triplet[2]) ** 2
            )
        )

    for left_index, left_name in enumerate(TIP_ORDER):
        for right_name in TIP_ORDER[left_index + 1 :]:
            left_tip = tip_triplets[left_name]
            right_tip = tip_triplets[right_name]
            features.append(
                math.sqrt(
                    (left_tip[0] - right_tip[0]) ** 2
                    + (left_tip[1] - right_tip[1]) ** 2
                    + (left_tip[2] - right_tip[2]) ** 2
                )
            )
    return features


def classify_with_model(classifier_bundle, landmarks, handedness_label):
    if classifier_bundle is None:
        return None

    features = flatten_landmarks(landmarks, handedness_label)
    probabilities = classifier_bundle["model"].predict_proba([features])[0]
    best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    confidence = float(probabilities[best_index])
    label = classifier_bundle["labels"][best_index]

    if confidence < 0.60 or is_private_signal(label):
        return GesturePrediction("", confidence, "model")

    return GesturePrediction(label, confidence, "model")


def predict_gesture(classifier_bundle, landmarks, handedness_label, width, height):
    model_prediction = classify_with_model(classifier_bundle, landmarks, handedness_label)
    if model_prediction and model_prediction.label:
        metrics = compute_hand_metrics(landmarks, handedness_label, width, height)
        return model_prediction, metrics, {}

    metrics = compute_hand_metrics(landmarks, handedness_label, width, height)
    rule_prediction, rule_scores = rule_based_prediction(metrics)

    if model_prediction and not rule_prediction.label and model_prediction.confidence > 0.50:
        return model_prediction, metrics, rule_scores

    return rule_prediction, metrics, rule_scores


def ensure_dataset_header():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    feature_names = feature_columns()
    headers = ["label", "handedness", *feature_names]

    if DATASET_PATH.exists():
        with DATASET_PATH.open("r", newline="", encoding="utf-8") as file_obj:
            reader = csv.reader(file_obj)
            existing_header = next(reader, [])
        if existing_header == headers:
            return

        legacy_path = DATA_DIR / f"gesture_samples_legacy_{int(time.time())}.csv"
        DATASET_PATH.replace(legacy_path)

    with DATASET_PATH.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(headers)


def append_sample(label, landmarks, handedness_label):
    if is_private_signal(label):
        return

    ensure_dataset_header()
    features = flatten_landmarks(landmarks, handedness_label)
    with DATASET_PATH.open("a", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow([label, handedness_label, *features])


def draw_info_line(panel, text, y, color=(240, 240, 240), scale=0.55, thickness=1):
    cv2.putText(
        panel,
        text,
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_landmark_indices(frame, landmarks):
    height, width = frame.shape[:2]
    for index, landmark in enumerate(landmarks):
        x_pos, y_pos = pixel_coords(landmark, width, height)
        cv2.circle(frame, (x_pos, y_pos), 10, LANDMARK_INDEX_BG_COLOR, -1)
        cv2.putText(
            frame,
            str(index),
            (x_pos - 6, y_pos + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            LANDMARK_INDEX_COLOR,
            1,
            cv2.LINE_AA,
        )


def prediction_display_name(prediction):
    if not prediction or not prediction.label:
        return "-"
    if is_private_signal(prediction.label):
        return PRIVATE_SIGNAL_DISPLAY
    return GESTURE_INFO.get(prediction.label, {}).get("display", prediction.label.upper())


def prediction_color(prediction):
    if not prediction or not prediction.label:
        return (200, 200, 200)
    if is_private_signal(prediction.label):
        return PRIVATE_SIGNAL_COLOR
    return GESTURE_INFO.get(prediction.label, {}).get("color", (220, 220, 220))


def draw_hand_label(frame, landmarks, handedness, prediction):
    height, width = frame.shape[:2]
    x_positions = []
    y_positions = []
    for landmark in landmarks:
        x_pos, y_pos = pixel_coords(landmark, width, height)
        x_positions.append(x_pos)
        y_positions.append(y_pos)

    anchor_x = max(min(x_positions) - 4, 8)
    anchor_y = max(min(y_positions) - 14, 28)
    short_hand = "L" if handedness == "Left" else "R" if handedness == "Right" else handedness[:1]
    label_text = short_hand
    if prediction and prediction.label:
        label_text = f"{short_hand}: {prediction_display_name(prediction)}"

    text_size, _ = cv2.getTextSize(
        label_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        2,
    )
    cv2.rectangle(
        frame,
        (anchor_x - 6, anchor_y - text_size[1] - 8),
        (anchor_x + text_size[0] + 8, anchor_y + 6),
        (22, 22, 22),
        -1,
    )
    cv2.putText(
        frame,
        label_text,
        (anchor_x, anchor_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        prediction_color(prediction),
        2,
        cv2.LINE_AA,
    )


def draw_side_panel(
    frame,
    stable_prediction,
    raw_prediction,
    handedness,
    hand_count,
    finger_count,
    fps,
    collection_label,
    challenge_state,
    classifier_bundle,
    sample_count,
):
    height = frame.shape[0]
    panel = frame[:, -PANEL_WIDTH:].copy()
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (PANEL_WIDTH, height), (28, 28, 36), -1)
    cv2.addWeighted(overlay, 0.88, panel, 0.12, 0, panel)

    cv2.putText(
        panel,
        "Painel de Controle",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    label = stable_prediction.label or "-"
    display_name = GESTURE_INFO.get(label, {}).get("display", label.upper())
    color = GESTURE_INFO.get(label, {}).get("color", (220, 220, 220))
    source = stable_prediction.source if stable_prediction.label else raw_prediction.source

    draw_info_line(panel, f"Gesto estavel: {display_name}", 72, color, 0.6, 2)
    draw_info_line(panel, f"Confianca: {stable_prediction.confidence:.2f}", 100)
    draw_info_line(panel, f"Origem: {source}", 126)
    draw_info_line(panel, f"Mao principal: {handedness or '-'}", 152)
    draw_info_line(panel, f"Maos detectadas: {hand_count}", 178)
    draw_info_line(panel, f"Dedos levantados: {finger_count}", 204)
    draw_info_line(panel, f"FPS: {fps:.1f}", 230)
    draw_info_line(panel, f"Modelo treinado: {'sim' if classifier_bundle else 'nao'}", 256)
    draw_info_line(panel, f"Amostras CSV: {sample_count}", 282)

    cv2.line(panel, (18, 300), (PANEL_WIDTH - 18, 300), (80, 80, 90), 1)
    draw_info_line(panel, "Coleta de dados", 326, (255, 220, 180), 0.62, 2)
    draw_info_line(
        panel,
        f"Rotulo atual: {GESTURE_INFO[collection_label]['display']}",
        354,
        GESTURE_INFO[collection_label]["color"],
        0.56,
        2,
    )
    draw_info_line(panel, "1-6 troca o rotulo", 378)
    draw_info_line(panel, "C salva a mao principal", 402)
    draw_info_line(panel, "T treina o modelo local", 426)

    cv2.line(panel, (18, 444), (PANEL_WIDTH - 18, 444), (80, 80, 90), 1)
    draw_info_line(panel, "Modo desafio", 470, (190, 230, 255), 0.62, 2)
    target_display = (
        GESTURE_INFO[challenge_state.target_label]["display"]
        if challenge_state.target_label
        else "-"
    )
    draw_info_line(panel, f"Alvo: {target_display}", 496)
    draw_info_line(panel, f"Pontuacao: {challenge_state.score}", 520)
    draw_info_line(panel, challenge_state.status_text[:32], 544)
    draw_info_line(panel, "G inicia/pausa desafio", 568)

    cv2.line(panel, (18, 586), (PANEL_WIDTH - 18, 586), (80, 80, 90), 1)
    draw_info_line(panel, "Atalhos", 612, (220, 220, 220), 0.62, 2)
    draw_info_line(panel, "Q sair", 638)

    frame[:, -PANEL_WIDTH:] = panel


def draw_main_overlay(frame, stable_prediction):
    if not stable_prediction.label:
        cv2.putText(
            frame,
            "Mostre um gesto na frente da camera",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return

    if is_private_signal(stable_prediction.label):
        cv2.putText(
            frame,
            PRIVATE_SIGNAL_DISPLAY,
            (24, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            PRIVATE_SIGNAL_COLOR,
            3,
            cv2.LINE_AA,
        )
        return

    info = GESTURE_INFO[stable_prediction.label]
    cv2.putText(
        frame,
        info["display"],
        (24, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        info["color"],
        3,
        cv2.LINE_AA,
    )


def count_dataset_samples():
    if not DATASET_PATH.exists():
        return 0
    count = 0
    with DATASET_PATH.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            if not is_private_signal(row["label"]):
                count += 1
    return count


def train_local_classifier():
    if joblib is None:
        raise RuntimeError("Instale joblib e scikit-learn para treinar o modelo.")

    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if not DATASET_PATH.exists():
        raise RuntimeError("Nenhum dataset encontrado. Colete amostras primeiro.")

    rows = []
    labels = []
    with DATASET_PATH.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        feature_names = [name for name in (reader.fieldnames or []) if name not in {"label", "handedness"}]
        for row in reader:
            if is_private_signal(row["label"]):
                continue
            labels.append(row["label"])
            rows.append([float(row[name]) for name in feature_names])

    if len(rows) < 12:
        raise RuntimeError("Colete pelo menos 12 amostras antes de treinar.")

    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        raise RuntimeError("Colete ao menos 2 gestos diferentes para treinar.")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    classifier = VotingClassifier(
        estimators=[
            (
                "svc",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "svc",
                            SVC(
                                C=8.0,
                                gamma="scale",
                                probability=True,
                                class_weight="balanced",
                                random_state=42,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=400,
                    random_state=42,
                    class_weight="balanced_subsample",
                    max_features="sqrt",
                    min_samples_leaf=1,
                ),
            ),
            (
                "et",
                ExtraTreesClassifier(
                    n_estimators=400,
                    random_state=42,
                    class_weight="balanced",
                    max_features="sqrt",
                ),
            ),
        ],
        voting="soft",
    )
    classifier.fit(rows, labels)
    joblib.dump(classifier, CLASSIFIER_PATH)
    LABELS_PATH.write_text(json.dumps(list(classifier.classes_)), encoding="utf-8")
    return classifier.classes_


def handle_collection_key(key, current_label):
    return COLLECTION_KEYS.get(key, current_label)


def create_extended_canvas(frame):
    height, width = frame.shape[:2]
    canvas = np.zeros((height, width + PANEL_WIDTH, 3), dtype=frame.dtype)
    canvas[:, :width] = frame
    return canvas


def run_app(camera_index):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    cap = open_camera(camera_index)
    if cap is None:
        raise RuntimeError("Nao foi possivel abrir a webcam.")

    classifier_bundle = load_classifier()
    hand_landmarker = create_hand_landmarker()
    hand_smoothers = {}
    logger = SessionLogger()
    challenge_state = ChallengeState()
    collection_label = "ok"
    sample_count = count_dataset_samples()
    status_message = "Pressione G para desafio ou C para coletar"
    frame_index = 0
    prev_time = time.perf_counter()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = hand_landmarker.detect_for_video(mp_image, frame_index)
            frame_index += 1

            handedness = ""
            hand_count = 0
            finger_count = 0
            raw_prediction = GesturePrediction("", 0.0, "none")
            stable_prediction = GesturePrediction("", 0.0, "none")
            primary_hand_result = None
            active_smoother_keys = set()

            if result.hand_landmarks and result.handedness:
                hand_count = len(result.hand_landmarks)

                for hand_index, (landmarks, handedness_info) in enumerate(
                    zip(result.hand_landmarks, result.handedness)
                ):
                    handedness_label = handedness_info[0].category_name or f"Hand{hand_index + 1}"
                    smoother_key = handedness_label
                    if smoother_key in active_smoother_keys:
                        smoother_key = f"{handedness_label}_{hand_index}"
                    active_smoother_keys.add(smoother_key)

                    if smoother_key not in hand_smoothers:
                        hand_smoothers[smoother_key] = GestureSmoother()

                    raw_candidate, metrics, _ = predict_gesture(
                        classifier_bundle,
                        landmarks,
                        handedness_label,
                        frame.shape[1],
                        frame.shape[0],
                    )
                    stable_candidate, _ = hand_smoothers[smoother_key].update(raw_candidate)

                    drawing_utils.draw_landmarks(
                        frame,
                        landmarks,
                        HandLandmarksConnections.HAND_CONNECTIONS,
                        HAND_LANDMARK_SPEC,
                        HAND_CONNECTION_SPEC,
                    )
                    draw_landmark_indices(frame, landmarks)
                    draw_hand_label(frame, landmarks, handedness_label, stable_candidate)

                    candidate_result = {
                        "handedness": handedness_label,
                        "raw_prediction": raw_candidate,
                        "stable_prediction": stable_candidate,
                        "metrics": metrics,
                        "landmarks": landmarks,
                    }

                    candidate_score = (
                        stable_candidate.confidence if stable_candidate.label else raw_candidate.confidence
                    )
                    primary_score = -1.0
                    if primary_hand_result is not None:
                        primary_stable = primary_hand_result["stable_prediction"]
                        primary_raw = primary_hand_result["raw_prediction"]
                        primary_score = (
                            primary_stable.confidence if primary_stable.label else primary_raw.confidence
                        )

                    if primary_hand_result is None or candidate_score >= primary_score:
                        primary_hand_result = candidate_result

                for smoother_key, smoother in hand_smoothers.items():
                    if smoother_key not in active_smoother_keys:
                        smoother.update(None)
            else:
                for smoother in hand_smoothers.values():
                    smoother.update(None)

            if primary_hand_result is not None:
                handedness = primary_hand_result["handedness"]
                raw_prediction = primary_hand_result["raw_prediction"]
                stable_prediction = primary_hand_result["stable_prediction"]
                finger_count = primary_hand_result["metrics"]["finger_count"]

            public_raw_prediction = sanitize_public_prediction(raw_prediction)
            public_stable_prediction = sanitize_public_prediction(stable_prediction)

            challenge_state.update(public_stable_prediction)
            logger.log(public_stable_prediction)

            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            canvas = create_extended_canvas(frame)
            draw_main_overlay(canvas, stable_prediction)
            draw_info_line(
                canvas,
                status_message[:50],
                frame.shape[0] - 20,
                (220, 220, 220),
                0.55,
                1,
            )
            draw_side_panel(
                canvas,
                public_stable_prediction,
                public_raw_prediction,
                handedness,
                hand_count,
                finger_count,
                fps,
                collection_label,
                challenge_state,
                classifier_bundle,
                sample_count,
            )

            cv2.imshow("Reconhecimento de Gestos", canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key in COLLECTION_KEYS:
                collection_label = handle_collection_key(key, collection_label)
                status_message = (
                    f"Rotulo de coleta alterado para {GESTURE_INFO[collection_label]['display']}"
                )
            elif key == ord("c"):
                if primary_hand_result is not None:
                    if is_collectible_prediction(raw_prediction):
                        append_sample(
                            collection_label,
                            primary_hand_result["landmarks"],
                            handedness,
                        )
                        sample_count += 1
                        status_message = (
                            f"Amostra salva para {GESTURE_INFO[collection_label]['display']}"
                        )
                    else:
                        status_message = "Gesto atual nao pode ser salvo no dataset"
                else:
                    status_message = "Nenhuma mao detectada para salvar"
            elif key == ord("g"):
                challenge_state.toggle()
                status_message = challenge_state.status_text
            elif key == ord("t"):
                try:
                    trained_labels = train_local_classifier()
                    classifier_bundle = load_classifier()
                    status_message = (
                        "Modelo treinado com: " + ", ".join(label.upper() for label in trained_labels)
                    )
                except Exception as exc:  # pragma: no cover - UI feedback
                    status_message = str(exc)
    finally:
        hand_landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    run_app(args.camera_index)


if __name__ == "__main__":
    main()
