import os
import urllib.request

import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.set_page_config(page_title="MYRRAR MVP", layout="centered")
st.title("MYRRAR – Skin Tone Analysis (MVP)")

# --- Landmark indices for safe skin regions ---
LEFT_CHEEK = [234, 93, 132, 58, 172]
RIGHT_CHEEK = [454, 323, 361, 288, 397]
FOREHEAD = [10, 338, 297, 332, 284]  # was [10, 338, 297, 332, 284, 251, 389, 356]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = "face_landmarker.task"

DEBUG_SHOW_LANDMARKS = True  # show 468 points + polygons + bbox

@st.cache_resource(show_spinner=True)
def load_face_landmarker():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


def extract_region_mask(image, landmarks, indices):
    h, w, _ = image.shape
    points = np.array(
        [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices],
        dtype=np.int32,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, points, 255)
    return mask


def estimate_skin_tone(l_channel_mean):
    if l_channel_mean > 75:
        return "Light"
    if l_channel_mean > 60:
        return "Medium"
    if l_channel_mean > 45:
        return "Tan"
    return "Deep"


def bbox_from_landmarks(landmarks, w, h, pad=0.02):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min = max(0, int((min(xs) - pad) * w))
    x_max = min(w, int((max(xs) + pad) * w))
    y_min = max(0, int((min(ys) - pad) * h))
    y_max = min(h, int((max(ys) + pad) * h))
    return x_min, y_min, x_max, y_max


def draw_debug_overlays(image_bgr, result, landmarks,
                        box_color=(0, 0, 255),  # red (BGR)
                        lm_color=(255, 0, 0),   # blue (BGR)
                        poly_color=(0, 255, 255),  # yellow for polygons
                        box_thickness=2,
                        lm_radius=1):
    dbg = image_bgr.copy()
    h, w, _ = dbg.shape

    # Try detector box first
    drew_box = False
    if getattr(result, "face_detections", None):
        for det in result.face_detections:
            rel_box = det.bounding_box
            x_min = int(rel_box.origin_x)
            y_min = int(rel_box.origin_y)
            x_max = x_min + int(rel_box.width)
            y_max = y_min + int(rel_box.height)
            cv2.rectangle(dbg, (x_min, y_min), (x_max, y_max), box_color, box_thickness)
            drew_box = True

    # Fallback: derive bbox from landmarks
    if not drew_box and landmarks:
        x_min, y_min, x_max, y_max = bbox_from_landmarks(landmarks, w, h)
        cv2.rectangle(dbg, (x_min, y_min), (x_max, y_max), box_color, box_thickness)

    # Landmarks
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(dbg, (cx, cy), lm_radius, lm_color, -1, lineType=cv2.LINE_AA)

    # Polygons for our sampling regions
    for region in (LEFT_CHEEK, RIGHT_CHEEK, FOREHEAD):
        pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in region], np.int32)
        cv2.polylines(dbg, [pts], isClosed=True, color=poly_color, thickness=1, lineType=cv2.LINE_AA)

    return dbg


uploaded = st.file_uploader("Upload a clear, front-facing selfie", type=["jpg", "png", "jpeg"])

if uploaded:
    image_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_landmarker = load_face_landmarker()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        st.error("No face detected. Please upload a clear selfie.")
    else:
        landmarks = result.face_landmarks[0]  # list of 468 NormalizedLandmark

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for region in (LEFT_CHEEK, RIGHT_CHEEK, FOREHEAD):
            mask |= extract_region_mask(image, landmarks, region)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        skin_pixels = l_channel[mask == 255]

        if skin_pixels.size < 500:
            st.error("Could not reliably sample skin. Try better lighting or move hair off the forehead.")
        else:
            mean_l = float(np.mean(skin_pixels))
            tone = estimate_skin_tone(mean_l)

            overlay = image.copy()
            overlay[mask == 255] = (
                overlay[mask == 255] * 0.6 + np.array([0, 255, 0]) * 0.4
            )

            st.image(
                cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_BGR2RGB),
                caption="Sampled Skin Regions",
            )
            st.success(f"Estimated Skin Tone: **{tone}**")
            st.caption("Early-stage estimate. Lighting and makeup may affect results.")

            if DEBUG_SHOW_LANDMARKS:
                dbg = draw_debug_overlays(
                    image,
                    result,
                    landmarks,
                    box_color=(0, 0, 255),   # red bbox
                    lm_color=(255, 0, 0),    # blue landmarks
                    poly_color=(0, 255, 255) # yellow polygons
                )
                st.image(cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB),
                         caption="Debug: bbox (red), landmarks (blue), regions (yellow)")