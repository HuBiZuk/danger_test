# utils.py
import streamlit as st
import os
import json
import numpy as np

# ⚠️ Streamlit 1.22.0에서는 이 패치가 필요 없으므로 제거하거나 주석 처리합니다.
# def apply_streamlit_patch():
#     # from streamlit.elements import image as st_image
#     # st_image.original_image_to_url = st_image.image_to_url
#     pass


def init_directories():
    os.makedirs("videos", exist_ok=True)
    os.makedirs("settings", exist_ok=True)


def save_settings(video_name, settings):
    file_path = os.path.join("settings", f"{video_name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)


def load_settings(video_name):
    file_path = os.path.join("settings", f"{video_name}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return { # 기본 설정
        "zones": [],
        "fire_check": False,
        "warning_distance": 30,
        "extension_threshold": 0.85,
        "angle_threshold": 130,
        "hip_ratio": 0.2,
        "detection_mode": "Algorithm",
        "vis_options": {
            "alert_only": False,
            "skeleton": True,
            "zones": True,
            "bbox": True,
            "label": True,
            "wrist_dot": True,
            "text": True
        }
    }


def get_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


def calculate_angle(p1, p2, p3):
    # p2가 꼭짓점
    vec1 = np.array(p1) - np.array(p2)
    vec2 = np.array(p3) - np.array(p2)

    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0  # 0으로 나누는 오류 방지

    cosine_angle = dot_product / (magnitude1 * magnitude2)
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) # 클리핑으로 부동소수점 오류 방지
    angle_deg = np.degrees(angle_rad)
    return angle_deg