import streamlit as st
import os
import json
import numpy as np


# 👇 [수정 1] Streamlit 1.34+ 버전 호환성 패치 (캔버스 에러 방지)
def apply_streamlit_patch():
    """
    최신 Streamlit 버전에서 streamlit-drawable-canvas 라이브러리가
    작동하지 않는 문제(image_to_url 에러)를 해결하는 패치입니다.
    """
    try:
        import streamlit.elements.image as st_image
        from streamlit.elements.lib import image_utils

        # 호환성 래퍼 함수 정의 (인자 개수 및 순서 맞춤)
        def custom_image_to_url(image, width=None, clamp=False, channels="RGB", output_format="JPEG", image_id=None):
            # width에 int가 들어오면 에러가 나므로 None으로 고정하고, 나머지 인자는 순서대로 전달
            return image_utils.image_to_url(
                image,
                None,  # width 자리에 None 전달
                clamp,
                channels,
                output_format,
                image_id
            )

        # 패치 적용
        st_image.image_to_url = custom_image_to_url

    except ImportError:
        pass  # 구버전이거나 경로가 다르면 무시
    except Exception as e:
        print(f"Streamlit Patch Error: {e}")


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

    # 👇 [수정 2] 기본 설정값에 낙상 관련 설정(fall_check, fall_ratio) 추가
    return {
        "zones": [],
        "fire_check": False,
        "fall_check": True,  # 낙상 감지 기본 켜기
        "fall_ratio": 1.2,  # 낙상 민감도 기본값
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
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


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
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # 클리핑으로 부동소수점 오류 방지
    angle_deg = np.degrees(angle_rad)
    return angle_deg