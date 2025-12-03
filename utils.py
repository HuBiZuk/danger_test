# utils.py
import os
import json
import math
import numpy as np
import streamlit.elements.image as st_image


# 호환성 패치
def apply_streamlit_patch():
    if not hasattr(st_image, 'original_image_to_url'):
        st_image.original_image_to_url = st_image.image_to_url

    def simple_patch(image, width=None, clamp=False, channels="RGB", output_format="JPEG", image_id=None,
                     allow_emoji=False):
        return st_image.original_image_to_url(image, width, clamp, channels, output_format, image_id)

    st_image.image_to_url = simple_patch


# 폴더 생성 (없으면 에러나므로 필수)
def init_directories():
    if not os.path.exists('videos'): os.makedirs('videos')
    if not os.path.exists('settings'): os.makedirs('settings')


# 설정 불러오기
def load_settings(video_name):
    json_path = os.path.join('settings', f"{video_name}.json")
    # 기본값
    default_settings = {
        'zones': [],
        'warning_distance': 30,
        'angle_threshold': 130,
        'hip_ratio': 0.2,
        'extension_threshold': 0.85,
        'detection_mode': 'Algorithm',
        'vis_options': {
            'alert_only': False, 'bbox': True, 'label': True,
            'skeleton': True, 'zones': True, 'wrist_dot': True,
            'text': True
        }
    }

    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                saved = json.load(f)
                # 구역 데이터 유효성 검사
                if 'zones' in saved:
                    valid_zones = []
                    for z in saved['zones']:
                        if 'points' in z and len(z['points']) > 2:
                            valid_zones.append(z)
                    saved['zones'] = valid_zones

                # 없는 키 보충 (구버전 호환)
                for k, v in default_settings.items():
                    if k not in saved: saved[k] = v
                if 'vis_options' not in saved: saved['vis_options'] = default_settings['vis_options']
                return saved
        except:
            return default_settings
    return default_settings


# [중요] 설정을 파일로 저장하는 함수
def save_settings(video_name, settings):
    json_path = os.path.join('settings', f"{video_name}.json")
    with open(json_path, 'w') as f:
        json.dump(settings, f, indent=4)  # indent=4로 보기 좋게 저장


# 수학 계산 함수들 (Processor에서 사용)
def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle