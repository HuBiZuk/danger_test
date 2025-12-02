import torch
import streamlit as st
import cv2
import numpy as np
import os
import joblib
import pandas as pd
import json
import time
import math
from ultralytics import YOLO

# --- [초기 설정] ---
if not os.path.exists('videos'): os.makedirs('videos')
if not os.path.exists('settings'): os.makedirs('settings')


# --- [함수] 계산 ---
def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle


# --- [함수] 설정 ---
def load_settings(video_name):
    json_path = os.path.join('settings', f"{video_name}.json")
    default_settings = {
        'zone_x': 0.4, 'zone_y': 0.5, 'zone_w': 0.15, 'zone_h': 0.25,
        'padding': 50, 'angle_threshold': 130, 'hip_ratio': 0.2,
        'extension_threshold': 0.85,
        'detection_mode': 'Algorithm',
        'vis_alert_only': False,
        'vis_skeleton': True,
        'vis_bbox': True,
        'vis_class_label': True,
        'vis_zones': True,
        'vis_safe_line': True,
        'vis_wrist_text': True,
        'vis_wrist_dot': True
    }
    if os.path.exists(json_path):
        with open(json_path, 'r') as f: return json.load(f)
    return default_settings


def save_settings(video_name, settings):
    json_path = os.path.join('settings', f"{video_name}.json")
    with open(json_path, 'w') as f: json.dump(settings, f)


# --- [함수] 프레임 분석 ---
def process_frame(frame, yolo_model, custom_model, settings):
    frame = cv2.resize(frame, (800, 600))
    h, w, _ = frame.shape

    # 설정값 로드
    z_x, z_y = settings['zone_x'], settings['zone_y']
    z_w, z_h = settings['zone_w'], settings['zone_h']
    pad = settings['padding']
    ang_th = settings['angle_threshold']
    hip_r = settings['hip_ratio']
    ext_th = settings.get('extension_threshold', 0.85)
    mode = settings.get('detection_mode', 'Algorithm')

    # 시각화 옵션
    v_alert_only = settings.get('vis_alert_only', False)  # 감지 시에만 표시
    v_skel = settings.get('vis_skeleton', True)
    v_bbox = settings.get('vis_bbox', True)
    v_cls = settings.get('vis_class_label', True)
    v_zones = settings.get('vis_zones', True)
    v_line = settings.get('vis_safe_line', True)
    v_text = settings.get('vis_wrist_text', True)
    v_dot = settings.get('vis_wrist_dot', True)

    # 구역 좌표
    d_x1, d_y1 = int(z_x * w), int(z_y * h)
    d_x2, d_y2 = int((z_x + z_w) * w), int((z_y + z_h) * h)
    w_x1, w_y1 = max(0, d_x1 - pad), max(0, d_y1 - pad)
    w_x2, w_y2 = min(w, d_x2 + pad), min(h, d_y2 + pad)

    # YOLO 추론
    device = 0 if torch.cuda.is_available() else 'cpu'
    results = yolo_model(frame, verbose=False, conf=0.25, device=device)

    # [배경 설정]
    # '감지 시 표시'가 켜져 있으면 -> 무조건 원본 (뼈대 미리 그리기 X)
    # 꺼져 있으면 -> 뼈대 옵션에 따라 결정
    if v_alert_only:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        if v_skel:
            annotated_frame = results[0].plot(boxes=False, probs=False)
            image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    global_is_danger = False
    global_is_warning = False

    # 분석 및 그리기
    if results[0].keypoints is not None and results[0].boxes is not None:
        keypoints_data = results[0].keypoints.data.cpu().numpy()
        boxes_data = results[0].boxes.data.cpu().numpy()

        for box_info, kps in zip(boxes_data, keypoints_data):
            bx1, by1, bx2, by2, b_conf, b_cls = box_info
            class_name = yolo_model.names[int(b_cls)]

            p_danger = False
            p_warning = False
            p_arms_res = []

            # 팔 분석
            arms = [{'side': 'Right', 's': 6, 'e': 8, 'w': 10, 'h': 12},
                    {'side': 'Left', 's': 5, 'e': 7, 'w': 9, 'h': 11}]

            for arm in arms:
                res = {'valid': False}
                if len(kps) > arm['h'] and kps[arm['w']][2] >= 0.25:
                    s, e, w_pt, h_pt = kps[arm['s']], kps[arm['e']], kps[arm['w']], kps[arm['h']]
                    wx, wy = int(w_pt[0]), int(w_pt[1])
                    ex, ey = int(e[0]), int(e[1])
                    sx, sy = int(s[0]), int(s[1])

                    has_hip = h_pt[2] > 0.25
                    is_hand_low = False
                    safe_y = 0
                    hx, hy = 0, 0
                    if has_hip:
                        hx, hy = int(h_pt[0]), int(h_pt[1])
                        safe_y = hy - (abs(hy - sy) * hip_r)
                        is_hand_low = wy > safe_y

                    angle = calculate_angle((sx, sy), (ex, ey), (wx, wy))
                    len_u, len_l = get_distance((sx, sy), (ex, ey)), get_distance((ex, ey), (wx, wy))
                    ext_r = get_distance((sx, sy), (wx, wy)) / (len_u + len_l) if (len_u + len_l) > 0 else 0

                    is_algo = (angle > ang_th) or (ext_r > ext_th)
                    is_ai = False
                    if mode in ['AI', 'Both']:
                        inp = pd.DataFrame(
                            [{'rw_x': wx / w, 'rw_y': wy / h, 're_x': ex / w, 're_y': ey / h, 'rs_x': sx / w,
                              'rs_y': sy / h}])
                        is_ai = (custom_model.predict(inp)[0] == 1)

                    is_reach = is_algo if mode == 'Algorithm' else (is_ai if mode == 'AI' else (is_algo and is_ai))
                    if is_hand_low: is_reach = False

                    in_d = (d_x1 < wx < d_x2) and (d_y1 < wy < d_y2)
                    in_w = (w_x1 < wx < w_x2) and (w_y1 < wy < w_y2)

                    if in_d:
                        p_danger = True
                    elif in_w and is_reach:
                        p_warning = True

                    res = {
                        'valid': True, 'wx': wx, 'wy': wy, 'hx': hx, 'safe_y': safe_y,
                        'in_d': in_d, 'in_w': in_w, 'is_reach': is_reach, 'is_low': is_hand_low,
                        'angle': angle, 'side': arm['side'], 'has_hip': has_hip
                    }
                p_arms_res.append(res)

            if p_danger: global_is_danger = True
            if p_warning: global_is_warning = True

            # ============================================================
            # [핵심 수정] 시각화 필터링 (사람별로 그릴지 말지 결정)
            # ============================================================
            should_draw = True
            if v_alert_only:
                # 감지 시 표시 모드인데, 위험하지도 경고도 아니면 -> 안 그림
                if not (p_danger or p_warning):
                    should_draw = False

            if should_draw:
                # 1. 박스 & 라벨
                if v_bbox:
                    col = (255, 0, 0) if p_danger else ((255, 165, 0) if p_warning else (0, 255, 0))
                    status = "DANGER" if p_danger else ("WARNING" if p_warning else "SAFE")
                    thick = 4 if p_danger else (3 if p_warning else 2)

                    cv2.rectangle(image, (int(bx1), int(by1)), (int(bx2), int(by2)), col, thick)

                    if v_cls:
                        label = f"{class_name}: {status}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(image, (int(bx1), int(by1) - 25), (int(bx1) + tw + 10, int(by1)), col, -1)
                        t_col = (255, 255, 255) if (p_danger or p_warning) else (0, 0, 0)
                        cv2.putText(image, label, (int(bx1) + 5, int(by1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, t_col, 2)

                # 2. 팔 정보 (점, 선, 텍스트)
                for res in p_arms_res:
                    if not res['valid']: continue
                    wx, wy = res['wx'], res['wy']

                    # 안전선
                    if v_line and res['has_hip'] and res['side'] == 'Right':
                        cv2.line(image, (res['hx'] - 40, int(res['safe_y'])), (res['hx'] + 40, int(res['safe_y'])),
                                 (255, 255, 0), 2)

                    # 점
                    dot_col = (0, 255, 0)
                    if res['in_d']:
                        dot_col = (255, 0, 0)
                    elif res['in_w'] and res['is_reach']:
                        dot_col = (255, 165, 0)
                    elif res['is_low']:
                        dot_col = (0, 0, 255)

                    if v_dot:
                        rad = 8 if (res['in_d'] or (res['in_w'] and res['is_reach'])) else 5
                        cv2.circle(image, (wx, wy), rad, dot_col, -1)

                    # 텍스트
                    if v_text:
                        msg = "Safe"
                        if res['in_d']:
                            msg = "TOUCH!"
                        elif res['in_w'] and res['is_reach']:
                            msg = "REACHING"
                        elif res['is_low']:
                            msg = "Low"
                        cv2.putText(image, msg, (wx, wy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dot_col, 2)

    # 상단 상태바 (항상 표시)
    if global_is_danger:
        bar_txt, bar_col, txt_col = "DANGER: TOUCH DETECTED", (255, 0, 0), (255, 255, 255)
    elif global_is_warning:
        bar_txt, bar_col, txt_col = "WARNING: APPROACHING", (255, 165, 0), (0, 0, 0)
    else:
        bar_txt, bar_col, txt_col = "SYSTEM: SAFE", (0, 0, 0), (0, 255, 0)

    cv2.rectangle(image, (0, 0), (800, 60), bar_col, -1)
    cv2.putText(image, bar_txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, txt_col, 2)

    if v_zones:
        cv2.rectangle(image, (w_x1, w_y1), (w_x2, w_y2), (255, 255, 0), 2)
        cv2.rectangle(image, (d_x1, d_y1), (d_x2, d_y2), (255, 0, 0), 3)

    return image


# --- UI 실행 ---
st.set_page_config(layout="wide", page_title="AI 뮤지엄 관리 시스템")
st.title("🏛️ AI 전시품 보호 관리 시스템")

with st.sidebar:
    st.header("📂 영상 관리")
    up = st.file_uploader("새 영상 업로드", type=["mp4", "avi"])
    if up:
        ext = os.path.splitext(up.name)[1]
        path = os.path.join("videos", f"video_{int(time.time())}{ext}")
        with open(path, "wb") as f: f.write(up.getbuffer())
        st.success("저장 완료");
        st.rerun()
    st.markdown("---")
    v_list = [f for f in os.listdir("videos") if f.endswith(('.mp4', '.avi'))]
    sel_v = st.selectbox("🎥 영상 선택", v_list) if v_list else None

if sel_v:
    path = os.path.join("videos", sel_v)
    curr = load_settings(sel_v)
    if os.path.exists('model.pkl'):
        model = joblib.load('model.pkl')
    else:
        st.error("모델 없음"); st.stop()
    try:
        yolo = YOLO('yolov8n-pose.pt')
    except:
        st.error("YOLO 실패"); st.stop()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): st.error("영상 에러"); st.stop()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("⚙️ 설정")
        mode = st.radio("판단 기준", ["Algorithm", "AI", "Both"],
                        index=["Algorithm", "AI", "Both"].index(curr.get('detection_mode', 'Algorithm')),
                        horizontal=True)
        t1, t2, t3 = st.tabs(["📏 구역", "⚡ 감도", "👁️ 시각화"])
        with t1:
            zx = st.slider("가로", 0.0, 1.0, curr['zone_x'], 0.01)
            zy = st.slider("세로", 0.0, 1.0, curr['zone_y'], 0.01)
            zw = st.slider("너비", 0.05, 0.8, curr['zone_w'], 0.01)
            zh = st.slider("높이", 0.05, 0.8, curr['zone_h'], 0.01)
            pd_val = st.slider("경계", 0, 150, curr['padding'])
        with t2:
            eth = st.slider("비율", 0.5, 1.0, curr.get('extension_threshold', 0.85))
            ath = st.slider("각도", 0, 180, curr['angle_threshold'])
            hr = st.slider("안전높이", -0.5, 1.0, curr['hip_ratio'], 0.1)
        with t3:
            st.caption("🚨 : 경고 시에만 표시")
            v_alert = st.toggle("🚨 감지 시에만 표시", value=curr.get('vis_alert_only', False))
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                vb = st.checkbox("🧍 객체 박스", value=curr.get('vis_bbox', True))
                vl = st.checkbox("🏷️ 객체 이름표", value=curr.get('vis_class_label', True))
                vs = st.checkbox("🦴 뼈대", value=curr.get('vis_skeleton', True))
            with c2:
                vz = st.checkbox("🔲 구역 박스", value=curr.get('vis_zones', True))
                vd = st.checkbox("🔴 손목 점", value=curr.get('vis_wrist_dot', True))
                vt = st.checkbox("🔤 상태 글씨", value=curr.get('vis_wrist_text', True))
                vln = st.checkbox("➖ 안전선", value=curr.get('vis_safe_line', True))

        if st.button("💾 저장", use_container_width=True):
            save_settings(sel_v, {
                'zone_x': zx, 'zone_y': zy, 'zone_w': zw, 'zone_h': zh, 'padding': pd_val,
                'angle_threshold': ath, 'hip_ratio': hr, 'extension_threshold': eth, 'detection_mode': mode,
                'vis_alert_only': v_alert, 'vis_skeleton': vs, 'vis_bbox': vb, 'vis_class_label': vl,
                'vis_zones': vz, 'vis_safe_line': vln, 'vis_wrist_text': vt, 'vis_wrist_dot': vd
            });
            st.success("저장됨!")

    with col2:
        st.header("📹 모니터링")
        c1, c2 = st.columns([1, 4])
        with c1:
            run = st.checkbox("▶️ 재생", value=True)
        with c2:
            sf = st.slider("타임라인", 0, max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1), 0)

        frame_ph = st.empty()
        live_set = {
            'zone_x': zx, 'zone_y': zy, 'zone_w': zw, 'zone_h': zh, 'padding': pd_val,
            'angle_threshold': ath, 'hip_ratio': hr, 'extension_threshold': eth, 'detection_mode': mode,
            'vis_alert_only': v_alert, 'vis_skeleton': vs, 'vis_bbox': vb, 'vis_class_label': vl,
            'vis_zones': vz, 'vis_safe_line': vln, 'vis_wrist_text': vt, 'vis_wrist_dot': vd
        }

        cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
        if run:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                frame_ph.image(process_frame(frame, yolo, model, live_set), channels="RGB")
        else:
            ret, frame = cap.read()
            if ret: frame_ph.image(process_frame(frame, yolo, model, live_set), channels="RGB")
    cap.release()