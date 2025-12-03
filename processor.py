# processor.py
import cv2
import torch
import joblib
import os
import pandas as pd
import numpy as np
import streamlit as st
from ultralytics import YOLO
from utils import get_distance, calculate_angle


@st.cache_resource
def get_models():
    try:
        # 모델 파일 경로 확인
        yolo_path = 'yolov8n-pose.pt'
        # 다운로드 필요시 자동 다운로드됨 (Ultralytics 기능)

        yolo = YOLO(yolo_path)
        custom = joblib.load('model.pkl') if os.path.isfile('model.pkl') else None
        return yolo, custom
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        return None, None


def process_frame(frame, yolo_model, custom_model, settings):
    # 분석용 리사이즈
    frame = cv2.resize(frame, (800, 600))
    h, w, _ = frame.shape

    # 설정값 풀기
    zones = settings['zones']
    warn_dist = settings['warning_distance']
    ang_th = settings['angle_threshold']
    hip_r = settings['hip_ratio']
    ext_th = settings['extension_threshold']
    mode = settings['detection_mode']
    vis = settings['vis_options']

    device = 0 if torch.cuda.is_available() else 'cpu'
    results = yolo_model(frame, verbose=False, conf=0.25, device=device)

    # 배경 생성
    if vis['alert_only']:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        if vis['skeleton']:
            annotated_frame = results[0].plot(boxes=False, probs=False)
            image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    global_is_danger = False
    global_is_warning = False

    # 1. 구역 그리기
    active_polygons = []
    if vis['zones']:
        for i, z in enumerate(zones):
            if not z.get('active', True): continue

            pts = np.array(z['points']) * [w, h]
            pts = pts.astype(np.int32).reshape((-1, 1, 2))
            active_polygons.append(pts)

            if warn_dist > 0:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                k_size = int(warn_dist * 2) + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                expanded_mask = cv2.dilate(mask, kernel)
                contours, _ = cv2.findContours(expanded_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, (255, 255, 0), 2)

            cv2.polylines(image, [pts], True, (255, 0, 0), 2)
            start_pt = tuple(pts[0][0])
            cv2.putText(image, f"#{i + 1}", (start_pt[0], start_pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0),
                        2)

    # 2. 분석 로직
    if results[0].keypoints is not None and results[0].boxes is not None:
        keypoints_data = results[0].keypoints.data.cpu().numpy()
        boxes_data = results[0].boxes.data.cpu().numpy()

        for box_info, kps in zip(boxes_data, keypoints_data):
            bx1, by1, bx2, by2, b_conf, b_cls = box_info

            p_danger = False
            p_warning = False
            wrist_points = []

            # 팔 정의 (우, 좌)
            arms = [{'side': 'Right', 's': 6, 'e': 8, 'w': 10, 'h': 12},
                    {'side': 'Left', 's': 5, 'e': 7, 'w': 9, 'h': 11}]

            for arm in arms:
                if len(kps) > arm['h'] and kps[arm['w']][2] >= 0.25:
                    s, e, wrist, hip = kps[arm['s']], kps[arm['e']], kps[arm['w']], kps[arm['h']]
                    wx, wy = int(wrist[0]), int(wrist[1])
                    sx, sy = int(s[0]), int(s[1])
                    ex, ey = int(e[0]), int(e[1])

                    has_hip = hip[2] > 0.25
                    hy = int(hip[1]) if has_hip else 0
                    safe_y = hy - (abs(hy - sy) * hip_r) if has_hip else 0
                    is_low = (wy > safe_y) if has_hip else False

                    angle = calculate_angle((sx, sy), (ex, ey), (wx, wy))
                    len_u = get_distance((sx, sy), (ex, ey))
                    len_l = get_distance((ex, ey), (wx, wy))
                    ext_r = (get_distance((sx, sy), (wx, wy)) / (len_u + len_l)) if (len_u + len_l) > 0 else 0

                    is_algo = (angle > ang_th) or (ext_r > ext_th)
                    is_ai = False
                    if mode in ['AI', 'Both'] and custom_model:
                        inp = pd.DataFrame(
                            [{'rw_x': wx / w, 'rw_y': wy / h, 're_x': ex / w, 're_y': ey / h, 'rs_x': sx / w,
                              'rs_y': sy / h}])
                        try:
                            is_ai = (custom_model.predict(inp)[0] == 1)
                        except:
                            pass

                    is_reach = is_algo if mode == 'Algorithm' else (is_ai if mode == 'AI' else (is_algo and is_ai))
                    if is_low: is_reach = False

                    in_d = False
                    in_w = False
                    for poly_pts in active_polygons:
                        dist = cv2.pointPolygonTest(poly_pts, (wx, wy), True)
                        if dist >= 0:
                            in_d = True
                        elif dist >= -warn_dist:
                            in_w = True

                    if in_d:
                        p_danger = True
                    elif in_w and is_reach:
                        p_warning = True

                    wrist_points.append(
                        {'x': wx, 'y': wy, 'state': 'D' if in_d else ('W' if in_w and is_reach else 'S')})

            if p_danger: global_is_danger = True
            if p_warning: global_is_warning = True

            draw_box = True
            if vis['alert_only'] and not (p_danger or p_warning): draw_box = False

            if draw_box:
                color = (255, 0, 0) if p_danger else ((255, 165, 0) if p_warning else (0, 255, 0))
                if vis['bbox']:
                    cv2.rectangle(image, (int(bx1), int(by1)), (int(bx2), int(by2)), color, 2)
                    if vis['label']:
                        status = "TOUCH" if p_danger else ("REACH" if p_warning else "Safe")
                        cv2.putText(image, status, (int(bx1), int(by1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if vis['wrist_dot']:
                    for wp in wrist_points:
                        c = (0, 255, 0)
                        if wp['state'] == 'D':
                            c = (255, 0, 0)
                        elif wp['state'] == 'W':
                            c = (255, 165, 0)
                        cv2.circle(image, (wp['x'], wp['y']), 6, c, -1)

    # 상태바
    if global_is_danger:
        bar, msg, tc = (255, 0, 0), "DANGER: TOUCH DETECTED", (255, 255, 255)
    elif global_is_warning:
        bar, msg, tc = (255, 165, 0), "WARNING: APPROACHING", (0, 0, 0)
    else:
        bar, msg, tc = (50, 50, 50), "SYSTEM: SAFE", (0, 255, 0)

    cv2.rectangle(image, (0, 0), (w, 40), bar, -1)
    cv2.putText(image, msg, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tc, 2)

    return image