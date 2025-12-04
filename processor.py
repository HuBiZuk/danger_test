# processor.py
import cv2
import torch
import joblib
import os
import pandas as pd
import numpy as np
import streamlit as st # Streamlit을 사용하는 함수가 없더라도 @st.cache_resource 때문에 필요할 수 있습니다.
from ultralytics import YOLO
from utils import get_distance, calculate_angle


@st.cache_resource
def get_models():
    try:
        # 모델 파일 경로 확인
        yolo_path = 'yolov8n-pose.pt'   # 다운로드 필요시 자동 다운로드됨 (Ultralytics 기능)
        fire_path = 'smoke_fire_model_hsy_v2.pt'

        yolo = YOLO(yolo_path)

        # 화재모델 로드(파일이 없으면 경고 후 None)
        fire_model = YOLO(fire_path) if os.path.isfile(fire_path) else None
        if not fire_model: st.warning(f"⚠️{fire_path} 파일이 없어 화재 감지가 비활성화 됩니다.")

        custom = joblib.load('model.pkl') if os.path.isfile('model.pkl') else None

        return yolo, custom, fire_model

    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        return None, None


def process_frame(frame, yolo_model, custom_model, fire_model, settings):
    # 분석용 리사이즈
    frame = cv2.resize(frame, (800, 600))
    h, w, _ = frame.shape

    device = 0 if torch.cuda.is_available() else 'cpu'

    # 설정값 풀기
    zones = settings['zones']
    warn_dist = settings['warning_distance']
    ang_th = settings['angle_threshold']
    hip_r = settings['hip_ratio']
    ext_th = settings['extension_threshold']
    mode = settings['detection_mode']
    vis = settings['vis_options']

    # -----------------------------------------
    # 🔥 화재/연기 감지 로직
    # ------------------------------------------
    if settings.get('fire_check', False) and fire_model is not None:
        fire_results = fire_model(frame, verbose=False, conf=0.4, device=device)

        for box in fire_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_name =  fire_model.names[int(box.cls[0])]

            # 그리기(빨간색 박스)
            cv2.rectangle(frame, (x1,y1),(x2,y2),(9,9,255),2)
            cv2.putText(frame,f"{cls_name} {conf:2f}", (x1,y1 -10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
    # ---------------------------------------------------------

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
    for i, z in enumerate(zones):
        if not z.get('active', True): continue # 비활성화 구역 건너뜀

        # zones에 저장된 정규화된 좌표를 픽셀 좌표로 변환하여 사용
        pts = np.array(z['points']) * [w, h]
        pts = pts.astype(np.int32).reshape((-1, 1, 2))
        active_polygons.append(pts) # 모든 활성구역 데이터를 active_polygons에 추가


        if vis['zones']:
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

                    # 제한높이(limit) 선 그리기 (노란색) : 비율로 그림
                    if not vis['alert_only'] and vis['skeleton'] and has_hip and safe_y > 0:
                        torso_h = abs(hy - sy)          # 몸통 길이 계산
                        line_w = int(torso_h * 0.4)     # 선의 절반 길이를 몸통의 40%로 설정
                        line_w = max(10, line_w)        # 최소 길이는 10px

                        cv2.line(image, (sx - line_w, int(safe_y)), (sx + line_w, int(safe_y)), (0,255,255),2)
                        cv2.putText(image, "Limit", (sx - line_w, int(safe_y) -5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)


                    angle = calculate_angle((sx, sy), (ex, ey), (wx, wy))
                    len_u = get_distance((sx, sy), (ex, ey))
                    len_l = get_distance((ex, ey), (wx, wy))
                    ext_r = (get_distance((sx, sy), (wx, wy)) / (len_u + len_l)) if (len_u + len_l) > 0 else 0

                    is_algo = (angle > ang_th) or (ext_r > ext_th)
                    is_ai = False
                    if mode in ['AI', 'OR', 'AND'] and custom_model:
                        inp = pd.DataFrame(
                            [{'rw_x': wx / w, 'rw_y': wy / h, 're_x': ex / w, 're_y': ey / h, 'rs_x': sx / w,
                              'rs_y': sy / h}])
                        try:
                            is_ai = (custom_model.predict(inp)[0] == 1)
                        except:
                            pass

                    # 모드별 최종 판단로직 세분화
                    if mode == 'Algorithm':
                        is_reach = is_algo
                    elif mode == 'AI':
                        is_reach = is_ai
                    elif mode == 'OR':
                        is_reach = is_algo or is_ai
                    elif mode == 'AND':
                        is_reach = is_algo and is_ai
                    else:
                        is_reach = is_algo

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