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


# -----------------------------------------------------------
# [신규] 좌표 정규화 함수 (비율 기반)
# -----------------------------------------------------------
def get_norm_xy(kps):
    """
    절대 좌표를 '골반 중심' & '몸통 크기 비율'의 상대 좌표로 변환합니다.
    """
    data = kps.copy()  # (17, 3)

    # 1. 골반 중심점 (0,0 기준점)
    left_hip = data[11][:2]
    right_hip = data[12][:2]
    center = (left_hip + right_hip) / 2

    # 2. 척추 길이 (몸통 크기) 계산 = 스케일 기준
    left_sh = data[5][:2]
    right_sh = data[6][:2]
    center_sh = (left_sh + right_sh) / 2

    torso_size = np.linalg.norm(center_sh - center)
    scale = torso_size if torso_size > 10 else 1.0

    # 3. 정규화 (좌표 - 중심) / 스케일
    data[:, 0] = (data[:, 0] - center[0]) / scale
    data[:, 1] = (data[:, 1] - center[1]) / scale

    # 4. XY 추출 (Conf 제외)
    xy_only = data[:, :2]

    return xy_only.flatten()


def get_device():
    if torch.cuda.is_available():
        return 0
    else:
        return 'cpu'


@st.cache_resource
def get_models(model_name='yolov8n-pose.pt'):
    try:
        device_status = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        print(f"모델 로딩중...(현재장치: {device_status})")

        fire_path = 'smoke_fire_model_hsy_v2.pt'
        yolo = YOLO(model_name)
        fire_model = YOLO(fire_path) if os.path.isfile(fire_path) else None

        custom = joblib.load('model.pkl') if os.path.isfile('model.pkl') else None

        return yolo, custom, fire_model
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        return None, None, None


def process_frame(frame, yolo_model, custom_model, fire_model, settings):
    ai_result = {"safe": 0.0, "move": 0.0, "threat": 0.0, "is_active": False}
    ai_result_list = []

    device = get_device()
    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape

    # 버퍼 초기화
    if 'pose_buffer' not in st.session_state:
        st.session_state['pose_buffer'] = {}
    if 'threat_cooldown' not in st.session_state or not isinstance(st.session_state['threat_cooldown'], dict):
        st.session_state['threat_cooldown'] = {}

    # 설정값 풀기
    zones = settings['zones']
    warn_dist = settings['warning_distance']
    ang_th = settings['angle_threshold']
    hip_r = settings['hip_ratio']
    ext_th = settings['extension_threshold']
    mode = settings['detection_mode']
    vis = settings['vis_options']
    lock_duration = settings.get('lock_duration', 30)

    is_infinite_mode = (lock_duration > 1000)

    # 락 프레임 계산 30프레임 = 1초
    if is_infinite_mode:
        lock_frames = 999999
    else:
        lock_frames = lock_duration * 30

    # -----------------------------------------
    # 🔥 화재/연기 감지 로직 (기존 유지)
    # ------------------------------------------
    if settings.get('fire_check', False) and fire_model is not None:
        fire_results = fire_model(frame, verbose=False, conf=0.4, device=device)
        for box in fire_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_name = fire_model.names[int(box.cls[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            if 'fire' in cls_name.lower():
                cv2.putText(frame, "FIRE DETECTED!!!", (50, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    results = yolo_model.track(frame, persist=True, verbose=False, conf=0.15, device=device)    # track을 써야 객체마다 고유 ID 가 나옴

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
    global_is_fall = False

    # -----------------------------------------
    # 1. 구역 그리기 (복구됨: fillPoly, dilate)
    # -----------------------------------------
    active_polygons = []
    for i, z in enumerate(zones):
        if not z.get('active', True): continue

        pts = np.array(z['points']) * [w, h]
        pts = pts.astype(np.int32).reshape((-1, 1, 2))
        active_polygons.append(pts)

        if vis['zones']:
            # [복구] 경고 구역 확장 그리기
            if warn_dist > 0:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                k_size = int(warn_dist * 2) + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                expanded_mask = cv2.dilate(mask, kernel)
                contours, _ = cv2.findContours(expanded_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, (255, 255, 0), 2)  # 노란색 경계선


            # 기본 빨간 구역선
            cv2.polylines(image, [pts], True, (255, 0, 0), 2)
            start_pt = tuple(pts[0][0])
            cv2.putText(image, f"#{i + 1}", (start_pt[0], start_pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0),
                        2)

    # -----------------------------------------
    # 2. 분석 로직
    # -----------------------------------------
    if results[0].keypoints is not None and results[0].boxes is not None:
        keypoints_data = results[0].keypoints.data.cpu().numpy()
        boxes_data = results[0].boxes.data.cpu().numpy()

        for box_info, kps in zip(boxes_data, keypoints_data):
            if len(box_info) == 7:
                bx1, by1, bx2, by2,track_id, b_conf, b_cls = box_info[:7]
                track_id = int(track_id)
            else:   # 데이터 이상하면 건너뜀
                continue

            current_timer = st.session_state['threat_cooldown'].get(track_id, 0)

            p_danger = False;
            p_warning = False;
            is_fall = False
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
                    hx = int(hip[0]) if has_hip else 0

                    # [기존 기능] 낙상 감지
                    check_fall_algo = settings.get('fall_check', True)
                    fall_ratio = settings.get('fall_ratio', 1.2)
                    if has_hip and check_fall_algo:
                        body_w = abs(sx - hx)
                        body_h = abs(sy - hy)
                        if body_w > body_h * fall_ratio: is_fall = True
                        if hy <= sy: is_fall = True

                    # [기존 기능] Limit Line
                    safe_y = hy - (abs(hy - sy) * hip_r) if has_hip else 0
                    is_low = (wy > safe_y) if has_hip else False

                    if not vis['alert_only'] and vis['skeleton'] and has_hip and safe_y > 0:
                        torso_h = abs(hy - sy)
                        line_w = int(torso_h * 0.4)
                        line_w = max(10, line_w)
                        cv2.line(image, (sx - line_w, int(safe_y)), (sx + line_w, int(safe_y)), (0, 255, 255), 2)
                        cv2.putText(image, "Limit", (sx - line_w, int(safe_y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (0, 255, 255), 1)

                    # [기존 기능] 알고리즘 감지
                    angle = calculate_angle((sx, sy), (ex, ey), (wx, wy))
                    len_u = get_distance((sx, sy), (ex, ey))
                    len_l = get_distance((ex, ey), (wx, wy))
                    ext_r = (get_distance((sx, sy), (wx, wy)) / (len_u + len_l)) if (len_u + len_l) > 0 else 0
                    is_algo = (angle > ang_th) or (ext_r > ext_th)

                    is_ai_reach = False

                    # =========================================================
                    # 👇 [AI 판단 로직] 30프레임 버퍼 + 비율 정규화 + 슬라이더 적용
                    # =========================================================
                    if mode in ['AI', 'OR', 'AND'] and custom_model:
                        # 슬라이더 값 가져오기
                        ai_th_val = settings.get('ai_threshold', 0.7)

                        # 1. 비율 데이터 추출 (34 features)
                        current_pose = get_norm_xy(kps)

                        if track_id not in st.session_state['pose_buffer']: st.session_state['pose_buffer'][track_id] = []

                        st.session_state['pose_buffer'][track_id].append(current_pose)

                        # 30프레임
                        if len(st.session_state['pose_buffer'][track_id]) > 30:
                            st.session_state['pose_buffer'][track_id].pop(0)


                        # 2. 예측 및 판단
                        if len(st.session_state['pose_buffer'][track_id]) == 30:
                            try:
                                is_threat_now = False

                                seq_data = np.concatenate(st.session_state['pose_buffer'][track_id])
                                cols = [f"v{i}" for i in range(1020)]
                                inp = pd.DataFrame([seq_data], columns=cols)

                                # (1) 확률 계산
                                probs = custom_model.predict_proba(inp)[0]
                                p_safe = probs[0] if len(probs) > 0 else 0
                                p_move = probs[1] if len(probs) > 1 else 0
                                p_threat = probs[2] if len(probs) > 2 else 0

                                ai_result["is_active"] = True
                                ai_result["safe"] = float(p_safe)
                                ai_result["move"] = float(p_move)
                                ai_result["threat"] = float(p_threat)

                                # (2) 1등 라벨 확인
                                max_idx = np.argmax(probs)  # 0:Safe, 1:Move, 2:Threat

                                # 설정값 가져오기
                                ai_th_val = settings.get('ai_threshold', 0.7)

                                current_lock_timer = st.session_state['threat_cooldown'].get(track_id, 0)   # 쿨타임 상태 확인
                                is_new_threat = (max_idx == 2 and p_threat >= ai_th_val)    # 새 위협감지 확인

                                if is_new_threat:   # 위협이 감지되면 락 시간 갱신(리셋)
                                    st.session_state['threat_cooldown'][track_id] = lock_frames
                                    current_timer = lock_frames

                                # (3) 최종 상태 결정 및 텍스트/색상 설정
                                if current_lock_timer > 0:  # 위협 락걸림
                                    is_threat_now = True
                                    text_str = "THREAT (LOCKED)"
                                    text_color = (255, 0, 0)
                                    is_threat_now = False
                                    current_timer = current_lock_timer


                                elif max_idx == 1:  #  이동
                                    is_threat_now = False
                                    text_str = f"Move ({p_move * 100:.0f}%)"
                                    text_color = (255, 255, 0)  # 노란색

                                else:   # 안전
                                    is_threat_now = False
                                    text_str = f"Safe ({p_safe * 100:.0f}%)"
                                    text_color = (0, 255, 0)  # 초록색

                                # (4) 화면 표시
                                if vis['text']:
                                    # 머리 위 라벨 (기존 유지)
                                    cv2.putText(image, f"AI: {text_str}", (sx, sy - 30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

                                    # 왼쪽 하단 대시보드
                                    base_y = h - 120  # 바닥에서 120픽셀 위를 시작점으로 잡음

                                    # 검은 배경 박스
                                    cv2.rectangle(image, (10, base_y), (220, base_y + 80), (0, 0, 0), -1)

                                    # 1. Safe (초록)
                                    cv2.putText(image, f"Safe: {p_safe * 100:.0f}%", (20, base_y + 20), 1, 1,
                                                (0, 255, 0), 1)
                                    cv2.rectangle(image, (100, base_y + 10), (100 + int(p_safe * 100), base_y + 20),
                                                  (0, 255, 0), -1)

                                    # 2. Move (노랑)
                                    cv2.putText(image, f"Move: {p_move * 100:.0f}%", (20, base_y + 45), 1, 1,
                                                (0, 255, 255), 1)
                                    cv2.rectangle(image, (100, base_y + 35), (100 + int(p_move * 100), base_y + 45),
                                                  (0, 255, 255), -1)

                                    # 3. Threat (빨강)
                                    cv2.putText(image, f"Threat: {p_threat * 100:.0f}%", (20, base_y + 70), 1, 1,
                                                (0, 0, 255), 1)
                                    cv2.rectangle(image, (100, base_y + 60),
                                                  (100 + int(p_threat * 100), base_y + 70), (0, 0, 255), -1)

                                # 위험 신호 전달
                                if is_threat_now:
                                    is_ai_reach = True

                            except Exception as e:
                                print(f"에러발생: {e}")
                    # =========================================================

                    # 모드별 최종 판단 통합
                    if mode == 'Algorithm':
                        is_reach = is_algo
                    elif mode == 'AI':
                        is_reach = is_ai_reach
                    elif mode == 'OR':
                        is_reach = is_algo or is_ai_reach
                    elif mode == 'AND':
                        is_reach = is_algo and is_ai_reach
                    else:
                        is_reach = is_algo

                    if is_low: is_reach = False

                    # 구역 진입 체크
                    in_d = False;
                    in_w = False
                    for poly_pts in active_polygons:
                        dist = cv2.pointPolygonTest(poly_pts, (wx, wy), True)
                        if dist >= 0:
                            in_d = True
                        elif dist >= -warn_dist:
                            in_w = True

                    # [중요] 손이 제한선 아래가 아닐 때만 경고
                    if not is_low:
                        if in_d and is_reach:
                            p_danger = True
                            st.session_state['threat_cooldown'][track_id] = lock_frames

                        elif in_w and is_reach:
                            p_warning = True

                    wrist_points.append(
                        {'x': wx, 'y': wy, 'state': 'D' if in_d else ('W' if in_w and is_reach else 'S')})


            #==========================================
            # 최종 상태 판단 및 박스 그리기 로직
            #------------------------------------------

            # 현재 남은 쿨타임 확인
            is_locked_threat = False
            if current_timer > 0:
                is_locked_threat = True
                # [수정] 무한 아니면 감지된 객체 시간만 감소
                if not is_infinite_mode:
                    st.session_state['threat_cooldown'][track_id] -= 1

            # 전체 상태 플래그 업데이트
            if p_danger or is_locked_threat:
                global_is_danger = True
            elif p_warning:
                global_is_warning = True
            if is_fall: global_is_fall = True

            # 결과 리스트에 담기 (view 대시보드용)
            status_code = "Safe"
            if p_danger or is_locked_threat: status_code = "Threat"
            elif p_warning: status_code = "Reach"
            elif 'max_idx' in locals() and max_idx == 1: status_code = "Move"

            ai_result_list.append({
                "id": track_id,
                "status": status_code,
                "threat": p_threat if 'p_threat' in locals() else 0,
                "is_locked": is_locked_threat
            })

            # 박스 그리기 여부 결정
            draw_box = True
            if vis['alert_only'] and not (global_is_danger or is_locked_threat or global_is_warning or is_fall):
                draw_box = False

            if draw_box:
                # 색상 및 텍스트 우선순위 결정
                # 1순위: 낙상
                if is_fall:
                    c, txt = (255, 0, 255), f"FALL {track_id}"
                # 2순위: 위협 (현재 감지됨 OR 락 걸림) -> 무조건 빨강/THREAT
                elif p_danger or is_locked_threat:
                    c, txt = (255, 0, 0), f"THREAT {track_id}"
                    # 3순위: 접근 경고
                elif p_warning:
                    c, txt = (255, 165, 0), f"REACH {track_id}"
                # 4순위: 안전
                else:
                    c, txt = (0, 255, 0), f"Safe {track_id}"

                # 실제 그리기
                if vis['bbox']: cv2.rectangle(image, (int(bx1), int(by1)), (int(bx2), int(by2)), c, 2)
                if vis['label']: cv2.putText(image, txt, (int(bx1), int(by1) - 5), 1, 1.5, c, 2)

                # 손목 점 그리기
                if vis['wrist_dot']:
                    for wp in wrist_points:
                        wc = (0, 255, 0)
                        if wp['state'] == 'D':
                            wc = (255, 0, 0)
                        elif wp['state'] == 'W':
                            wc = (255, 165, 0)
                        cv2.circle(image, (wp['x'], wp['y']), 6, wc, -1)



    # 상단 상태바
    if global_is_fall:
        bar, msg = (255, 0, 255), "EMERGENCY: FALL DETECTED"
    elif global_is_danger:
        bar, msg = (255, 0, 0), "DANGER: TOUCH DETECTED"
    elif global_is_warning:
        bar, msg = (255, 165, 0), "WARNING: APPROACHING"
    else:
        bar, msg = (50, 50, 50), "SYSTEM: SAFE"

    cv2.rectangle(image, (0, 0), (w, 40), bar, -1)
    cv2.putText(image, msg, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return image, ai_result