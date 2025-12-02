import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import joblib
import pandas as pd
import math
from ultralytics import YOLO


# 각도 계산 함수
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle


st.set_page_config(layout="wide", page_title="AI 뮤지엄 가드 (Ratio Logic)")
st.title("🏛️ AI 전시품 보호 시스템 (비율 기반 높이 제어)")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("⚙️ 설정 패널")

    # 1. 구역 설정
    st.subheader("1. 구역 설정")
    st.info("빨간 박스를 전시품에 맞추세요.")
    zone_x = st.slider("가로 위치 (X)", 0.0, 1.0, 0.4, 0.01)
    zone_y = st.slider("세로 위치 (Y)", 0.0, 1.0, 0.5, 0.01)
    zone_w = st.slider("너비 (Width)", 0.05, 0.8, 0.15, 0.01)
    zone_h = st.slider("높이 (Height)", 0.05, 0.8, 0.25, 0.01)
    padding = st.slider("경계영역 확장 (Padding)", 0, 150, 50)

    st.markdown("---")

    # 2. 민감도 설정
    st.subheader("2. 동작 민감도")

    # [요청 1] 각도 최소값 0으로 조정
    angle_threshold = st.slider(
        "💪 팔 펴짐 각도 (낮을수록 민감)",
        min_value=0, max_value=180, value=120,
        help="0도에 가까우면 팔을 굽혀도 감지, 180도면 완전히 펴야 감지"
    )

    # [요청 2] 픽셀 대신 비율 사용
    st.subheader("3. 높이 필터 (비율 기반)")
    st.info("손이 '하늘색 선' 아래에 있으면 무조건 안전합니다.")

    hip_ratio = st.slider(
        "안전 높이 조절 (골반 기준)",
        min_value=-0.5, max_value=1.0, value=0.2, step=0.1,
        help="0.0=골반높이, 0.5=배꼽높이, 1.0=어깨높이. (이 값보다 아래면 무시)"
    )

    uploaded_file = st.file_uploader("CCTV 영상 업로드", type=["mp4", "avi"])

if uploaded_file is not None:
    # 모델 로드
    if os.path.exists('model.pkl'):
        custom_model = joblib.load('model.pkl')
    else:
        st.error("model.pkl 파일이 없습니다.")
        st.stop()

    try:
        yolo_model = YOLO('yolov8n-pose.pt')
    except:
        st.error("YOLO 로드 실패")
        st.stop()

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    with col2:
        st.header("📹 실시간 모니터링")
        st_frame = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame, (800, 600))
            h, w, _ = frame.shape

            # 구역 좌표
            d_x1, d_y1 = int(zone_x * w), int(zone_y * h)
            d_x2, d_y2 = int((zone_x + zone_w) * w), int((zone_y + zone_h) * h)
            w_x1, w_y1 = max(0, d_x1 - padding), max(0, d_y1 - padding)
            w_x2, w_y2 = min(w, d_x2 + padding), min(h, d_y2 + padding)

            results = yolo_model(frame, verbose=False, conf=0.5)
            annotated_frame = results[0].plot()
            image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            global_status = "SAFE"
            overlay_color = None
            is_danger = False
            is_warning = False

            if results[0].keypoints is not None:
                keypoints_data = results[0].keypoints.data.cpu().numpy()

                for kps in keypoints_data:
                    arms = [
                        {'side': 'Right', 's': 6, 'e': 8, 'w': 10, 'h': 12},
                        {'side': 'Left', 's': 5, 'e': 7, 'w': 9, 'h': 11}
                    ]

                    for arm in arms:
                        if len(kps) <= arm['h']: continue

                        s, e, w_pt, h_pt = kps[arm['s']], kps[arm['e']], kps[arm['w']], kps[arm['h']]

                        if w_pt[2] < 0.5: continue  # 손목 인식 실패 시 패스

                        wx, wy = int(w_pt[0]), int(w_pt[1])
                        ex, ey = int(e[0]), int(e[1])
                        sx, sy = int(s[0]), int(s[1])  # 어깨 Y
                        hx, hy = int(h_pt[0]), int(h_pt[1])  # 골반 Y

                        # ========================================================
                        # [핵심] 비율 기반 높이 필터 (Ratio Height Filter)
                        # ========================================================
                        # 1. 몸통 길이 계산 (어깨 ~ 골반)
                        torso_height = abs(hy - sy)

                        # 2. 안전 기준선(Threshold) 계산
                        # 골반 위치(hy)에서 몸통 길이 * 비율만큼 위(-)로 올라간 지점
                        # 예: 비율 0.2면 골반보다 몸통의 20%만큼 높은 곳
                        safe_y_limit = hy - (torso_height * hip_ratio)

                        # 3. 손의 위치 판단 (Y가 클수록 아래쪽)
                        # 손목Y(wy)가 기준선(safe_y_limit)보다 크면(아래면) 안전
                        is_hand_low = wy > safe_y_limit

                        # 시각화: 기준선을 하늘색으로 그려줌 (디버깅용)
                        if arm['side'] == 'Right':  # 한 번만 그리기 위해
                            cv2.line(image, (hx - 40, int(safe_y_limit)), (hx + 40, int(safe_y_limit)), (255, 255, 0),
                                     2)
                            cv2.putText(image, "Safe Limit", (hx + 45, int(safe_y_limit)), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.4, (255, 255, 0), 1)

                        # ========================================================

                        # AI 예측
                        input_data = pd.DataFrame([{
                            'rw_x': wx / w, 'rw_y': wy / h, 're_x': ex / w, 're_y': ey / h, 'rs_x': sx / w,
                            'rs_y': sy / h
                        }])
                        ai_pred = custom_model.predict(input_data)[0]

                        # 각도 계산
                        elbow_angle = calculate_angle((sx, sy), (ex, ey), (wx, wy))

                        # 최종 판단:
                        # (각도 만족) AND (AI 뻗음) AND (손이 기준선보다 높음!)
                        is_reaching = (elbow_angle > angle_threshold) and (ai_pred == 1) and (not is_hand_low)

                        # 상태 텍스트
                        status_msg = "Low" if is_hand_low else ("Reach" if is_reaching else "Safe")
                        t_color = (200, 200, 200) if is_hand_low else ((0, 0, 255) if is_reaching else (0, 255, 0))
                        cv2.putText(image, status_msg, (wx, wy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, t_color, 1)

                        in_danger = (d_x1 < wx < d_x2) and (d_y1 < wy < d_y2)
                        in_warning = (w_x1 < wx < w_x2) and (w_y1 < wy < w_y2)

                        if in_danger:
                            # 1순위: 빨간 박스 접촉 (무조건 위험)
                            is_danger = True
                            cv2.circle(image, (wx, wy), 20, (255, 0, 0), -1)
                            cv2.putText(image, "TOUCH!", (wx, wy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                        elif in_warning:
                            # 2순위: 노란 구역 + 뻗음 감지
                            if is_reaching:
                                is_warning = True
                                cv2.circle(image, (wx, wy), 15, (255, 165, 0), -1)
                                cv2.putText(image, "REACHING", (wx, wy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            (255, 165, 0), 2)
                            else:
                                cv2.circle(image, (wx, wy), 8, (0, 0, 255), -1)  # 파란점

            if is_danger:
                global_status = "DANGER (TOUCH)"
                overlay_color = (255, 0, 0)
            elif is_warning:
                global_status = "WARNING (APPROACH)"
                overlay_color = (255, 165, 0)

            cv2.rectangle(image, (w_x1, w_y1), (w_x2, w_y2), (255, 255, 0), 2)
            cv2.rectangle(image, (d_x1, d_y1), (d_x2, d_y2), (255, 0, 0), 3)

            if overlay_color:
                cv2.putText(image, global_status, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, overlay_color, 4)

            cv2.rectangle(image, (0, 0), (800, 50), (0, 0, 0), -1)
            cv2.putText(image, f"SYSTEM: {global_status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            st_frame.image(image, channels="RGB")

    cap.release()

    # streamlit run 3_app.py