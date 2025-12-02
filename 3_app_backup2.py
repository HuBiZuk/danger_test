#  streamlit run 3_app.py
import streamlit as st
import cv2
import numpy as np
import os
import joblib
import pandas as pd
import json
import time
from ultralytics import YOLO

# --- [초기 설정] 폴더 생성 ---
if not os.path.exists('videos'):
    os.makedirs('videos')
if not os.path.exists('settings'):
    os.makedirs('settings')


# --- [함수] 각도 계산 ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle


# --- [함수] 설정 로드 / 저장 ---
def load_settings(video_name):
    json_path = os.path.join('settings', f"{video_name}.json")
    default_settings = {
        'zone_x': 0.4, 'zone_y': 0.5, 'zone_w': 0.15, 'zone_h': 0.25,
        'padding': 50, 'angle_threshold': 120, 'hip_ratio': 0.2,
        'use_ai': True
    }
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return default_settings


def save_settings(video_name, settings):
    json_path = os.path.join('settings', f"{video_name}.json")
    with open(json_path, 'w') as f:
        json.dump(settings, f)


# --- [함수] 프레임 분석 (통합 로직) ---
def process_frame(frame, yolo_model, custom_model, settings):
    # 화면 크기 고정
    frame = cv2.resize(frame, (800, 600))
    h, w, _ = frame.shape

    # 설정값 풀기
    z_x, z_y = settings['zone_x'], settings['zone_y']
    z_w, z_h = settings['zone_w'], settings['zone_h']
    pad = settings['padding']
    ang_th = settings['angle_threshold']
    hip_r = settings['hip_ratio']
    use_ai = settings.get('use_ai', True)

    # 구역 좌표 계산
    d_x1, d_y1 = int(z_x * w), int(z_y * h)
    d_x2, d_y2 = int((z_x + z_w) * w), int((z_y + z_h) * h)
    w_x1, w_y1 = max(0, d_x1 - pad), max(0, d_y1 - pad)
    w_x2, w_y2 = min(w, d_x2 + pad), min(h, d_y2 + pad)

    # YOLO 추론
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
                if w_pt[2] < 0.5: continue

                wx, wy = int(w_pt[0]), int(w_pt[1])
                ex, ey = int(e[0]), int(e[1])
                sx, sy = int(s[0]), int(s[1])

                # 1. 높이 필터 (골반이 보일 때만 작동)
                has_hip = h_pt[2] > 0.5
                is_hand_low = False

                if has_hip:
                    hx, hy = int(h_pt[0]), int(h_pt[1])
                    torso_height = abs(hy - sy)
                    safe_y_limit = hy - (torso_height * hip_r)
                    is_hand_low = wy > safe_y_limit

                    # 안전선 그리기
                    if arm['side'] == 'Right':
                        cv2.line(image, (hx - 40, int(safe_y_limit)), (hx + 40, int(safe_y_limit)), (255, 255, 0), 2)
                        cv2.putText(image, "Limit", (hx + 45, int(safe_y_limit)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (255, 255, 0), 1)

                # 2. AI 예측
                cond_ai = True  # AI 미사용 시 기본 통과
                if use_ai:
                    input_data = pd.DataFrame([{
                        'rw_x': wx / w, 'rw_y': wy / h, 're_x': ex / w, 're_y': ey / h, 'rs_x': sx / w, 'rs_y': sy / h
                    }])
                    ai_pred = custom_model.predict(input_data)[0]
                    cond_ai = (ai_pred == 1)

                # 3. 각도 계산
                elbow_angle = calculate_angle((sx, sy), (ex, ey), (wx, wy))
                cond_angle = elbow_angle > ang_th

                # 4. 최종 판단
                # (각도 만족) AND (AI 만족) AND (손이 낮지 않음)
                is_reaching = cond_angle and cond_ai and (not is_hand_low)

                # 상태 텍스트 시각화
                if is_reaching:
                    status_msg = f"REACH ({int(elbow_angle)})"
                    t_color = (0, 0, 255)
                else:
                    if is_hand_low:
                        status_msg = "Low"
                    elif not has_hip:
                        status_msg = "NoHip"  # 골반 없음(근접)
                    elif not cond_angle:
                        status_msg = f"Bend({int(elbow_angle)})"
                    elif not cond_ai:
                        status_msg = "AI:0"
                    else:
                        status_msg = "Safe"
                    t_color = (200, 200, 200)

                cv2.putText(image, status_msg, (wx, wy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, t_color, 1)

                # 구역 체크
                in_danger_zone = (d_x1 < wx < d_x2) and (d_y1 < wy < d_y2)
                in_warning_zone = (w_x1 < wx < w_x2) and (w_y1 < wy < w_y2)

                if in_danger_zone:
                    is_danger = True
                    cv2.circle(image, (wx, wy), 20, (255, 0, 0), -1)
                    cv2.putText(image, "TOUCH!", (wx, wy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                elif in_warning_zone and is_reaching:
                    is_warning = True
                    cv2.circle(image, (wx, wy), 15, (255, 165, 0), -1)
                    cv2.putText(image, "REACHING", (wx, wy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                elif in_warning_zone:
                    cv2.circle(image, (wx, wy), 8, (0, 0, 255), -1)

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

    return image


# --- 페이지 설정 ---
st.set_page_config(layout="wide", page_title="AI 뮤지엄 관리 시스템")
st.title("🏛️ AI 전시품 보호 관리 시스템 (최종 완성판)")

# --- 사이드바 ---
with st.sidebar:
    st.header("📂 영상 관리")
    uploaded_file = st.file_uploader("새 영상 업로드", type=["mp4", "avi"])

    # 한글 파일명 오류 방지: 시간기반 저장
    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1]
        safe_filename = f"video_{int(time.time())}{file_ext}"
        save_path = os.path.join("videos", safe_filename)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"저장 완료: {safe_filename}")
        st.rerun()

    st.markdown("---")
    video_list = [f for f in os.listdir("videos") if f.endswith(('.mp4', '.avi'))]
    selected_video = None
    if video_list:
        selected_video = st.selectbox("🎥 분석할 영상 선택", video_list)
    else:
        st.warning("영상을 업로드 하세요")

# --- 메인 로직 ---
if selected_video:
    video_path = os.path.join("videos", selected_video)
    current_settings = load_settings(selected_video)

    # 모델 로드
    if os.path.exists('model.pkl'):
        custom_model = joblib.load('model.pkl')
    else:
        st.error("model.pkl 파일이 없습니다.")
        st.stop()

    try:
        yolo_model = YOLO('yolov8n-pose.pt')
    except:
        st.error("YOLO 모델 로드 실패")
        st.stop()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"영상을 열 수 없습니다: {video_path}")
        st.stop()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("⚙️ 구역 및 감도 설정")

        # 0. AI 옵션
        st.subheader("0. 감지 모드")
        use_ai = st.checkbox("🤖 AI 판단 사용 (오작동 시 해제)",
                             value=current_settings.get('use_ai', True))

        # 1. 구역 설정
        st.subheader("1. 구역 설정")
        z_x = st.slider("가로 (X)", 0.0, 1.0, current_settings['zone_x'], 0.01)
        z_y = st.slider("세로 (Y)", 0.0, 1.0, current_settings['zone_y'], 0.01)
        z_w = st.slider("너비", 0.05, 0.8, current_settings['zone_w'], 0.01)
        z_h = st.slider("높이", 0.05, 0.8, current_settings['zone_h'], 0.01)
        pad = st.slider("경계 확장", 0, 150, current_settings['padding'])

        st.markdown("---")
        # 2. 감도 설정
        st.subheader("2. 감도 설정")
        ang_th = st.slider("팔 펴짐 각도", 0, 180, current_settings['angle_threshold'])
        hip_r = st.slider("안전 높이 (골반 비율)", -0.5, 1.0, current_settings['hip_ratio'], 0.1)

        if st.button("💾 설정 저장"):
            new_settings = {
                'zone_x': z_x, 'zone_y': z_y, 'zone_w': z_w, 'zone_h': z_h,
                'padding': pad, 'angle_threshold': ang_th, 'hip_ratio': hip_r,
                'use_ai': use_ai
            }
            save_settings(selected_video, new_settings)
            st.success("설정 저장됨!")

    with col2:
        st.header("📹 실시간 모니터링")

        c1, c2 = st.columns([1, 4])
        with c1:
            run = st.checkbox("▶️ 재생", value=True)
        with c2:
            start_frame = st.slider("타임라인", 0, max(0, total_frames - 1), 0)

        st_frame = st.empty()

        live_settings = {
            'zone_x': z_x, 'zone_y': z_y, 'zone_w': z_w, 'zone_h': z_h,
            'padding': pad, 'angle_threshold': ang_th, 'hip_ratio': hip_r,
            'use_ai': use_ai
        }

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        if run:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                result_img = process_frame(frame, yolo_model, custom_model, live_settings)
                st_frame.image(result_img, channels="RGB")
        else:
            ret, frame = cap.read()
            if ret:
                result_img = process_frame(frame, yolo_model, custom_model, live_settings)
                st_frame.image(result_img, channels="RGB")
            else:
                st.info("로딩 중...")

    cap.release()
