# main.py
import streamlit as st
import cv2
import os
import time

# 모듈 임포트
import utils
import processor
import view

# 1. 초기화 및 설정
st.set_page_config(layout="wide", page_title="AI 전시품 보호 시스템 v3")
utils.init_directories()
utils.apply_streamlit_patch()  # 이미지 호환성 패치 적용

st.title("🏛️ AI 전시품 보호 관리 시스템")

# 2. 모델 로드
yolo_model, custom_model = processor.get_models()

# 3. 사이드바 (파일 선택)
sel_v = view.render_sidebar()

if not sel_v or not yolo_model:
    st.warning("영상을 선택하거나 모델을 확인해주세요.")
    st.stop()

# 4. 설정 로드
video_path = os.path.join("videos", sel_v)
curr_settings = utils.load_settings(sel_v)

# 5. 메인 레이아웃 (좌우 분할)
left_col, right_col = st.columns([1, 1], gap="medium")

# --- [왼쪽] 설정 및 편집 화면 ---
with left_col:
    tab1, tab2, tab3 = st.tabs(["📝 구역 관리", "⚡ 감도 설정", "👁️ 시각화 설정"])

    with tab1:
        view.render_zone_tab(sel_v, curr_settings, video_path)
    with tab2:
        wd, et, at, md = view.render_sensitivity_tab(sel_v, curr_settings)
    with tab3:
        check_alert, v_skel, v_zone, v_box, v_dot, v_txt = view.render_vis_tab(sel_v, curr_settings)

# --- [오른쪽] 모니터링 화면 ---
with right_col:
    st.subheader("📹 실시간 모니터링")

    col_p1, col_p2 = st.columns([3, 1])
    with col_p2:
        run_monitor = st.checkbox("▶ 재생", value=True)
    with col_p1:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = st.slider("탐색", 0, total_frames, 0, label_visibility="collapsed")

    st_screen = st.empty()

    # 실시간 설정을 반영하기 위한 딕셔너리 구성
    live_settings = curr_settings.copy()
    live_settings['warning_distance'] = wd
    live_settings['extension_threshold'] = et
    live_settings['angle_threshold'] = at
    live_settings['detection_mode'] = md
    live_settings['vis_options'] = {
        'alert_only': check_alert, 'skeleton': v_skel, 'zones': v_zone,
        'bbox': v_box, 'label': True, 'wrist_dot': v_dot, 'text': v_txt
    }

    # 재생 루프
    if run_monitor:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 무한 반복
                continue

            # processor 모듈에 위임
            out_img = processor.process_frame(frame, yolo_model, custom_model, live_settings)
            st_screen.image(out_img, channels="RGB")

            time.sleep(0.01)  # CPU 점유율 조절
    else:
        # 일시 정지 상태
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            out_img = processor.process_frame(frame, yolo_model, custom_model, live_settings)
            st_screen.image(out_img, channels="RGB")

    cap.release()

    # streamlit run main.py