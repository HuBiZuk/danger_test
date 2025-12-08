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

# utils.apply_streamlit_patch()  # 이미지 호환성 패치 적용 - Streamlit 1.22.0에서는 필요 없으므로 제거 또는 주석 처리

st.title("🏛️ AI 전시품 보호 관리 시스템")

# 2. 사이드바 (파일 선택)
sel_v, sel_model_name = view.render_sidebar()

# 3. 모델 로드
yolo_model, custom_model, fire_model = processor.get_models(sel_model_name)

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
    # st.tabs는 Streamlit 1.22.0에서 지원되지 않습니다. st.radio로 대체합니다.
    selected_tab = st.radio(
        "설정 탭 선택",
        ["📝 구역 관리", "⚡ 감도 설정", "👁️ 시각화 설정"],
        key="main_tabs",
        horizontal=True # 탭처럼 보이게 하기 위해 가로 정렬
    )

    if selected_tab == "📝 구역 관리":
        view.render_zone_tab(sel_v, curr_settings, video_path)
    elif selected_tab == "⚡ 감도 설정":
        wd, et, at, md, hr, fire_check, fall_check, fr, ai_th = view.render_sensitivity_tab(sel_v, curr_settings)
        st.session_state['wd'] = wd
        st.session_state['et'] = et
        st.session_state['at'] = at
        st.session_state['md'] = md
        st.session_state['hr'] = hr
        st.session_state['fire_check'] = fire_check
        st.session_state['fall_check'] = fall_check
        st.session_state['fr'] = fr
        st.session_state['ai_th'] = ai_th  # 👈 [추가] 세션에 저장


    elif selected_tab == "👁️ 시각화 설정":
        # render_vis_tab에서 리턴값을 받아야 함
        check_alert, v_skel, v_zone, v_box, v_dot, v_txt = view.render_vis_tab(sel_v, curr_settings)
        st.session_state['check_alert'] = check_alert
        st.session_state['v_skel'] = v_skel
        st.session_state['v_zone'] = v_zone
        st.session_state['v_box'] = v_box
        st.session_state['v_dot'] = v_dot
        st.session_state['v_txt'] = v_txt


# --- [오른쪽] 모니터링 화면 ---
with right_col:
    st.subheader("📹 실시간 모니터링")

    col_p1, col_p2 = st.columns([3, 1])
    with col_p2:
        run_monitor = st.checkbox("▶ 재생", value=True)
    with col_p1:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # st.slider의 label_visibility는 Streamlit 1.22.0에서 지원되지 않습니다.
        frame_idx = st.slider("탐색", 0, total_frames, 0) # label=""로 레이블 명시적 제거


    st_screen = st.empty()

    # 실시간 설정을 반영하기 위한 딕셔너리 구성
    live_settings = curr_settings.copy()

    live_settings['warning_distance'] = st.session_state.get('wd', curr_settings.get("warning_distance", 30))
    live_settings['extension_threshold'] = st.session_state.get('et', curr_settings.get("extension_threshold", 0.7))
    live_settings['angle_threshold'] = st.session_state.get('at', curr_settings.get("angle_threshold", 120))
    live_settings['detection_mode'] = st.session_state.get('md', curr_settings.get("detection_mode", "Algorithm"))
    live_settings['hip_ratio'] = st.session_state.get('hr', curr_settings.get("hip_ratio", 0.2))
    live_settings['fire_check'] = st.session_state.get('fire_check', curr_settings.get("fire_check", False))
    live_settings['fall_check'] = st.session_state.get('fall_check', curr_settings.get("fall_check", True))
    live_settings['fall_ratio'] = st.session_state.get('fr', curr_settings.get("fall_ratio", 1.2))
    live_settings['ai_threshold'] = st.session_state.get('ai_th', curr_settings.get("ai_threshold", 0.7))




    live_settings['vis_options'] = curr_settings.get('vis_options', {
        'alert_only': False, 'skeleton': True, 'zones': True,
        'bbox': True, 'label': True, 'wrist_dot': True, 'text': True
    })
    live_settings['vis_options']['alert_only'] = st.session_state.get('check_alert', live_settings['vis_options']['alert_only'])
    live_settings['vis_options']['skeleton'] = st.session_state.get('v_skel', live_settings['vis_options']['skeleton'])
    live_settings['vis_options']['zones'] = st.session_state.get('v_zone', live_settings['vis_options']['zones'])
    live_settings['vis_options']['bbox'] = st.session_state.get('v_box', live_settings['vis_options']['bbox'])
    live_settings['vis_options']['wrist_dot'] = st.session_state.get('v_dot', live_settings['vis_options']['wrist_dot'])
    live_settings['vis_options']['text'] = st.session_state.get('v_txt', live_settings['vis_options']['text'])


    # 재생 루프
    if run_monitor:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 무한 반복
                continue

            # processor 모듈에 위임
            out_img = processor.process_frame(frame, yolo_model, custom_model, fire_model, live_settings)
            st_screen.image(out_img, channels="RGB")

            time.sleep(0.01)  # CPU 점유율 조절
    else:
        # 일시 정지 상태
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            out_img = processor.process_frame(frame, yolo_model, custom_model, fire_model, live_settings)
            st_screen.image(out_img, channels="RGB")

    cap.release()

    # streamlit run main.py