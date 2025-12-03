import streamlit as st
import cv2
import numpy as np
import os
import joblib
import pandas as pd
import json
import time
import math
import torch
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas
from PIL import Image

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


# --- [함수] 설정 (다중구역) ---
def load_settings(video_name):
    json_path = os.path.join('settings', f"{video_name}.json")
    default_settings = {
        'zones': [{'id': 1, 'x': 0.4, 'y': 0.4, 'w': 0.2, 'h': 0.2, 'active': True}],
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
        try:
            with open(json_path, 'r') as f:
                saved = json.load(f)
                if 'zones' not in saved:
                    saved['zones'] = [{
                        'id': 1,
                        'x': saved.get('zone_x', 0.4),
                        'y': saved.get('zone_y', 0.4),
                        'w': saved.get('zone_w', 0.2),
                        'h': saved.get('zone_h', 0.2),
                        'active': True
                    }]
                return saved
        except:
            return default_settings
    return default_settings


def save_settings(video_name, settings):
    json_path = os.path.join('settings', f"{video_name}.json")
    with open(json_path, 'w') as f: json.dump(settings, f)


# --- [함수] 프레임 분석 ---
def process_frame(frame, yolo_model, custom_model, settings):
    frame = cv2.resize(frame, (800, 600))
    h, w, _ = frame.shape

    # 설정값 로드
    zones = settings.get('zones', [])
    pad = settings['padding']
    ang_th = settings['angle_threshold']
    hip_r = settings['hip_ratio']
    ext_th = settings.get('extension_threshold', 0.85)
    mode = settings.get('detection_mode', 'Algorithm')

    # 시각화 변수
    v_alert_only = settings.get('vis_alert_only', False)
    v_skel = settings.get('vis_skeleton', True)
    v_bbox = settings.get('vis_bbox', True)
    v_cls = settings.get('vis_class_label', True)
    v_line = settings.get('vis_safe_line', True)
    v_dot = settings.get('vis_wrist_dot', True)
    v_text = settings.get('vis_wrist_text', True)
    v_zones = settings.get('vis_zones', True)

    # YOLO 추론
    device = 0 if torch.cuda.is_available() else 'cpu'
    results = yolo_model(frame, verbose=False, conf=0.25, device=device)

    # 배경 이미지 생성
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

    # 분석
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

                    # 엉덩이 및 높이 계산
                    has_hip = h_pt[2] > 0.25
                    is_hand_low = False
                    safe_y = 0
                    hx, hy = 0, 0
                    if has_hip:
                        hx, hy = int(h_pt[0]), int(h_pt[1])
                        safe_y = hy - (abs(hy - sy) * hip_r)
                        is_hand_low = wy > safe_y

                    # 팔 뻗음 감지
                    angle = calculate_angle((sx, sy), (ex, ey), (wx, wy))
                    len_u, len_l = get_distance((sx, sy), (ex, ey)), get_distance((ex, ey), (wx, wy))
                    ext_r = get_distance((sx, sy), (wx, wy)) / (len_u + len_l) if (len_u + len_l) > 0 else 0

                    is_algo = (angle > ang_th) or (ext_r > ext_th)
                    is_ai = False
                    if mode in ['AI', 'Both'] and custom_model:
                        inp = pd.DataFrame(
                            [{'rw_x': wx / w, 'rw_y': wy / h, 're_x': ex / w, 're_y': ey / h, 'rs_x': sx / w,
                              'rs_y': sy / h}])
                        is_ai = (custom_model.predict(inp)[0] == 1)

                    is_reach = is_algo if mode == 'Algorithm' else (is_ai if mode == 'AI' else (is_algo and is_ai))
                    if is_hand_low: is_reach = False

                    # 다중구역 충돌체크
                    in_any_d = False
                    in_any_w = False

                    for zone in zones:
                        if not zone.get('active', True): continue

                        zx1 = int(zone['x'] * w)
                        zy1 = int(zone['y'] * h)
                        zx2 = int((zone['x'] + zone['w']) * w)
                        zy2 = int((zone['y'] + zone['h']) * h)

                        wx1, wy1 = max(0, zx1 - pad), max(0, zy1 - pad)
                        wx2, wy2 = min(w, zx2 + pad), min(h, zy2 + pad)

                        if (zx1 < wx < zx2) and (zy1 < wy < zy2):
                            in_any_d = True
                        elif (wx1 < wx < wx2) and (wy1 < wy < wy2):
                            in_any_w = True

                    if in_any_d:
                        p_danger = True
                    elif in_any_w and is_reach:
                        p_warning = True

                    res = {
                        'valid': True, 'wx': wx, 'wy': wy, 'hx': hx, 'safe_y': safe_y,
                        'in_d': in_any_d, 'in_w': in_any_w, 'is_reach': is_reach, 'is_low': is_hand_low,
                        'side': arm['side'], 'has_hip': has_hip
                    }
                p_arms_res.append(res)

            if p_danger: global_is_danger = True
            if p_warning: global_is_warning = True

            # 시각화 그리기
            should_draw = True
            if v_alert_only and not (p_danger or p_warning): should_draw = False

            if should_draw:
                if v_bbox:
                    col = (255, 0, 0) if p_danger else ((255, 165, 0) if p_warning else (0, 255, 0))
                    status = "DANGER" if p_danger else ("WARNING" if p_warning else "SAFE")
                    thick = 4 if p_danger else (3 if p_warning else 2)
                    cv2.rectangle(image, (int(bx1), int(by1)), (int(bx2), int(by2)), col, thick)
                    if v_cls:
                        label = f"{class_name}: {status}"
                        cv2.rectangle(image, (int(bx1), int(by1) - 20), (int(bx1) + 150, int(by1)), col, -1)
                        cv2.putText(image, label, (int(bx1), int(by1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 255), 2)

                for res in p_arms_res:
                    if not res['valid']: continue
                    wx, wy = res['wx'], res['wy']

                    if v_line and res['has_hip'] and res['side'] == 'Right':
                        cv2.line(image, (res['hx'] - 40, int(res['safe_y'])), (res['hx'] + 40, int(res['safe_y'])),
                                 (255, 255, 0), 2)

                    dot_col = (0, 255, 0)
                    msg = "Safe"

                    if res['in_d']:
                        dot_col, msg = (255, 0, 0), "TOUCH!"
                    elif res['in_w'] and res['is_reach']:
                        dot_col, msg = (255, 165, 0), "REACH"
                    elif res['is_low']:
                        dot_col, msg = (0, 0, 255), "LOW"

                    if v_dot:
                        radius = 8 if (res['in_d'] or (res['in_w'] and res['is_reach'])) else 5
                        cv2.circle(image, (wx, wy), radius, dot_col, -1)

                    if v_text:
                        cv2.putText(image, msg, (wx, wy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dot_col, 2)

    # 상태바
    if global_is_danger:
        bar_col, txt, t_col = (255, 0, 0), "DANGER: TOUCH DETECTED", (255, 255, 255)
    elif global_is_warning:
        bar_col, txt, t_col = (255, 165, 0), "WARNING: APPROACHING", (0, 0, 0)
    else:
        bar_col, txt, t_col = (50, 50, 50), "SYSTEM: SAFE", (0, 255, 0)

    cv2.rectangle(image, (0, 0), (w, 50), bar_col, -1)
    cv2.putText(image, txt, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, t_col, 2)

    if v_zones:
        for idx, zone in enumerate(zones):
            if not zone.get('active', True): continue

            zx1 = int(zone['x'] * w)
            zy1 = int(zone['y'] * h)
            zx2 = int((zone['x'] + zone['w']) * w)
            zy2 = int((zone['y'] + zone['h']) * h)

            cv2.rectangle(image, (zx1, zy1), (zx2, zy2), (255, 0, 0), 2)
            cv2.putText(image, f"#{idx + 1}", (zx1, zy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            wx1, wy1 = max(0, zx1 - pad), max(0, zy1 - pad)
            wx2, wy2 = min(w, zx2 + pad), min(h, zy2 + pad)
            cv2.rectangle(image, (wx1, wy1), (wx2, wy2), (255, 255, 0), 1)

    return image


# --- UI 실행 ---
st.set_page_config(layout="wide", page_title="AI 뮤지엄 관리 시스템")
st.title("🏛️ AI 전시품 보호 관리 시스템")


# 1. 모델로드
@st.cache_resource
def get_models():
    try:
        yolo = YOLO('yolov8n-pose.pt')
        custom = joblib.load('model.pkl') if os.path.isfile('model.pkl') else None
        return yolo, custom
    except Exception as e:
        return None, None


yolo_model, custom_model = get_models()
if not yolo_model:
    st.error("모델 로드 실패. 'yolov8n-pose.pt' 파일이 있는지 확인하세요.")
    st.stop()

# 사이드바
with st.sidebar:
    st.header("📂 영상 관리")
    up = st.file_uploader("새 영상 업로드", type=["mp4", "avi"])
    if up:
        path = os.path.join("videos", f"video_{int(time.time())}.mp4")
        with open(path, "wb") as f: f.write(up.getbuffer())
        st.success("저장 완료");
        st.rerun()

    v_list = [f for f in os.listdir("videos") if f.endswith(('.mp4', '.avi'))]
    sel_v = st.selectbox("🎥 영상 선택", v_list) if v_list else None

if sel_v:
    video_path = os.path.join("videos", sel_v)
    curr_settings = load_settings(sel_v)

    # 탭 구성
    tab1, tab2, tab3 = st.tabs(["📏 구역 편집 (Canvas)", "⚡ 감도 설정", "👁️ 시각화 설정"])

    # ---[탭 1] 구역 편집(Canvas) ---
    with tab1:
        st.info("💡 사각형을 그리고 드래그/리사이즈 하세요. 설정 후 '구역 저장' 버튼을 꼭 누르세요.")

        col_canvas, col_ctrl = st.columns([3, 1])

        # 캔버스용 초기 데이터 준비
        canvas_w, canvas_h = 600, 450
        initial_drawing = {"version": "4.4.0", "objects": []}

        # 배경 이미지
        cap_temp = cv2.VideoCapture(video_path)
        ret, bg_frame = cap_temp.read()
        cap_temp.release()
        bg_img = None
        if ret:
            bg_frame = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)
            bg_img = Image.fromarray(cv2.resize(bg_frame, (canvas_w, canvas_h)))

            if 'canvas_key' not in st.session_state: st.session_state['canvas_key'] = 0

            for i, z in enumerate(curr_settings['zones']):
                initial_drawing['objects'].append({
                    "type": "rect",
                    "left": z['x'] * canvas_w,
                    "top": z['y'] * canvas_h,
                    "width": z['w'] * canvas_w,
                    "height": z['h'] * canvas_h,
                    "fill": "rgba(255, 0, 0, 0.2)",
                    "stroke": "red",
                    "strokeWidth": 2
                })

        with col_canvas:
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.2)",
                stroke_color="red",
                background_image=bg_img,
                update_streamlit=True,
                height=canvas_h,
                width=canvas_w,
                drawing_mode="transform",
                initial_drawing=initial_drawing if st.session_state.get('reset_canvas', False) else None,
                key=f"canvas_{st.session_state['canvas_key']}",
            )

            model_sel = st.radio("도구선택", ["편집/이동 (transform)", "새 구역 그리기(Rect)"], horizontal=True)
            if model_sel == "새 구역 그리기(Rect)":
                st.caption("왼쪽 툴바에서 'Rect'를 선택해 그리세요. 다 그리면 '편집/이동'으로 돌아오세요.")

        with col_ctrl:
            st.subheader("구역 관리(최대 20개)")
            parsed_zones = []
            if canvas_result.json_data:
                objs = canvas_result.json_data["objects"]
                for i, obj in enumerate(objs):
                    if obj["type"] == "rect":
                        parsed_zones.append({
                            'id': i + 1,
                            'x': obj["left"] / canvas_w,
                            'y': obj["top"] / canvas_h,
                            'w': obj["width"] / canvas_w,
                            'h': obj["height"] / canvas_h,
                            'active': True
                        })

            if len(parsed_zones) > 20: parsed_zones = parsed_zones[:20]

            final_zones = []
            for i, pz in enumerate(parsed_zones):
                is_active = True
                if i < len(curr_settings['zones']):
                    is_active = curr_settings['zones'][i].get('active', True)

                act = st.toggle(f"구역 #{i + 1}", value=is_active, key=f"tg_{i}")
                pz['active'] = act
                final_zones.append(pz)

            st.divider()
            if st.button("💾구역 저장 적용", use_container_width=True):
                curr_settings['zones'] = final_zones
                save_settings(sel_v, curr_settings)
                st.success("구역 저장됨")
                st.rerun()

            if st.button("🔄모두 지우기", use_container_width=True):
                curr_settings['zones'] = []
                save_settings(sel_v, curr_settings)
                st.session_state['reset_canvas'] = True
                st.session_state['canvas_key'] += 1
                st.rerun()

    # --- [탭 2] 감도설정 ---
    with tab2:
        col_list = st.columns([1, 1])
        c1, c2 = col_list[0], col_list[1]
        with c1:
            pad_val = st.slider("경계 여유값(Padding)", 0, 100, curr_settings['padding'])
            eth = st.slider("팔 뻗음 비율 (Extension)", 0.5, 1.0, curr_settings.get('extension_threshold', 0.85))
        with c2:
            ath = st.slider("팔 각도 임계값", 90, 180, curr_settings['angle_threshold'])
            hr = st.slider("허리 대비 손 높이", -0.5, 1.0, curr_settings['hip_ratio'])

        det_mode = st.radio("판단 모드", ["Algorithm", "AI", "Both"],
                            index=["Algorithm", "AI", "Both"].index(curr_settings.get('detection_mode', 'Algorithm')))

        if st.button("감도 설정 저장", use_container_width=True):
            curr_settings.update(
                {'padding': pad_val, 'extension_threshold': eth, 'angle_threshold': ath, 'hip_ratio': hr,
                 'detection_mode': det_mode})
            save_settings(sel_v, curr_settings)
            st.success("감도 저장됨")

    # --- [탭 3] 시각화 설정 ---
    with tab3:
        v_alert = st.checkbox("🚨 위험 시에만 표시", value=curr_settings.get('vis_alert_only', False))
        col_list_vis = st.columns([1, 1])
        c1, c2 = col_list_vis[0], col_list_vis[1]
        with c1:
            vb = st.checkbox("객체 박스", value=curr_settings.get('vis_bbox', True))
            vl = st.checkbox("이름표", value=curr_settings.get('vis_class_label', True))
            vs = st.checkbox("뼈대", value=curr_settings.get('vis_skeleton', True))
        with c2:
            vz = st.checkbox("구역 표시", value=curr_settings.get('vis_zones', True))
            vd = st.checkbox("손목 점", value=curr_settings.get('vis_wrist_dot', True))
            vln = st.checkbox("안전선", value=curr_settings.get('vis_safe_line', True))
            vt = st.checkbox("상태 텍스트", value=curr_settings.get('vis_wrist_text', True))

        if st.button("시각화 설정 저장", use_container_width=True):
            curr_settings.update({
                'vis_alert_only': v_alert, 'vis_bbox': vb, 'vis_class_label': vl,
                'vis_skeleton': vs, 'vis_zones': vz, 'vis_wrist_dot': vd, 'vis_safe_line': vln, 'vis_wrist_text': vt
            })
            save_settings(sel_v, curr_settings)
            st.success("시각화 저장됨")

    # ---[메인] 모니터링 ---
    st.markdown("---")
    st.header("📹실시간 모니터링")

    col_vid, col_stat = st.columns([3, 1])
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with col_vid:
        run_monitor = st.checkbox("▶️재생 시작", value=True)
        sf = st.slider("타임라인", 0, total_frames, 0)
        st_frame = st.empty()

    live_settings = curr_settings.copy()
    live_settings.update({
        'padding': pad_val, 'extension_threshold': eth, 'angle_threshold': ath, 'hip_ratio': hr,
        'detection_mode': det_mode,
        'vis_alert_only': v_alert, 'vis_bbox': vb, 'vis_class_label': vl, 'vis_skeleton': vs, 'vis_zones': vz,
        'vis_wrist_dot': vd, 'vis_safe_line': vln, 'vis_wrist_text': vt
    })

    if run_monitor:
        cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            res_img = process_frame(frame, yolo_model, custom_model, live_settings)
            st_frame.image(res_img, channels="RGB")
            time.sleep(0.02)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
        ret, frame = cap.read()
        if ret:
            res_img = process_frame(frame, yolo_model, custom_model, live_settings)
            st_frame.image(res_img, channels="RGB")

    cap.release()