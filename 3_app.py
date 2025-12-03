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

# ==============================================================================
# [호환성 패치] Streamlit 1.32.0+ 호환
# ==============================================================================
import streamlit.elements.image as st_image

if not hasattr(st_image, 'original_image_to_url'):
    st_image.original_image_to_url = st_image.image_to_url


def simple_patch(image, width=None, clamp=False, channels="RGB", output_format="JPEG", image_id=None,
                 allow_emoji=False):
    return st_image.original_image_to_url(image, width, clamp, channels, output_format, image_id)


st_image.image_to_url = simple_patch

# --- [초기 설정] ---
if not os.path.exists('videos'): os.makedirs('videos')
if not os.path.exists('settings'): os.makedirs('settings')


# --- [함수] 설정 관리 ---
def load_settings(video_name):
    json_path = os.path.join('settings', f"{video_name}.json")
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
                # 데이터 유효성 검사
                if 'zones' in saved:
                    valid_zones = []
                    for z in saved['zones']:
                        if 'points' in z and len(z['points']) > 2:
                            valid_zones.append(z)
                    saved['zones'] = valid_zones

                for k, v in default_settings.items():
                    if k not in saved: saved[k] = v
                if 'vis_options' not in saved: saved['vis_options'] = default_settings['vis_options']
                return saved
        except:
            return default_settings
    return default_settings


def save_settings(video_name, settings):
    json_path = os.path.join('settings', f"{video_name}.json")
    with open(json_path, 'w') as f: json.dump(settings, f, indent=4)


# --- [함수] 계산 및 분석 ---
def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle


def process_frame(frame, yolo_model, custom_model, settings):
    # 분석용 리사이즈 (800x600 고정)
    frame = cv2.resize(frame, (800, 600))
    h, w, _ = frame.shape

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

            # 정규화 좌표(0~1) -> 픽셀 좌표
            pts = np.array(z['points']) * [w, h]
            pts = pts.astype(np.int32).reshape((-1, 1, 2))
            active_polygons.append(pts)

            # (1) 경고 영역 (노란색 팽창)
            if warn_dist > 0:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                k_size = int(warn_dist * 2) + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                expanded_mask = cv2.dilate(mask, kernel)
                contours, _ = cv2.findContours(expanded_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, (255, 255, 0), 2)

            # (2) 위험 영역 (빨간색 실선)
            cv2.polylines(image, [pts], True, (255, 0, 0), 2)

            start_pt = tuple(pts[0][0])
            cv2.putText(image, f"#{i + 1}", (start_pt[0], start_pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0),
                        2)

    # 2. 포즈 분석
    if results[0].keypoints is not None and results[0].boxes is not None:
        keypoints_data = results[0].keypoints.data.cpu().numpy()
        boxes_data = results[0].boxes.data.cpu().numpy()

        for box_info, kps in zip(boxes_data, keypoints_data):
            bx1, by1, bx2, by2, b_conf, b_cls = box_info

            p_danger = False
            p_warning = False
            wrist_points = []

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

                    # 충돌 체크
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


# ==============================================================================
# [메인 UI]
# ==============================================================================
st.set_page_config(layout="wide", page_title="AI 전시품 보호 시스템 v3")
st.title("🏛️ AI 전시품 보호 관리 시스템")


# 1. 모델 로드
@st.cache_resource
def get_models():
    try:
        yolo = YOLO('yolov8n-pose.pt')
        custom = joblib.load('model.pkl') if os.path.isfile('model.pkl') else None
        return yolo, custom
    except:
        return None, None


yolo_model, custom_model = get_models()

# 사이드바
with st.sidebar:
    st.header("📂 파일 관리")
    up = st.file_uploader("영상 업로드", type=["mp4", "avi"])
    if up:
        p = os.path.join("videos", f"vid_{int(time.time())}.mp4")
        with open(p, "wb") as f: f.write(up.getbuffer())
        st.success("업로드 완료")
        st.rerun()

    v_list = [f for f in os.listdir("videos") if f.endswith(('.mp4', '.avi'))]
    sel_v = st.selectbox("영상 선택", v_list) if v_list else None

if not sel_v or not yolo_model:
    st.warning("영상을 선택하거나 모델을 확인해주세요.")
    st.stop()

# 설정 로드
video_path = os.path.join("videos", sel_v)
curr_settings = load_settings(sel_v)

left_col, right_col = st.columns([1, 1], gap="medium")

# ==========================================
# [왼쪽] 설정 탭
# ==========================================
with left_col:
    tab1, tab2, tab3 = st.tabs(["📝 구역 관리", "⚡ 감도 설정", "👁️ 시각화 설정"])

    # --- [탭 1] 구역 관리 (통합됨) ---
    with tab1:
        # 1. 캔버스 상태 관리 (편집 vs 그리기)
        if 'draw_mode_state' not in st.session_state:
            st.session_state['draw_mode_state'] = 'transform'  # 기본은 편집(이동/수정)
        if 'cv_key' not in st.session_state:
            st.session_state['cv_key'] = 0

        # 2. 버튼 컨트롤
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            # 버튼을 누르면 그리기 모드로 전환
            if st.button("➕ 새 구역 그리기", use_container_width=True):
                st.session_state['draw_mode_state'] = 'polygon'
                st.session_state['cv_key'] += 1  # 캔버스 리로드하여 모드 적용
                st.rerun()
        with col_btn2:
            if st.button("🗑️ 전체 삭제", use_container_width=True):
                curr_settings['zones'] = []
                save_settings(sel_v, curr_settings)
                st.session_state['draw_mode_state'] = 'transform'
                st.session_state['cv_key'] += 1
                st.rerun()

        # 현재 모드 안내
        if st.session_state['draw_mode_state'] == 'polygon':
            st.info("🖌️ **그리기 모드**: 점을 찍어 다각형을 완성하세요. (완료 후 아래 '저장' 버튼 클릭)")
        else:
            st.info("✋ **편집 모드**: 구역을 선택하여 이동하거나 크기를 조절하세요.")

        # 3. 캔버스 배경 및 크기
        cw, ch = 600, 450
        if 'canvas_bg' not in st.session_state or st.session_state.get('last_vid') != sel_v:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame)
                st.session_state['canvas_bg'] = img_pil.resize((cw, ch))
            else:
                st.session_state['canvas_bg'] = None
            st.session_state['last_vid'] = sel_v

        bg_resized = st.session_state['canvas_bg']

        # 4. 초기 데이터 생성 (좌표 오차 보정 로직 포함)
        initial_drawing = {"version": "4.4.0", "objects": []}

        # 저장된 구역을 불러와서 Path 객체로 캔버스에 배치
        # 주의: initial_drawing은 key가 변경될 때만 캔버스에 적용됨
        for z in curr_settings['zones']:
            pts = z['points']
            if not pts: continue

            # 정규화 -> 픽셀
            poly_pts = np.array(pts) * [cw, ch]

            # Bounding Box (왼쪽 상단 기준점) 계산
            min_x = np.min(poly_pts[:, 0])
            min_y = np.min(poly_pts[:, 1])

            # Path 명령어 (Bounding Box 기준 상대 좌표)
            path_cmds = []
            path_cmds.append(["M", poly_pts[0][0] - min_x, poly_pts[0][1] - min_y])
            for p in poly_pts[1:]:
                path_cmds.append(["L", p[0] - min_x, p[1] - min_y])
            path_cmds.append(["Z"])

            initial_drawing['objects'].append({
                "type": "path",
                "path": path_cmds,
                "fill": "rgba(255, 0, 0, 0.2)",
                "stroke": "red",
                "strokeWidth": 2,
                "left": min_x,  # 실제 위치
                "top": min_y,  # 실제 위치
                "originX": "left",
                "originY": "top"
            })

        # 5. 캔버스 렌더링
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.2)",
            stroke_color="red",
            stroke_width=2,
            background_image=bg_resized,
            update_streamlit=True,
            height=ch, width=cw,
            drawing_mode=st.session_state['draw_mode_state'],  # 현재 모드 반영
            initial_drawing=initial_drawing,
            key=f"canvas_{st.session_state['cv_key']}"
        )

        # 6. 저장 및 적용 (핵심 좌표 보정 로직)
        if st.button("💾 구역 저장 (적용)", use_container_width=True, type="primary"):
            new_zones = []
            if canvas_result.json_data and "objects" in canvas_result.json_data:
                for obj in canvas_result.json_data["objects"]:

                    # 1) 공통 속성 추출
                    left = obj.get('left', 0)
                    top = obj.get('top', 0)
                    scaleX = obj.get('scaleX', 1)
                    scaleY = obj.get('scaleY', 1)
                    points = []

                    # 2) Path (기존 구역) 좌표 복원
                    if obj["type"] == "path":
                        for cmd in obj["path"]:
                            if cmd[0] == 'M' or cmd[0] == 'L':
                                # Path는 Bounding Box(left, top) 기준 상대 좌표임
                                rel_x = cmd[1]
                                rel_y = cmd[2]
                                # 절대 좌표 = Box위치 + (상대좌표 * 배율)
                                abs_x = left + (rel_x * scaleX)
                                abs_y = top + (rel_y * scaleY)
                                points.append([abs_x / cw, abs_y / ch])

                    # 3) Polygon (새로 그린 구역) 좌표 복원
                    elif obj["type"] == "polygon":
                        # Polygon도 Fabric.js 버전에 따라 offset 처리가 필요할 수 있음.
                        # st_canvas에서는 보통 left/top이 바운딩박스 시작점.
                        # pathOffset 등을 고려해야 하지만, 가장 안전한 방법은 아래와 같음:

                        # points 배열 내부는 보통 (0,0) 근처의 값들이거나 상대값
                        # 하지만 Fabric 객체는 항상 left, top을 가짐.
                        for p in obj["points"]:
                            rel_x = p.get('x', 0)
                            rel_y = p.get('y', 0)

                            # 공식 적용
                            abs_x = left + (rel_x * scaleX)
                            abs_y = top + (rel_y * scaleY)

                            points.append([abs_x / cw, abs_y / ch])

                    if len(points) > 2:
                        new_zones.append({'points': points, 'active': True})

            # 데이터 저장
            curr_settings['zones'] = new_zones
            save_settings(sel_v, curr_settings)

            # 저장 후에는 항상 '편집 모드'로 복귀 + 캔버스 리로드 (그래야 Path로 변환되어 보임)
            st.session_state['draw_mode_state'] = 'transform'
            st.session_state['cv_key'] += 1
            st.rerun()

        # 7. 리스트 관리 (깜빡임 방지 적용)
        st.markdown("---")
        st.write("📋 **구역 목록**")

        if not curr_settings['zones']:
            st.caption("구역이 없습니다. '새 구역 그리기'를 눌러보세요.")
        else:
            zones_to_keep = []
            delete_occurred = False

            for i, z in enumerate(curr_settings['zones']):
                c1, c2, c3 = st.columns([0.2, 0.6, 0.2])
                with c1:
                    st.write(f"#{i + 1}")
                with c2:
                    curr_act = z.get('active', True)
                    # 키를 고유하게 주어 상태 유지
                    new_act = st.toggle("활성", value=curr_act, key=f"ac_{i}")
                    if new_act != curr_act:
                        z['active'] = new_act
                        curr_settings['zones'][i] = z
                        save_settings(sel_v, curr_settings)
                        # 여기서는 cv_key 변경 없이 rerun -> 깜빡임 없음
                        st.rerun()
                with c3:
                    if st.button("🗑️", key=f"del_{i}"):
                        delete_occurred = True
                        continue
                zones_to_keep.append(z)

            if delete_occurred:
                curr_settings['zones'] = zones_to_keep
                save_settings(sel_v, curr_settings)
                st.session_state['cv_key'] += 1  # 삭제는 리로드 필요
                st.rerun()

    # --- [탭 2] 감도 설정 ---
    with tab2:
        st.subheader("경고/위험 판단 기준")
        wd = st.slider("⚠️ 경고 감지 거리 (픽셀)", 0, 200, curr_settings.get('warning_distance', 30))
        et = st.slider("팔 뻗음 비율 (Extension)", 0.5, 1.0, curr_settings['extension_threshold'])
        at = st.slider("팔 각도 임계값 (Angle)", 90, 180, curr_settings['angle_threshold'])
        md = st.radio("판단 모드", ["Algorithm", "AI", "Both"],
                      index=["Algorithm", "AI", "Both"].index(curr_settings['detection_mode']))

        if st.button("감도 저장"):
            curr_settings.update(
                {'warning_distance': wd, 'extension_threshold': et, 'angle_threshold': at, 'detection_mode': md})
            save_settings(sel_v, curr_settings)
            st.success("저장됨")

    # --- [탭 3] 시각화 설정 ---
    with tab3:
        st.subheader("화면 표시 옵션")
        vo = curr_settings['vis_options']
        check_alert = st.checkbox("🚨 위험 시에만 표시", value=vo['alert_only'])
        v_skel = st.checkbox("뼈대 (Skeleton)", value=vo['skeleton'])
        v_zone = st.checkbox("구역 (Zones)", value=vo['zones'])
        v_box = st.checkbox("객체 박스 (BBox)", value=vo['bbox'])
        v_dot = st.checkbox("손목 점 (Dot)", value=vo['wrist_dot'])
        v_txt = st.checkbox("상태 텍스트 (Text)", value=vo['text'])

        if st.button("시각화 옵션 저장"):
            vo.update(
                {'alert_only': check_alert, 'skeleton': v_skel, 'zones': v_zone, 'bbox': v_box, 'wrist_dot': v_dot,
                 'text': v_txt})
            save_settings(sel_v, curr_settings)
            st.success("저장됨")

# ==========================================
# [오른쪽] 모니터링 화면
# ==========================================
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


    def run_live_processing(frame_img):
        live_settings = curr_settings.copy()
        live_settings['warning_distance'] = wd
        live_settings['extension_threshold'] = et
        live_settings['angle_threshold'] = at
        live_settings['detection_mode'] = md
        live_settings['vis_options'] = {
            'alert_only': check_alert, 'skeleton': v_skel, 'zones': v_zone,
            'bbox': v_box, 'label': True, 'wrist_dot': v_dot, 'text': v_txt
        }
        return process_frame(frame_img, yolo_model, custom_model, live_settings)


    if run_monitor:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            out_img = run_live_processing(frame)
            st_screen.image(out_img, channels="RGB")
            time.sleep(0.01)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            out_img = run_live_processing(frame)
            st_screen.image(out_img, channels="RGB")
    cap.release()