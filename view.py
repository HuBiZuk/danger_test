# view.py

import streamlit as st
import cv2
import time
import os
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from utils import save_settings, load_settings


# ===============================================================
# ① 단일 삭제 콜백 (key 충돌 제거, UI 즉시 반영)
# ===============================================================
def delete_zone_callback(i, video_name):
    settings = load_settings(video_name)
    if "zones" in settings and 0 <= i < len(settings["zones"]):
        del settings["zones"][i]
        save_settings(video_name, settings)

    # 🔥 강제 canvas 리렌더
    st.session_state["cv_key"] += 1
    st.session_state["force_rerun"] = True


# ===============================================================
# 사이드바
# ===============================================================
def render_sidebar():
    with st.sidebar:
        st.header("📂 파일 관리")

        upload_file = st.file_uploader("영상 업로드", type=["mp4", "avi"])
        if upload_file:
            path = os.path.join("videos", f"vid_{int(time.time())}.mp4")
            with open(path, "wb") as f:
                f.write(upload_file.getbuffer())
            st.success("업로드 완료")
            time.sleep(1)
            st.experimental_rerun()

        # -----------------------
        # AI 모델 선택 박스
        # ----------------------
        st.markdown("---")
        st.subheader("AI모델 선택")

        model_option = [
            "yolo11n-pose.pt",  # Nano (빠름, 추천)
            "yolo11s-pose.pt",  # Small
            "yolo11m-pose.pt",  # Medium
            "yolo11l-pose.pt",  # Large
            "yolo11x-pose.pt",  # XLarge (매우 정밀)

            # --- [YOLOv8] 기존 ---
            "yolov8n-pose.pt",  # 기존 사용 모델
        ]

        # index=5 는 'yolov8n-pose.pt' 기본값으로 설정
        select_model = st.selectbox("YOLO 포즈 모델", model_option, index=0)
        st.caption("※ v11은 성능이 더 좋으며, v9/v10은 포즈 미지원")
        st.markdown("---")
        # ---------------------------------------------------------

        # 영상 목록 로드
        video_list = [f for f in os.listdir("videos") if f.endswith((".mp4", ".avi"))]
        video_list.sort()

        video_list.insert(0, "실시간 카메라")

        if video_list:
            sel_video = st.selectbox("영상 선택", video_list)
            return sel_video, select_model

        return None, select_model


# ===============================================================
# 구역 관리 탭
# ===============================================================
def render_zone_tab(sel_v, curr_settings, video_path):
    # 세션 초기화
    if "draw_mode_state" not in st.session_state:
        st.session_state["draw_mode_state"] = "transform"

    if "cv_key" not in st.session_state:
        st.session_state["cv_key"] = 0

    st.info("💡 새 구역 그리기 → 점 찍기 → 시작점 클릭해 닫기 → 저장 버튼 클릭")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("➕ 새 구역 그리기"):
            st.session_state["draw_mode_state"] = "polygon"
            st.session_state["cv_key"] += 1
            st.experimental_rerun()

    with c2:
        if st.button("🗑️ 전체 삭제"):
            curr_settings["zones"] = []
            save_settings(sel_v, curr_settings)
            st.session_state["draw_mode_state"] = "transform"
            st.session_state["cv_key"] += 1
            st.experimental_rerun()

    cw, ch = 600, 450

    # ===============================================================
    # 영상 15번째 프레임 로딩
    # ===============================================================
    if "canvas_bg" not in st.session_state or st.session_state.get("last_vid") != sel_v:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 15)
        ret, frame = cap.read()

        # 영상이 짧아서 15번째 프레임 없으면 5번째 프레임 읽기
        if not ret:
            cap:set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        cap.release()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            st.session_state["canvas_bg"] = img.resize((cw, ch))
        else:
            st.session_state["canvas_bg"] = None

        st.session_state["last_vid"] = sel_v

    # ===============================================================
    # 기존 zones 로드
    # ===============================================================
    initial_drawing = {"version": "4.4.0", "objects": []}

    for z in curr_settings["zones"]:
        pts = z["points"]
        if not pts: continue

        # 활성 상태에따라 색상 변경 (활성: 빨강, 비활성: 회색)
        is_active = z.get("active", True)
        stroke_color = "red" if is_active else "gray"
        fill_color = "rgba(255,0,0,0.3)" if is_active else "rgba(128,128,128,0.1)"

        poly = np.array(pts) * [cw, ch]
        min_x = np.min(poly[:, 0])
        min_y = np.min(poly[:, 1])
        path_cmds = [["M", poly[0][0] - min_x, poly[0][1] - min_y]]
        for p in poly[1:]:
            path_cmds.append(["L", p[0] - min_x, p[1] - min_y])
        path_cmds.append(["Z"])

        initial_drawing["objects"].append({
            "type": "path",
            "path": path_cmds,
            "fill": fill_color,
            "stroke": stroke_color,
            "strokeWidth": 2,
            "left": min_x,
            "top": min_y,
            "originX": "left",
            "originY": "top"
        })

    # ===============================================================
    # Canvas
    # ===============================================================
    canvas = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_color="red",
        stroke_width=2,
        background_image=st.session_state["canvas_bg"],
        height=ch,
        width=cw,
        drawing_mode=st.session_state["draw_mode_state"],
        initial_drawing=initial_drawing,
        update_streamlit=True,
        key=f"canvas_{st.session_state['cv_key']}"
    )

    # ===============================================================
    # 🔥 첫 저장 시 좌표 튐 문제 해결 (Path 객체 처리)
    # ===============================================================
    if st.button("💾 구역 저장 (적용)", type="primary", use_container_width=True):

        new_zones = []

        if canvas.json_data and "objects" in canvas.json_data:
            for obj in canvas.json_data["objects"]:

                points = []

                if obj["type"] == "path":
                    left = obj["left"]
                    top = obj["top"]
                    scaleX = obj.get("scaleX", 1.0)
                    scaleY = obj.get("scaleY", 1.0)

                    origin_x = obj.get("originX", "left")
                    origin_y = obj.get("originY", "top")

                    is_path_coords_absolute = (origin_x == "center" and origin_y == "center")

                    for cmd in obj["path"]:
                        if cmd[0] in ["M", "L"]:
                            abs_x = 0
                            abs_y = 0

                            if is_path_coords_absolute:
                                abs_x = cmd[1]
                                abs_y = cmd[2]
                            else:
                                abs_x = left + cmd[1] * scaleX
                                abs_y = top + cmd[2] * scaleY

                            points.append([abs_x / cw, abs_y / ch])

                if len(points) > 2:
                    new_zones.append({"points": points, "active": True})

        curr_settings["zones"] = new_zones
        save_settings(sel_v, curr_settings)

        st.session_state["draw_mode_state"] = "transform"
        st.session_state["cv_key"] += 1

        st.experimental_rerun()

    # ===============================================================
    # 구역 목록 (삭제 및 활성/비활성)
    # ===============================================================
    st.markdown("---")
    st.subheader("📋 구역 목록")

    if not curr_settings["zones"]:
        st.caption("구역 없음")
    else:
        for i, z in enumerate(curr_settings["zones"]):
            c1, c2, c3 = st.columns([0.2, 0.6, 0.2])

            with c1:
                st.write(f"#{i + 1}")

            with c2:
                is_active = z.get("active", True)
                changed = st.checkbox("활성", value=is_active, key=f"act_{i}")

                if changed != is_active:
                    curr_settings["zones"][i]["active"] = changed
                    save_settings(sel_v, curr_settings)
                    st.experimental_rerun()

            with c3:
                st.button(
                    "🗑️",
                    key=f"delbtn_{i}",
                    on_click=delete_zone_callback,
                    args=(i, sel_v)
                )

    if st.session_state.get("force_rerun"):
        st.session_state["force_rerun"] = False
        st.experimental_rerun()


# ===============================================================
# 감도 설정 (노란 구역 크기 'wd' 및 AI 설정 포함)
# ===============================================================
def render_sensitivity_tab(sel_v, curr_settings):
    st.subheader("경고/위험 판단 기준")

    # 화재감지 체크박스
    fire_check = st.checkbox("🔥 화재 / 연기 감지 모드 켜기", value=curr_settings.get("fire_check", False))

    st.markdown("---")

    # 판단모드 옵션 변경
    mode_options = ["Algorithm", "AI", "OR", "AND"]
    current_mode = curr_settings.get("detection_mode", "Algorithm")

    if current_mode == "Both": current_mode = "AND"
    if current_mode not in mode_options: current_mode = "Algorithm"

    md = st.radio("판단 모드", mode_options,
                  index=mode_options.index(current_mode),
                  horizontal=True)

    # 1. 경고 감지 거리 (이게 노란 박스 크기 결정!)
    wd = st.slider("⚠️ 경고 감지 거리 (노란 구역)", 0, 200, curr_settings.get("warning_distance", 30))

    et = st.slider("팔 뻗음 비율", 0.5, 1.0, curr_settings["extension_threshold"])
    at = st.slider("팔 각도 임계값", 90, 180, curr_settings["angle_threshold"])
    hr = st.slider("골반기준 손 높이 상한 비율", 0.0, 1.0, curr_settings.get("hip_ratio", 0.2), 0.05)

    st.markdown("---")

    # 낙상 감지
    fall_check = st.checkbox("🤸 낙상 감지 켜기", value=curr_settings.get("fall_check", True))
    fr = st.slider("낙상 기울기 비율(낮을수록 민감)", 0.5, 2.0,
                   curr_settings.get("fall_ratio", 1.2), 0.1,
                   disabled=not fall_check)

    st.markdown("---")

    # [신규] AI 민감도 슬라이더
    st.markdown("##### 🧠 AI 민감도 설정")
    ai_th = st.slider("AI 위협 민감도 (낮을수록 예민)", 0.1, 1.0,
                      curr_settings.get("ai_threshold", 0.7), 0.05)

    if st.button("감도 저장"):
        curr_settings.update({
            "fire_check": fire_check,
            "fall_check": fall_check,
            "fall_ratio": fr,
            "warning_distance": wd,
            "extension_threshold": et,
            "angle_threshold": at,
            "detection_mode": md,
            "hip_ratio": hr,
            "ai_threshold": ai_th  # 저장 항목
        })
        save_settings(sel_v, curr_settings)
        st.success("저장됨")

    # 리턴값 9개 (wd가 포함되어야 노란박스가 그려짐)
    return wd, et, at, md, hr, fire_check, fall_check, fr, ai_th


# ===============================================================
# 시각화 설정
# ===============================================================
def render_vis_tab(sel_v, curr_settings):
    st.subheader("화면 표시 옵션")

    # 기본값 안전 처리
    vo = curr_settings.get("vis_options", {
        "alert_only": False, "skeleton": True, "zones": True,
        "bbox": True, "wrist_dot": True, "text": True
    })

    c_alert = st.checkbox("🚨 위험 시에만 표시", value=vo.get("alert_only", False))
    c_sk = st.checkbox("뼈대 표시", value=vo.get("skeleton", True))
    c_zn = st.checkbox("구역 표시", value=vo.get("zones", True))
    c_bb = st.checkbox("객체 박스", value=vo.get("bbox", True))
    c_dot = st.checkbox("손목 점", value=vo.get("wrist_dot", True))
    c_txt = st.checkbox("상태 텍스트", value=vo.get("text", True))

    if st.button("시각화 옵션 저장"):
        vo.update({
            "alert_only": c_alert,
            "skeleton": c_sk,
            "zones": c_zn,
            "bbox": c_bb,
            "wrist_dot": c_dot,
            "text": c_txt
        })
        curr_settings["vis_options"] = vo
        save_settings(sel_v, curr_settings)
        st.success("저장됨")

    return c_alert, c_sk, c_zn, c_bb, c_dot, c_txt

def  draw_ai_dashborad(ai_result):
    if ai_result and ai_result["is_active"]:
        st.markdown("### 📊 AI 실시간 분석")
        c1, c2, c3 = st.columns(3)
        c1.metric("Safe", f"{ai_result['safe']*100:.1f}%")
        c2.metric("Move", f"{ai_result['move']*100:.1f}%")
        t_val = ai_result['threat'] * 100
        c3.metric("Threat", f"{t_val:.0f}%", delta="위험" if t_val > 50 else "안전", delta_color="inverse")
