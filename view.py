# view.py

import streamlit as st  # ⚠️ 이 줄이 가장 상단에 있어야 함!
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

        #-----------------------
        # AI 모델 선택 박스
        #----------------------
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
        select_model = st.selectbox("YOLO 포즈 모델", model_option, index=5)
        st.caption("※ v11은 성능이 더 좋으며, v9/v10은 포즈 미지원")
        st.markdown("---")
        #---------------------------------------------------------

        # 영상 목록 로드
        video_list = [f for f in os.listdir("videos") if f.endswith((".mp4", ".avi"))]
        video_list.sort()

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
    # 영상 첫 프레임 로딩
    # ===============================================================
    if "canvas_bg" not in st.session_state or st.session_state.get("last_vid") != sel_v:
        cap = cv2.VideoCapture(video_path)
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
    # 🔥 첫 저장 시 좌표 튐 문제 해결 (originX/Y에 따른 Path 좌표 해석)
    # ===============================================================
    if st.button("💾 구역 저장 (적용)", type="primary", use_container_width=True):

        new_zones = []

        if canvas.json_data and "objects" in canvas.json_data:
            for obj in canvas.json_data["objects"]:

                points = []
                # ------------------------------------
                # Case A: 새로 그린 polygon (구역설정 특어진 원인: path로 그려지는데 plygon 좌표로 그려서 좌표 안맞음)
                # ------------------------------------
                """
                # 삭제
                if obj["type"] == "polygon":
                    st.write("--- 새로 그린 Polygon 객체 디버깅 시작 ---")
                    st.json(obj)  # obj 딕셔너리의 전체 내용을 JSON 형태로 출력
                    st.write("--- 새로 그린 Polygon 객체 디버깅 종료 ---")

                    left = obj["left"]
                    top = obj["top"]
                    scaleX = obj["scaleX"]
                    scaleY = obj["scaleY"]

                    for p in obj["points"]:
                        # 현재로서는 가장 단순한 형태의 변환 로직을 유지.
                        abs_x = left + p["x"]  # scaleX 곱하기 제거 상태 유지
                        abs_y = top + p["y"]  # scaleY 곱하기 제거 상태 유지
                        points.append([abs_x / cw, abs_y / ch])
                """

                # ------------------------------------
                # ✅ Path 객체 처리 (새로 그린 polygon도 이 타입으로 반환됨)
                # ------------------------------------
                if obj["type"] == "path":  # ⚠️모든 도형은 이 블록에서 처리.
                    left = obj["left"]
                    top = obj["top"]
                    scaleX = obj.get("scaleX", 1.0)  # scaleX, scaleY가 없을 경우 기본값 1.0
                    scaleY = obj.get("scaleY", 1.0)  # (JSON에 있었지만, 안전하게 get으로 처리)

                    # originX와 originY를 확인하여 좌표 해석 방식을 결정
                    # 기본값은 'left', 'top'이며, 없으면 이렇게 가정
                    origin_x = obj.get("originX", "left")
                    origin_y = obj.get("originY", "top")

                    # originX/Y가 'center'인 경우, path 좌표가 이미 절대 캔버스 좌표일 가능성이 높음
                    # (JSON 분석 결과, 'center'일 때 path 좌표가 절대 좌표였음)
                    is_path_coords_absolute = (origin_x == "center" and origin_y == "center")

                    for cmd in obj["path"]:
                        if cmd[0] in ["M", "L"]:  # Path 명령 중 이동(M) 또는 선(L)만 처리
                            abs_x = 0
                            abs_y = 0

                            if is_path_coords_absolute:
                                # origin이 'center'이고 path 좌표가 이미 절대값인 경우
                                # left/top/scaleX/Y는 건드리지 않고 path 좌표를 직접 사용
                                abs_x = cmd[1]
                                abs_y = cmd[2]
                            else:
                                # origin이 'left'/'top'이거나 다른 경우, path 좌표는 left/top 기준 상대값
                                # 우리가 initial_drawing에서 생성한 path 객체들이 이 경우에 해당
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
    # 구역 목록 (삭제 즉시 반영 + 기존 기능 유지)
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
# 감도 설정
# ===============================================================
def render_sensitivity_tab(sel_v, curr_settings):
    st.subheader("경고/위험 판단 기준")

    # 화재감지 체크박스
    fire_check = st.checkbox("🔥 화재 / 연기 감지 모드 켜기", value=curr_settings.get("fire_check", False))

    st.markdown("---")  # 구분선

    # 판단모드 옵션 변경
    mode_options = ["Algorithm", "AI", "OR", "AND"]
    current_mode = curr_settings.get("detection_mode","Algorithm")

    # 기존 설정 호환성 처리(기존Both 저장되있을시 AND로 처리
    if current_mode == "Both":
        current_mode = "AND"
    if current_mode not in mode_options:
        current_mode = "Algorithm"

    md = st.radio("판단 모드", mode_options,
                  index=mode_options.index(current_mode),
                  horizontal=True)
    wd = st.slider("⚠️ 경고 감지 거리", 0, 200, curr_settings.get("warning_distance", 30))
    et = st.slider("팔 뻗음 비율", 0.5, 1.0, curr_settings["extension_threshold"])
    at = st.slider("팔 각도 임계값", 90, 180, curr_settings["angle_threshold"])
    hr = st.slider("골반기준 손 높이 상한 비율", 0.0, 1.0, curr_settings.get("hip_ratio", 0.2), 0.05)

    st.markdown("---")
    fall_check = st.checkbox("🤸 낙상 감지 켜기", value=curr_settings.get("fall_check", True))
    fr = st.slider("낙상 기울기 비율(낮을수록 민감", 0.5, 2.0,
                   curr_settings.get("fall_ratio", 1.2), 0.1,
                   disabled=not fall_check)

    if st.button("감도 저장"):
        curr_settings.update({
            "fire_check": fire_check,
            "fall_check": fall_check,
            "fall_ratio": fr,
            "warning_distance": wd,
            "extension_threshold": et,
            "angle_threshold": at,
            "detection_mode": md,
            "hip_ratio": hr
        })
        save_settings(sel_v, curr_settings)
        st.success("저장됨")

    return wd, et, at, md, hr, fire_check, fall_check, fr


# ===============================================================
# 시각화 설정
# ===============================================================
def render_vis_tab(sel_v, curr_settings):
    st.subheader("화면 표시 옵션")

    vo = curr_settings["vis_options"]

    c_alert = st.checkbox("🚨 위험 시에만 표시", value=vo["alert_only"])
    c_sk = st.checkbox("뼈대 표시", value=vo["skeleton"])
    c_zn = st.checkbox("구역 표시", value=vo["zones"])
    c_bb = st.checkbox("객체 박스", value=vo["bbox"])
    c_dot = st.checkbox("손목 점", value=vo["wrist_dot"])
    c_txt = st.checkbox("상태 텍스트", value=vo["text"])

    if st.button("시각화 옵션 저장"):
        vo.update({
            "alert_only": c_alert,
            "skeleton": c_sk,
            "zones": c_zn,
            "bbox": c_bb,
            "wrist_dot": c_dot,
            "text": c_txt
        })
        save_settings(sel_v, curr_settings)
        st.success("저장됨")

    return c_alert, c_sk, c_zn, c_bb, c_dot, c_txt