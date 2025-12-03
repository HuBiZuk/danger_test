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
            st.rerun()

        video_list = [f for f in os.listdir("videos") if f.endswith((".mp4", ".avi"))]
        video_list.sort(reverse=True)

        if video_list:
            return st.selectbox("영상 선택", video_list)
        return None


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
            st.rerun()

    with c2:
        if st.button("🗑️ 전체 삭제"):
            curr_settings["zones"] = []
            save_settings(sel_v, curr_settings)
            st.session_state["draw_mode_state"] = "transform"
            st.session_state["cv_key"] += 1
            st.rerun()

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
    # 기존 zones → canvas object로 복원
    # ===============================================================
    initial_drawing = {"version": "4.4.0", "objects": []}

    for z in curr_settings["zones"]:
        pts = z["points"]
        if not pts:
            continue

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
            "fill": "rgba(255, 0, 0, 0.3)",
            "stroke": "red",
            "strokeWidth": 2,
            "left": min_x,
            "top": min_y,
            "originX": "left",
            "originY": "top",
            "scaleX": 1,
            "scaleY": 1
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
    # 🔥 첫 저장 시 좌표 튐 문제 완전 해결 (Fabric.js 원리 그대로 적용)
    # ===============================================================
    if st.button("💾 구역 저장 (적용)", type="primary", use_container_width=True):

        new_zones = []

        if canvas.json_data and "objects" in canvas.json_data:
            for obj in canvas.json_data["objects"]:

                points = []

                # ------------------------------------
                # Case A: 새로 그린 polygon
                # ------------------------------------
                if obj["type"] == "polygon":
                    left = obj["left"]
                    top = obj["top"]
                    scaleX = obj["scaleX"]
                    scaleY = obj["scaleY"]
                    off_x = obj["pathOffset"]["x"]
                    off_y = obj["pathOffset"]["y"]

                    for p in obj["points"]:
                        abs_x = left + (p["x"] + off_x) * scaleX
                        abs_y = top + (p["y"] + off_y) * scaleY
                        points.append([abs_x / cw, abs_y / ch])

                # ------------------------------------
                # Case B: 로드된 path(불러온 도형)
                # ------------------------------------
                elif obj["type"] == "path":
                    left = obj["left"]
                    top = obj["top"]
                    scaleX = obj["scaleX"]
                    scaleY = obj["scaleY"]

                    for cmd in obj["path"]:
                        if cmd[0] in ["M", "L"]:
                            abs_x = left + cmd[1] * scaleX
                            abs_y = top + cmd[2] * scaleY
                            points.append([abs_x / cw, abs_y / ch])

                if len(points) > 2:
                    new_zones.append({"points": points, "active": True})

        curr_settings["zones"] = new_zones
        save_settings(sel_v, curr_settings)

        st.session_state["draw_mode_state"] = "transform"
        st.session_state["cv_key"] += 1
        st.rerun()

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
                changed = st.toggle("활성", value=is_active, key=f"act_{i}")

                if changed != is_active:
                    curr_settings["zones"][i]["active"] = changed
                    save_settings(sel_v, curr_settings)
                    st.rerun()

            with c3:
                st.button(
                    "🗑️",
                    key=f"delbtn_{i}",   # 🔥 key 충돌 방지
                    on_click=delete_zone_callback,
                    args=(i, sel_v)
                )

    if st.session_state.get("force_rerun"):
        st.session_state["force_rerun"] = False
        st.rerun()


# ===============================================================
# 감도 설정
# ===============================================================
def render_sensitivity_tab(sel_v, curr_settings):
    st.subheader("경고/위험 판단 기준")

    wd = st.slider("⚠️ 경고 감지 거리", 0, 200, curr_settings.get("warning_distance", 30))
    et = st.slider("팔 뻗음 비율", 0.5, 1.0, curr_settings["extension_threshold"])
    at = st.slider("팔 각도 임계값", 90, 180, curr_settings["angle_threshold"])
    md = st.radio("판단 모드", ["Algorithm", "AI", "Both"],
                  index=["Algorithm", "AI", "Both"].index(curr_settings["detection_mode"]))

    if st.button("감도 저장"):
        curr_settings.update({
            "warning_distance": wd,
            "extension_threshold": et,
            "angle_threshold": at,
            "detection_mode": md
        })
        save_settings(sel_v, curr_settings)
        st.success("저장됨")

    return wd, et, at, md


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
