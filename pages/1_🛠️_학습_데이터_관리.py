import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 상위 폴더 모듈 경로 설정
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import data_augmenter  # 데이터 증강 모듈

# 페이지 설정
st.set_page_config(layout="wide", page_title="학습 데이터 관리")
st.title("🛠️ AI 학습 데이터 관리 및 모델 훈련")


# -------------------------------------------------------------------------
# [핵심 함수] 픽셀 좌표 -> 신체 비율(Ratio) 변환
# -------------------------------------------------------------------------
def convert_to_ratio(df):
    """
    절대 좌표(픽셀)로 된 데이터를 몸통 크기 기준 비율 데이터로 변환합니다.
    - 해상도, 거리, 체격 차이를 없애줍니다.
    """
    # 좌표 컬럼만 추출
    v_cols = [c for c in df.columns if c.startswith('v')]
    meta_cols = [c for c in df.columns if c not in v_cols]

    if not v_cols: return df

    coords = df[v_cols].values
    new_coords_list = []

    for row in coords:
        # 17개 관절, 2개 좌표(x,y)로 구조화
        # 행 하나를 (-1, 17, 2)로 reshape하면 [프레임수, 17개관절, 2좌표]가 됨
        try:
            frames = row.reshape(-1, 17, 2)
        except ValueError:
            # 컬럼 개수가 17*2의 배수가 아니면 변환 불가 (그대로 반환)
            return df

        normalized_frames = []

        for frame in frames:
            # frame shape: (17, 2)
            # 5:왼어깨, 6:오른어깨, 11:왼골반, 12:오른골반

            # 1. 골반 중심점 (0,0 기준점)
            l_hip = frame[11]
            r_hip = frame[12]
            center = (l_hip + r_hip) / 2

            # 2. 척추 길이 (몸통 크기) 계산 = 스케일 기준
            l_sh = frame[5]
            r_sh = frame[6]
            center_sh = (l_sh + r_sh) / 2

            # 몸통 길이 (골반~어깨 거리)
            torso_size = np.linalg.norm(center_sh - center)

            # 3. 크기 정규화 (몸통 크기를 1.0으로 맞춤)
            # 이미 비율 데이터거나 노이즈인 경우(크기가 너무 작음)는 1.0으로 처리해 에러 방지
            scale = torso_size if torso_size > 10 else 1.0

            # 변환 공식: (내좌표 - 중심점) / 몸통길이
            norm_frame = (frame - center) / scale
            normalized_frames.append(norm_frame)

        # 다시 1줄로 펴기
        new_row = np.array(normalized_frames).flatten()
        new_coords_list.append(new_row)

    # 데이터프레임 재구성
    df_new = pd.DataFrame(new_coords_list, columns=v_cols)
    df_meta = df[meta_cols]  # 메타데이터 유지

    # 인덱스 리셋 후 병합
    df_new.reset_index(drop=True, inplace=True)
    df_meta.reset_index(drop=True, inplace=True)

    return pd.concat([df_new, df_meta], axis=1)


# 화면 분할
left_col, right_col = st.columns([1, 1], gap="large")

# ==============================================================================
# [왼쪽] 데이터 증강 도구 (비율 변환 기능 포함)
# ==============================================================================
with left_col:
    st.subheader("1️⃣ 데이터 증강 (Augmentation)")
    st.info("원본 CSV를 업로드하면 '비율 데이터'로 변환 후 증강합니다.")

    uploaded_file = st.file_uploader("원본 데이터(CSV) 업로드", type=["csv"])

    if uploaded_file:
        try:
            df_origin = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df_origin = pd.read_csv(uploaded_file, encoding='cp949')
        except Exception as e:
            st.error(f"파일 읽기 오류: {e}")
            st.stop()

        st.write(f"📂 원본 데이터: **{len(df_origin)}** 행")

        st.markdown("##### ⚙️ 클래스별 증강 배율 설정")
        col_n, col_m, col_t = st.columns(3)
        with col_n:
            n_fac = st.number_input("Neutral (정지)", min_value=1, value=1)
        with col_m:
            m_fac = st.number_input("Movement (이동)", min_value=1, value=2)
        with col_t:
            t_fac = st.number_input("Threat (위협)", min_value=1, value=10)

        # 🚀 증강 실행 버튼
        if st.button("🚀 변환 및 증강 실행", type="primary"):
            try:
                with st.spinner("1단계: 신체 비율 데이터로 변환 중..."):
                    # 👇 [핵심] 여기서 비율 변환 함수 호출
                    df_ratio = convert_to_ratio(df_origin)

                with st.spinner("2단계: 데이터 증강 처리 중..."):
                    # 변환된 df_ratio를 넣어서 증강 수행
                    df_aug = data_augmenter.run_augmentation(
                        df_ratio,
                        neutral_factor=n_fac,
                        movement_factor=m_fac,
                        threat_factor=t_fac
                    )

                    # 파일 저장 (파일명에 ratio_aug_ 붙임)
                    save_name = f"ratio_aug_{uploaded_file.name}"
                    df_aug.to_csv(save_name, index=False)

                    st.success(f"✅ 작업 완료! 총 **{len(df_aug)}** 행")
                    st.success(f"저장됨: `{save_name}`")

                    with st.expander("결과 데이터 미리보기 (값 범위를 확인하세요)"):
                        st.dataframe(df_aug.head())
                        st.caption("※ 값이 -1.5 ~ 1.5 사이의 소수점이면 비율 변환이 잘 된 것입니다.")

            except Exception as e:
                st.error(f"작업 중 오류 발생: {e}")

# ==============================================================================
# [오른쪽] 모델 학습 도구
# ==============================================================================
with right_col:
    st.subheader("2️⃣ 모델 학습 (Training)")
    st.info("증강된 데이터('ratio_aug_...')를 선택하여 학습하세요.")

    # 파일 목록 (ratio_aug_로 시작하는 파일만 필터링 추천)
    csv_files = [f for f in os.listdir(".") if f.endswith(".csv") and "aug_" in f]

    if not csv_files:
        st.warning("학습할 데이터 파일이 없습니다.")
    else:
        target_file = st.selectbox("학습에 사용할 파일 선택", csv_files)

        st.markdown("##### ⚙️ 학습 파라미터")
        n_estimators = st.slider("트리 개수 (Estimators)", 10, 200, 100)

        if st.button("🔥 모델 학습 시작", type="primary"):
            try:
                with st.spinner(f"'{target_file}' 로딩 중..."):
                    df_train = pd.read_csv(target_file)

                    feature_cols = [c for c in df_train.columns if c.startswith('v')]
                    X = df_train[feature_cols]
                    y = df_train['label']

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                with st.spinner("AI 모델 학습 중..."):
                    model = RandomForestClassifier(n_estimators=n_estimators, random_state=0)

                    # 1차원 배열 변환 (경고 방지)
                    model.fit(X_train, y_train.values.ravel())

                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)

                st.success(f"🎉 학습 완료! 정확도: **{acc * 100:.2f}%**")

                # 모델 저장
                model_save_path = "model.pkl"
                joblib.dump(model, model_save_path)
                real_path = os.path.abspath(model_save_path)

                st.success(f"💾 모델 저장 완료! 위치: `{real_path}`")

                with st.expander("상세 결과 보고서"):
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.json(report)

            except Exception as e:
                st.error(f"학습 오류: {e}")