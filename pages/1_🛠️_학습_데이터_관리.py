from distutils.command.upload import upload

import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 상위 폴더의 모듈을 불러오기 위한 경로 설정
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import data_augmenter

# 페이지 설정
st.set_page_config(layout="wide", page_title="학습 데이터 관리")

st.title("🛠️ AI 학습 데이터 관리 및 모델 훈련")

# 화면 분할(왼쪽: 증강 / 오른쪽: 학습)
left_col, right_col = st.columns([1,1], gap="large")

# =========================
# [왼쪽] 데이터 증강 도구
# =========================
with left_col:
    st.subheader("1️⃣ 데이터 증강 (Augmentation)")
    st.info("원본 CSV를 업로드 하여 데이터를 증폭 시킵니다.")

    # 1. 파일 업로드
    uploaded_file = st.file_uploader("원본 CSV 업로드", type=["csv"])

    if uploaded_file:
        df_origin = pd.read_csv(uploaded_file)
        st.write(f"📂 원본 데이터: **{len(df_origin)}** 행")

        st.markdown("##### ⚙️ 클래스별 증강 배율 설정")
        col_n, col_m, col_t = st.columns(3)
        with col_n:
            n_fac = st.number_input("Neutral(정지)", min_value=1, value=1, help="데이터가 많으므로 유지")
        with col_m:
            m_fac = st.number_input("Movement (이동)", min_value=1, value=2, help="2배 증강")
        with col_t:
            t_fac = st.number_input("Threat (위협)", min_value=1, value=10, help="10배 증강")

        # 2. 증강 실행 버튼
        if st.button("🚀 데이터 증강 실행", type="primary"):
             try:
                 with st.spinner("증강 처리중..."):
                    # data_augmenter 모듈 호출
                    df_aug = data_augmenter.run_augmentation(
                        df_origin,
                        neutral_factor=n_fac,
                        movement_factor=m_fac,
                        threat_factor=t_fac,
                    )
                    # 파일 저장(현재 실행 위치인 루트 폴더에 저장됨)
                    save_name = f"aug_{uploaded_file.name}"
                    df_aug.to_csv(save_name, index=False)

                    st.success(f"✅ 증강 완료! 총 **{len(df_aug)}** 행")
                    st.success(f"파일 저장됨: `{save_name}`")

                    # 미리보기
                    with st.expander("증강 데이터 미리보기"):
                        st.dataframe(df_aug.head())

             except Exception as e:
                 st.error(f"에러발생: {e}")

# ==================================
# [오른쪽] 모델 학습 도구
# =================================
with right_col:
    st.subheader("2️⃣ 모델 학습 (Training)")
    st.info("증강된 데이터를 선택하여 AI 모델을 재학습 시킵니다.")

    # 1. 학습용 파일 선택 (현재 폴더의 CSV중 'aug_'로 시작하는 것들)
    csv_files = [f for  f in os.listdir(".") if f.endswith(".csv") and f.startswith("aug_")]

    if not csv_files:
        st.warning("증강된 데이터 파일('aug_*.csv')이 없습니다. 먼저 왼쪽에서 증강을 수행하세요.")
    else:
        target_file = st.selectbox("학습에 사용할 파일 선택", csv_files)

        st.markdown("##### ⚙️ 학습 파라미터")
        n_estimators = st.slider("트리 개수(Estinators)", 10, 200, 100)

        # 2. 학습 시작 버튼
        if st.button("🔥 모델 학습 시작", type="primary"):
            try:
                with st.spinner(f"`{target_file}` 데이터를 로딩중..."):
                    df_train = pd.read_csv(target_file)

                    # 데이터 전처리(X 좌표 Y 라밸)
                    # V 로 시작하는 컬럼만 입력 데이터로 사용
                    feature_cols = [c for c in df_train.columns if c.startswith('v')]
                    X = df_train[feature_cols]
                    y = df_train[['label']]

                    # 학습 / 검증 데이터 분리
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                with st.spinner("AI 모델 학습 중..."):
                    # 모델 생성 및 학습(RandomForest 사용)
                    model = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
                    model.fit(X_train, y_train.values.ravel())

                    # 성능 평가
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)

                st.success(f"🎉 학습 완료! 정확도: **{acc*100:.2f}%**")

                # 모델저장
                model_save_path = "model.pkl"
                joblib.dump(model, model_save_path)
                st.success(f"💾 모델 저장 완료: `{model_save_path}`")
                st.caption("이제 'main'페이지로 졸아가서 새로고침하면 적용 됩니다.")

                with st.expander("상세 결과 보고서"):
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.json(report)

            except Exception as e:
                st.error(f"학습 중 오류 발생: {e}")


