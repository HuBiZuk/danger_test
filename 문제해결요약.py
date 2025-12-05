"""
[1203_1204]
# Streamlit 버전 호환성 문제 해결:
    streamlit과 streamlit-drawable-canvas 라이브러리의 버전 불일치로 인한 AttributeError (image_to_url, toggle, rerun 등) 문제가 발생했습니다.
    이를 해결하기 위해 streamlit을 1.22.0 버전으로, streamlit-drawable-canvas를 0.9.0 버전으로 낮추어 호환되는 환경을 구축했습니다.
    특히, streamlit run 명령어가 사용하는 실제 Python 환경(Python311)을 파악하여 해당 환경에 정확한 버전의 라이브러리들을 설치하는 것이 중요했습니다.
    구버전 Streamlit(1.22.0)에서 작동하도록 st.toggle을 st.checkbox로, st.rerun()을 st.experimental_rerun()으로, st.slider의 label_visibility 및 중복 label 인자를 수정했습니다.
    구역 그리기 좌표 틀어짐 문제의 근본 원인 파악 및 해결:


# "그릴 땐 정상인데 저장 시 좌표가 틀어지는" 현상의 원인
    streamlit-drawable-canvas가 drawing_mode="polygon"으로 그린 도형을 type: "polygon"이 아닌 type: "path" 객체로 반환한다는 사실을 파악하지 못했기 때문이었습니다.
    가장 결정적인 단서는 st.json(obj) 디버깅을 통해 얻은 RAW JSON 데이터였습니다. 이 데이터를 통해 path 객체가 originX: "center", originY: "center"일 때 obj["path"]의 좌표들이 이미 절대 캔버스 픽셀 좌표임을 확인했습니다.
    이를 바탕으로 view.py의 좌표 계산 로직을 수정하여, path 객체의 originX/originY 속성 값에 따라 obj["path"] 내부의 좌표(cmd[1], cmd[2])를 다르게 해석하도록 변경했습니다:
    originX/Y가 center인 경우: abs_x = cmd[1], abs_y = cmd[2] (절대 좌표를 그대로 사용)
그렇지 않은 경우 (기존에 initial_drawing으로 로드되는 path 객체): abs_x = left + cmd[1] * scaleX, abs_y = top + cmd[2] * scaleY (상대 좌표를 left/top 기준 스케일 적용하여 변환)

[AI생성좌표 손뻗음 쓰러짐 감지 학습 모델]
C:\Users\niceguysm\AppData\Local\Programs\Python\Python311\python.exe C:\Python_workspace\danger_test\백업\2-2_train_yolo.py
데이터 로드 성공: 6000개 샘플
모델 학습 중...

📊 정확도: 91.75%

[상세 리포트]
              precision    recall  f1-score   support

        Safe       0.89      0.88      0.89       432
       Reach       0.87      0.88      0.88       387
        Fall       0.99      0.99      0.99       381

    accuracy                           0.92      1200
   macro avg       0.92      0.92      0.92      1200
weighted avg       0.92      0.92      0.92      1200

💾 'model.pkl' 저장 완료! (8개 입력 특성)


"""