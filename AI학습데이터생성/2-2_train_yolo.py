import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 데이터 로드
try:
    df = pd.read_csv('final_data_v2.csv')
    print(f"데이터 로드 성공: {len(df)}개 샘플")
except FileNotFoundError:
    print("❌ 'final_data_v2.csv' 파일이 없습니다. 데이터 생성 코드를 먼저 실행해주세요.")
    exit()

# 2. 입력(X)와 정답(y) 분리
# 기존 6개 좌표 + 골반(rh_x, rh_y) 2개 추가 = 총 8개 특성
features = ['rw_x', 'rw_y', 're_x', 're_y', 'rs_x', 'rs_y', 'rh_x', 'rh_y']
X = df[features]
y = df['label']

# 3. 학습용/테스트용 데이터 분리 (8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 모델 생성 및 학습 (Random Forest)
print("모델 학습 중...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. 성능 평가
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n📊 정확도: {acc * 100:.2f}%")
print("\n[상세 리포트]")
# 0:Safe, 1:Reach(Danger), 2:Fall(낙상)
print(classification_report(y_test, y_pred, target_names=['Safe', 'Reach', 'Fall']))

# 6. 모델 저장
joblib.dump(model, 'model.pkl')
print("💾 'model.pkl' 저장 완료! (8개 입력 특성)")