import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 데이터 로드
try:
    df = pd.read_csv('fall_data.csv')
except FileNotFoundError:
    print("fall_data.csv가 없습니다. 1단계를 실행하세요.")
    exit()

# 입력(X), 정답(y)
X = df.drop('label', axis=1)
y = df['label']

# 학습 (RandomForest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 저장
joblib.dump(model, 'fall_model.pkl')
print("쓰러짐 감지 모델(fall_model.pkl) 생성 완료!")