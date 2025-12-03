import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

print("최종 모델 학습 시작...")

try:
    df = pd.read_csv('../final_data.csv')
except:
    print("final_data.csv가 없습니다.")
    exit()

X = df[['rw_x', 'rw_y', 're_x', 're_y', 'rs_x', 'rs_y']]
y = df['label']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, '../model.pkl')
print("모델 갱신 완료! (차렷 자세 구분 가능)")