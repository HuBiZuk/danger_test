import pandas as pd
import numpy as np
import math

# 데이터 개수 (많을수록 좋음)
NUM_SAMPLES = 5000
data = []


def rotate_point(cx, cy, angle_deg, length):
    """중심점(cx, cy)에서 각도와 거리만큼 떨어진 점의 좌표를 계산"""
    rad = math.radians(angle_deg)
    x = cx + length * math.cos(rad)
    y = cy + length * math.sin(rad)
    return x, y


print("YOLO 맞춤형 데이터 생성 중... (어깨-팔꿈치-손목)")

for _ in range(NUM_SAMPLES):
    # 1. 어깨 위치 (화면 어디에나 있을 수 있음 -> 위치 의존성 제거)
    sx = np.random.uniform(0.1, 0.9)
    sy = np.random.uniform(0.1, 0.9)

    # 팔 길이 (화면 비율 기준)
    arm_len = 0.15  # 상박
    forearm_len = 0.15  # 하박

    # === [CASE 0: Safe (안전)] ===
    # 팔이 굽혀져 있는 상태 (전화 받기, 머리 긁기, 뒷짐 등)

    # 어깨 -> 팔꿈치 각도 (랜덤)
    angle_s_e = np.random.uniform(0, 360)
    ex, ey = rotate_point(sx, sy, angle_s_e, arm_len)

    # 팔꿈치 -> 손목 각도 (Safe: 팔꿈치 각도가 급격하게 꺾임)
    # 팔이 펴지는 방향에서 60~140도 정도 꺾여야 안전
    bend = np.random.uniform(60, 140)
    # 방향을 랜덤으로(안쪽으로 굽히거나 바깥으로 굽히거나)
    if np.random.choice([True, False]):
        angle_e_w = angle_s_e + bend
    else:
        angle_e_w = angle_s_e - bend

    wx, wy = rotate_point(ex, ey, angle_e_w, forearm_len)

    # 데이터 저장 (YOLO가 주는 순서: 손목 -> 팔꿈치 -> 어깨 순으로 컬럼명 매칭)
    # 3_app.py에서 사용하는 컬럼명: ['rw_x', 'rw_y', 're_x', 're_y', 'rs_x', 'rs_y']
    data.append({
        'rw_x': wx, 'rw_y': wy,
        're_x': ex, 're_y': ey,
        'rs_x': sx, 'rs_y': sy,
        'label': 0  # 안전
    })

    # === [CASE 1: Danger (위험)] ===
    # 팔을 쭉 뻗은 상태 (전시품 터치 시도)

    # 어깨 -> 팔꿈치 각도 (랜덤)
    angle_s_e_d = np.random.uniform(0, 360)
    ex_d, ey_d = rotate_point(sx, sy, angle_s_e_d, arm_len)

    # 팔꿈치 -> 손목 각도 (Danger: 거의 일직선)
    # -15도 ~ +15도 오차범위 내에서 쭉 뻗음
    straight_variance = np.random.uniform(-15, 15)
    angle_e_w_d = angle_s_e_d + straight_variance

    wx_d, wy_d = rotate_point(ex_d, ey_d, angle_e_w_d, forearm_len)

    data.append({
        'rw_x': wx_d, 'rw_y': wy_d,
        're_x': ex_d, 're_y': ey_d,
        'rs_x': sx, 'rs_y': sy,
        'label': 1  # 위험
    })

# CSV 저장
df = pd.DataFrame(data)
df.to_csv('yolo_action_data.csv', index=False)
print("완료! 'yolo_action_data.csv' 파일 생성됨.")