import pandas as pd
import numpy as np
import math

NUM_SAMPLES = 5000
data = []


def rotate_point(cx, cy, angle_deg, length):
    rad = math.radians(angle_deg)
    return cx + length * math.cos(rad), cy + length * math.sin(rad)


print("방향성(차렷 자세)을 고려한 데이터 생성 중...")

for _ in range(NUM_SAMPLES):
    sx = np.random.uniform(0.2, 0.8)
    sy = np.random.uniform(0.2, 0.6)  # 어깨는 보통 화면 상단~중간

    arm_len = 0.15
    forearm_len = 0.15

    # ==========================================
    # [CASE 0: Safe (안전)]
    # 1. 팔을 굽힘 OR 2. 팔을 펴고 아래로 내림(차렷)
    # ==========================================

    is_hanging_down = np.random.choice([True, False])

    if is_hanging_down:
        # [안전-A] 차렷 자세 (팔이 펴져 있지만 아래를 향함)
        # 이미지 좌표계: 0도(우), 90도(하), 180도(좌), 270도(상)
        # 아래쪽(70도 ~ 110도 사이)
        angle_s_e = np.random.uniform(70, 110)
        ex, ey = rotate_point(sx, sy, angle_s_e, arm_len)

        # 팔꿈치->손목도 비슷하게 아래로 (거의 일자)
        angle_e_w = angle_s_e + np.random.uniform(-10, 10)
        wx, wy = rotate_point(ex, ey, angle_e_w, forearm_len)

    else:
        # [안전-B] 팔을 아무 방향이나 굽히고 있음 (전화, 뒷짐)
        angle_s_e = np.random.uniform(0, 360)
        ex, ey = rotate_point(sx, sy, angle_s_e, arm_len)

        # 굽힘 (90도 전후)
        bend = np.random.uniform(50, 130)
        if np.random.choice([True, False]):
            angle_e_w = angle_s_e + bend
        else:
            angle_e_w = angle_s_e - bend
        wx, wy = rotate_point(ex, ey, angle_e_w, forearm_len)

    data.append({
        'rw_x': wx, 'rw_y': wy, 're_x': ex, 're_y': ey, 'rs_x': sx, 'rs_y': sy,
        'label': 0  # 안전
    })

    # ==========================================
    # [CASE 1: Danger (위험)]
    # 팔을 쭉 뻗음 AND (아래쪽 방향이 아님)
    # ==========================================

    # 방향: 위(270), 왼쪽(180), 오른쪽(0) 근처.
    # 즉, 아래쪽(60~120도)을 제외한 나머지 각도
    valid_angles = list(range(120, 360)) + list(range(0, 60))
    angle_s_e_d = np.random.choice(valid_angles)

    # 랜덤 노이즈 추가
    angle_s_e_d += np.random.uniform(-10, 10)

    ex_d, ey_d = rotate_point(sx, sy, angle_s_e_d, arm_len)

    # 팔 펴짐 (거의 일자)
    angle_e_w_d = angle_s_e_d + np.random.uniform(-15, 15)
    wx_d, wy_d = rotate_point(ex_d, ey_d, angle_e_w_d, forearm_len)

    data.append({
        'rw_x': wx_d, 'rw_y': wy_d, 're_x': ex_d, 're_y': ey_d, 'rs_x': sx, 'rs_y': sy,
        'label': 1  # 위험
    })

df = pd.DataFrame(data)
df.to_csv('final_data.csv', index=False)
print("완료! 이제 AI는 '차렷 자세'를 안전하다고 판단합니다.")