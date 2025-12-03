# 쓰러짐 가상 데이터 생성

import pandas as pd
import numpy as np
import random

# 데이터 개수 (많을수록 좋음)
NUM_SAMPLES = 5000
data = []

print("가상 쓰러짐 데이터 생성 중... (BBox 정규화 적용)")

for _ in range(NUM_SAMPLES):
    # 랜덤 노이즈 (사람마다 자세가 조금씩 다르니까)
    noise = lambda: np.random.uniform(-0.05, 0.05)

    # 자세 타입 결정 (0:서있음, 1:완전누움, 2:웅크려넘어짐)
    pose_type = np.random.choice(['stand', 'lying', 'crumpled'])

    row = {}

    if pose_type == 'stand':
        # === [CASE 0: 서 있거나 걷기] ===
        # 특징: 박스가 세로로 김 (Ratio < 1.0)
        # 좌표: 어깨는 위(0.1), 발은 아래(0.9)

        row['label'] = 0
        row['bbox_ratio'] = np.random.uniform(0.3, 0.8)  # 세로로 긴 박스

        # Y좌표 (위에서 아래로)
        sy = np.random.uniform(0.1, 0.2)  # 어깨 Y
        hy = np.random.uniform(0.4, 0.5)  # 골반 Y
        ky = np.random.uniform(0.6, 0.7)  # 무릎 Y
        ay = np.random.uniform(0.8, 0.95)  # 발목 Y

        # X좌표 (중앙 정렬)
        cx = 0.5
        row['ls_x'] = cx - 0.15 + noise();
        row['ls_y'] = sy + noise()
        row['rs_x'] = cx + 0.15 + noise();
        row['rs_y'] = sy + noise()
        row['lh_x'] = cx - 0.1 + noise();
        row['lh_y'] = hy + noise()
        row['rh_x'] = cx + 0.1 + noise();
        row['rh_y'] = hy + noise()
        row['lk_x'] = cx - 0.1 + noise();
        row['lk_y'] = ky + noise()
        row['rk_x'] = cx + 0.1 + noise();
        row['rk_y'] = ky + noise()
        row['la_x'] = cx - 0.1 + noise();
        row['la_y'] = ay + noise()
        row['ra_x'] = cx + 0.1 + noise();
        row['ra_y'] = ay + noise()

    elif pose_type == 'lying':
        # === [CASE 1-A: 완전히 뻗어 누움] ===
        # 특징: 박스가 가로로 김 (Ratio > 1.2)
        # 좌표: 모든 Y좌표가 중간(0.5) 근처에 모여있음

        row['label'] = 1
        row['bbox_ratio'] = np.random.uniform(1.2, 2.5)  # 가로로 긴 박스

        # Y좌표 (모두 비슷함)
        base_y = 0.5
        row['ls_y'] = base_y + noise()
        row['rs_y'] = base_y + noise()
        row['lh_y'] = base_y + noise()
        row['rh_y'] = base_y + noise()
        row['lk_y'] = base_y + noise()
        row['rk_y'] = base_y + noise()
        row['la_y'] = base_y + noise()
        row['ra_y'] = base_y + noise()

        # X좌표 (왼쪽 머리 -> 오른쪽 발, 또는 반대)
        if np.random.choice([True, False]):
            # 머리(좌) -> 발(우)
            row['ls_x'] = 0.1;
            row['rs_x'] = 0.1
            row['lh_x'] = 0.4;
            row['rh_x'] = 0.4
            row['lk_x'] = 0.7;
            row['rk_x'] = 0.7
            row['la_x'] = 0.9;
            row['ra_x'] = 0.9
        else:
            # 머리(우) -> 발(좌)
            row['ls_x'] = 0.9;
            row['rs_x'] = 0.9
            row['lh_x'] = 0.6;
            row['rh_x'] = 0.6
            row['lk_x'] = 0.3;
            row['rk_x'] = 0.3
            row['la_x'] = 0.1;
            row['ra_x'] = 0.1

        # 노이즈 추가
        for k in row:
            if k != 'label' and k != 'bbox_ratio':
                row[k] += noise()

    else:  # crumpled
        # === [CASE 1-B: 털썩 주저앉음/웅크림] ===
        # 특징: 박스가 정사각형에 가까움 (Ratio ≈ 1.0)
        # 좌표: 어깨가 바닥(y=1)에 가까워짐 (상체가 무너짐)

        row['label'] = 1
        row['bbox_ratio'] = np.random.uniform(0.8, 1.3)

        # 상체가 낮아짐 (어깨 Y가 0.4~0.6 까지 내려옴)
        sy = np.random.uniform(0.4, 0.7)
        hy = np.random.uniform(0.6, 0.8)
        ky = np.random.uniform(0.7, 0.9)
        ay = np.random.uniform(0.8, 0.95)

        cx = 0.5
        row['ls_x'] = cx - 0.2 + noise();
        row['ls_y'] = sy + noise()
        row['rs_x'] = cx + 0.2 + noise();
        row['rs_y'] = sy + noise()
        row['lh_x'] = cx - 0.15 + noise();
        row['lh_y'] = hy + noise()
        row['rh_x'] = cx + 0.15 + noise();
        row['rh_y'] = hy + noise()
        row['lk_x'] = cx - 0.1 + noise();
        row['lk_y'] = ky + noise()
        row['rk_x'] = cx + 0.1 + noise();
        row['rk_y'] = ky + noise()
        row['la_x'] = cx - 0.1 + noise();
        row['la_y'] = ay + noise()
        row['ra_x'] = cx + 0.1 + noise();
        row['ra_y'] = ay + noise()

    data.append(row)

# 저장
df = pd.DataFrame(data)
df.to_csv('fall_data.csv', index=False)
print("완료! 'fall_data.csv' 파일 생성됨.")
print("특징: 박스 비율(Ratio)과 박스 내부 상대 좌표(0~1)로 학습합니다.")