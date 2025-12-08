import pandas as pd
import numpy as np
import math

NUM_SAMPLES = 6000  # 데이터 개수 증가
data = []


def rotate_point(cx, cy, angle_deg, length):
    rad = math.radians(angle_deg)
    return cx + length * math.cos(rad), cy + length * math.sin(rad)


print("데이터 생성 시작... (상체, 손뻗음, 낙상 포함)")

for _ in range(NUM_SAMPLES):
    # ---------------------------------------------------------
    # 1. 기본 신체 스펙 설정 (정규화된 좌표 0.0 ~ 1.0 기준)
    # ---------------------------------------------------------
    torso_len = np.random.uniform(0.15, 0.25)  # 몸통 길이
    arm_len = 0.12
    forearm_len = 0.12

    # 상황 결정 (0:안전, 1:손뻗음, 2:쓰러짐)
    # 비율 조정: 안전 40%, 손뻗음 30%, 쓰러짐 30%
    scenario = np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3])

    # ---------------------------------------------------------
    # CASE 2: [낙상/쓰러짐] - 몸통이 가로로 누움
    # ---------------------------------------------------------
    if scenario == 2:
        # 어깨 위치 (바닥 근처나 화면 중앙 등 다양하게)
        sx = np.random.uniform(0.2, 0.8)
        sy = np.random.uniform(0.4, 0.9)

        # 몸통 각도: 누워있음 (0도=우측, 180도=좌측) +- 30도 변동
        if np.random.choice([True, False]):
            body_angle = np.random.normal(0, 20)  # 머리 좌측, 다리 우측
        else:
            body_angle = np.random.normal(180, 20)  # 머리 우측, 다리 좌측

        # 골반 위치 계산 (어깨 기준 몸통 각도로 배치)
        rh_x, rh_y = rotate_point(sx, sy, body_angle, torso_len)

        # 팔 위치: 힘없이 쳐지거나 바닥에 놓임 (몸통 각도와 비슷하거나 중력 방향)
        arm_angle = body_angle + np.random.uniform(-40, 40)
        ex, ey = rotate_point(sx, sy, arm_angle, arm_len)

        forearm_angle = arm_angle + np.random.uniform(-30, 30)
        wx, wy = rotate_point(ex, ey, forearm_angle, forearm_len)

        label = 2  # 쓰러짐

    # ---------------------------------------------------------
    # 서 있는 상태 (안전 or 손뻗음) - 몸통이 세로
    # ---------------------------------------------------------
    else:
        # 어깨 위치: 상체만 나오는 경우(Zoom)를 고려해 화면 꽉 차게도 설정
        sx = np.random.uniform(0.2, 0.8)
        sy = np.random.uniform(0.1, 0.7)

        # 몸통 각도: 서 있음 (90도=아래쪽) +- 10도 기울기
        body_angle = np.random.normal(90, 10)
        rh_x, rh_y = rotate_point(sx, sy, body_angle, torso_len)

        # [CASE 0: 안전] - 차렷 or 뒷짐 or 통화
        if scenario == 0:
            is_hanging_down = np.random.choice([True, False])

            if is_hanging_down:
                # 차렷: 팔이 몸통 방향(아래)과 유사 (70~110도)
                angle_s_e = np.random.normal(90, 15)
                ex, ey = rotate_point(sx, sy, angle_s_e, arm_len)
                angle_e_w = angle_s_e + np.random.uniform(-10, 10)
                wx, wy = rotate_point(ex, ey, angle_e_w, forearm_len)
            else:
                # 굽힘: 각도는 자유로우나 팔꿈치가 접혀있음
                angle_s_e = np.random.uniform(0, 360)
                ex, ey = rotate_point(sx, sy, angle_s_e, arm_len)

                bend = np.random.uniform(60, 140)  # 많이 접힘
                angle_e_w = angle_s_e + (bend if np.random.choice([True, False]) else -bend)
                wx, wy = rotate_point(ex, ey, angle_e_w, forearm_len)

            label = 0

        # [CASE 1: 위험/손뻗음] - 팔이 펴져있고 + 몸통에서 멀어짐
        else:
            # 팔 각도: 아래쪽(차렷)을 제외한 모든 방향
            # 몸통이 90도니까, 팔은 0~60도(우상) or 120~180(좌상) or 200~340(위)
            valid_angles = list(range(130, 360)) + list(range(0, 50))
            angle_s_e = np.random.choice(valid_angles)

            ex, ey = rotate_point(sx, sy, angle_s_e, arm_len)

            # 팔이 펴짐 (팔꿈치 각도 차이 적음)
            angle_e_w = angle_s_e + np.random.uniform(-20, 20)
            wx, wy = rotate_point(ex, ey, angle_e_w, forearm_len)

            label = 1

    # 데이터 추가 (골반 rh_x, rh_y 추가됨)
    data.append({
        'rw_x': wx, 'rw_y': wy,  # 오른 손목
        're_x': ex, 're_y': ey,  # 오른 팔꿈치
        'rs_x': sx, 'rs_y': sy,  # 오른 어깨
        'rh_x': rh_x, 'rh_y': rh_y,  # 오른 골반 (New!)
        'label': label
    })

# 저장
df = pd.DataFrame(data)
df.to_csv('final_data_v2.csv', index=False)
print(f"생성 완료! 총 {NUM_SAMPLES}개")
print("컬럼 구성: rw_x, rw_y, re_x, re_y, rs_x, rs_y, rh_x, rh_y, label")
print("라벨 설명: 0(안전), 1(손뻗음-위험), 2(쓰러짐-낙상)")