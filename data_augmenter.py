import pandas as pd
import numpy as np


# --- 헬퍼 함수: x, y 좌표 인덱스 분리 ---
def get_xy_indices(num_cols):
    """
    x: 0, 2, 4... (짝수 인덱스)
    y: 1, 3, 5... (홀수 인덱스)
    """
    x_indices = np.arange(0, num_cols, 2)
    y_indices = np.arange(1, num_cols, 2)
    return x_indices, y_indices


# === 증강기법 ===
# --- 1. 크기조절 (Scaling) ---
def apply_scaling(data, scale_factor, x_indices, y_indices):
    scaled = data.copy()
    scaled[:, x_indices] *= scale_factor
    scaled[:, y_indices] *= scale_factor
    return scaled


# --- 2. 위치 이동(Translation) ---
def apply_translation(data, delta_x, delta_y, x_indices, y_indices):
    translated = data.copy()
    translated[:, x_indices] += delta_x
    translated[:, y_indices] += delta_y
    return translated


# --- 3. 노이즈 추가 (Jittering) ---
def apply_jittering(data, x_indices, y_indices, noise_scale=1.0):
    noise = np.random.normal(0, noise_scale, data.shape)
    jittered = data.copy()
    jittered[:, x_indices] += noise[:, x_indices]
    jittered[:, y_indices] += noise[:, y_indices]
    return jittered


# --- 4. 좌우 반전(Mirroring) ---
def apply_mirroring_2d(data, x_indices, max_x):
    mirrored = data.copy()
    mirrored[:, x_indices] = max_x - mirrored[:, x_indices]
    return mirrored


# --- [메인] 증강 배율 적용 ---
def run_augmentation(df_original, neutral_factor, movement_factor, threat_factor):
    # 1. 컬럼 분리 ('v'로 시작하는 좌표 vs 나머지 메타데이터)
    v_cols = [c for c in df_original.columns if c.startswith('v')]
    other_cols = [c for c in df_original.columns if c not in v_cols]

    if not v_cols:
        raise ValueError("데이터에 'v'로 시작하는 좌표 컬럼이 없습니다.")
    if 'label' not in df_original.columns:
        raise ValueError("'label' 컬럼이 필요합니다.")

    # 2. 데이터를 Numpy 배열로 변환 (인덱스 에러 원천 차단)
    try:
        coords = df_original[v_cols].values.astype(float)  # 좌표값
    except ValueError:
        # 혹시 v컬럼에 문자가 섞여있을 경우 대비
        print("⚠️ 주의: 좌표 데이터에 숫자가 아닌 값이 포함되어 있어 강제 변환합니다.")
        coords = df_original[v_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values

    meta = df_original[other_cols].values  # 나머지 정보

    num_coords = len(v_cols)
    x_idx, y_idx = get_xy_indices(num_coords)

    # 화면 크기 추정
    try:
        max_val = np.nanmax(coords[:, x_idx])
        max_x = 1920.0 if max_val > 2.0 else 1.0
    except:
        max_x = 1920.0

    # 3. 라벨 텍스트 처리 (숫자 0, 1, 2 포함)
    label_col_idx = other_cols.index('label')
    labels = meta[:, label_col_idx].astype(str)
    labels = np.char.strip(labels)  # 공백 제거

    print(f"👉 [DEBUG] 발견된 라벨 목록: {np.unique(labels)}")

    # 키워드 정의 (숫자 '0', '1', '2' 추가됨)
    threat_kws = ['손뻗기', '손 뻗기', '주머니', '절도', '던지기', '주먹', '밀치기', '공격', '위협', 'threat', '2']
    move_kws = ['걷기', '이동', '뒷걸음', '회전', '통화하며', 'movement', '1']
    neutral_kws = ['정지', '뒷짐', '팔짱', '앉은', '앉아', '쪼그려', '핸드폰', '머리', '얼굴', '기본', 'neutral', '0']

    # 벡터 검색 함수
    def check_keywords(label_arr, keywords):
        cond = np.zeros(len(label_arr), dtype=bool)
        for kw in keywords:
            cond |= np.char.find(label_arr, kw) != -1
        return cond

    # 그룹 분류
    is_threat = check_keywords(labels, threat_kws)
    is_move = check_keywords(labels, move_kws) & (~is_threat)
    is_neutral = check_keywords(labels, neutral_kws) & (~is_threat) & (~is_move)

    # 분류 안 된 나머지는 Neutral로 처리
    is_others = ~(is_threat | is_move | is_neutral)
    if np.any(is_others):
        print(f"⚠️ 분류되지 않은 데이터 {np.sum(is_others)}개는 'Neutral'로 처리합니다.")
        is_neutral |= is_others

    idx_threat = np.where(is_threat)[0]
    idx_move = np.where(is_move)[0]
    idx_neutral = np.where(is_neutral)[0]

    print(f"📊 분류 결과 - Threat: {len(idx_threat)}, Move: {len(idx_move)}, Neutral: {len(idx_neutral)}")

    # 4. 내부 증강 함수 (배열 기반 - 고속)
    def augment_group(indices, factor):
        if len(indices) == 0: return None, None

        src_coords = coords[indices]
        src_meta = meta[indices]

        out_coords_list = [src_coords]
        out_meta_list = [src_meta]

        num_aug = int(factor) - 1

        if num_aug > 0:
            for _ in range(num_aug):
                new_coords = src_coords.copy()

                # 랜덤 변환 적용
                if np.random.rand() < 0.3:
                    new_coords = apply_mirroring_2d(new_coords, x_idx, max_x)

                s = np.random.uniform(0.9, 1.1)
                new_coords = apply_scaling(new_coords, s, x_idx, y_idx)

                dx = np.random.uniform(-max_x * 0.05, max_x * 0.05)
                dy = np.random.uniform(-max_x * 0.02, max_x * 0.02)
                new_coords = apply_translation(new_coords, dx, dy, x_idx, y_idx)

                noise_val = 2.0 if max_x > 2.0 else 0.005
                new_coords = apply_jittering(new_coords, x_idx, y_idx, noise_val)

                out_coords_list.append(new_coords)
                out_meta_list.append(src_meta)

        return np.vstack(out_coords_list), np.vstack(out_meta_list)

    # 5. 실행
    tc, tm = augment_group(idx_threat, threat_factor)
    mc, mm = augment_group(idx_move, movement_factor)
    nc, nm = augment_group(idx_neutral, neutral_factor)

    # 6. 결과 병합
    final_c_list = []
    final_m_list = []

    if tc is not None: final_c_list.append(tc); final_m_list.append(tm)
    if mc is not None: final_c_list.append(mc); final_m_list.append(mm)
    if nc is not None: final_c_list.append(nc); final_m_list.append(nm)

    if not final_c_list: return df_original

    final_coords = np.vstack(final_c_list)
    final_meta = np.vstack(final_m_list)

    # DataFrame 복원
    df_c = pd.DataFrame(final_coords, columns=v_cols)
    df_m = pd.DataFrame(final_meta, columns=other_cols)

    return pd.concat([df_c, df_m], axis=1)