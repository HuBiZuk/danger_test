# 1_extract_data.py
import cv2
import mediapipe as mp
import pandas as pd

# 설정
VIDEO_PATH = 'test_video.mp4'  # 생성한 영상 파일명 확인 필수!
OUTPUT_CSV = 'pose_data.csv'

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(VIDEO_PATH)
data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # 이름표를 확실하게 지정합니다 (rw_x, rw_y, re_x, re_y)
        row = {
            'rw_x': landmarks[16].x,  # Right Wrist X
            'rw_y': landmarks[16].y,  # Right Wrist Y
            're_x': landmarks[14].x,  # Right Elbow X
            're_y': landmarks[14].y,  # Right Elbow Y
            'label': 0
        }
        data.append(row)

cap.release()
df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)
print("1단계 완료: pose_data.csv 생성됨")