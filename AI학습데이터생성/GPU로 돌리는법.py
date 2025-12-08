# 터미널 입력
"""
1단계: 내 파이썬이 GPU를 인식하는지 확인

    python -c "import torch; print(torch.cuda.is_available())"

    결과가 **False**라고 뜨면? 👉 100% CPU 버전이 설치된 상태입니다. (2단계로 가세요)
    결과가 **True**라고 뜨면? 👉 설정만 조금 바꾸면 됩니다. (3단계로 가세요)

2단계: CPU 버전 삭제하고 GPU 버전 재설치 (가장 중요)
    1. 기존 버전 삭제
        pip uninstall torch torchvision torchaudio -y

    2. GPU 버전 설치 (CUDA 12.1 버전 기준)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

3. 설치 후 다시 확인
다시 python -c "import torch; print(torch.cuda.is_available())"를 쳐서 **True**가 나오는지 꼭 확인하세요!
-----------------------------------------------
3단계: 코드에서 "GPU 써!"라고 명령하기
3_app.py 코드에서 YOLO 모델을 돌릴 때, device=0 (0번 그래픽카드) 옵션을 명시해주면 확실하게 GPU를 씁니다.
수정할 부분 (두 군데)
YOLO 모델 로드 부분 (try-except 블록)
(사실 로드할 때는 큰 상관없지만, 추론할 때가 중요합니다.)
process_frame 함수 내부의 추론 부분 (여기가 핵심!)
results = yolo_model(..., device=0) 을 추가합니다.
▼ 수정된 3_app.py 코드 (해당 부분만)

    # ... (생략) ...

    # --- [함수] 프레임 분석 ---
    def process_frame(frame, yolo_model, custom_model, settings):
        # ... (생략) ...

        # [수정] device=0 옵션 추가! (이게 있어야 GPU로 돌아갑니다)
        # 만약 GPU가 없어서 에러나면 device='cpu'로 바꾸거나 device 옵션을 지우세요.
        results = yolo_model(frame, verbose=False, conf=0.25, device=0)

        annotated_frame = results[0].plot()

        # ... (나머지 코드는 그대로) ...
-----------------------------------------------------------------
💡 학원 컴퓨터에서 할 일 (딱 2개)
학원 컴퓨터에 가서 프로젝트를 실행할 때, 이 2가지만 기억하세요.
드라이버 확인 (선택):
NVIDIA 그래픽 드라이버가 너무 옛날 거면 인식이 안 될 수 있습니다. 가능하다면 업데이트하세요. (안 해도 CPU로 돌아가게 막아놨으니 안심하세요.)
설치 명령어 (필수):
학원 컴퓨터에서도 GPU용 PyTorch를 깔아야 합니다. 터미널에 이걸 입력하세요.
code
Bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
(만약 학원 인터넷이 느려서 다운로드가 너무 오래 걸리면, 그냥 pip install ultralytics만 해도 기본 CPU 버전으로 설치되어 작동은 합니다.)
결론: 위 코드로 수정하시면, GTX 1060이 인식되면 쌩쌩 돌아가고, 인식이 안 돼도 죽지 않고 CPU로 돌아갑니다. 안심하고 가져가세요! 👍
"""