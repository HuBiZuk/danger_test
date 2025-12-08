"""
# 커밋 & 푸쉬
1. 변경된 파일 담기
    git add .

2. 설명 적고 저장하기 (커밋)
    git commit -m "(1208 AI증강데이터 학습구현)"

3. 깃허브에 올리기 (푸쉬)
    git push origin main

💡 혹시 에러가 난다면?
    git push -f origin main

----------------------------
# 풀
상황 1: 아예 처음 다운로드 받을 때 (학원 컴퓨터 등)
    git clone https://github.com/HuBiZuk/danger_test.git

상황 2: 이미 폴더가 있는데, 최신 내용으로 업데이트할 때
    git pull origin main

🚨 (비상용) "충돌 나서 안 받아져요!" 할 때
    # 1. 깃허브 최신 정보 가져오기
    git fetch --all

    # 2. 내 컴퓨터 상태를 깃허브(main)랑 똑같이 강제로 맞추기
    git reset --hard origin/main


"""