# 동영상 다운로드
# !wget https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4 -O BigBuckBunny_320x180.mp4

import cv2

# 파일 읽기
filepath = 'highway.mp4'

# 동영상 읽기
try:
    cap = cv2.VideoCapture(filepath)
    print('success!')
except:
    print('fail')
    # break

# 객체 감지기 생성
object_detector = cv2.createBackgroundSubtractorMOG2()

# 초당 1프레임 저장
while True:
    ret, frame = cap.read()
    
    # 프레임이 안 읽힌 경우는 통과
    if not ret:
        break
    
    # 흑백변환
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 마스크 생성
    mask = object_detector.apply(frame_gray)

    # 배경 영상 받아오기
    background = object_detector.getBackgroundImage()

    # 좌표 가져와서 바운딩 박스 그리기
    cnt, _, stats, _ = cv2.connectedComponentsWithStats(mask)

    for i in range(1, cnt):
        x, y, w, h, area = stats[i]

        if area < 80:
            continue

        cv2.rectangle(frame, (x, y, w, h), (0, 0, 255), 2)

    # 시각화
    cv2.imshow('frame', frame)
    cv2.imshow('background', background)
    cv2.imshow('mask', mask)

    # 중간에 나가는 키 설정
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows