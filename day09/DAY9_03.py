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

# 캐스케이드 분류기 설정
car_casecade = cv2.CascadeClassifier('cars.xml')

# 초당 1프레임 저장
while True:
    ret, frame = cap.read()
    
    # 프레임이 안 읽힌 경우는 통과
    if not ret:
        break

    # 관심영역 설정
    roi = frame[60:360, 200:500]

    # roi = frame[:, :] // 전체영역

    # 캐스케이드 분류기로 객체좌표를 가져와서 박스표시
    cars = car_casecade.detectMultiScale(roi)
    for (x, y, w, h) in cars:
        # cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(roi, (x, y, w, h), (0, 0, 255), 2)
        print(x, y)

    # 시각화
    cv2.imshow('frame', frame)
    cv2.imshow('roi', roi)

    # 중간에 나가는 키 설정
    if cv2.waitKey(1) == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows