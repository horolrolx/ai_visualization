# casscade classifier를 이용한 객체감지
import cv2
from tracker import *

# 파일읽기
filepath = 'highway.mp4'

# 트래커 생성
tracker = EuclideanDistTracker()

# 영상 읽기
cap = cv2.VideoCapture(filepath)

# 영상이 안 열리는 경우 에러메세지
if not cap.isOpened():
    print("Video is unavailable :", filepath) 
    exit(0)

# 프레임마다 반복
while True:
    ret, frame = cap.read()    
    
    if not ret:
        break

    # 캐스케이드 분류기 생성
    car_casscade = cv2.CascadeClassifier('cars.xml') 
    cars = car_casscade.detectMultiScale(frame)
    
    # 바운딩 박스 표시
    detections = []
    for (x, y, w, h) in cars:
        # cv2.rectangle(frame, (x, y, w, h), (0, 0, 255), 2)
        detections.append([x,y,w,h])  
        # print(x, y)
    
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(frame, str(id), (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,255), 2)
        cv2.rectangle(frame, (x, y, w, h), (0, 0, 255), 2)     
    

    # 시각화
    cv2.imshow('frame', frame)

    # 중간에 나가는 키 설정
    if cv2.waitKey(20) == 27: # ESC
        break

# 비디오, 윈도우 종료
cap.release()
cv2.destroyAllWindows()