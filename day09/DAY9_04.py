# 동영상 다운로드
# !wget https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4 -O BigBuckBunny_320x180.mp4

import cv2
import numpy as np
from tracker import EuclideanDistTracker

# 파일 읽기
filepath = 'highway.mp4'

# 트레커 생성
tracker = EuclideanDistTracker()

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
    
    # 관심영역 설정
    roi = frame[60:360, 220:500]

    # 마스크 생성
    mask = object_detector.apply(roi)
    
    # 마스크에 임계값 적용
    _, mask = cv2.threshold(mask, 230, 255, cv2.THRESH_BINARY)

    # 외곽선 좌표 추출
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])
            
            cv2.rectangle(roi, (x, y, w, h), (0, 0, 255), 2)

    boxes_ids = tracker.update(detections)
    
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.rectangle(roi, (x, y, w, h), (0, 255, 0), 2)
        cv2.putText(roi, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 시각화
    cv2.imshow('frame', frame)
    # cv2.imshow('background', background)
    cv2.imshow('mask', mask)
    cv2.imshow('roi', roi)

    # 중간에 나가는 키 설정
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows