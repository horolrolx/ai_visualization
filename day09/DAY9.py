# 동영상 다운로드
# !wget https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4 -O BigBuckBunny_320x180.mp4

import cv2

# 파일 읽기
filepath = 'BigBuckBunny_320x180.mp4'

# 동영상 읽기
try:
    cap = cv2.VideoCapture(filepath)
    print('success!')
except:
    print('fail')
    # break

# 영상이 안 열리는 경우 에러 메시지
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if not cap.isOpened():
    print("Video is unavailable :", filepath)
    exit(0)

# 전체 프레임 갯수 저장
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 초당 프레임 갯수 저장
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 파일을 저장한 번호 저장
count = 0

# 초당 1프레임 저장
while True:
    ret, frame = cap.read()
    if int(cap.get(1)) % int(fps) == 0:
        cv2.imwrite('BigBuckBunny_320x180.mp4' + '/frame{}.jpg'.format(cap.get(1)), frame)

    print('Saved frame : {}.jpg'.format(cap.get(1)))
    count += 1

    # 마지막 프레임에 도달하면 종료
    if cap.get(1) == length:
        break

    # ESC 키를 누르면 종료
    key = cv2.waitKey(33)
    if key == 27: # ESC
        break 

cap.release()
cv2.destroyAllWindows()