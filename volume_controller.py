import cv2
import time
import numpy as np
import math
from hand_tracking_module import hand_detector
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
detector = hand_detector()

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]
vol = volume.GetMasterVolumeLevel()
vol_bar = np.interp(vol, [min_vol, max_vol], [400, 150])
vol_prc = np.interp(vol, [min_vol, max_vol], [0, 100])
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = detector.find_hand(frame)
    landmarks = detector.find_position(frame)
    if landmarks:
        x1, y1 = landmarks[4][1], landmarks[4][2]
        x2, y2 = landmarks[8][1], landmarks[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (x1, y1), 8, (255, 0, 0), -1)
        cv2.circle(frame, (x2, y2), 8, (255, 0, 0), -1)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

        length = math.hypot(x2 - x1, y2 - y1)
        vol = np.interp(length, [20, 300], [min_vol, max_vol])
        vol_bar = np.interp(length, [20, 300], [400, 150])
        vol_prc = np.interp(length, [20, 300], [0, 100])
        if length < 20:
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
        volume.SetMasterVolumeLevel(vol, None)

    # Drawing Volume Bar
    cv2.rectangle(frame, (50, 150), (80, 400), (0, 255, 0), 3)
    cv2.rectangle(frame, (50, int(vol_bar)), (80, 400), (0, 255, 0), -1)
    cv2.putText(frame, f"{int(vol_prc)} %", (40, 440), cv2.FONT_HERSHEY_PLAIN,
                2, (255, 0, 0), 2)
    cTime = time.time()
    fps = int(1 / (cTime - pTime))
    pTime = cTime
    cv2.putText(frame, f"FPS: {fps}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2
                , (255, 0, 0), 2)
    cv2.imshow("cam", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
