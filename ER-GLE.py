import cv2
import numpy as np
import mediapipe as mp

import DetectEmotion as de
import CompareCelebrity as cc


roih, roiw = 360, 360

cap = cv2.VideoCapture(0)

# Main
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # cv2.imshow('test', frame)

    # 콜 오면
    # fh, fw, fc = frame.shape
    # rx = (fw - roiw) / 2 
    # ry = (fh - roih) / 2
    # roi = frame[ry:ry+fw, rx:rx+fh]
    # cbAcc, cbImg, cbName = cc.CompareCelebrity(roi)

    output_image = de.DetectEmotion(frame)

    output_image = np.array(output_image, dtype=np.uint8)
    cv2.imshow('Image Binary Mask Crop Test', output_image)

    # 버퍼 전송
    # return output_image

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()