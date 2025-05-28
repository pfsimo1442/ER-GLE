import cv2
import numpy as np
import mediapipe as mp

import Nukki as nk



nkColor = (255, 0, 255)

cpSize = (320, 320)
cpSize_3 = (320, 320, 3) 

cap = cv2.VideoCapture(0)

# Main
while True:
    ret, frame = cap.read()
    if not ret:
        break

    RGB_frame = cv2.resize(RGB_frame, cpSize)
    output_image = nk.Nukki(RGB_frame, nkColor)
    
    cv2.imshow('Image Binary Mask Crop Test', output_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()