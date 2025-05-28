import cv2
import numpy as np
import mediapipe as mp

import BackGroundRemove as br



brColor = (255, 0, 255)

cpSize = (320, 320)
cpSize_3 = (320, 320, 3) 

cap = cv2.VideoCapture(0)

# Main
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, cpSize)
    output_image = br.BackGroundRemove(frame, brColor)
    
    cv2.imshow('Image Binary Mask Crop Test', output_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()