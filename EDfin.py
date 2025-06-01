from fer import FER
from fer.utils import draw_annotations
import numpy as np
import cv2

cap = cv2.VideoCapture(0) 

detector = FER(mtcnn=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    emotion = detector.detect_emotions(frame)
    edframe = draw_annotations(frame, emotion)
    
    cv2.imshow("ASDF", edframe)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break