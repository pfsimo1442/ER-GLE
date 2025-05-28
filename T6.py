import cv2
import numpy as np
import mediapipe as mp

change_background_mp = mp.solutions.selfie_segmentation
change_bg_segment = change_background_mp.SelfieSegmentation()

cap = cv2.VideoCapture(0)
#sample_img = cv2.imread('Resources/Imgs/sample.png')
bg_img = cv2.imread('Resources/Imgs/Background.png')
bg_img = cv2.resize(bg_img, (640, 480))

while True:
    ret, frame = cap.read()

    #RGB_sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #result = change_bg_segment.process(RGB_sample_img)
    result = change_bg_segment.process(RGB_frame)

    binary_mask = result.segmentation_mask > 0.75

    binary_mask_3 = np.dstack((binary_mask,binary_mask,binary_mask))
    output_image = np.where(binary_mask_3, frame, 255)

    output_image = np.where(binary_mask_3, frame, bg_img)     
    
    cv2.imshow('Image Binary Mask Crop Test', output_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()