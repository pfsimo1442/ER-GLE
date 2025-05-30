import cv2
import numpy as np
import mediapipe as mp

change_background_mp = mp.solutions.selfie_segmentation
change_bg_segment = change_background_mp.SelfieSegmentation()

brColor = (255, 0, 255)
brAccuLim = 0.9

def BackGroundRemove(img):
    result = change_bg_segment.process(img)

    binary_mask = result.segmentation_mask > brAccuLim
    binary_mask_3 = np.dstack((binary_mask, binary_mask, binary_mask))

    edited_img = np.where(binary_mask_3, img, brColor)

    return edited_img