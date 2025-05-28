import cv2
import numpy as np
import mediapipe as mp

change_background_mp = mp.solutions.selfie_segmentation
change_bg_segment = change_background_mp.SelfieSegmentation()

nkAccuLim = 0.75

def Nukki(img, color_bg):
    result = change_bg_segment.process(img)

    binary_mask = result.segmentation_mask > nkAccuLim
    binary_mask_3 = np.dstack((binary_mask, binary_mask, binary_mask))

    edited_img = np.where(binary_mask_3, img, color_bg)

    return edited_img