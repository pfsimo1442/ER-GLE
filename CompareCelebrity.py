import os
import cv2

cbFolder = 'Resources/CelebrityImgs'
subdir_names = os.listdir(cbFolder)
for file_name in subdir_names:
    csv = cv2.imread(cbFolder + '/' + file_name)

def CompareCelebrities(image):
    a = 0