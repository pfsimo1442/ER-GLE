import os
import cv2

import RemoveBackGround as rb
import CompareImg as ci

cpSize = (320, 320)
cpSize_3 = (320, 320, 3) 

cbFolder = 'Resources/Imgs'
subdir_names = os.listdir(cbFolder)

def CompareCelebrity(image):
    bstAcc = 0.0
    bstImg = None

    cv2.imshow(image)
    img = rb.BackGroundRemove(image)
    img = cv2.resize(img, cpSize)

    for file_name in subdir_names:
        cbImage = cv2.imread(cbFolder + '/' + file_name)
        cbImg = rb.BackGroundRemove(cbImage)
        cbImg = cv2.resize(cbImg, cpSize)

        acc = ci.CompareEachBhat(img, cbImg)

        if bstAcc < acc:
            bstAcc = acc
            bstImg = cbImage

    # bstAcc 0.0 ~ 1.0
    return bstAcc, bstImg, "asdf"