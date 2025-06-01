import cv2
import CompareCelebrity as cc

imgPath = ".\Resources\SampleImgs\Einstain.jpg"
frame = cv2.imread(imgPath)

acc, img, name = cc.CompareCelebrity(frame)

print(acc)
while True:
    cv2.imshow("asdf", img)