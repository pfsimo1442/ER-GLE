import cv2
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN

ImgPath = '.\Resources\SampleImgs\SI.jpg'

def extract_face(filename, required_size=(224, 224)):
	img = cv2.imread(filename)
	detector = MTCNN()

	for results in detector.detect_faces(img):
		x, y, w, h = results['box']
		cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), thickness=1)
	
	return img

# load the photo and extract the face
pixels = extract_face(ImgPath)
# plot the extracted face

while True:
	cv2.imshow("ASDF", pixels)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# cap.release()
cv2.destroyAllWindows()