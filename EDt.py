from fer import FER
from fer.utils import draw_annotations
from mtcnn.mtcnn import MTCNN
import numpy as np
import cv2

"""
emotions_dict = {
    "angry": {"tr": "Kizgin", "de": "Wutend"},
    "disgust": {"tr": "Igrenme", "de": "der Ekel"},
    "fear": {"tr": "Korku", "de": "Furcht"},
    "happy": {"tr": "Mutluluk", "de": "Glucklich"},
    "sad": {"tr": "Uzuntu", "de": "Traurig"},
    "surprise": {"tr": "Saskinlik", "de": "Uberraschung"},
    "neutral": {"tr": "Notr", "de": "Neutral"},
}

def draw_scores(
    frame: np.ndarray,
    emotions: dict,
    bounding_box: dict,
    lang: str = "en",
    size_multiplier: int = 1,
) -> np.ndarray:
    # Draw scores for each emotion under faces.
    GRAY = (211, 211, 211)
    GREEN = (0, 255, 0)
    x, y, w, h = bounding_box

    for idx, (emotion, score) in enumerate(emotions.items()):
        color = GRAY if score < 0.01 else GREEN

        if lang != "en":
            emotion = emotions_dict[emotion][lang]

        emotion_score = "{}: {}".format(
            emotion, "{:.2f}".format(score) if score >= 0.01 else ""
        )
        cv2.putText(
            frame,
            emotion_score,
            (
                x,
                y + h + (15 * size_multiplier) + idx * (15 * size_multiplier),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * size_multiplier,
            color,
            1 * size_multiplier,
            cv2.LINE_AA,
        )
    return frame

def draw_annotations(
    frame: np.ndarray,
    faces: list,
    boxes=True,
    scores=True,
    color: tuple = (0, 155, 255),
    lang: str = "en",
    size_multiplier: int = 1,
) -> np.ndarray:
    # Draws boxes around detected faces. Faces is a list of dicts with `box` and `emotions`.
    if not len(faces):
        return frame

    for face in faces:
        x, y, w, h = face["box"]
        emotions = face["emotions"]

        if boxes:
            cv2.rectangle(
                frame,
                (x, y, w, h),
                color,
                2,
            )

        if scores:
            frame = draw_scores(frame, emotions, (x, y, w, h), lang, size_multiplier)
    return frame
"""

img = cv2.imread(".\Resources\SampleImgs\SI.jpg")  
detector = FER(mtcnn=True)
emotion = detector.detect_emotions(img)
img = draw_annotations(img, emotion)

while True:
	cv2.imshow("ASDF", img)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break