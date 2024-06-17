import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mpHands = mp.solutions.hands

cap = cv2.VideoCapture(0)
hands = mpHands.Hands()

while True:
    data, image = cap.read()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks, mpHands.HAND_CONNECTIONS)
    cv2.imshow('HandTracker', image)
    cv2.waitKey(1)
