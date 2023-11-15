import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

previousTime = 0
currentTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector(detectionConfidence=int(0.5), trackConfidence=int(0.5))

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    # To show only hands
    #img = detector.findHands(img, draw = False)
    lmList = detector.findPosition(img, draw = False)
    if len(lmList) != 0:
        print(lmList[4])

    currentTime = time.time()
    FPScount = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, str(int(FPScount)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)