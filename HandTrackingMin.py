import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# For hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# for hand tracking lines
mpDraw = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0



while True:
    success, img = cap.read()
    # Converting image into RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Getting X and Y values for Hand
    #print(results.multi_hand_landmarks)

    # using for loop check if there are one hand or multi hands
    if results.multi_hand_landmarks:
        for handLMS in results.multi_hand_landmarks:
            for id, lm in enumerate(handLMS.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 4:
                   cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLMS, mpHands.HAND_CONNECTIONS)

    currentTime = time.time()
    FPScount = 1 / (currentTime - previousTime)
    previousTime = currentTime

    #cv2.putText(img, str(int(FPScount)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)