import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, mxHands=2, detectionConfidence = 0.5, trackConfidence = 0.5):
        self.mode = mode
        self.mxHands = mxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        # For hand detection
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.mxHands, self.detectionConfidence, self.trackConfidence)

        # for hand tracking lines
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw = True):
        # Converting image into RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # using for loop check if there are one hand or multi hands
        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw = True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

        return lmList


def main():
    previousTime = 0
    currentTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector(detectionConfidence=int(0.5), trackConfidence=int(0.5))

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])


        currentTime = time.time()
        FPScount = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(img, str(int(FPScount)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()