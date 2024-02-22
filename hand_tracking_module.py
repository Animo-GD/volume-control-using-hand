import cv2
import mediapipe as mp
import time


class hand_detector:
    def __init__(self, static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.static_image_mode,
            self.max_num_hands,
            self.model_complexity,
            self.min_detection_confidence,
            self.min_tracking_confidence,
        )
        self.mpDraw = mp.solutions.drawing_utils

    def find_hand(self, image, draw=True):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for lm in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, lm, self.mpHands.HAND_CONNECTIONS)
        return image

    def find_position(self,image,hand_number=0,draw=True):
        landmarks = []
        if self.results.multi_hand_landmarks:
            current_hand = self.results.multi_hand_landmarks[hand_number]
            for id, l in enumerate(current_hand.landmark):
                h, w, c = image.shape
                cx, cy = int(l.x * w), int(l.y * h)
                landmarks.append([id,cx,cy])
        return landmarks
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    detector = hand_detector()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = detector.find_hand(frame)
        landmarks = detector.find_position(frame)
        if landmarks:
            print(landmarks[4])

        cTime = time.time()
        fps = int(1 / (cTime - pTime))
        pTime = cTime
        cv2.putText(frame, f"{fps}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2
                    , (255, 0, 0), 2)
        cv2.imshow("cam", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
