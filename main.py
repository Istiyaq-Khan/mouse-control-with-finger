import cv2
import mediapipe as mp
import pyautogui
import math

cap = cv2.VideoCapture(0)
hand = mp.solutions.hands.Hands(max_num_hands=1)
draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

def calc_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # মিরর ইমেজ
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hand.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            # Index fingertip (point 8)
            x = int(landmarks[8].x * screen_width)
            y = int(landmarks[8].y * screen_height)
            pyautogui.moveTo(x, y)

            # Point 4 (thumb tip), point 6 (index base), point 20 (pinky tip)
            x4, y4 = int(landmarks[4].x * screen_width), int(landmarks[4].y * screen_height)
            x6, y6 = int(landmarks[6].x * screen_width), int(landmarks[6].y * screen_height)
            x20, y20 = int(landmarks[20].x * screen_width), int(landmarks[20].y * screen_height)

            # Distance check
            if calc_distance(x4, y6, x6, y6) < 40:
                pyautogui.click()  # Left Click
                pyautogui.sleep(0.2)

            elif calc_distance(x4, y20, x20, y20) < 40:
                pyautogui.rightClick()  # Right Click
                pyautogui.sleep(0.2)

            draw.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Mouse Control", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
