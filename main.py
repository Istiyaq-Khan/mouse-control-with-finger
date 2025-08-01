import cv2  # OpenCV – for webcam image processing  
import mediapipe as mp   # MediaPipe – for hand tracking 
import pyautogui  # pyautogui – to control mouse movements/clicks 
import math   # math – for calculating distances

cap = cv2.VideoCapture(0)  # Start webcam (0 is usually the default cam)
hand = mp.solutions.hands.Hands(max_num_hands=1)   # Detect only 1 hand  
draw = mp.solutions.drawing_utils   # For drawing hand landmarks  

screen_width, screen_height = pyautogui.size()  # Get your screen size for mapping hand movement 

def calc_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)   # Euclidean distance formula 

while True:
    success, img = cap.read()    # Read one frame from the webcam  
    img = cv2.flip(img, 1)  # Flip horizontally (mirror image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # Convert to RGB for MediaPipe  
    result = hand.process(img_rgb)    # Process the image to detect hand landmarks  

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark  # All hand keypoints

            # Index fingertip (point 8)
            x = int(landmarks[8].x * screen_width)
            y = int(landmarks[8].y * screen_height)
            pyautogui.moveTo(x, y)   # Move mouse to fingertip position
            
            #Get positions of thumb tip, index base, pinky tip
            # Point 4 (thumb tip), point 6 (index base), point 20 (pinky tip)
            x4, y4 = int(landmarks[4].x * screen_width), int(landmarks[4].y * screen_height)
            x6, y6 = int(landmarks[6].x * screen_width), int(landmarks[6].y * screen_height)
            x20, y20 = int(landmarks[20].x * screen_width), int(landmarks[20].y * screen_height)

            # Distance check
            if calc_distance(x4, y6, x6, y6) < 40:    # Thumb close to index base → Left click  
                pyautogui.click()  # Left Click
                pyautogui.sleep(1)   # Small delay to avoid multiple clicks  

            elif calc_distance(x4, y20, x20, y20) < 40:  # Thumb close to pinky → Right click  
                pyautogui.rightClick()  # Right Click
                pyautogui.sleep(1)  
            #Draw hand landmarks on the image
            draw.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Mouse Control", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
