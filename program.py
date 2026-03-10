import cv2
import mediapipe as mp
import math
import time
from pynput.mouse import Controller, Button
import numpy as np

# Initialize the webcam and mouse controller
cap = cv2.VideoCapture(0)
mouse = Controller()

# Get screen dimensions
screen_width, screen_height = 1920, 1080
smoothening = 10
scroll_multiplier = 0.05 

# Variables for smoothing cursor movement and scrolling
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
prev_scroll_y = 0
prev_click_time = 0

# Define a virtual "active area" within the camera view
active_area_padding_x = 100
active_area_padding_y = 100

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Loop to continuously read frames
while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera. Is the camera being used by another app?")
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    h, w, c = img.shape

    cv2.rectangle(img, (active_area_padding_x, active_area_padding_y),
                  (w - active_area_padding_x, h - active_area_padding_y),
                  (255, 0, 255), 2)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            
            index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]

            # --- GESTURE LOGIC (simplified) ---
            
            # 1. SCROLLING (Index and Middle fingers up, others down)
            if (index_tip.y < index_pip.y) and (middle_tip.y < middle_pip.y) and \
               (ring_tip.y > ring_pip.y): # Check that ring finger is down
                
                scroll_y = int((index_tip.y + middle_tip.y) / 2 * h)
                if prev_scroll_y != 0:
                    scroll_delta = scroll_y - prev_scroll_y
                    mouse.scroll(0, scroll_delta * scroll_multiplier)
                prev_scroll_y = scroll_y
                cv2.putText(img, "SCROLLING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # ** NEW: Reset scroll position if not in scroll gesture **
            else:
                prev_scroll_y = 0
            
            # 2. DOUBLE CLICK (Fist)
            if (index_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y) and \
                 (middle_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y) and \
                 (ring_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y):
                if time.time() - prev_click_time > 1:
                    mouse.click(Button.left, 2)
                    prev_click_time = time.time()
                    cv2.putText(img, "DOUBLE CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 3. SINGLE CLICK & CURSOR MOVEMENT
            elif prev_scroll_y == 0: # Only move cursor if not scrolling
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                dist = math.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)
                
                cv2.putText(img, f"Dist: {dist:.2f}", (index_x, index_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if dist < 40: # Adjust this value as needed
                    mouse.click(button=Button.left)
                    cv2.circle(img, (index_x, index_y), 15, (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, "CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                else:
                    x = np.interp(index_x, [active_area_padding_x, w - active_area_padding_x], [0, screen_width])
                    y = np.interp(index_y, [active_area_padding_y, h - active_area_padding_y], [0, screen_height])
                    
                    curr_x = prev_x + (x - prev_x) / smoothening
                    curr_y = prev_y + (y - prev_y) / smoothening
                    
                    mouse.position = (curr_x, curr_y)
                
                prev_x, prev_y = curr_x, curr_y

    # Display the resulting frame
    cv2.imshow("Hand Cursor", img)
    
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
