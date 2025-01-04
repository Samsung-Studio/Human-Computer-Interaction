import cv2
import numpy as np
import mediapipe as mp
from pynput.mouse import Button, Controller
import time

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the mouse controller
mouse = Controller()

# Screen dimensions
screen_width, screen_height = 1920, 1080  # Replace with your screen resolution
cam_width, cam_height = 640, 480  # Webcam resolution

# State variables
right_click = False
double_click = False
last_click_time = 0

# For smoothing mouse movement
smooth_x, smooth_y = 0, 0
alpha = 0.7  # Smoothing factor (higher = smoother but less responsive)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a natural mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    result = hands.process(framergb)

    # Post-process the results
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            landmarks = []
            for lm in handLms.landmark:
                lmx = int(lm.x * cam_width)
                lmy = int(lm.y * cam_height)
                landmarks.append((lmx, lmy))

            # Draw hand landmarks on the frame
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            # Palm center (using wrist point)
            palm_center = landmarks[0]
            forefinger = landmarks[8]
            thumb = landmarks[4]

            # Map palm center to screen coordinates
            target_x = np.interp(palm_center[0], [0, cam_width], [0, screen_width])
            target_y = np.interp(palm_center[1], [0, cam_height], [0, screen_height])

            # Smooth movement using exponential moving average
            smooth_x = alpha * smooth_x + (1 - alpha) * target_x
            smooth_y = alpha * smooth_y + (1 - alpha) * target_y

            # Move the mouse smoothly
            mouse.position = (int(smooth_x), int(smooth_y))

            # Detect pinch for right-click
            pinch_distance = np.linalg.norm(np.array(thumb) - np.array(forefinger))
            if pinch_distance < 30 and not right_click:
                mouse.click(Button.right, 1)
                right_click = True
            elif pinch_distance >= 30:
                right_click = False

            # Detect double-click
            current_time = time.time()
            if pinch_distance < 30:
                if current_time - last_click_time < 0.5:
                    mouse.click(Button.left, 2)
                    double_click = True
                last_click_time = current_time
            else:
                double_click = False

    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()