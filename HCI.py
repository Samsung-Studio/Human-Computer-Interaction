import cv2
import numpy as np
import mediapipe as mp
import pyautogui

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Screen dimensions
screen_width, screen_height = pyautogui.size()

ret = True
while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width, _ = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # Process the result
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract landmark positions
            landmarks = []
            for lm in hand_landmarks.landmark:
                lmx = int(lm.x * frame_width)
                lmy = int(lm.y * frame_height)
                landmarks.append((lmx, lmy))

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

            # Palm center (middle point between index and little finger base)
            palm_center = landmarks[9]
            index_tip = landmarks[8]
            thumb_tip = landmarks[4]

            # Map palm center to screen coordinates
            screen_x = np.interp(palm_center[0], [0, frame_width], [0, screen_width])
            screen_y = np.interp(palm_center[1], [0, frame_height], [0, screen_height])

            # Move the mouse cursor
            pyautogui.moveTo(screen_x, screen_y, duration=0.1)

            # Detect pinch gesture for right-click
            pinch_distance = np.linalg.norm(np.array(index_tip) - np.array(thumb_tip))
            if pinch_distance < 30:  # Adjust threshold as needed
                pyautogui.rightClick()
                cv2.putText(frame, "Right Click", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()