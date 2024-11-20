import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import screen_brightness_control as sbc
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Gesture functions
def move_mouse(x, y):
    pyautogui.moveTo(x, y)

def left_click():
    pyautogui.click()

def right_click():
    pyautogui.click(button='right')

def adjust_volume(change):
    # Get current volume range
    volume_range = volume.GetVolumeRange()
    current_volume = volume.GetMasterVolumeLevel()
    new_volume = np.clip(current_volume + change, volume_range[0], volume_range[1])
    volume.SetMasterVolumeLevel(new_volume, None)

def adjust_brightness(change):
    # Adjust brightness by a specific percentage
    current_brightness = sbc.get_brightness(display=0)[0]
    new_brightness = np.clip(current_brightness + change, 0, 100)
    sbc.set_brightness(new_brightness)

# Main function to capture gestures and map to actions
def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for natural interaction
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmark positions for gesture recognition
                landmarks = hand_landmarks.landmark
                index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                # Map gestures
                # Gesture for Mouse Move: Index finger only
                if landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y < index_tip.y < middle_tip.y:
                    x, y = int(index_tip.x * screen_width), int(index_tip.y * screen_height)
                    move_mouse(x, y)

                # Gesture for Left Click: Thumb tip close to index tip
                if np.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y) < 0.05:
                    left_click()

                # Gesture for Right Click: Index finger straight, middle finger bent
                if middle_tip.y > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y:
                    right_click()

                # Gesture for Volume Up: Palm open, fingers spread (approximate check)
                if all(landmark.y < landmarks[mp_hands.HandLandmark.WRIST].y for landmark in hand_landmarks.landmark[:5]):
                    adjust_volume(1.0)  # Increase volume by small steps

                # Gesture for Volume Down: Fist closed (approximate check)
                if all(landmark.y > landmarks[mp_hands.HandLandmark.WRIST].y for landmark in hand_landmarks.landmark[:5]):
                    adjust_volume(-1.0)  # Decrease volume by small steps

                # Gesture for Brightness Up: Thumb up
                if thumb_tip.y < landmarks[mp_hands.HandLandmark.THUMB_IP].y:
                    adjust_brightness(10)  # Increase brightness by 10%

                # Gesture for Brightness Down: Thumb down
                if thumb_tip.y > landmarks[mp_hands.HandLandmark.THUMB_IP].y:
                    adjust_brightness(-10)  # Decrease brightness by 10%

        # Display the frame for debugging
        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()