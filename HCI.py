# Gesture Recognition -->

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Directories -->

TRAIN_DIR = "dataset/train"
VALID_DIR = "dataset/validation"
MODEL_PATH = "gesture_model.h5"

# Model Parameters -->

IMG_HEIGHT, IMG_WIDTH = 64, 64
BATCH_SIZE = 32
EPOCHS = 20

# Data Augmentation -->

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Model Architecture -->

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(train_generator.num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and Train the Model -->

model = create_model()
model.summary()

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)

print(f"Model saved to {MODEL_PATH}")

# Hand Detection -->

import cv2
import numpy as np

# Preprocessing Functions -->

def preprocess_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# Hand Detection -->

def detect_hand(frame):
    preprocessed = preprocess_frame(frame)
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        return max_contour
    return None

# Kalman Filter -->

import numpy as np

class KalmanFilter:
    def __init__(self):
        self.state = np.zeros((4, 1))  # [x, y, dx, dy]
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.P = np.eye(4) * 1000
        self.Q = np.eye(4)
        self.R = np.eye(2) * 10

    def predict(self):
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, measurement):
        y = measurement - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state += np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def get_state(self):
        return self.state[:2].flatten()

# HCI Controller -->

import pyautogui
from kalman_filter import KalmanFilter

def interpret_gesture(gesture):
    if gesture == 'Palm':
        return 'Move'
    elif gesture == 'Fist':
        return 'Click'
    elif gesture == 'VolumeUp':
        return 'Volume_Up'
    elif gesture == 'VolumeDown':
        return 'Volume_Down'
    return None

def execute_action(action, position=None):
    if action == 'Move' and position:
        pyautogui.moveTo(position[0], position[1])
    elif action == 'Click':
        pyautogui.click()
    elif action == 'Volume_Up':
        pyautogui.press('volumeup')
    elif action == 'Volume_Down':
        pyautogui.press('volumedown')

# Main -->

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from hand_detection import detect_hand
from kalman_filter import KalmanFilter
from hci_controller import interpret_gesture, execute_action

MODEL_PATH = "gesture_model.h5"
model = load_model(MODEL_PATH)

cap = cv2.VideoCapture(0)
kalman = KalmanFilter()

gesture_mapping = {0: 'Palm', 1: 'Fist', 2: 'VolumeUp', 3: 'VolumeDown'}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hand_contour = detect_hand(frame)
    if hand_contour is not None:
        x, y, w, h = cv2.boundingRect(hand_contour)
        roi = frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, (64, 64))
        roi = roi / 255.0
        roi = np.expand_dims(roi, axis=0)

        prediction = model.predict(roi)
        gesture_index = np.argmax(prediction)
        gesture = gesture_mapping.get(gesture_index, None)
        action = interpret_gesture(gesture)

        if action:
            kalman.predict()
            kalman.update(np.array([[x + w // 2], [y + h // 2]]))
            position = kalman.get_state()
            execute_action(action, position)

    cv2.imshow('Hand Gesture Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()