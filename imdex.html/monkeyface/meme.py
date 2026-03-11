import cv2
import mediapipe as mp
import numpy as np
import os

IMAGE_PATHS = {
    "THUMBS_UP": "thumbs_up.jpg",
    "POINTING": "pointing.jpg",
    "NEUTRAL": "neutral.jpg",
    "THINKING": "thinking.jpg"
}

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def load_and_resize_image(path, target_height):
    """Loads an image and resizes it to match the camera frame height."""
    full_path = os.path.join(os.getcwd(), path)
    img = cv2.imread(full_path)
    if img is None:
        print(f"Error: Failed to load image {path}. Make sure the file exists in the same directory.")
        return None

    ratio = target_height / img.shape[0]
    target_width = int(img.shape[1] * ratio)
    return cv2.resize(img, (target_width, target_height))


def classify_gesture(hand_landmarks):
    """Classifies hand gestures (Thumbs Up and Pointing)."""

    y_thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    y_index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    y_middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    y_ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    y_pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

    y_middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y

    is_thumb_up = y_thumb_tip < y_middle_pip
    are_fingers_down = (
        y_index_tip > y_middle_pip and
        y_middle_tip > y_middle_pip and
        y_ring_tip > y_middle_pip and
        y_pinky_tip > y_middle_pip
    )

    if is_thumb_up and are_fingers_down:
        return "THUMBS_UP"

    is_index_up = y_index_tip < y_middle_pip
    is_other_fingers_down = (
        y_middle_tip > y_middle_pip and
        y_ring_tip > y_middle_pip and
        y_pinky_tip > y_middle_pip
    )
    is_thumb_down = y_thumb_tip > y_middle_pip

    if is_index_up and is_other_fingers_down and is_thumb_down:
        return "POINTING"

    return "NEUTRAL"


def check_thinking_gesture(hand_landmarks, face_landmarks, frame_width, frame_height):
    """Checks whether the index finger is near the mouth/nose (Thinking Gesture)."""
    if not hand_landmarks or not face_landmarks:
        return False

    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_x = int(index_tip.x * frame_width)
    index_y = int(index_tip.y * frame_height)

    nose_tip = face_landmarks.landmark[4]
    nose_x = int(nose_tip.x * frame_width)
    nose_y = int(nose_tip.y * frame_height)

    distance = np.sqrt((index_x - nose_x) ** 2 + (index_y - nose_y) ** 2)

    MAX_DISTANCE = 50

    y_middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    y_middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y

    is_middle_finger_down = y_middle_tip > y_middle_pip

    if distance < MAX_DISTANCE and is_middle_finger_down:
        return True

    return False


# --- MAIN PROGRAM ---
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)

print("Gesture Tracker running. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    current_gesture = "NEUTRAL"
    hand_landmarks_data = None
    face_landmarks_data = None

    if hand_results.multi_hand_landmarks:
        hand_landmarks_data = hand_results.multi_hand_landmarks[0]

    if face_results.multi_face_landmarks:
        face_landmarks_data = face_results.multi_face_landmarks[0]

    if hand_landmarks_data:

        if hand_landmarks_data and face_landmarks_data:
            if check_thinking_gesture(hand_landmarks_data, face_landmarks_data, frame_width, frame_height):
                current_gesture = "THINKING"

        if current_gesture == "NEUTRAL":
            current_gesture = classify_gesture(hand_landmarks_data)

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks_data,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
        )

    gesture_image = load_and_resize_image(IMAGE_PATHS[current_gesture], frame_height)

    if gesture_image is not None:
        output_frame = np.concatenate((frame, gesture_image), axis=1)

        cv2.putText(
            output_frame,
            f"Gesture: {current_gesture.replace('_', ' ')}",
            (frame_width + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
    else:
        output_frame = frame
        cv2.putText(
            output_frame,
            "LOAD IMAGE FAILED - Check file names!",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow('Gesture & Image Pairing', output_frame)

    key = cv2.waitKey(5)
    if key == ord('q') or key == 27:
        break

hands.close()
face_mesh.close()
cap.release()
cv2.destroyAllWindows()