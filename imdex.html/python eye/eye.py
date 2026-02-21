import cv2
import pygame
import os

# ---------------- SOUND SETUP ----------------
pygame.mixer.init()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sound_path = os.path.join(BASE_DIR, "alarm.wav")

alarm_sound = pygame.mixer.Sound(sound_path)
alarm_playing = False

# ---------------- CASCADE LOAD ----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                    'haarcascade_eye_tree_eyeglasses.xml')

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# ---------------- STATE VARIABLES ----------------
closed_frames = 0
OPEN_EYES_FRAME_LIMIT = 5     
CLOSED_EYES_THRESHOLD = 12    

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes_found = False

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(30, 30)
        )

        if len(eyes) > 0:
            eyes_found = True

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

    # ---------------- LOGIC ----------------
    if eyes_found:
        closed_frames = 0
        status = "Eyes Open"
    else:
        closed_frames += 1
        status = "Eyes Closed?"

    # If eyes closed continuously
    if closed_frames >= CLOSED_EYES_THRESHOLD:
        status = "SLEEPING!"
        cv2.putText(frame, "WAKE UP!", (180,200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)

        if not alarm_playing:
            alarm_sound.play(-1)
            alarm_playing = True
    else:
        if alarm_playing:
            alarm_sound.stop()
            alarm_playing = False

    cv2.putText(frame, status, (30,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Eye Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()