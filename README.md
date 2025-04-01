import cv2
import os
import time
from datetime import datetime
from playsound import playsound
import threading

# Setup directory for saving images
save_dir = "detected_faces"
os.makedirs(save_dir, exist_ok=True)

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Load sound alert (you can replace with your own sound file)
sound_path = "alert.wav"

def play_sound():
    threading.Thread(target=playsound, args=(sound_path,), daemon=True).start()

# Flags
saved_faces = set()  # to avoid saving duplicates

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    face_count = 0

    for (x, y, w, h) in faces:
        face_count += 1
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Blur face for privacy
        # frame[y:y+h, x:x+w] = cv2.GaussianBlur(frame[y:y+h, x:x+w], (99, 99), 30)

        # Eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

        # Smile
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.5, 15)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

        # Save if not already saved
        face_key = f"{x}_{y}_{w}_{h}"
        if face_key not in saved_faces:
            saved_faces.add(face_key)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"face_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[+] Face saved: {filename}")
            play_sound()

    # Face count display
    cv2.putText(frame, f"Faces: {face_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Timestamp
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, time_now, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Show result
    cv2.imshow("Advanced Face Detector", frame)

    key = cv2.waitKey(10)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        filename = os.path.join(save_dir, f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[+] Screenshot saved manually: {filename}")
        play_sound()

cap.release()
cv2.destroyAllWindows()
