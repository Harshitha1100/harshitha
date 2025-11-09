import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# -----------------------------
# Load trained model
# -----------------------------
model = load_model("emotion_model.h5")
emotion_labels = ['Happy', 'Sad', 'Neutral']  # Update if needed

# -----------------------------
# Load Haarcascade for face detection
# -----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -----------------------------
# Prepare folder to save captured images
# -----------------------------
save_dir = "captured_faces"
os.makedirs(save_dir, exist_ok=True)

# -----------------------------
# Start webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

frame_count = 0
frame_skip = 5  # Predict every 5 frames for performance
last_emotion = "Neutral"  # Store last predicted emotion

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_gray = gray_frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_gray, (48, 48))
        face_input = np.expand_dims(face_resized, axis=0)
        face_input = np.expand_dims(face_input, axis=-1)
        face_input = face_input / 255.0

        frame_count += 1
        if frame_count % frame_skip == 0:
            prediction = model.predict(face_input, verbose=0)
            emotion_index = np.argmax(prediction)
            last_emotion = emotion_labels[emotion_index]  # update last emotion

        # Draw rectangle and emotion using last predicted emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, last_emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show webcam feed
    cv2.imshow("Emotion Detection", frame)

    # Press 'q' to quit, 'c' to capture photo
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c') and len(faces) > 0:
        # Save the first detected face
        x, y, w, h = faces[0]
        face_crop = frame[y:y+h, x:x+w]
        save_path = os.path.join(save_dir, f"{last_emotion}_{frame_count}.jpg")
        cv2.imwrite(save_path, face_crop)
        print(f"📸 Saved {save_path}")

cap.release()
cv2.destroyAllWindows()
print("👋 Webcam closed safely")
