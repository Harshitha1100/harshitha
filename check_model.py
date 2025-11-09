import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
print("Loading model...")
model = load_model("emotion_model.h5")
print("✅ Model loaded successfully!")

# Emotion labels (edit if your dataset differs)
emotion_labels = ['Happy', 'Sad', 'Neutral']

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)
    face = face / 255.0

    # Predict emotion
    prediction = model.predict(face)
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]

    # Display emotion on screen
    cv2.putText(frame, f'Emotion: {emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show webcam feed
    cv2.imshow("Facial Expression Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
