import cv2
import numpy as np
import pandas as pd  # <-- Added this import
from tensorflow.keras.models import load_model

# --- CONSTANTS ---
IMG_HEIGHT = 48
IMG_WIDTH = 48
MODEL_PATH = 'my_traffic_sign_model.h5' # Path to your saved model

# --- LOAD THE TRAINED MODEL ---
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- LOAD CLASS NAMES DYNAMICALLY FROM CSV FILE ---
# This block replaces the old manual dictionary.
try:
    data = pd.read_csv('traffic_sign.csv')
    class_names = data.set_index('ClassId')['SignName'].to_dict()
    print("Class names loaded successfully from CSV.")
except FileNotFoundError:
    print("Error: 'traffic_sign.csv' not found. Please make sure it's in the same folder as your script.")
    exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# --- INITIALIZE WEBCAM ---
cap = cv2.VideoCapture(0) # 0 is the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press 'q' to quit.")

while True:
    # Read a frame from the webcam
    success, frame = cap.read()
    if not success:
        print("Error: Failed to capture frame.")
        break

    # --- PREPROCESS THE FRAME ---
    # 1. Resize the frame to match the model's input size
    image_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

    # 2. Normalize pixel values to be between 0 and 1
    image_normalized = image_resized / 255.0

    # 3. Reshape the image to (1, height, width, channels) for the model
    image_reshaped = np.reshape(image_normalized, (1, IMG_HEIGHT, IMG_WIDTH, 3))

    # --- MAKE A PREDICTION ---
    # model.predict can be slow, using __call__ is faster for single images
    prediction = model(image_reshaped, training=False)
    predicted_class_id = np.argmax(prediction)
    confidence = np.max(prediction) * 100 # Get confidence score

    # Get the sign name from the dictionary
    predicted_sign_name = class_names.get(predicted_class_id, "Unknown Class")

    # --- DISPLAY THE RESULT ON THE FRAME ---
    # Only display the prediction if confidence is above a threshold (e.g., 50%)
    if confidence > 50:
        text = f"{predicted_sign_name} ({confidence:.2f}%)"
        # Draw a black background rectangle for better text visibility
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (5, 5), (15 + text_width, 35 + text_height), (0,0,0), -1)
        # Put the white text on the black background
        cv2.putText(frame, text, (10, 30 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show the frame in a window
    cv2.imshow("Traffic Sign Recognition", frame)

    # --- EXIT CONDITION ---
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("Application closed.")