import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('isl_model.h5')

# Labels corresponding to your dataset (A-Z)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
          'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize MediaPipe Hands to detect multiple hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)  # Detect up to two hands
mp_drawing = mp.solutions.drawing_utils

# OpenCV to capture the video feed
cap = cv2.VideoCapture(0)  # 0 for default camera

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to detect hands
    results = hands.process(image_rgb)

    predictions = []  # Store predictions from both hands

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the frame for visualization
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box around the hand
            h, w, c = frame.shape  # Height, Width, and Channels of the frame
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y

            # Extract hand region and preprocess
            hand_img = frame[y_min:y_max, x_min:x_max]  # Crop hand region
            if hand_img.size > 0:  # Ensure the image size is valid
                hand_img = cv2.resize(hand_img, (128, 128))  # Resize to match model input shape
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)  # Convert to Grayscale
                hand_img = np.expand_dims(hand_img, axis=-1)  # Add channel dimension for grayscale
                hand_img = hand_img / 255.0  # Normalize pixel values
                hand_img = np.expand_dims(hand_img, axis=0)  # Add batch dimension

                # Model prediction
                prediction = model.predict(hand_img)
                predictions.append(prediction[0])  # Append prediction

    # Combine predictions from both hands
    if predictions:
        combined_prediction = np.mean(predictions, axis=0)  # Average predictions
        predicted_label = labels[np.argmax(combined_prediction)]  # Get the most probable label

        # Display the combined prediction on the screen
        cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame using matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Turn off axis
    plt.pause(0.001)  # Pause to allow for the display update

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
hands.close()
plt.close()  # Close the matplotlib window