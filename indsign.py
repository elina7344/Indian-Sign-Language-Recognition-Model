import cv2
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the Hand Landmarker with two-hand support
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to display the alphabet at the bottom of the screen
def display_alphabet(image, alphabet):
    height, width, _ = image.shape
    position = (int(width / 2 - 50), int(height - 30))  # Bottom-center
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (255, 0, 0)
    thickness = 3
    cv2.putText(image, alphabet, position, font, font_scale, font_color, thickness)

# Function to draw a single green bounding box around both hands
def draw_single_bounding_box(image, landmarks_left, landmarks_right):
    image_height, image_width, _ = image.shape
    
    # Initialize the min and max coordinates to cover both hands
    x_min, y_min = image_width, image_height
    x_max, y_max = 0, 0
    
    # Iterate over the left hand landmarks to update min/max coordinates
    for lm in landmarks_left.landmark:
        x, y = int(lm.x * image_width), int(lm.y * image_height)
        if x < x_min: x_min = x
        if y < y_min: y_min = y
        if x > x_max: x_max = x
        if y > y_max: y_max = y

    # Iterate over the right hand landmarks to update min/max coordinates
    for lm in landmarks_right.landmark:
        x, y = int(lm.x * image_width), int(lm.y * image_height)
        if x < x_min: x_min = x
        if y < y_min: y_min = y
        if x > x_max: x_max = x
        if y > y_max: y_max = y
    
    # Draw a single green rectangle around both hands
    cv2.rectangle(image, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 3)

# Function to recognize gestures from two hands
def recognize_gesture(landmarks_left, landmarks_right):
    # Example for 'A' gesture: Thumbs of both hands touching each other, palms facing forward and down
    thumb_left = landmarks_left.landmark[4]  # Thumb tip on left hand
    thumb_right = landmarks_right.landmark[4]  # Thumb tip on right hand
    wrist_left = landmarks_left.landmark[0]  # Wrist on left hand
    wrist_right = landmarks_right.landmark[0]  # Wrist on right hand

    # Check if both thumbs are close to each other (within a threshold)
    thumbs_touched = abs(thumb_left.x - thumb_right.x) < 0.05 and abs(thumb_left.y - thumb_right.y) < 0.05
    
    # Check if both wrists are approximately at the same height (indicating palms facing forward)
    wrists_aligned = abs(wrist_left.y - wrist_right.y) < 0.1

    # Check if thumbs are above the wrists (indicating thumbs pointing upwards)
    thumbs_above_wrists = thumb_left.y < wrist_left.y and thumb_right.y < wrist_right.y

    if thumbs_touched and wrists_aligned and thumbs_above_wrists:
        return "A"
    
    # Add more gesture logic for B to Z if needed
    
    return None  # Return None if no gesture is recognized

# Start video capture from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hand landmarks
    result = hands.process(image_rgb)

    # If landmarks are detected and there are two hands
    if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
        # Extract landmarks for both hands
        landmarks_left = result.multi_hand_landmarks[0]
        landmarks_right = result.multi_hand_landmarks[1]
        
        # Draw landmarks for both hands
        mp_drawing.draw_landmarks(image, landmarks_left, mp_hands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, landmarks_right, mp_hands.HAND_CONNECTIONS)
        
        # Draw a single bounding box around both hands
        draw_single_bounding_box(image, landmarks_left, landmarks_right)
        
        # Recognize the gesture
        gesture = recognize_gesture(landmarks_left, landmarks_right)
        
        if gesture:
            # Display the corresponding alphabet at the bottom of the screen
            display_alphabet(image, gesture)
    
    # Show the video feed
    cv2.imshow('Two-Hand Sign Language Recognition', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Exit when 'ESC' is pressed
        break

# Release resources
cap.release()
cv2.destroyAllWindows()