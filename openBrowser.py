#Iteration of Hand Recognition for Web Opening and Interaction
import cv2
import mediapipe as mp
import os
import psutil

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Function to check if Brave is running
def is_brave_running():
    for process in psutil.process_iter(['name']):
        if process.info['name'] == "brave.exe":  # Adjust to your system's Brave process name
            return process
    return None

# Use MediaPipe Hands
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    brave_opened = False  # Track if Brave is currently opened
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the image horizontally for a mirrored view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get landmark positions
                landmarks = hand_landmarks.landmark

                # Calculate the distance between the tip of the thumb and the tip of the pinky
                thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
                thumb_pinky_distance = abs(thumb_tip.x - pinky_tip.x) + abs(thumb_tip.y - pinky_tip.y)

                # Detect hand gesture
                if thumb_pinky_distance > 0.2:  # Open hand
                    gesture = "Open Hand"
                    if not brave_opened:  # If Brave is not already opened
                        os.startfile("C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe")
                        brave_opened = True
                else:  # Closed hand
                    gesture = "Closed Hand"
                    if brave_opened:  # If Brave is open, close it
                        brave_process = is_brave_running()
                        if brave_process:
                            brave_process.terminate()  # Terminate the Brave process
                        brave_opened = False

                # Display gesture
                cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("Hand Gesture Recognition", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
