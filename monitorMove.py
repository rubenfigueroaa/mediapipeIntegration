import cv2
import mediapipe as mp
import pyautogui
import pygetwindow as gw
import time

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Screen dimensions for scaling
screen_width, screen_height = pyautogui.size()

# Variables to track horizontal and vertical movement
prev_x = None
prev_y = None
window_name = "Brave"  # Name of the Brave browser window
sensitivity = 2  # Lower sensitivity to avoid fast movement
smoothing_factor = 0.2  # Smoothing factor to make the movement more gradual
delta_x_avg = 0  # Variable to store averaged horizontal movement
delta_y_avg = 0  # Variable to store averaged vertical movement

# Use MediaPipe Hands
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
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

                # Get x and y coordinates of the wrist
                wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y

                # Track horizontal and vertical movement
                if prev_x is not None and prev_y is not None:
                    # Calculate the movement deltas and apply sensitivity
                    delta_x = (wrist_x - prev_x) * screen_width * sensitivity
                    delta_y = (wrist_y - prev_y) * screen_height * sensitivity

                    # Apply smoothing to both movements
                    delta_x_avg = smoothing_factor * delta_x + (1 - smoothing_factor) * delta_x_avg
                    delta_y_avg = smoothing_factor * delta_y + (1 - smoothing_factor) * delta_y_avg

                    # Move the browser tab based on both horizontal and vertical movements
                    active_window = None
                    for window in gw.getWindowsWithTitle(window_name):
                        if window.isActive:
                            active_window = window
                            break

                    if active_window:
                        # Get current position of the window
                        current_x, current_y = active_window.left, active_window.top
                        new_x = current_x + int(delta_x_avg)
                        new_y = current_y + int(delta_y_avg)

                        # Ensure the new position stays within screen bounds
                        new_x = max(0, min(new_x, screen_width - active_window.width))
                        new_y = max(0, min(new_y, screen_height - active_window.height))

                        # Move the active Brave browser window
                        active_window.moveTo(new_x, new_y)

                # Update previous x and y coordinateqs
                prev_x = wrist_x
                prev_y = wrist_y

        # Show the frame
        cv2.imshow("Hand Movement Control", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()