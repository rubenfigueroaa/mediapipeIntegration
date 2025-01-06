import cv2
import mediapipe as mp
import os
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize Pycaw for system volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get the current volume range
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

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

                # Get landmark positions
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Calculate the Euclidean distance between thumb tip and index tip
                distance = math.sqrt(
                    (thumb_tip.x - index_tip.x) ** 2 +
                    (thumb_tip.y - index_tip.y) ** 2 +
                    (thumb_tip.z - index_tip.z) ** 2
                )

                # Map the distance to the volume range
                volume_level = min_vol + (max_vol - min_vol) * min(max(distance, 0.1), 0.3) / 0.2
                volume.SetMasterVolumeLevel(volume_level, None)

                # Display the volume level
                volume_percent = int((volume_level - min_vol) / (max_vol - min_vol) * 100)
                cv2.putText(frame, f"Volume: {volume_percent}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("Hand Gesture Volume Control", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
