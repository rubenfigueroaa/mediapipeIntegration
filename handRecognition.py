#Import Libraries
import cv2
import mediapipe as mp

#Initialize mediapipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#Open the webcam 0 Default Camera
cap = cv2.VideoCapture(0)

#Using mediapipe.hands Module
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands =1,
    min_detection_confidence= 0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break

        #Flip image for mirrored view
        frame = cv2.flip(frame,1)
        rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        #Frame processing
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                #Draw hand landmarks
                mp_drawing.draw_landmarks(frame,hand_landmarks,
                                          mp_hands.HAND_CONNECTIONS)

                #GET LANDMARK POSITION
                landmarks = hand_landmarks.landmark

                #Calculate the distance between tip of thumb and tip of pinky
                thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
                thumb_pinky_distance = abs(thumb_tip.x-pinky_tip.x)+ abs(thumb_tip.y-pinky_tip.y)

                #Threshold to differentiate open and close hand
                if thumb_pinky_distance > 0.2:
                    gesture = "Open Hand"
                else:
                    gesture = "Closed Hand"

                #Display Gesture
                cv2.putText(frame,gesture,(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),
                            2,cv2.LINE_AA)
        #Show the frame
        cv2.imshow("Hand Gesture Recognition",frame)

        #Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#Release Resources
cap.release()
cv2.destroyAllWindows()


