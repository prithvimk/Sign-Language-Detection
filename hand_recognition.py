import cv2
import mediapipe as mp

# initializing mediapipe
mpHands = mp.solutions.hands    # this performs the hand recognition
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)    #this line configures the model
mpDraw = mp.solutions.drawing_utils #this line draws the detected keypoints

# initializing webcam for video capture
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)  # flip frame vertically

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)   # get hand landmark predictions

    # post processing the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            """ for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy]) """

            # drawing landmarks on the frame
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS, mpDraw.DrawingSpec(color=(3, 252, 244), thickness=2, circle_radius=2),
            mpDraw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))






    cv2.imshow('Output', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

