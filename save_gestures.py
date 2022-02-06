import os
import pandas as pd
import cv2
import mediapipe as mp

GESTURES_PATH = 'gestures.csv'
start_capture_flag = False
stop_execution_flag = False

if os.path.exists(GESTURES_PATH):
    os.remove(GESTURES_PATH)

mpHands = mp.solutions.hands    # this performs the hand recognition
# this line configures the model
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# initializing webcam for video capture
cap = cv2.VideoCapture(0)
img_no = 0
TOTAL_DATAPOINTS = 50
frames = 0
gesture_name = ''
landmarks = []
gesture_data = []

while True:
    _, frame = cap.read()
    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)  # flip frame vertically

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)   # get hand landmark predictions

    keypress = cv2.waitKey(1)
    if keypress == ord('c'):
        if start_capture_flag == False:
            start_capture_flag = True
            gesture_name = str(input('Enter Gesture Name: '))
        else:
            start_capture_flag = False
            frames = 0
            img_no = 0
    elif keypress == ord('q'):
        break

    if start_capture_flag == True:
        frames += 1
    if img_no == TOTAL_DATAPOINTS:
        gesture_data += landmarks

        landmarks = []
        start_capture_flag = False
        frames = 0
        img_no = 0

    # post processing the result
    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            # drawing landmarks on the frame
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS, mpDraw.DrawingSpec(color=(3, 252, 244), thickness=2, circle_radius=2),
                                  mpDraw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            if frames > 50:
                img_no += 1
                for point, lm in zip(mpHands.HandLandmark, handslms.landmark):
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([str(point), lmx, lmy, gesture_name])

                cv2.putText(frame, "Capturing...", (30, 60),
                            cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
                cv2.putText(frame, str(img_no), (30, 400),
                            cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))

    cv2.imshow('Output', frame)


cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(gesture_data, columns=[
    'hand_landmark', 'lmx', 'lmy', 'gesture'])
df.to_csv('gestures.csv', index=False)