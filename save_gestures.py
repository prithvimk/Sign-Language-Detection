import os
import pandas as pd
import cv2
import mediapipe as mp
import argparse



GESTURES_PATH = 'gestures.csv'
start_capture_flag = False
stop_execution_flag = False

parser = argparse.ArgumentParser(
    description='Save gesture keypoints in gestures.csv')
parser.add_argument('-n', '--new', action='store_true',
                    help='Overwrite the previously collected data.')

args = parser.parse_args()

if args.new:
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
handpoints = ['HandLandmark.WRIST_lmx', 'HandLandmark.WRIST_lmy', 'HandLandmark.THUMB_CMC_lmx', 'HandLandmark.THUMB_CMC_lmy', 'HandLandmark.THUMB_MCP_lmx', 'HandLandmark.THUMB_MCP_lmy', 'HandLandmark.THUMB_IP_lmx', 'HandLandmark.THUMB_IP_lmy', 'HandLandmark.THUMB_TIP_lmx', 'HandLandmark.THUMB_TIP_lmy', 'HandLandmark.INDEX_FINGER_MCP_lmx', 'HandLandmark.INDEX_FINGER_MCP_lmy', 'HandLandmark.INDEX_FINGER_PIP_lmx', 'HandLandmark.INDEX_FINGER_PIP_lmy', 'HandLandmark.INDEX_FINGER_DIP_lmx', 'HandLandmark.INDEX_FINGER_DIP_lmy', 'HandLandmark.INDEX_FINGER_TIP_lmx', 'HandLandmark.INDEX_FINGER_TIP_lmy', 'HandLandmark.MIDDLE_FINGER_MCP_lmx', 'HandLandmark.MIDDLE_FINGER_MCP_lmy', 'HandLandmark.MIDDLE_FINGER_PIP_lmx',
              'HandLandmark.MIDDLE_FINGER_PIP_lmy', 'HandLandmark.MIDDLE_FINGER_DIP_lmx', 'HandLandmark.MIDDLE_FINGER_DIP_lmy', 'HandLandmark.MIDDLE_FINGER_TIP_lmx', 'HandLandmark.MIDDLE_FINGER_TIP_lmy', 'HandLandmark.RING_FINGER_MCP_lmx', 'HandLandmark.RING_FINGER_MCP_lmy', 'HandLandmark.RING_FINGER_PIP_lmx', 'HandLandmark.RING_FINGER_PIP_lmy', 'HandLandmark.RING_FINGER_DIP_lmx', 'HandLandmark.RING_FINGER_DIP_lmy', 'HandLandmark.RING_FINGER_TIP_lmx', 'HandLandmark.RING_FINGER_TIP_lmy', 'HandLandmark.PINKY_MCP_lmx', 'HandLandmark.PINKY_MCP_lmy', 'HandLandmark.PINKY_PIP_lmx', 'HandLandmark.PINKY_PIP_lmy', 'HandLandmark.PINKY_DIP_lmx', 'HandLandmark.PINKY_DIP_lmy', 'HandLandmark.PINKY_TIP_lmx', 'HandLandmark.PINKY_TIP_lmy']
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
                lmks = []
                for point, lm in zip(mpHands.HandLandmark, handslms.landmark):
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    lmks += [lmx, lmy]
                landmarks.append(lmks + [gesture_name])

                cv2.putText(frame, "Capturing...", (30, 60),
                            cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
                cv2.putText(frame, str(img_no), (30, 400),
                            cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))

    cv2.imshow('Output', frame)


cap.release()
cv2.destroyAllWindows()
handpoints.append('gesture_name')
df = pd.DataFrame(gesture_data, columns=handpoints)

try:
    df = pd.concat([df, pd.read_csv(GESTURES_PATH)])
finally:
    df.to_csv(GESTURES_PATH, index=False)