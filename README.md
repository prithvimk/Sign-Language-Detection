# Sign Language Detection

**A Work in Progress**

This project aims to create a Machine Learning model that can translate Indian Sign Language to English text and act as a simple medium of communication for people unfamiliar with sign language.

The hand recognition is done using [MediaPipe Hands solution](https://google.github.io/mediapipe/solutions/hands.html) in Python.

Tutorials that I referred:
1. [Real-time Hand Gesture Recognition using TensorFlow & OpenCV](https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/)
2. [Python: Hand landmark estimation with MediaPipe](https://techtutorialsx.com/2021/04/10/python-hand-landmark-estimation/)

Currently, only dataset creation has been implemented ([save_gestures.py](save_gestures.py)) 

**Instructions to create dataset**

1. Create virtual environment using
   ```virtualenv``` and activate it.
2. Run: ```pip install -r requirements.txt```
3. To just play around with the hand detection, run [hand_recognition.py](hand_recognition.py)
4. To start creating the dataset, run [save_gestures.py](save_gestures.py)
   
   To overwrite old data, run: ```python save_gestures.py --new```. If you already have a ```gestures.csv``` file in the working directory, then new data will be added to that file by default.
5. Press 'C' on your keyboard to start capturing the gesture. 
6. Enter the name of the gesture in the terminal.
7. Raise your hand in front of the camera while making the gesture and it will automatically start capturing pixel coordinates of the landmarks that are being detected.
8. After number of datapoints recorded equals ```   TOTAL_DATAPOINTS```, code will stop capturing.
9.  Press 'C' to start recording a new gesture or press 'Q' to terminate the program.

## **To-Do**

1. Study more about ISL and decide what changes need to be made.
2. Test out different machine learning models and architectures.
3. Work on deployment.
