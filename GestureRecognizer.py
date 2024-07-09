import numpy as np
import cv2
import math
import time
import pickle

height = 760
width = 1280

class mpHands:
    import mediapipe as mp  
    def __init__(self, maxHands=2, tol1=int(0.5), tol2=int(0.5)):
        self.hands = self.mp.solutions.hands.Hands(False, maxHands, tol1, tol2)

    def Marks(self, frame):
        myHands = []
        handsType = []
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frameRGB)
        if results.multi_hand_landmarks:
            for hand in results.multi_handedness:
                handType = hand.classification[0].label
                handsType.append(handType)
                
            for handLandMarks in results.multi_hand_landmarks:
                myHand = []
                for Landmark in handLandMarks.landmark:
                    myHand.append((int(Landmark.x * width), int(Landmark.y * height)))
                myHands.append(myHand)
        return myHands, handsType

def distFinder(handData):
    distMatrix = np.zeros([len(handData), len(handData)], dtype='float')
    palmSize = math.sqrt((handData[0][0] - handData[9][0]) ** 2 + 
                                                (handData[0][1] - handData[9][1]) ** 2)
    for row in range(len(handData)):
        for column in range(len(handData)):
            distMatrix[row][column] = (math.sqrt((handData[row][0] - handData[column][0]) ** 2 + 
                                                (handData[row][1] - handData[column][1]) ** 2))/palmSize
    return distMatrix

def ErrFinder(gestMatrix, unknownMatrix, keyPoints):
    error = 0
    for row in keyPoints:
        for column in keyPoints:
            error += abs(gestMatrix[row][column] - unknownMatrix[row][column])
    return error

def GestRecognizer(frame, handData, distArray):
    unknownGesture = distFinder(handData[0])
    minError = float('inf')
    recognizedGest = None
    for indx, gesturesLM in enumerate(distArray):
        Err = ErrFinder(gesturesLM, unknownGesture, keyPoints)
        if Err < minError:
             minError = Err
             recognizedGest = nameArray[indx]
    if minError <= 10:
        cv2.putText(frame, recognizedGest, (100, 80), font, fontSize, fontColor, 2)
    else:
        cv2.putText(frame, "Unknown", (100, 80), font, fontSize, fontColor, 2)


cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 120)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

findHands = mpHands(2)
keyPoints = [0, 4, 5, 9, 13, 17, 8, 12, 16, 20]
time.sleep(0.5)
Training = int(input('Enter 1 to train, Enter 0 to recognize gesture\n'))
gesture_count = 0
distArray = []
nameArray = []
Trained = False
font = cv2.FONT_HERSHEY_SIMPLEX
fontSize = 1
fontColor = (0, 255, 255)
time.sleep(0.5)
if Training == 1:
    gesture_num = int(input('How many gestures would you like to record?\n'))
while True:
    ignore, frame = cam.read()
    frame = cv2.flip(frame, 1)
    handData, handsType = findHands.Marks(frame)
    if Training == 1:
        if handData:
            print('Show your gesture, press T when ready\n')
            if cv2.waitKey(1) & 0xff == ord('t'):
                knownGesture = distFinder(handData[0])
                distArray.append(knownGesture)
                gesture_name = input('What would you like to name this gesture?\n')
                nameArray.append(gesture_name)
                gesture_count += 1
                if gesture_count == gesture_num:
                    trainingName = input('Enter Filename for training data (Press Enter for Default)')
                    if trainingName == '':
                        trainingName = 'Default'
                    trainingName = f'{trainingName}.pkl'
                    Training = 0
                    if trainingName:
                        with open(trainingName, 'wb') as file:
                             pickle.dump(nameArray, file)
                             pickle.dump(distArray, file)
    elif Training == 0:
        if not Trained:
            trainingName = input('What training data would you like to use? (Press Enter for Default)')
            if trainingName == '':
                trainingName = 'Default'
            trainingName = f'{trainingName}.pkl'
            with open(trainingName, 'rb') as file:
                nameArray = pickle.load(file)
                distArray = pickle.load(file)
            Trained = True
        if Trained:
            if handData:
                GestRecognizer(frame, handData, distArray)
            
    for hand, handType in zip(handData, handsType):
        handColor = (255, 0, 0) if handType == "Right" else (0, 0, 255)
        for i in keyPoints:
            cv2.circle(frame, hand[i], 25, handColor, 3)
    
    cv2.imshow('Webcam 1', frame)
    cv2.moveWindow('Webcam 1', 0, 0)
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
