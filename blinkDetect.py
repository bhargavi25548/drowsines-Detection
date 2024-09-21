import dlib
import sys
import cv2
import time
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread
import pygame
import queue

FACE_DOWNSAMPLE_RATIO = 1.5
RESIZE_HEIGHT = 460

thresh = 0.27
modelPath = "shape_predictor_70_face_landmarks.dat"
sound_path = "alarm.wav"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)

leftEyeIndex = [36, 37, 38, 39, 40, 41]
rightEyeIndex = [42, 43, 44, 45, 46, 47]

blinkCount = 0
drowsy = 0
state = 0
blinkTime = 0.15  # 150ms
drowsyTime = 1.5  # 1200ms
ALARM_ON = False
GAMMA = 1.5
threadStatusQ = queue.Queue()

invGamma = 1.0 / GAMMA
table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(0, 256)]).astype("uint8")

# Initialize pygame mixer
pygame.mixer.init()

def gamma_correction(image):
    return cv2.LUT(image, table)

def histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

def soundAlert(path, threadStatusQ):
    pygame.mixer.music.load(path)
    while True:
        if not threadStatusQ.empty():
            FINISHED = threadStatusQ.get()
            if FINISHED:
                pygame.mixer.music.stop()
                break
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def checkEyeStatus(landmarks):
    mask = np.zeros(frame.shape[:2], dtype=np.float32)
    hullLeftEye = [(landmarks[i][0], landmarks[i][1]) for i in leftEyeIndex]
    hullRightEye = [(landmarks[i][0], landmarks[i][1]) for i in rightEyeIndex]

    cv2.fillConvexPoly(mask, np.int32(hullLeftEye), 255)
    cv2.fillConvexPoly(mask, np.int32(hullRightEye), 255)

    leftEAR = eye_aspect_ratio(hullLeftEye)
    rightEAR = eye_aspect_ratio(hullRightEye)

    ear = (leftEAR + rightEAR) / 2.0

    eyeStatus = 1 if ear >= thresh else 0
    return eyeStatus

def checkBlinkStatus(eyeStatus):
    global state, blinkCount, drowsy
    if state >= 0 and state <= falseBlinkLimit:
        if eyeStatus:
            state = 0
        else:
            state += 1
    elif state >= falseBlinkLimit and state < drowsyLimit:
        if eyeStatus:
            blinkCount += 1
            state = 0
        else:
            state += 1
    else:
        if eyeStatus:
            state = 0
            drowsy = 1
            blinkCount += 1
        else:
            drowsy = 1

def getLandmarks(im):
    imSmall = cv2.resize(im, None,
                         fx=1.0 / FACE_DOWNSAMPLE_RATIO,
                         fy=1.0 / FACE_DOWNSAMPLE_RATIO,
                         interpolation=cv2.INTER_LINEAR)

    rects = detector(imSmall, 0)
    if len(rects) == 0:
        return 0

    newRect = dlib.rectangle(int(rects[0].left() * FACE_DOWNSAMPLE_RATIO),
                             int(rects[0].top() * FACE_DOWNSAMPLE_RATIO),
                             int(rects[0].right() * FACE_DOWNSAMPLE_RATIO),
                             int(rects[0].bottom() * FACE_DOWNSAMPLE_RATIO))

    points = [(p.x, p.y) for p in predictor(im, newRect).parts()]
    return points

capture = cv2.VideoCapture(0)

for i in range(10):
    ret, frame = capture.read()

totalTime = 0.0
validFrames = 0
dummyFrames = 100

print("Calibration in Progress!")
while validFrames < dummyFrames:
    validFrames += 1
    t = time.time()
    ret, frame = capture.read()
    height, width = frame.shape[:2]
    IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
    frame = cv2.resize(frame, None,
                       fx=1 / IMAGE_RESIZE,
                       fy=1 / IMAGE_RESIZE,
                       interpolation=cv2.INTER_LINEAR)

    adjusted = histogram_equalization(frame)

    landmarks = getLandmarks(adjusted)
    timeLandmarks = time.time() - t

    if landmarks == 0:
        validFrames -= 1
        cv2.putText(frame, "Unable to detect face, Please check proper lighting", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "or decrease FACE_DOWNSAMPLE_RATIO", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1,
                    cv2.LINE_AA)
        cv2.imshow("Blink Detection Demo", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            sys.exit()

    else:
        totalTime += timeLandmarks

print("Calibration Complete!")

spf = totalTime / dummyFrames
print("Current SPF (seconds per frame) is {:.2f} ms".format(spf * 1000))

drowsyLimit = drowsyTime / spf
falseBlinkLimit = blinkTime / spf
print("false blink limit:{}".format(falseBlinkLimit))

if __name__ == "__main__":
    vid_writer = cv2.VideoWriter('output-low-light-2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
                                 (frame.shape[1], frame.shape[0]))
    alarm_off_time = 0
    while True:
        try:
            t = time.time()
            ret, frame = capture.read()
            height, width = frame.shape[:2]
            IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
            frame = cv2.resize(frame, None,
                               fx=1 / IMAGE_RESIZE,
                               fy=1 / IMAGE_RESIZE,
                               interpolation=cv2.INTER_LINEAR)

            adjusted = histogram_equalization(frame)

            landmarks = getLandmarks(adjusted)
            if landmarks == 0:
                validFrames -= 1
                cv2.putText(frame, "Unable to detect face, Please check proper lighting", (10, 30),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "or decrease FACE_DOWNSAMPLE_RATIO", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255),
                            1, cv2.LINE_AA)
                cv2.imshow("Blink Detection Demo", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            eyeStatus = checkEyeStatus(landmarks)
            checkBlinkStatus(eyeStatus)

            for i in leftEyeIndex:
                cv2.circle(frame, (landmarks[i][0], landmarks[i][1]), 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            for i in rightEyeIndex:
                cv2.circle(frame, (landmarks[i][0], landmarks[i][1]), 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            if drowsy:
                if not ALARM_ON:
                    ALARM_ON = True
                    alarm_off_time = time.time() + 4  # Alarm off after 4 seconds
                    threadStatusQ.put(not ALARM_ON)
                    thread = Thread(target=soundAlert, args=(sound_path, threadStatusQ,))
                    thread.setDaemon(True)
                    thread.start()

                if time.time() > alarm_off_time:
                    ALARM_ON = False
                    # Check if eyes are still closed to potentially re-enable alarm
                    if eyeStatus == 0:
                        ALARM_ON = True
            else:
                cv2.putText(frame, "Blinks : {}".format(blinkCount), (460, 80), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                ALARM_ON = False

            cv2.imshow("Blink Detection Demo", frame)
            vid_writer.write(frame)

            k = cv2.waitKey(1)
            if k == ord('r'):
                state = 0
                drowsy = 0
                ALARM_ON = False
                threadStatusQ.put(not ALARM_ON)

            elif k == 27:
                break

        except Exception as e:
            print(e)

    capture.release()
    vid_writer.release()
    cv2.destroyAllWindows()
