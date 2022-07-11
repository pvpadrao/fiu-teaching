import cv2
import numpy as np
from djitellopy import tello
import time

drone = tello.Tello()
drone.connect()
print(drone.get_battery())
drone.streamon()
drone.takeoff()

# left/right (-100/100), forward/backward (-100/100), up/down (-100/100), yaw_velocity (-100/100)
# increase height a little bit to ease face detection
drone.send_rc_control(0, 0, 28, 0)
time.sleep(1)
# width, height of the image
w, h = 360, 240
# forward/backward range
fbRange = [6200, 6800]
# PID controller
pid = [0.4, 0.4, 0]
# previous Error
pError = 0


def findFace(img):
    faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
    # convert to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # scale = 1.2, nearest neighbor = 8
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)
    # list of multiple detected faces with the information of the center point
    myFaceListC = []

    myFaceListArea = []

    for (x, y, w, h) in faces:
        # draw rectangle around the faces: image, start point, end point, color, thickness
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # center values used to rotate
        # x center
        cx = x + w // 2
        # y center of face
        cy = y + h // 2

        # area is used to move robot forward or backwards.
        # if we get closer, area will increase. Otherwise, it decreases.
        area = w * h
        # draw a circle in the center of the detected face
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)

    if len(myFaceListArea) != 0:
        # index of the max area of detected face
        i = myFaceListArea.index(max(myFaceListArea))

        return img, [myFaceListC[i], myFaceListArea[i]]

    else:

        return img, [[0, 0], 0]


def trackFace(info, w, pid, pError):
    area = info[1]
    x, y = info[0]
    fb = 0
    # how far away the object is from the center of the image
    error = x - w // 2

    # PID Controller for the yaw movement
    yaw_speed = pid[0] * error + pid[1] * (error - pError)

    yaw_speed = int(np.clip(yaw_speed, -100, 100))

    if fbRange[0] < area < fbRange[1]:  # green zone

        fb = 0

    elif area > fbRange[1]:  # too close, move away

        fb = -20

    elif area < fbRange[0] and area != 0:  # too far, move closer
        fb = 20
    # if we get nothing detected, then we need to stop
    if x == 0:
        yaw_speed = 0
        error = 0
    drone.send_rc_control(0, fb, 0, yaw_speed)

    return error


while True:
    img = drone.get_frame_read().frame
    img = cv2.resize(img, (w, h))
    img, info = findFace(img)
    pError = trackFace(info, w, pid, pError)
    cv2.imshow("Output", img)
    # if the q key is pressed, it breaks the while loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        drone.land()
        break
