from ultralytics import YOLO
import cv2

import numpy
import cvzone

model = YOLO('../yolo-WEIGHT/yolov8n.pt')
img = cv2.VideoCapture(0)
img.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
img.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)

while True:

    _, cap = img.read()
    #img = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
    result = model(cap, show=False)
    for r in result:
        boxes = r.boxes
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0]
            print(f'{x1},{y1},{x2},{y2}')
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cap1=cv2.rectangle(cap, (x1, y1), (x2, y2), (0, 200, 53), 3)
    cv2.imshow('new', cap1)
   #cv2.imshow('new1', img)
    key = cv2.waitKey(1)
    if 27 == key:
        break
