from ultralytics import YOLO
import cv2
import cvzone
from sort import *
import numpy as np

model = YOLO("../yolo-WEIGHT/yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              'teddy bear', "hair drier", "toothbrush"
              ]
traker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [400, 297, 673, 297]
img = cv2.VideoCapture("../videos/Videos/cars.mp4")
totalcar = []

while True:
    #Reading the Mask
    mask = cv2.imread("../img/Screenshot 2023-07-06 170905.jpg")
    mask1 = cv2.resize(mask, (1280, 720))
    deteaction = np.empty((0, 5))
    _, cap1 = img.read()
    cap = cv2.bitwise_and(cap1, mask1)
    #Detection using Yolo
    result = model(cap, stream=True)
    limits = [380, 305, 727, 305]
    cv2.line(cap1, (380, 305), (727, 305), (0, 0, 255), 2)
    for b in result:
        box = b.boxes
        for v in box:
            x1, y1, x2, y2 = v.xyxy[0]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            conf = v.conf[0]
            conf = conf * 100
            conf = int(conf)
            cls = int(v.cls[0])

            currentcls = classNames[cls]

            if currentcls == 'car' and conf > 30:
                currentarray = np.array([x1, y1, x2, y2, conf])
                deteaction = np.vstack((deteaction, currentarray))
    #Tracking the cars
    result_array = traker.update(deteaction)

    for r in result_array:
        x1, y1, x2, y2, id1 = map(int, r)

        w = x2 - x1
        h = y2 - y1
        cx = x1 + w // 2
        cy = y1 + h // 2
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:

            if totalcar.count(id1) ==  0:
                totalcar.append(id1)
                print(f'{totalcar}totalcar')

                cv2.putText(cap1,str(len(totalcar)),(100,100),cv2.FONT_HERSHEY_PLAIN,10,color=(24,2,255) ,thickness=3)
                cv2.line(cap1, (380, 305), (727, 305), (0, 255, 0), 2)
        cvzone.cornerRect(cap1, (x1, y1, w, h), colorC=(255, 253, 111), rt=1, t=2)
        cvzone.putTextRect(cap1, f'{id1}', (x1, y1), 1, 1, [253, 22, 5], font=cv2.FONT_HERSHEY_COMPLEX)
        cv2.imshow("cap", cap1)
        print(x1, y1, x2, y2)

    cv2.waitKey(1)
