'''from ultralytics import YOLO
import cv2
import math
import cvzone
import streamlit as st

st.title("Welcome to Plastic Bottle Detection")
cap = cv2.VideoCapture(0)  #for webcam
#cap = st.camera_input(cv2.VideoCapture(0))
cap.set(3, 1280)

cap.set(4, 960)

#cap = cv2.VideoCapture("../videos/car1.mp4") #for video

model = YOLO("weights/last.pt")
className = ["Monkey","bicycle","car", "motorbike", "aeroplane", "bus", "train", "truck",
             "traffic light", "fire hydrant","stop sign","parking meter","bench", "bird","cat",
             "dog", "horse","sheep","cow","elephant","bear", "zebra", "giraffe", "backpack", "umbrella",
             "hqandbag", "tie", "suitcae","fribee","skis","snowboard","sport ball","kite", "baseball bat",
             "baseball glove","skate board","surf board", "tennis racket","bottle","wine glass","cup"
             "fork","kife","spoon","bowl","banana","apple","sandwitch","orange","broccoli",
             "carroy","hot dog", "pizza", "donut","cake","chair","sofa","pottedplant","bed",
             "diningtable", "toilet","tvremote","laptop","mouse", "remote", "keyboard","cell phone",
             "microvave", "Phone", "toaster", "sink","refreezeratoe","book","loack","vase","scissor",
             "teddy bear", "hair drier","tooth brush"]
className = ["bottle"]
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            #box
            x1, y1, x2, y2 = box.xyxy[0]
            #x1, y1, w, h = box.xywh[0]

            #bbox = int(x1), int(y1), int(w), int(h)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #w, h = x2 - x1, y2 - y2
            #print( x1,y1,x2,y2)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            #cvzone.cornerRect(img,(x1,y1,w,h))

            #confidence
            conf = math.ceil(box.conf[0]*100)/100

            #class
            cls = int(box.cls[0])
            currentClass = className[cls]
            #if currentClass=="car" or currentClass=="bus" and conf>0.4:
            cvzone.putTextRect(img, f'{className[cls]} {conf}', (max(0, x1), max(35, y1)), scale=2, thickness=2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)


            print(conf)

    #st.video(cv2.imshow("image", img))
    cv2.imshow("image", img)
    cv2.waitKey(1)'''

import cv2
import math
import cvzone
import streamlit as st
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

st.title("Welcome to Plastic Bottle Detection")

model = YOLO("weights/last.pt")
className = ["bottle"]

cap = st.camera_input(cv2.VideoCapture(0))

while cap.isOpened():
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            currentClass = className[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cvzone.putTextRect(frame, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=2, thickness=2)
    st.image(frame, channels="RGB")