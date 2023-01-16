# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 10:22:32 2023

@author: Veena
"""
import cv2
from yolov7 import YOLOv7
import os


input_filename = os.listdir(os.path.join(os.getcwd(), "input"))[1]
cap = cv2.VideoCapture(os.path.join(os.getcwd(), "input", 
                                    str(input_filename)))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

relevant_frames = [i for i in range (1, total_frames) if divmod(i, int(fps)*1)[1]==0]
relevant_frames= relevant_frames+[1, total_frames-1]
relevant_frames.sort()

model_path = os.path.join(os.getcwd(), "models", "yolov7_384x640.onnx")

yolov7_detector = YOLOv7(model_path, conf_thres=0.60, iou_thres=0.5)

ppl = []
for num in relevant_frames:
    print(num)
    cap.set(cv2.CAP_PROP_POS_FRAMES,num)
    ret, frame = cap.read()
    if cv2.waitKey(1) == ord('q'):
        break
    try:
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    boxes, scores, class_ids = yolov7_detector(frame)
    combined_img = yolov7_detector.draw_detections(frame)
    cv2.imwrite(os.path.join(os.getcwd(), "output", 
                         str(input_filename)+'_'+str(num)+".jpg"), 
            combined_img)
    try:
        ppl.append(list(class_ids).count(0))
        
    except: 
        print("No Persons were detected!")
   
     

f = open(os.path.join(os.getcwd(), "output", 
                      str(input_filename)+'_'+"output.txt"),"w")
f.write(str(max(ppl)))
f.close()
