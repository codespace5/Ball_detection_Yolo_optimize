import cv2
from openvino.runtime import Core
import numpy as np
from utils import detect
from ruamel import yaml


data = yaml.safe_load(open('yolov8n/metadata.yaml'))
seg_model_path = 'yolov8n/yolov8n-seg.xml'
label_map = data['names']

core = Core()
seg_ov_model = core.read_model(seg_model_path)
device = "CPU"  # GPU
if device != "CPU":
    seg_ov_model.reshape({0: [1, 3, 640, 640]})
seg_compiled_model = core.compile_model(seg_ov_model, device)
cap = cv2.VideoCapture("a.mp4")
while True:
    _, image = cap.read()
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detect(input_image, seg_compiled_model)[0]
    if len(detections['det']) == 0:
        continue
    dets = detections['det']
    segs = detections['segment']
    for i , seg_cnt in enumerate(segs):
        if dets[i][5] == 32:
            (x,y),radius = cv2.minEnclosingCircle(seg_cnt)
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(image,center,radius,(0,255,0),2)

    res = cv2.resize(image, (1500, 1000))
    cv2.imshow("sd", res)

    cv2.waitKey(1)