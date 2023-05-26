import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from IPython import display
# display.clear_output()

import ultralytics
# ultralytics.checks()

from ultralytics import YOLO

model = '/AIHCM/AI_Member/workspace/hungtd/yolov8n.pt' # path to my pretrained weight 
# (either yolov8n.pt or yolov8s.pt for download original pretrained weight)
data = '/AIHCM/AI_Member/workspace/hungtd/datasets/data/data.yaml'
epochs = '800'
imgsz = '416'
os.system(f"yolo task=detect mode=train model={model} data={data} epochs={epochs} imgsz={imgsz} plots=True")