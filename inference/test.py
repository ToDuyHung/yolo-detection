from ultralytics import YOLO
import numpy as np
from PIL import Image


if __name__ == "__main__":
    checkpoint=r"api\model\checkpoint\best.onnx"
    model = YOLO(checkpoint)
    image_path = r"cloth.png"
    imgs = [np.array(Image.open(image_path).convert('RGB'))]
    results = model.predict(source=imgs, conf=0.25, device='cpu')
    for index, result in enumerate(results):
        xyxy_list = result.boxes.xyxy.tolist()
        conf_list = result.boxes.conf.tolist()
        cls_id_list = result.boxes.cls.tolist()
        print(xyxy_list, conf_list, cls_id_list)