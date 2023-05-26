import os
import math
import numpy as np
import cv2
import time
from tqdm import tqdm
from glob import glob
from typing import List
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from IPython import display
display.clear_output()

import onnxruntime
from ultralytics import YOLO
from PIL import Image

from inference_utils import xywh2xyxy, nms
from constant import fashion_condition, label_list

BOUNDING_BOX = "bounding_box"
CATEGORY = "category" 
CONFIDENCE = "confidence"


class ObjectDetection:
    def __init__(self, checkpoint, conf_thres=0.25, iou_thres=0.5, device='-1', use_onnx=False, save_image=True):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.save_image = save_image
        self.use_onnx = use_onnx
        self.fashion_condition = fashion_condition
        self.label_list = label_list
        self.model_size = 416

        if self.save_image:
            self.storage = r"inference\storage"
            os.makedirs(self.storage, exist_ok = True)

        if device in ['cpu', 'CPU', '-1', None, '']:
            self.device = "cpu"
            providers = ["CPUExecutionProvider"]
        else:
            self.device = f"cuda:{os.environ['CUDA_VISIBLE_DEVICES']}"
            providers = ["CUDAExecutionProvider"]

        if not self.use_onnx and '.pt' in checkpoint: # pytorch
            self.model = YOLO(checkpoint)
        elif self.use_onnx and '.onnx' in checkpoint: # onnx
            self.model = onnxruntime.InferenceSession(checkpoint, providers=providers)
            self.get_input_details()
            self.get_output_details()
        
    def __call__(self, images: List[np.ndarray]):
        return self.detect_objects(images, self.save_image)
        
    def get_input_details(self):
        model_inputs = self.model.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

    def get_output_details(self):
        model_outputs = self.model.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def process_input(self, images):
        height_width_list, image_list = [], []
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h0, w0 = image.shape[:2]
            height_width_list.append((h0, w0))
            image = cv2.resize(image, (self.model_size, self.model_size))
            if self.use_onnx:
                image = image / 255.0
                image = image.transpose(2, 0, 1)
                image = image[np.newaxis, :, :, :].astype(np.float32)
            image_list.append(image)
        if self.use_onnx:
            image_list = np.concatenate(image_list, axis=0)
            print(image_list.shape)
        return height_width_list, image_list
        
    def detect_objects(self, images, save_image=True):
        height_width_list, input_list = self.process_input(images)
        output_dict = {}

        if not self.use_onnx:
            results = self.model.predict(source=input_list, conf=self.conf_threshold, device=self.device)
            for index, result in enumerate(results):
                xyxy_list = result.boxes.xyxy.tolist()
                conf_list = result.boxes.conf.tolist()
                cls_id_list = result.boxes.cls.tolist()
                output_dict[index] = (xyxy_list, conf_list, cls_id_list)     
        else:
            results = self.model.run(self.output_names, {self.input_names[0]: input_list})[0]
            for index, result in enumerate(results):
                result = np.expand_dims(result, axis=0)
                predictions = np.squeeze(result).T
                scores = np.max(predictions[:, 4:], axis=1)
                predictions = predictions[scores > self.conf_threshold, :]
                scores = scores[scores > self.conf_threshold]
                if len(scores) == 0:
                    return [], [], []
                class_ids = np.argmax(predictions[:, 4:], axis=1)
                boxes = self.extract_boxes(predictions)
                indices = nms(boxes, scores, self.iou_threshold)
                xyxy_list, conf_list, cls_id_list = boxes[indices], scores[indices], class_ids[indices]
                output_dict[index] = (xyxy_list, conf_list, cls_id_list)     

        if save_image:
            pass
        print(output_dict)
        outputs = self.process_output(height_width_list, output_dict)
        return outputs

    def extract_boxes(self, predictions):
        boxes = predictions[:, :4]
        boxes = self.rescale_boxes(boxes)
        boxes = xywh2xyxy(boxes)
        return boxes

    def rescale_boxes(self, boxes):
        input_shape = np.array([self.model_size, self.model_size, self.model_size, self.model_size])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.model_size, self.model_size, self.model_size, self.model_size])
        return boxes
    
    def process_output(self, height_width_list, output_dict):
        outputs = {}
        for k, v in output_dict.items():
            print('Index', k)
            xyxy_list, conf_list, cls_id_list = v
            h0, w0 = height_width_list[k]
            final_result = []
            for (xyxy, conf, cls_id) in zip(xyxy_list, conf_list, cls_id_list):
                cls_name = self.label_list[int(cls_id)]
                print(cls_name, xyxy, conf)
                x1 = float(xyxy[0] / self.model_size * w0)
                y1 = float(xyxy[1] / self.model_size * h0)
                x2 = float(xyxy[2] / self.model_size * w0)
                y2 = float(xyxy[3] / self.model_size * h0)
                conf = float(conf)
                box = [x1, y1, x2, y2]
                if cls_name in self.fashion_condition:
                    final_result.append({BOUNDING_BOX: box, CATEGORY: [cls_name], CONFIDENCE: conf})
            outputs[str(k)] = final_result
        return outputs
    

if __name__ == "__main__":
    
    storage = r"inference\storage"
    model = YOLO(r"api\model\checkpoint\best.pt")
    # model.export(format="onnx") 
    
    onnx_checkpoint = r"api\model\checkpoint\best.onnx"
    # onnx_checkpoint = r"api\model\checkpoint\best_.onnx"
    conf_thres = 0.25
    iou_thres = 0.6
    image_path = r"cloth.png"
    model = ObjectDetection(checkpoint=r"api\model\checkpoint\best.pt", conf_thres=0.25, iou_thres=0.5, device='-1', use_onnx=False, save_image=True)
    # model = ObjectDetection(checkpoint=r"api\model\checkpoint\best.onnx", conf_thres=0.25, iou_thres=0.5, device='-1', use_onnx=True, save_image=True)
    imgs = [np.array(Image.open(image_path).convert('RGB'))] * 2
    outputs = model(imgs)
    print(outputs)

    if outputs:
        for index, img in enumerate(imgs):
            print(index)
            result = outputs[str(index)]
            for (id, ele) in enumerate(result):
                xyxy = ele['bounding_box'] 
                conf = ele['confidence'] 
                cls_name = ele['category'][0]
                
                x1 = float(xyxy[0])
                y1 = float(xyxy[1])
                x2 = float(xyxy[2])
                y2 = float(xyxy[3])
                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 1)
                cv2.putText(img, cls_name, (int(x1), int(y1)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
            img_pil = Image.fromarray(img)
            img_pil.save(rf"inference\storage_nononnx\{index}.jpg")
            # img_pil.save(rf"inference\storage\{index}.jpg")

    # [model.predict(source=imgs, conf=conf_thres, device="cpu") for _ in range(10)] # warmup

    # s = time.time()
    # # # [model(imgs) for _ in range(30)]
    # [model.predict(source=imgs, conf=conf_thres, device="cpu") for _ in range(2)]
    # print(time.time() - s)

    # s = time.time()
    # input_ = imgs * 2
    # print(type(input_))
    # results = model.predict(source=input_, conf=conf_thres, device="cpu")
    # for result in results:
    #     # print(type(result))
    #     xyxy_list = result.boxes.xyxy.tolist()
    #     conf_list = result.boxes.conf.tolist()
    #     cls_id_list = result.boxes.cls.tolist()
    #     print(len(xyxy_list))
    #     for (id, (xyxy, conf, cls_id)) in enumerate(zip(xyxy_list, conf_list, cls_id_list)):
    #         cls_name = label_list[int(cls_id)]
    #         # print(cls_id, cls_name, conf)
    #     # break
    # print(time.time() - s)


    