# import torch

# print(torch.cuda.is_available())

# from ultralytics import YOLO
# model = YOLO('model/checkpoint/best.pt')
# print(model.predict('https://ultralytics.com/images/bus.jpg', device='cuda:0'))

import requests
from time import perf_counter
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import base64
import cv2
import numpy as np

def convert():
    img = Image.open("cloth.png") # path to file
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data) # bytes
    base64_str = base64_str.decode("utf-8") # str
    return base64_str

def call_api():
    base64_str = convert()

    url = "http://172.29.13.23:8080/process/object-detection"

    payload="{\r\n  \"index\": \"string\",\r\n  \"data\": [\r\n    {\r\n      \"input_type\": \"image_base64\",\r\n      \"image_url\": \"" + base64_str + "\"\r\n    }\r\n  ]\r\n}"
    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    # print(response.text)
    return response.json()

if __name__ == "__main__":
    # warm up
    for i in range(10):
        call_api()

    avg_time = 0
    for i in tqdm(range(10)):
        s = perf_counter()
        for j in range(50):
            call_api()
        e = perf_counter()
        avg_time += (e - s) / 50

    print("Elapsed time average", avg_time / 10)

    # test output
    # img = Image.open('blackshirt.jpg')
    # cv2_img = np.array(img)
    # result = call_api()
    # print(result)
    # for (id, ele) in enumerate(result['data']['0']):
    #         xyxy = ele['bounding_box'] 
    #         conf = ele['confidence'] 
    #         cls_name = ele['category'][0]
            
    #         x1 = float(xyxy[0])
    #         y1 = float(xyxy[1])
    #         x2 = float(xyxy[2])
    #         y2 = float(xyxy[3])
    #         cv2_img = cv2.rectangle(cv2_img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 1)
    #         cv2.putText(cv2_img, cls_name, (int(x1), int(y1)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
    # img_pil = Image.fromarray(cv2_img)
    # img_pil.save('storage/test.jpg')
