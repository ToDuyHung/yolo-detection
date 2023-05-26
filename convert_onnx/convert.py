from IPython import display
display.clear_output()

from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO(r"api\model\checkpoint\best.pt")
    model.export(format="onnx")

    '''
    1. Để export model YOLO format ONNX sử dụng GPU:
    - Cần cài đặt thư viện onnxruntime-gpu thay vì onnxruntime
    2. Để có thể dùng batch prediction, cần export model YOLO dynamic input bằng cách:
    - Ctrl click vào hàm export trong model.export để chuyển đến file model.py trong lib
    - Ctrl click vào class Exporter trong file model.py để chuyển đến file exporter.py trong lib
    - Ctrl F để tìm hàm _export_onnx, chỉnh biến dynamic = True (biến này mặc định là False)
    '''
    