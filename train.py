import torch
from ultralytics import YOLO
import ultralytics

print(torch.cuda.is_available())
print('----------------------')
print(ultralytics.checks())

if __name__ == '__main__':
    # model = YOLO('yolov8n.pt')
    model = YOLO('runs/detect/train/weights/best.pt')
    model.train(resume=True, data='C:/Users/user/Desktop/yolo_v8/Fish-44/data.yaml', imgsz=640, batch=4, epochs=30, device=0)
    # 사용할 gpu를 숫자로 지정하면 됨
    # resume 옵션을 사용하는 경우 이전 학습에 이어서 작업 가능
