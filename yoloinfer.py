# 추론 과정
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')
results = model.predict(source='fish.jpg', show=False, save=True)
# gpu로 학습한 경우에 show를 true로 지정하면 에러가 발생함

for result in results:
    boxes = result.boxes
    print(boxes)

# print(results[0].xywh) # 이미지 영역 = bbox의 좌표값
# print(results[0].cls)
# results가 list로 들어가는 이유 : 추론을 여러 개 작업할 수 있기 때문에 결과 값이 항상 list로 나옴

print(boxes.xywh)
print(boxes.cls)