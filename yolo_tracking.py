import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        results = model.track(frame, persist=True)
        # tracking 시작 - tracking number를 자동으로 부여하고 추적함
        frame = results[0].plot()

    cv2.imshow('tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
