import time
from .settings import CLASSES, MODEL_PATH, IMAGE_SIZE, CONFIDENCE

class YoloDetector:
    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO(MODEL_PATH)
        if tuple(self.model.names[i] for i in range(len(self.model.names))) != CLASSES:
            raise ValueError('Wrong checkpoint taxonomy')
        self.model.to('cuda:0')

    def predict(self, frame):
        started = time.perf_counter()
        result = self.model.predict(frame, imgsz=IMAGE_SIZE, conf=CONFIDENCE,
                                    device=0, half=False, verbose=False)[0]
        boxes = result.boxes.cpu()
        detections = [dict(class_id=int(c), class_name=CLASSES[int(c)],
                           confidence=float(s), bbox_xyxy=b)
                      for b, s, c in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())]
        return dict(model='curated_yolo26m', width=frame.width, height=frame.height,
                    detections=detections, predict_ms=(time.perf_counter()-started)*1000)
