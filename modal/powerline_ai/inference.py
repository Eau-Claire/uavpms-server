import time
from .postprocess import postprocess
from .settings import CLASSES, MODEL_NAME, MODEL_PATH, IMAGE_SIZE, NMS_IOU, POSTPROCESS


class YoloDetector:
    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO(MODEL_PATH)
        if tuple(self.model.names[i] for i in range(len(self.model.names))) != CLASSES:
            raise ValueError('Wrong checkpoint taxonomy')
        self.model.to('cuda:0')

    def predict(self, frame):
        started = time.perf_counter()
        # Predict at the lowest per-class threshold; postprocess applies each class's own threshold.
        result = self.model.predict(frame, imgsz=IMAGE_SIZE, conf=POSTPROCESS.min_confidence,
                                    iou=NMS_IOU, device=0, half=False, verbose=False)[0]
        boxes = result.boxes.cpu()
        raw = [dict(class_id=int(c), class_name=CLASSES[int(c)],
                    confidence=float(s), bbox_xyxy=b)
               for b, s, c in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())]
        detections = postprocess(raw, POSTPROCESS)
        return dict(model=MODEL_NAME, width=frame.width, height=frame.height,
                    detections=detections, raw_count=len(raw),
                    predict_ms=(time.perf_counter()-started)*1000)
