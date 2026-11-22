import time
from .postprocess import postprocess
from .settings import CLASSES, MODEL_NAME, MODEL_PATH, IMAGE_SIZE, LETTERBOX_RECT, NMS_IOU, POSTPROCESS


class YoloDetector:
    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO(MODEL_PATH)
        if tuple(self.model.names[i] for i in range(len(self.model.names))) != CLASSES:
            raise ValueError('Wrong checkpoint taxonomy')
        self.model.to('cuda:0')
        self.served = 0

    def predict(self, frame, decode_ms=None):
        """Detect on a decoded RGB PIL image.

        timing (ms): decode (measured by the caller), Ultralytics preprocess / inference /
        NMS from result.speed, filter = powerline_ai post-processing. predict_ms keeps its
        original meaning: everything inside this call.
        """
        started = time.perf_counter()
        # Predict at the lowest per-class threshold; postprocess applies each class's own threshold.
        result = self.model.predict(frame, imgsz=IMAGE_SIZE, conf=POSTPROCESS.min_confidence,
                                    iou=NMS_IOU, rect=LETTERBOX_RECT, device=0, half=False, verbose=False)[0]
        boxes = result.boxes.cpu()
        raw = [dict(class_id=int(c), class_name=CLASSES[int(c)],
                    confidence=float(s), bbox_xyxy=b)
               for b, s, c in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())]
        filter_started = time.perf_counter()
        detections = postprocess(raw, POSTPROCESS)
        done = time.perf_counter()
        self.served += 1
        speed = result.speed
        return dict(model=MODEL_NAME, width=frame.width, height=frame.height,
                    detections=detections, raw_count=len(raw),
                    predict_ms=(done - started) * 1000,
                    timing=dict(decode_ms=decode_ms, preprocess_ms=speed['preprocess'],
                                inference_ms=speed['inference'], nms_ms=speed['postprocess'],
                                filter_ms=(done - filter_started) * 1000))
