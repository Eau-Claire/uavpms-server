"""Persistent Docker API for the curated six-class YOLO26m checkpoint."""
import asyncio
import hmac
import io
import os
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from PIL import Image, UnidentifiedImageError

CLASSES = ('insulator', 'insulator_broken', 'insulator_flashover',
           'bird_nest', 'foreign_object', 'broken_strand')
MODEL_PATH = Path(os.getenv('MODEL_PATH', '/models/best.pt'))
DEVICE = os.getenv('DEVICE', 'cpu')
IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', '640'))
CONFIDENCE = float(os.getenv('CONFIDENCE', '0.25'))
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def decode_image(data: bytes):
    if len(data) > MAX_BYTES:
        raise HTTPException(413, 'Image exceeds 10 MiB')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in ('JPEG', 'PNG'):
                    raise HTTPException(415, 'Use JPEG or PNG')
                if image.width * image.height > MAX_PIXELS:
                    raise HTTPException(413, 'Image exceeds 20 MP')
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                return image.convert('RGB')
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(415, 'Invalid or oversized image') from exc


class Detector:
    def __init__(self):
        from ultralytics import YOLO
        if not MODEL_PATH.is_file():
            raise RuntimeError(f'Checkpoint not found: {MODEL_PATH}')
        self.model = YOLO(str(MODEL_PATH), task='detect')
        actual = tuple(self.model.names[i] for i in range(len(self.model.names)))
        if actual != CLASSES:
            raise RuntimeError(f'Wrong checkpoint taxonomy: {actual!r}')
        self.model.to(DEVICE)
        self.lock = asyncio.Lock()

    async def predict(self, data: bytes):
        image = decode_image(data)
        try:
            async with self.lock:
                return await asyncio.to_thread(self._predict, image)
        finally:
            image.close()

    def _predict(self, image):
        started = time.perf_counter()
        result = self.model.predict(image, imgsz=IMAGE_SIZE, conf=CONFIDENCE,
                                    device=DEVICE, half=False, verbose=False)[0]
        boxes = result.boxes.cpu()
        detections = [dict(class_id=int(class_id), class_name=CLASSES[int(class_id)],
                           confidence=round(float(score), 6),
                           bbox_xyxy=[round(float(value), 3) for value in box])
                      for box, score, class_id in
                      zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())]
        return dict(model='curated_yolo26m', width=image.width, height=image.height,
                    detections=detections,
                    predict_ms=round((time.perf_counter() - started) * 1000, 2))


state = {'detector': None}


@asynccontextmanager
async def lifespan(_):
    state['detector'] = await asyncio.to_thread(Detector)
    yield


app = FastAPI(title='UAVPMS Detection API', version='1.0.0', lifespan=lifespan)


@app.get('/health')
async def health():
    return {'status': 'ok'}


@app.get('/ready')
async def ready():
    if state['detector'] is None:
        raise HTTPException(503, 'Model is not loaded')
    return {'status': 'ready', 'model': 'curated_yolo26m', 'device': DEVICE}


@app.post('/detect')
async def detect(request: Request):
    expected = os.getenv('DETECT_API_TOKEN')
    scheme, _, supplied = request.headers.get('authorization', '').partition(' ')
    if not expected:
        raise HTTPException(503, 'DETECT_API_TOKEN is not configured')
    if scheme.lower() != 'bearer' or not hmac.compare_digest(supplied, expected):
        raise HTTPException(401, 'Invalid bearer token')
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_BYTES:
            raise HTTPException(413, 'Image exceeds 10 MiB')
        data.extend(chunk)
    if not data:
        raise HTTPException(422, 'Empty image body')
    return await asyncio.wait_for(state['detector'].predict(bytes(data)), timeout=150)
