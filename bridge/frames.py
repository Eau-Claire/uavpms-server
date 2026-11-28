"""Video only: keep one detection per defect and save its frame to MinIO.

Video frames are sampled without tracking, so one defect usually appears in several consecutive samples.
Detections of the same class whose boxes overlap (IoU >= MATCH_IOU) in nearby samples are treated as the same
defect. For each defect we keep the detection with the highest confidence, upload that frame (clean, without
drawn boxes) and send its URL back as ImageUrl. Boxes are normalized to that frame, so they match it exactly.
"""
import io
import os

MATCH_IOU = 0.3
MAX_GAP = 2  # a defect may be missed in one sample and still be recognised in the next one


def iou(a, b):
    """IoU of two normalized boxes {X, Y, Width, Height}."""
    left = max(a['X'], b['X'])
    top = max(a['Y'], b['Y'])
    right = min(a['X'] + a['Width'], b['X'] + b['Width'])
    bottom = min(a['Y'] + a['Height'], b['Y'] + b['Height'])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = a['Width'] * a['Height'] + b['Width'] * b['Height'] - intersection
    return intersection / union


def best_per_defect(frames):
    """frames: list of (sample number, detections) in time order.
    Returns one detection per defect: the one with the highest confidence."""
    defects = []  # each: {'class', 'box' (last seen), 'last_sample', 'best'}
    for sample, frame_detections in frames:
        recent = [d for d in defects if sample - d['last_sample'] <= MAX_GAP]

        # Match the most overlapping pairs first; each detection and each defect is used at most once.
        pairs = []
        for i, detection in enumerate(frame_detections):
            for j, defect in enumerate(recent):
                if detection['Class'] == defect['class']:
                    overlap = iou(detection['BoundingBox'], defect['box'])
                    if overlap >= MATCH_IOU:
                        pairs.append((overlap, i, j))
        pairs.sort(reverse=True)

        matched_detections = set()
        matched_defects = set()
        for _, i, j in pairs:
            if i in matched_detections or j in matched_defects:
                continue
            matched_detections.add(i)
            matched_defects.add(j)
            detection = frame_detections[i]
            defect = recent[j]
            defect['box'] = detection['BoundingBox']
            defect['last_sample'] = sample
            if detection['Confidence'] > defect['best']['Confidence']:
                defect['best'] = detection

        for i, detection in enumerate(frame_detections):
            if i not in matched_detections:
                defects.append({'class': detection['Class'], 'box': detection['BoundingBox'],
                                'last_sample': sample, 'best': detection})

    return [defect['best'] for defect in defects]


def save_best_frames(detections, request, folder, store):
    """Upload each needed frame once (folder/<FrameIndex>.jpg) and set ImageUrl on the detections."""
    urls = {}
    for detection in detections:
        frame_index = detection['FrameIndex']
        if frame_index not in urls:
            jpeg = (folder / f'{frame_index}.jpg').read_bytes()
            urls[frame_index] = store.save(request, frame_index, jpeg)
        detection['ImageUrl'] = urls[frame_index]


def storage_from_env():
    """MinIO settings, same variable names as the BE .env. Returns None when not configured (frames not saved)."""
    required = ['MINIO_ENDPOINT', 'MINIO_ACCESS_KEY', 'MINIO_SECRET_KEY', 'MINIO_BUCKET', 'MINIO_PUBLIC_BASE_URL']
    if not all(os.environ.get(name) for name in required):
        return None
    return {
        'endpoint': os.environ['MINIO_ENDPOINT'],
        'access_key': os.environ['MINIO_ACCESS_KEY'],
        'secret_key': os.environ['MINIO_SECRET_KEY'],
        'bucket': os.environ['MINIO_BUCKET'],
        'secure': os.environ.get('MINIO_USE_SSL', 'false').lower() == 'true',
        'public_base_url': os.environ['MINIO_PUBLIC_BASE_URL'].rstrip('/'),
    }


class FrameStore:
    def __init__(self, settings):
        from minio import Minio
        self.bucket = settings['bucket']
        self.public_base_url = settings['public_base_url']
        self.client = Minio(settings['endpoint'], access_key=settings['access_key'],
                            secret_key=settings['secret_key'], secure=settings['secure'])

    def save(self, request, frame_index, jpeg):
        """Upload one frame and return its public URL. The key is fixed per job, so a retry overwrites it."""
        key = f"frames/{request['MediaId']}/{request['RequestId']}/{frame_index}.jpg"
        self.client.put_object(self.bucket, key, io.BytesIO(jpeg), len(jpeg), content_type='image/jpeg')
        # The BE stores this URL as the key "<bucket>/frames/..." (StorageKeys.ToKey), like other media.
        return f'{self.public_base_url}/{self.bucket}/{key}'
