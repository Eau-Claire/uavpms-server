"""Pure mapping helpers for the future BE-side bridge; no network side effects."""
import math
from uuid import UUID, uuid5, NAMESPACE_URL

def normalized_box(xyxy, width, height):
    if len(xyxy) != 4 or not all(math.isfinite(v) for v in [*xyxy, width, height]):
        raise ValueError('Invalid coordinates')
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image dimensions')
    x1,y1,x2,y2=xyxy
    x1,x2=[min(1.,max(0.,v/width)) for v in (x1,x2)]
    y1,y2=[min(1.,max(0.,v/height)) for v in (y1,y2)]
    if x2<=x1 or y2<=y1:raise ValueError('Degenerate box')
    return dict(X=x1,Y=y1,Width=x2-x1,Height=y2-y1)

def detection_id(request_id, frame_index, detection_index):
    """Deterministic only for a persisted, ordered prediction result."""
    UUID(str(request_id))
    return str(uuid5(NAMESPACE_URL, f'{request_id}:{frame_index}:{detection_index}'))
