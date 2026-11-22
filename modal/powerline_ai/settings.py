import json
from pathlib import Path

from .postprocess import CLASSES, PostprocessConfig

MODEL_NAME = 'seed85_curated_yolo26m'
MODEL_PATH = '/models/best.pt'
IMAGE_SIZE = 640
NMS_IOU = 0.6
# Square letterbox, as in training and threshold tuning. Ultralytics' default for a single
# image is a minimal rectangle, which lowered val recall from 0.840 to 0.813 for this model.
LETTERBOX_RECT = False
# UAV cameras: DJI writes MPO (multi-picture JPEG); full-resolution frames reach ~50 MP / ~50 MiB.
IMAGE_FORMATS = ('JPEG', 'MPO', 'PNG')
MAX_BYTES = 50 * 1024 * 1024
MAX_PIXELS = 60_000_000

# thresholds.json (written by tune_thresholds.py) overrides the defaults in PostprocessConfig.
_THRESHOLDS = Path(__file__).with_name('thresholds.json')


def load_postprocess_config(path: Path = _THRESHOLDS) -> PostprocessConfig:
    if not path.is_file():
        return PostprocessConfig()
    raw = json.loads(path.read_text(encoding='utf-8'))
    allowed = set(PostprocessConfig.__dataclass_fields__)
    unknown = set(raw) - allowed - {'_source'}
    if unknown:
        raise ValueError(f'Unknown keys in {path.name}: {sorted(unknown)}')
    bad = set(raw.get('class_confidence', {})) - set(CLASSES)
    if bad:
        raise ValueError(f'Unknown classes in {path.name}: {sorted(bad)}')
    base = PostprocessConfig().class_confidence
    raw = {k: v for k, v in raw.items() if k in allowed}
    raw['class_confidence'] = {**base, **raw.get('class_confidence', {})}
    return PostprocessConfig(**raw)


POSTPROCESS = load_postprocess_config()
