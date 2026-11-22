"""Retest of the three endpoint defects found by the first MF04 AI run (2026-11-22_1323).

DEF-009  DJI MPO files rejected with 415      -> resend every MPO image of the test split
DEF-010  10 MiB / 20 MP limit rejected UAV files -> resend every test image above those limits
DEF-011  single-image predict used a rect letterbox -> the endpoint must return the same boxes
         as a local square-letterbox (rect=False) predict of the same checkpoint and thresholds

Needs ultralytics + modal + fastapi + Pillow (yolov11/.venv) and the checkpoint in models/best.pt.

  python perf/retest_defects.py --dataset <finding_nemo_human_reviewed_6cls> --out perf/results/<run>
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from powerline_ai.media import decode_image  # noqa: E402
from powerline_ai.postprocess import CLASSES, overlap, postprocess  # noqa: E402
from powerline_ai.settings import IMAGE_SIZE, LETTERBOX_RECT, NMS_IOU, load_postprocess_config  # noqa: E402

APP, ENV = 'powerline-yolo26m-benchmark', 'main'
OLD_BYTES, OLD_PIXELS = 10 * 1024 * 1024, 20_000_000  # limits of the first deployment


def endpoint():
    import modal
    return modal.Cls.from_name(APP, 'Detector', environment_name=ENV)()


def call(det, data: bytes):
    try:
        return det.predict_jpeg.remote(data), None
    except Exception as exc:  # the endpoint raises HTTPException-like errors for 413/415
        return None, f'{type(exc).__name__}: {exc}'


def same_boxes(a, b, iou=0.98, conf=0.01):
    """Every detection in a has one partner in b with the same class, IoU >= iou and |conf diff| <= conf."""
    if len(a) != len(b):
        return False
    left = list(b)
    for d in a:
        hit = next((e for e in left if e['class_id'] == d['class_id']
                    and overlap(d['bbox_xyxy'], e['bbox_xyxy'])[0] >= iou
                    and abs(d['confidence'] - e['confidence']) <= conf), None)
        if hit is None:
            return False
        left.remove(hit)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--weights', type=Path, default=Path(__file__).resolve().parents[1] / 'models' / 'best.pt')
    p.add_argument('--landscape', type=int, default=40, help='val images used for the DEF-011 comparison')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    det = endpoint()
    report = {}

    test = sorted((a.dataset / 'images' / 'test').iterdir())
    mpo, big = [], []
    for f in test:
        with Image.open(f) as im:
            if im.format == 'MPO':
                mpo.append(f)
            if f.stat().st_size > OLD_BYTES or im.width * im.height > OLD_PIXELS:
                big.append(f)

    for key, files in (('DEF-009_mpo', mpo), ('DEF-010_large', big)):
        rows = []
        for f in files:
            r, err = call(det, f.read_bytes())
            with Image.open(f) as im:
                size = f'{im.width}x{im.height}'
            rows.append(dict(image=f.name, mib=round(f.stat().st_size / 2**20, 1), size=size, ok=err is None,
                             error=err, detections=None if r is None else len(r['detections']),
                             server_ms=None if r is None else round(sum(v or 0 for v in r['timing'].values()), 1)))
            print(key, rows[-1])
        report[key] = dict(images=len(rows), passed=sum(x['ok'] for x in rows),
                           result='Passed' if rows and all(x['ok'] for x in rows) else 'Failed', rows=rows)

    # DEF-011: local square-letterbox predict vs endpoint, on landscape images (the ones the bug hit).
    from ultralytics import YOLO
    model = YOLO(str(a.weights))
    cfg = load_postprocess_config()
    land = []
    for f in sorted((a.dataset / 'images' / 'val').iterdir()):
        with Image.open(f) as im:
            if im.width >= 1.3 * im.height:
                land.append(f)
        if len(land) >= a.landscape:
            break
    rows = []
    for f in land:
        data = f.read_bytes()
        frame = decode_image(data)
        local = {}
        for rect in (False, True):
            res = model.predict(frame, imgsz=IMAGE_SIZE, conf=cfg.min_confidence, iou=NMS_IOU, rect=rect,
                                device='cpu', verbose=False)[0]
            b = res.boxes.cpu()
            raw = [dict(class_id=int(c), class_name=CLASSES[int(c)], confidence=float(s), bbox_xyxy=x)
                   for x, s, c in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist())]
            local[rect] = postprocess(raw, cfg)
        r, err = call(det, data)
        remote = [] if r is None else r['detections']
        rows.append(dict(image=f.name, error=err, endpoint=len(remote), local_square=len(local[False]),
                         local_rect=len(local[True]), match_square=err is None and same_boxes(remote, local[False]),
                         match_rect=err is None and same_boxes(remote, local[True])))
        print('DEF-011', rows[-1])
    report['DEF-011_letterbox'] = dict(
        deployed_rect=LETTERBOX_RECT, images=len(rows),
        endpoint_matches_square=sum(x['match_square'] for x in rows),
        endpoint_matches_rect=sum(x['match_rect'] for x in rows),
        rect_differs_from_square=sum(x['local_square'] != x['local_rect'] for x in rows),
        result='Passed' if rows and not LETTERBOX_RECT and all(x['match_square'] for x in rows) else 'Failed',
        rows=rows)
    report['_env'] = dict(checkpoint_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),
                          note='local predict on CPU; endpoint on Modal T4 (FP32). Match = same class, '
                               'IoU >= 0.98, confidence within 0.01.')
    (a.out / 'retest_defects.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for k, v in report.items():
        if not k.startswith('_'):
            print(k, v['result'], {x: y for x, y in v.items() if x not in ('rows', 'result')})


if __name__ == '__main__':
    main()
