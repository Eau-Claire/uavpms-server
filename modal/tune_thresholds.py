"""Tune per-class confidence thresholds on the validation split, with the deployment post-processing.

Runs the checkpoint once on val at a low confidence, then for each class sweeps its
threshold (others fixed) and keeps the value with the best F1 at IoU >= 0.5 after
postprocess(). Two coordinate-descent passes, because cross-class suppression couples classes.
Writes powerline_ai/thresholds.json, which the deployment loads at start-up.

Never tune on test. Example (Windows, from uavpms-server/modal, training .venv):
  python tune_thresholds.py --weights models/best.pt ^
    --images D:/.../finding_nemo_human_reviewed_6cls/images/val ^
    --labels D:/.../finding_nemo_human_reviewed_6cls/labels/val
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from powerline_ai.postprocess import CLASSES, PostprocessConfig, postprocess

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
GRID = [round(0.05 * i, 2) for i in range(2, 19)]  # 0.10 .. 0.90


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--nms-iou', type=float, default=0.6)
    p.add_argument('--device', default='0')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--match-iou', type=float, default=0.5)
    p.add_argument('--out', type=Path, default=Path(__file__).with_name('powerline_ai') / 'thresholds.json')
    return p.parse_args()


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def read_gt(path, w, h):
    out = []
    if path.is_file():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            p = line.split()
            if len(p) == 5:
                c, x, y, bw, bh = int(float(p[0])), *map(float, p[1:])
                out.append((c, [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
    return out


def score(samples, cfg, match_iou):
    """Per-class TP/FP/FN after postprocess, class-aware greedy matching."""
    tp, fp, fn = ([0] * len(CLASSES) for _ in range(3))
    for gt, raw in samples:
        dets = postprocess(raw, cfg)
        used = set()
        for d in dets:  # already sorted by confidence
            best, bi = 0.0, None
            for gi, (c, box) in enumerate(gt):
                if c == d['class_id'] and gi not in used:
                    v = iou(box, d['bbox_xyxy'])
                    if v > best:
                        best, bi = v, gi
            if bi is not None and best >= match_iou:
                used.add(bi); tp[d['class_id']] += 1
            else:
                fp[d['class_id']] += 1
        for gi, (c, _) in enumerate(gt):
            if gi not in used:
                fn[c] += 1
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def report(title, samples, cfg, match_iou):
    tp, fp, fn = score(samples, cfg, match_iou)
    print(f'\n{title}')
    print(f'  {"class":20} {"thr":>5} {"P":>6} {"R":>6} {"F1":>6} {"FP":>6} {"FN":>6}')
    for i, c in enumerate(CLASSES):
        p, r, f = prf(tp[i], fp[i], fn[i])
        print(f'  {c:20} {cfg.threshold(c):5.2f} {p:6.3f} {r:6.3f} {f:6.3f} {fp[i]:6d} {fn[i]:6d}')
    p, r, f = prf(sum(tp), sum(fp), sum(fn))
    print(f'  {"ALL":20} {"":5} {p:6.3f} {r:6.3f} {f:6.3f} {sum(fp):6d} {sum(fn):6d}')


def main():
    a = parse_args()
    from PIL import Image
    from ultralytics import YOLO
    model = YOLO(str(a.weights))
    if tuple(model.names[i] for i in range(len(model.names))) != CLASSES:
        raise SystemExit(f'Wrong taxonomy: {model.names}')
    images = sorted(p for p in a.images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    samples = []
    for s in range(0, len(images), a.batch):
        chunk = images[s:s + a.batch]
        for img, r in zip(chunk, model.predict([str(p) for p in chunk], imgsz=a.imgsz, conf=0.05,
                                               iou=a.nms_iou, device=a.device, verbose=False)):
            b = r.boxes.cpu()
            raw = [dict(class_id=int(c), class_name=CLASSES[int(c)], confidence=float(sc), bbox_xyxy=bx)
                   for bx, sc, c in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist())]
            h, w = r.orig_shape
            samples.append((read_gt(a.labels / f'{img.stem}.txt', w, h), raw))
        print(f'\rpredicted {len(samples)}/{len(images)}', end='', flush=True)

    flat = {c: 0.25 for c in CLASSES}
    report('BEFORE: conf 0.25 for every class, no post-processing',
           samples, PostprocessConfig(class_confidence=flat, enabled=False), a.match_iou)
    cfg = PostprocessConfig(class_confidence=dict(PostprocessConfig().class_confidence))
    for _ in range(2):
        for i, c in enumerate(CLASSES):
            best = None
            for t in GRID:
                trial = replace(cfg, class_confidence={**cfg.class_confidence, c: t})
                tp, fp, fn = score(samples, trial, a.match_iou)
                f1 = prf(tp[i], fp[i], fn[i])[2]
                if best is None or f1 > best[0] + 1e-9:
                    best = (f1, t)
            cfg = replace(cfg, class_confidence={**cfg.class_confidence, c: best[1]})
            print(f'  {c:20} -> {best[1]:.2f} (F1 {best[0]:.3f})')
    report('AFTER: tuned per-class thresholds + post-processing', samples, cfg, a.match_iou)

    a.out.write_text(json.dumps({
        '_source': {'weights': str(a.weights), 'images': str(a.images), 'match_iou': a.match_iou,
                    'objective': 'per-class F1 after postprocess, val split'},
        'class_confidence': cfg.class_confidence,
    }, indent=2), encoding='utf-8')
    print('\nWrote', a.out)


if __name__ == '__main__':
    main()
