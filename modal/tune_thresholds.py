"""Choose the deployment post-processing from data: per-class thresholds and which steps to enable.

1. Run the checkpoint once on a labelled split at conf 0.05 and cache the raw boxes.
2. For each combination of the two optional steps (same-class merge, cross-class
   resolution), tune every class threshold by coordinate descent on that class's F1
   (IoU >= 0.5, after post-processing).
3. Keep the combination with the best macro-F1 (each class weighted equally, so rare
   defects count as much as insulators) and write powerline_ai/thresholds.json.

Tune on val only. Use --evaluate once on test to confirm the chosen file, never to re-tune.

  python tune_thresholds.py --weights models/best.pt --images <ds>/images/val --labels <ds>/labels/val
  python tune_thresholds.py --weights models/best.pt --images <ds>/images/test --labels <ds>/labels/test \\
      --evaluate powerline_ai/thresholds.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from powerline_ai.postprocess import CLASSES, PostprocessConfig, overlap, postprocess
from powerline_ai.settings import NMS_IOU, IMAGE_SIZE, load_postprocess_config

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
GRID = [round(0.05 * i, 2) for i in range(2, 19)]  # 0.10 .. 0.90
RAW_CONF = 0.05
DEPLOYED_BEFORE = PostprocessConfig(class_confidence={c: 0.25 for c in CLASSES},
                                    merge_same_class=False, resolve_cross_class=False)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--evaluate', type=Path, help='only report this thresholds.json on the given split')
    p.add_argument('--cache', type=Path, help='raw prediction cache (default: next to --images)')
    p.add_argument('--device', default='0', help="'0' for GPU, 'cpu' for CPU")
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--match-iou', type=float, default=0.5)
    p.add_argument('--out', type=Path, default=Path(__file__).with_name('powerline_ai') / 'thresholds.json')
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_gt(path: Path, w: int, h: int):
    out = []
    if path.is_file():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            p = line.split()
            if len(p) == 5:
                c, x, y, bw, bh = int(float(p[0])), *map(float, p[1:])
                out.append([c, [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]])
    return out


def predict_split(a, weights_sha):
    """Raw detections for every image, cached on disk and keyed by checkpoint hash and settings."""
    cache = a.cache or Path(__file__).with_name('.cache') / f'raw_{a.images.name}_{weights_sha[:12]}.json'
    cache.parent.mkdir(parents=True, exist_ok=True)
    key = dict(weights_sha256=weights_sha, images=str(a.images.resolve()), imgsz=IMAGE_SIZE,
               nms_iou=NMS_IOU, raw_conf=RAW_CONF)
    if cache.is_file():
        data = json.loads(cache.read_text(encoding='utf-8'))
        if data.get('key') == key:
            print(f'Using cached predictions: {cache}')
            return data['samples']
    from ultralytics import YOLO
    model = YOLO(str(a.weights))
    if tuple(model.names[i] for i in range(len(model.names))) != CLASSES:
        raise SystemExit(f'Wrong taxonomy: {model.names}')
    images = sorted(p for p in a.images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    samples = []
    for s in range(0, len(images), a.batch):
        chunk = images[s:s + a.batch]
        results = model.predict([str(p) for p in chunk], imgsz=IMAGE_SIZE, conf=RAW_CONF, iou=NMS_IOU,
                                device=a.device, verbose=False)
        for img, r in zip(chunk, results):
            b = r.boxes.cpu()
            h, w = r.orig_shape
            raw = [dict(class_id=int(c), class_name=CLASSES[int(c)], confidence=round(float(sc), 5),
                        bbox_xyxy=[round(v, 2) for v in bx])
                   for bx, sc, c in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist())]
            samples.append(dict(image=img.name, gt=read_gt(a.labels / f'{img.stem}.txt', w, h), raw=raw))
        print(f'\rpredicted {len(samples)}/{len(images)}', end='', flush=True)
    print()
    cache.write_text(json.dumps(dict(key=key, samples=samples)), encoding='utf-8')
    return samples


def score(samples, cfg, match_iou):
    """Per-class TP/FP/FN after post-processing; class-aware greedy matching by confidence."""
    n = len(CLASSES)
    tp, fp, fn = [0] * n, [0] * n, [0] * n
    for s in samples:
        gt = s['gt']
        used = set()
        for d in postprocess(s['raw'], cfg):
            best, bi = 0.0, None
            for gi, (c, box) in enumerate(gt):
                if c == d['class_id'] and gi not in used:
                    v = overlap(box, d['bbox_xyxy'])[0]
                    if v > best:
                        best, bi = v, gi
            if bi is not None and best >= match_iou:
                used.add(bi)
                tp[d['class_id']] += 1
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


def summary(samples, cfg, match_iou):
    tp, fp, fn = score(samples, cfg, match_iou)
    per_class = {c: dict(zip(('precision', 'recall', 'f1'), prf(tp[i], fp[i], fn[i])), fp=fp[i], fn=fn[i])
                 for i, c in enumerate(CLASSES)}
    p, r, f = prf(sum(tp), sum(fp), sum(fn))
    macro = sum(v['f1'] for v in per_class.values()) / len(CLASSES)
    return dict(per_class=per_class, micro=dict(precision=p, recall=r, f1=f, fp=sum(fp), fn=sum(fn)),
                macro_f1=macro)


def print_summary(title, cfg, s):
    print(f'\n{title}  [merge={cfg.merge_same_class} cross_class={cfg.resolve_cross_class}]')
    print(f'  {"class":20} {"thr":>5} {"P":>6} {"R":>6} {"F1":>6} {"FP":>6} {"FN":>6}')
    for c, v in s['per_class'].items():
        print(f'  {c:20} {cfg.threshold(c):5.2f} {v["precision"]:6.3f} {v["recall"]:6.3f} {v["f1"]:6.3f} '
              f'{v["fp"]:6d} {v["fn"]:6d}')
    m = s['micro']
    print(f'  {"ALL (micro)":20} {"":5} {m["precision"]:6.3f} {m["recall"]:6.3f} {m["f1"]:6.3f} '
          f'{m["fp"]:6d} {m["fn"]:6d}')
    print(f'  macro-F1 {s["macro_f1"]:.4f}')


def tune_thresholds(samples, cfg, match_iou, passes=2):
    for _ in range(passes):
        for i, c in enumerate(CLASSES):
            best_f1, best_t = -1.0, cfg.threshold(c)
            for t in GRID:
                trial = replace(cfg, class_confidence={**cfg.class_confidence, c: t})
                tp, fp, fn = score(samples, trial, match_iou)
                f1 = prf(tp[i], fp[i], fn[i])[2]
                if f1 > best_f1 + 1e-9:
                    best_f1, best_t = f1, t
            cfg = replace(cfg, class_confidence={**cfg.class_confidence, c: best_t})
    return cfg


def main():
    a = parse_args()
    weights_sha = sha256(a.weights)
    samples = predict_split(a, weights_sha)
    samples = [dict(s, raw=[d for d in s['raw'] if d['confidence'] >= GRID[0]]) for s in samples]
    print(f'{len(samples)} images, {sum(len(s["gt"]) for s in samples)} GT boxes, '
          f'{sum(len(s["raw"]) for s in samples)} raw boxes >= {GRID[0]}')

    before = summary(samples, DEPLOYED_BEFORE, a.match_iou)
    print_summary('BEFORE (conf 0.25 for all classes, no post-processing)', DEPLOYED_BEFORE, before)
    if a.evaluate:
        cfg = load_postprocess_config(a.evaluate)
        print_summary(f'EVALUATE {a.evaluate}', cfg, summary(samples, cfg, a.match_iou))
        return

    results = []
    for merge in (False, True):
        for cross in (False, True):
            cfg = replace(PostprocessConfig(), merge_same_class=merge, resolve_cross_class=cross)
            cfg = tune_thresholds(samples, cfg, a.match_iou)
            s = summary(samples, cfg, a.match_iou)
            print_summary('TUNED', cfg, s)
            results.append((s['macro_f1'], cfg, s))
    # Ties go to the combination with fewer enabled steps: a step must earn its place.
    best_f1, best_cfg, best = max(results, key=lambda x: (
        round(x[0], 6), -(x[1].merge_same_class + x[1].resolve_cross_class)))
    print('\nCandidates (macro-F1): ' + ', '.join(
        f'merge={c.merge_same_class}/cross={c.resolve_cross_class}: {f:.4f}' for f, c, _ in results))
    print_summary('CHOSEN', best_cfg, best)

    a.out.write_text(json.dumps({
        # Dataset-relative split name only; no machine-specific absolute paths in a committed file.
        '_source': dict(weights=a.weights.name, weights_sha256=weights_sha,
                        split='/'.join(a.images.resolve().parts[-3:]),
                        match_iou=a.match_iou, objective='per-class F1 thresholds, step choice by macro-F1',
                        before=dict(macro_f1=round(before['macro_f1'], 4), micro=before['micro']),
                        chosen=dict(macro_f1=round(best['macro_f1'], 4), micro=best['micro'])),
        'class_confidence': best_cfg.class_confidence,
        'merge_same_class': best_cfg.merge_same_class,
        'resolve_cross_class': best_cfg.resolve_cross_class,
    }, indent=2), encoding='utf-8')
    print('\nWrote', a.out)


if __name__ == '__main__':
    main()
