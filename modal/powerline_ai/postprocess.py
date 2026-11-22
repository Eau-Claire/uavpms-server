"""Deployment post-processing for raw YOLO detections. Pure Python, no torch.

Ultralytics NMS only suppresses overlaps within one class, so a single object can
come back as two classes, and one defect can come back as two same-class boxes.
Steps, in order:
  1. per-class confidence threshold
  2. merge same-class boxes that overlap (IoU) or nest (intersection / smaller area)
  3. cross-class: two defect boxes on the same object -> keep the more confident one;
     an insulator box nearly identical to a defect box is dropped (the defect wins)
  4. sort by confidence and cap the count
Detections are dicts with class_id, class_name, confidence, bbox_xyxy (pixels).
"""
from __future__ import annotations

from dataclasses import dataclass, field

CLASSES = ('insulator', 'insulator_broken', 'insulator_flashover',
           'bird_nest', 'foreign_object', 'broken_strand')
INSULATOR = 0
DEFECTS = frozenset(range(1, len(CLASSES)))


@dataclass(frozen=True)
class PostprocessConfig:
    class_confidence: dict = field(default_factory=lambda: {
        'insulator': 0.40, 'insulator_broken': 0.35, 'insulator_flashover': 0.35,
        'bird_nest': 0.40, 'foreign_object': 0.35, 'broken_strand': 0.30})
    merge_iou: float = 0.45        # same class: merge when IoU >= this
    merge_containment: float = 0.70  # same class: merge when overlap covers >= this of the smaller box
    cross_class_iou: float = 0.55  # two different defect classes on one object
    insulator_iou: float = 0.80    # insulator box almost identical to a defect box
    max_detections: int = 50
    merge_same_class: bool = True
    resolve_cross_class: bool = True

    def threshold(self, class_name: str) -> float:
        return float(self.class_confidence.get(class_name, 0.25))

    @property
    def min_confidence(self) -> float:
        return min(self.threshold(c) for c in CLASSES)


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def overlap(a, b):
    """Return (IoU, intersection / smaller area)."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    if inter <= 0:
        return 0.0, 0.0
    aa, bb = _area(a), _area(b)
    union = aa + bb - inter
    return (inter / union if union > 0 else 0.0), inter / max(min(aa, bb), 1e-9)


def _merge_same_class(dets, cfg):
    """Greedy by confidence: each kept box absorbs same-class boxes it overlaps or contains."""
    out = []
    for d in sorted(dets, key=lambda x: -x['confidence']):
        for k in out:
            if k['class_id'] != d['class_id']:
                continue
            iou, contain = overlap(k['bbox_xyxy'], d['bbox_xyxy'])
            if iou >= cfg.merge_iou or contain >= cfg.merge_containment:
                a, b = k['bbox_xyxy'], d['bbox_xyxy']
                k['bbox_xyxy'] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                k['merged'] = k.get('merged', 1) + d.get('merged', 1)
                break
        else:
            out.append(dict(d))
    return out


def _resolve_cross_class(dets, cfg):
    kept = []
    for d in sorted(dets, key=lambda x: -x['confidence']):
        drop = False
        for k in kept:
            if k['class_id'] == d['class_id']:
                continue
            iou, _ = overlap(k['bbox_xyxy'], d['bbox_xyxy'])
            both_defects = k['class_id'] in DEFECTS and d['class_id'] in DEFECTS
            if both_defects and iou >= cfg.cross_class_iou:
                drop = True  # k is more confident (sorted), d is the conflicting label
            elif d['class_id'] == INSULATOR and k['class_id'] in DEFECTS and iou >= cfg.insulator_iou:
                drop = True
            if drop:
                break
        if not drop:
            kept.append(d)
    # An insulator kept before a less confident but near-identical defect box: the defect wins.
    defects = [d for d in kept if d['class_id'] in DEFECTS]
    return [d for d in kept if d['class_id'] != INSULATOR or not any(
        overlap(d['bbox_xyxy'], x['bbox_xyxy'])[0] >= cfg.insulator_iou for x in defects)]


def postprocess(dets, cfg: PostprocessConfig):
    dets = [d for d in dets if d['confidence'] >= cfg.threshold(d['class_name'])]
    if cfg.merge_same_class:
        dets = _merge_same_class(dets, cfg)
    if cfg.resolve_cross_class:
        dets = _resolve_cross_class(dets, cfg)
    dets = [{**d, 'merged': d.get('merged', 1)} for d in dets]
    return sorted(dets, key=lambda x: -x['confidence'])[:cfg.max_detections]
