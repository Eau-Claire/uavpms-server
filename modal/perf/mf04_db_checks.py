"""Check what the backend stored for e2e uploads against the model output for the same images.

For every uploaded image (from mf04_e2e jobs.jsonl) this compares:
  - label mapping: each non-insulator detection the model returns must be stored once with the mapped
    DefectCategory; insulator must create nothing (AiLabel:insulator = NONE);
  - bounding boxes: stored {x, y, width, height} (normalized, top-left) vs the model box normalized to
    the frame the bridge sent; IoU >= 0.95;
  - GPS: InspectionMedia.CaptureLocation equals the EXIF GPS of the file, or is NULL when the file has
    none (never a tower position); DetectedAnomalies.Location is NULL (the bridge sends no GPS).

The model is called through the Modal SDK with the bytes the bridge would send (OpenCV decode,
downscale above 20 MP, JPEG re-encode), so differences come from the backend, not from preprocessing.
Run locally (needs modal, opencv, Pillow); the database rows come from a JSON dump made on the VPS:

  ssh vps "docker exec -i uavpms-db psql -U <user> -d <db> -At" < perf/mf04_db_dump.sql > dump.json
  python perf/mf04_db_checks.py --dump dump.json --jobs <run>/jobs.jsonl ... --images <test images dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'bridge'))
from processor import encode  # noqa: E402  same preprocessing as the bridge

APP, ENV = 'powerline-yolo26m-benchmark', 'main'
MAPPING = {'insulator': None, 'insulator_broken': 'BROKEN_INSULATOR', 'insulator_flashover': 'FLASHOVER',
           'bird_nest': 'BIRD_NEST', 'foreign_object': 'FOREIGN_OBJECT', 'broken_strand': 'BROKEN_STRAND'}


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def exif_gps(path: Path):
    """(lat, lon) from EXIF GPS, or None."""
    try:
        gps = Image.open(path).getexif().get_ifd(0x8825)
    except Exception:
        return None
    if not gps or 2 not in gps or 4 not in gps:
        return None

    def deg(v, ref):
        d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        return -d if ref in ('S', 'W') else d
    return deg(gps[2], gps.get(1, 'N')), deg(gps[4], gps.get(3, 'E'))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dump', type=Path, required=True)
    p.add_argument('--jobs', type=Path, nargs='+', required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()

    import modal
    det = modal.Cls.from_name(APP, 'Detector', environment_name=ENV)()
    db = json.loads(a.dump.read_text())
    media_db = {m['id']: m for m in db['media']}
    anomalies = {}
    for x in db['anomalies']:
        anomalies.setdefault(x['media_id'], []).append(x)

    rows, totals = [], Counter()
    for jobs in a.jobs:
        for line in jobs.read_text().splitlines():
            job = json.loads(line)
            path, mid = a.images / job['file'], job['media_id']
            out = det.predict_jpeg.remote(encode(cv2.imread(str(path))))
            w, h = out['width'], out['height']
            model = [dict(cls=d['class_name'], code=MAPPING[d['class_name']],
                          box=[d['bbox_xyxy'][0] / w, d['bbox_xyxy'][1] / h,
                               (d['bbox_xyxy'][2] - d['bbox_xyxy'][0]) / w, (d['bbox_xyxy'][3] - d['bbox_xyxy'][1]) / h])
                     for d in out['detections']]
            expected = [m for m in model if m['code']]
            stored = anomalies.get(mid, [])
            # greedy match by category and IoU
            left, matched, ious = list(stored), 0, []
            for m in sorted(expected, key=lambda m: -m['box'][2] * m['box'][3]):
                best = max((s for s in left if s['category'] == m['code']),
                           key=lambda s: iou(m['box'], s['box']), default=None)
                if best is not None:
                    v = iou(m['box'], best['box'])
                    ious.append(v)
                    if v >= 0.95:
                        matched += 1
                        left.remove(best)
            gps = exif_gps(path)
            md = media_db.get(mid, {})
            db_gps = (md.get('lat'), md.get('lon')) if md.get('lat') is not None else None
            gps_ok = (gps is None and db_gps is None) or (
                gps is not None and db_gps is not None and abs(gps[0] - db_gps[0]) < 1e-5 and abs(gps[1] - db_gps[1]) < 1e-5)
            row = dict(file=job['file'], media_id=mid,
                       model_classes=dict(Counter(m['cls'] for m in model)),
                       stored_categories=dict(Counter(s['category'] for s in stored)),
                       insulator_from_model=sum(m['cls'] == 'insulator' for m in model),
                       expected=len(expected), stored=len(stored), matched=matched,
                       min_iou=round(min(ious), 4) if ious else None,
                       exif_gps=gps, db_gps=db_gps, gps_ok=gps_ok,
                       anomaly_location_null=all(s['location'] is None for s in stored))
            row['mapping_ok'] = row['matched'] == row['expected'] == row['stored']
            rows.append(row)
            totals['images'] += 1
            totals['model_detections'] += len(model)
            totals['insulator_from_model'] += row['insulator_from_model']
            totals['expected_defects'] += len(expected)
            totals['stored_defects'] += len(stored)
            totals['matched_iou_095'] += matched
            totals['images_with_exif_gps'] += gps is not None
            print(json.dumps({k: row[k] for k in ('file', 'model_classes', 'stored_categories', 'matched',
                                                   'expected', 'stored', 'min_iou', 'gps_ok')}))
    checks = dict(
        mapping_and_boxes=all(r['mapping_ok'] for r in rows),
        insulator_creates_no_defect=not any(s['category'] in (None, 'NONE') for v in anomalies.values() for s in v)
        and totals['insulator_from_model'] > 0,
        gps_exif_or_null=all(r['gps_ok'] for r in rows),
        anomaly_location_null=all(r['anomaly_location_null'] for r in rows))
    summary = dict(totals=totals, checks=checks, result='Passed' if all(checks.values()) else 'Failed',
                   failures=[r for r in rows if not (r['mapping_ok'] and r['gps_ok'])][:20])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'db_checks.json').write_text(json.dumps(dict(summary=summary, rows=rows), indent=2, default=str))
    print(json.dumps(dict(totals=totals, checks=checks, result=summary['result']), indent=2))


if __name__ == '__main__':
    main()
