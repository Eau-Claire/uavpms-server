"""Score the deployed endpoint on a split with and without images flagged by audit_split_leakage.py.

Runs every image of the split through Detector.predict_jpeg once, saves detections to
detections_<split>.jsonl, then reports P/R/F1 (IoU 0.5) for: all images, and all images minus
those whose near-duplicate distance to train is <= d for each d in --distances.

  python perf/leakage_impact.py --dataset <ds> --split test \
      --pairs <yolov11>/artifacts/dataset_audit/leakage_<name>/leakage_pairs.csv --out perf/results/<run>
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mf04_ai_tests import APP, ENV, IMAGE_EXTS, accuracy, read_gt, rounded


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--distances', type=int, nargs='+', default=[0, 2, 4])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    import modal
    detector = modal.Cls.from_name(APP, 'Detector', environment_name=ENV)()
    images = sorted(x for x in (a.dataset / 'images' / a.split).iterdir() if x.suffix.lower() in IMAGE_EXTS)
    cache = a.out / f'detections_{a.split}.jsonl'
    if cache.is_file():
        results = {json.loads(line)['image']: json.loads(line) for line in cache.open(encoding='utf-8')}
    else:
        with ThreadPoolExecutor(8) as pool:
            outs = list(pool.map(lambda x: (x.name, detector.predict_jpeg.remote(x.read_bytes())), images))
        results = {n: dict(image=n, width=r['width'], height=r['height'], detections=r['detections']) for n, r in outs}
        a.out.mkdir(parents=True, exist_ok=True)
        cache.write_text(''.join(json.dumps(v) + '\n' for v in results.values()), encoding='utf-8')

    # Smallest train distance per image of this split.
    dist = {}
    with a.pairs.open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['split_a'] == 'train' and r['split_b'] == a.split:
                d = int(r['dhash_distance'])
                dist[r['image_b']] = min(d, dist.get(r['image_b'], 99))

    def score(names):
        pairs = []
        for n in names:
            r = results[n]
            pairs.append((read_gt(a.dataset / 'labels' / a.split / f'{Path(n).stem}.txt', r['width'], r['height']),
                          r['detections']))
        return accuracy(pairs)

    report = {'all': dict(images=len(results), **score(results))}
    for d in a.distances:
        keep = [n for n in results if dist.get(n, 99) > d]
        report[f'exclude_train_near_dup_d<={d}'] = dict(images=len(keep), excluded=len(results) - len(keep), **score(keep))
    leaked = [n for n in results if dist.get(n, 99) <= max(a.distances)]
    report[f'only_flagged_d<={max(a.distances)}'] = dict(images=len(leaked), **score(leaked))
    (a.out / f'leakage_impact_{a.split}.json').write_text(json.dumps(rounded(report), indent=2), encoding='utf-8')
    for k, v in report.items():
        m = v['micro']
        fo = v['per_class']['foreign_object']
        print(f'{k:34} images={v["images"]:5}  P={m["precision"]:.4f} R={m["recall"]:.4f} F1={m["f1"]:.4f} '
              f'macroF1={m["macro_f1"]:.4f}  FO F1={fo["f1"]:.4f} (tp {fo["tp"]}, fn {fo["fn"]})')


if __name__ == '__main__':
    main()
