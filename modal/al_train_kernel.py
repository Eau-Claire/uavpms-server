"""Kaggle kernel: fine-tune a challenger from the champion and evaluate both on the fixed gate set.

Inputs (attached Kaggle datasets, found automatically under /kaggle/input):
  - the cycle bundle (pushed by al_cycle.py): cycle.json, champion.pt, feedback/images, feedback/labels
  - Finding Nemo (tydapchai/finding-nemo-human-reviewed-6cls): replay pool (train), selection (val), gate set (test)
Outputs (/kaggle/working/result): challenger.pt, thresholds.json (tuned on val with the serving code shipped in the
bundle), metrics.json, train_results.csv, kernel.log

The gate set is never trained on: the kernel stops if a feedback image has the same SHA-256 as a gate image.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

INPUT = Path(os.environ.get('AL_INPUT', '/kaggle/input'))
WORK = Path(os.environ.get('AL_WORK', '/kaggle/working'))
RESULT = WORK / 'result'
CLASSES = ['insulator', 'insulator_broken', 'insulator_flashover', 'bird_nest', 'foreign_object', 'broken_strand']
DEFECTS = CLASSES[1:]
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
LOG = []


def log(msg: str):
    line = f'{time.strftime("%H:%M:%S")} {msg}'
    print(line, flush=True)
    LOG.append(line)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def find_one(pattern: str) -> Path:
    hits = sorted(INPUT.rglob(pattern))
    if len(hits) != 1:
        raise SystemExit(f'expected exactly one {pattern} under {INPUT}, found {[str(h) for h in hits]}')
    return hits[0]


def images_in(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def link_pair(img: Path, label_dir: Path, out_img: Path, out_lbl: Path, prefix: str):
    name = prefix + img.name
    (out_img / name).symlink_to(img)
    lbl = label_dir / (img.stem + '.txt')
    (out_lbl / (prefix + img.stem + '.txt')).write_text(lbl.read_text() if lbl.exists() else '')


def evaluate(weights: Path, data_yaml: Path, imgsz: int) -> dict:
    from ultralytics import YOLO
    r = YOLO(str(weights)).val(data=str(data_yaml), split='test', imgsz=imgsz, batch=16, plots=False,
                               verbose=False, project=str(WORK / 'eval'), name=weights.stem, exist_ok=True)
    ap50 = [float(x) for x in r.box.ap50]
    ap = [float(x) for x in r.box.ap]
    per_class = {c: {'ap50': round(ap50[i], 4), 'ap50_95': round(ap[i], 4)} for i, c in enumerate(CLASSES)}
    rd = r.results_dict
    return {
        'weights_sha256': sha256(weights),
        'precision': round(float(rd['metrics/precision(B)']), 4),
        'recall': round(float(rd['metrics/recall(B)']), 4),
        'mAP50': round(float(rd['metrics/mAP50(B)']), 4),
        'mAP50_95': round(float(rd['metrics/mAP50-95(B)']), 4),
        'MEAN_5': round(sum(per_class[c]['ap50'] for c in DEFECTS) / len(DEFECTS), 4),
        'per_class': per_class,
    }


def main():
    t0 = time.time()
    RESULT.mkdir(parents=True, exist_ok=True)
    cycle_path = find_one('cycle.json')
    bundle = cycle_path.parent
    cfg = json.loads(cycle_path.read_text())
    champion = bundle / 'champion.pt'
    if sha256(champion) != cfg['champion_sha256']:
        raise SystemExit('champion.pt does not match cycle.json')
    nemo = find_one('finding_nemo_human_reviewed_6cls/data.yaml').parent
    log(f'cycle {cfg["cycle_id"]}: bundle {bundle}, Finding Nemo {nemo}')

    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', f'ultralytics=={cfg["ultralytics"]}'], check=True)
    import torch
    log(f'torch {torch.__version__}, cuda {torch.version.cuda}, gpus {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}')

    fb_img, fb_lbl = bundle / 'feedback' / 'images', bundle / 'feedback' / 'labels'
    feedback = images_in(fb_img)
    gate_imgs = images_in(nemo / 'images' / 'test')
    # Feedback must not contain test (gate) or val (checkpoint selection + threshold tuning) images.
    held_out = {sha256(p): split for split in ('test', 'val') for p in images_in(nemo / 'images' / split)}
    leaked = [(p.name, held_out[h]) for p in feedback if (h := sha256(p)) in held_out]
    if leaked:
        raise SystemExit(f'{len(leaked)} feedback images are in held-out splits, refusing to train: {leaked[:5]}')
    log(f'feedback {len(feedback)} images; gate set {len(gate_imgs)} images; no overlap with val/test')

    # Training set = all feedback + a seeded replay sample of the original curated train split.
    ds = WORK / 'al_dataset'
    if ds.exists():
        shutil.rmtree(ds)
    for split in ('train', 'val', 'test'):
        (ds / 'images' / split).mkdir(parents=True)
        (ds / 'labels' / split).mkdir(parents=True)
    for p in feedback:
        link_pair(p, fb_lbl, ds / 'images' / 'train', ds / 'labels' / 'train', 'fb_')
    pool = images_in(nemo / 'images' / 'train')
    rng = random.Random(cfg['seed'])
    replay = rng.sample(pool, min(cfg['replay_images'], len(pool)))
    for p in replay:
        link_pair(p, nemo / 'labels' / 'train', ds / 'images' / 'train', ds / 'labels' / 'train', 'rp_')
    for split in ('val', 'test'):   # val: checkpoint selection; test: fixed gate set
        for p in images_in(nemo / 'images' / split):
            link_pair(p, nemo / 'labels' / split, ds / 'images' / split, ds / 'labels' / split, '')
    data_yaml = ds / 'data.yaml'
    data_yaml.write_text('path: ' + str(ds) + '\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'
                         + ''.join(f'  {i}: {c}\n' for i, c in enumerate(CLASSES)))
    log(f'train set: {len(feedback)} feedback + {len(replay)} replay')

    from ultralytics import YOLO
    model = YOLO(str(champion))
    if [model.names[i] for i in range(len(model.names))] != CLASSES:
        raise SystemExit(f'champion taxonomy mismatch: {model.names}')
    tr = cfg['train']
    model.train(data=str(data_yaml), epochs=tr['epochs'], imgsz=tr['imgsz'], batch=tr['batch'], optimizer='AdamW',
                lr0=tr['lr0'], lrf=tr['lrf'], cos_lr=True, warmup_epochs=tr['warmup_epochs'], weight_decay=5e-4,
                close_mosaic=tr['close_mosaic'], amp=True, seed=cfg['seed'], deterministic=True, workers=2,
                device=0, plots=False, project=str(WORK / 'runs'), name='challenger', exist_ok=True)
    run = WORK / 'runs' / 'challenger'
    best = run / 'weights' / 'best.pt'
    shutil.copy2(best, RESULT / 'challenger.pt')
    if (run / 'results.csv').exists():
        shutil.copy2(run / 'results.csv', RESULT / 'train_results.csv')

    # Per-class thresholds for the challenger, tuned on val with the exact post-processing the endpoint uses.
    code = bundle / 'code'
    subprocess.run([sys.executable, str(code / 'tune_thresholds.py'), '--weights', str(RESULT / 'challenger.pt'),
                    '--images', str(ds / 'images' / 'val'), '--labels', str(ds / 'labels' / 'val'),
                    '--out', str(RESULT / 'thresholds.json'), '--cache', str(WORK / 'tune_cache.json'), '--device', '0'],
                   check=True, cwd=code, env={**os.environ, 'PYTHONPATH': str(code)})
    log('thresholds tuned on val')

    log('evaluating champion and challenger on the gate set')
    metrics = {
        'cycle_id': cfg['cycle_id'],
        'gate_set': {'name': 'finding_nemo_human_reviewed_6cls/test', 'images': len(gate_imgs)},
        'train_set': {'feedback_images': len(feedback), 'replay_images': len(replay), 'seed': cfg['seed']},
        'champion': evaluate(champion, data_yaml, tr['imgsz']),
        'challenger': evaluate(RESULT / 'challenger.pt', data_yaml, tr['imgsz']),
        'challenger_thresholds_sha256': sha256(RESULT / 'thresholds.json'),
        'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        'ultralytics': cfg['ultralytics']},
        'duration_s': round(time.time() - t0),
    }
    (RESULT / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    log(f'champion MEAN_5 {metrics["champion"]["MEAN_5"]} mAP50-95 {metrics["champion"]["mAP50_95"]} | '
        f'challenger MEAN_5 {metrics["challenger"]["MEAN_5"]} mAP50-95 {metrics["challenger"]["mAP50_95"]}')
    (RESULT / 'kernel.log').write_text('\n'.join(LOG) + '\n')


def keep_only_result():
    """Kaggle publishes the whole working directory as the kernel output: leave only result/ (a few files)."""
    for p in WORK.iterdir():
        if p == RESULT:
            continue
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:   # leave a readable reason in the output, then fail the kernel
        RESULT.mkdir(parents=True, exist_ok=True)
        (RESULT / 'kernel.log').write_text('\n'.join(LOG) + f'\nFAILED: {exc!r}\n')
        raise
    finally:
        keep_only_result()
