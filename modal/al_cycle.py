"""Automatic active-learning cycle: Kaggle retraining, fixed gate, Modal deploy, smoke test, automatic rollback.

  python al_cycle.py init                         # once: create finding_model/ on Camber with the current champion
  python al_cycle.py cycle    --feedback <dir>    # full cycle; deploys only if the gate passes
  python al_cycle.py fetch    --cycle-id <id>     # resume after an interruption (no re-push)
  python al_cycle.py gate     --result <dir>      # recompute the gate report for a fetched result
  python al_cycle.py rollback                     # redeploy the previous champion (also automatic on failure)

Run with a Python that has the `modal` package (e.g. yolov11/.venv/bin/python); `kaggle` and `camber` must be on
PATH and authenticated (CAMBER_API_KEY in the environment or in --env-file).

Feedback directory (exported by the backend after human review; YOLO format, 6 classes):
  <dir>/images/*.jpg|png   <dir>/labels/<same stem>.txt   (empty .txt = reviewed image with no object)

Safety: nothing changes in production unless the challenger passes the gate on the fixed test set. The previous
champion is archived (locally and on Camber) before the swap. After `modal deploy` the endpoint must report the
new model and thresholds hashes and answer valid detections on smoke images; otherwise the previous champion is
restored and redeployed automatically. Only one cycle runs at a time (lock file). Every cycle, passed or not, is
recorded in stash://…/finding_model/registry.json with its metrics and gate report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODAL_DIR = HERE
CHAMPION = MODAL_DIR / 'models' / 'best.pt'
MANIFEST = MODAL_DIR / 'models' / 'manifest.json'
THRESHOLDS = MODAL_DIR / 'powerline_ai' / 'thresholds.json'
HISTORY = MODAL_DIR / 'models' / 'history'
KERNEL_SRC = HERE / 'al_train_kernel.py'
WORK = HERE / '.al_work'   # git-ignored scratch
CLASSES = ['insulator', 'insulator_broken', 'insulator_flashover', 'bird_nest', 'foreign_object', 'broken_strand']
DEFECTS = CLASSES[1:]
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
NEMO_DATASET = 'tydapchai/finding-nemo-human-reviewed-6cls'
RETRY_BASE_S = 10

DEFAULTS = {
    'kaggle_user': None,               # default: the authenticated user
    'dataset_slug': 'uavpms-al-cycle',
    'kernel_slug': 'uavpms-al-train',
    'accelerators': ['NvidiaTeslaT4', 'NvidiaTeslaP100'],   # next one is tried when the queue is too long
    'queue_timeout_s': 1200,
    'min_feedback_images': 30,
    'replay_images': 2000,
    'seed': 0,
    'ultralytics': '8.4.172',
    # Gentle fine-tune: frozen backbone (first 10 layers), low LR, last epoch (no checkpoint selection on the gate set).
    'train': {'epochs': 5, 'imgsz': 640, 'batch': 16, 'lr0': 5e-5, 'lrf': 0.1, 'warmup_epochs': 0, 'close_mosaic': 2,
              'freeze': 10},
    # Gate: the challenger must not be worse than the champion on the fixed gate set.
    'gate': {'min_mean5_gain': 0.0, 'max_map50_95_drop': 0.005, 'max_defect_ap50_drop': 0.02},
    # Fixed floor: the original seed85 model on the gate set (Finding Nemo val, 2,271 images). Every promoted model must
    # also stay within the same tolerances of it, so small per-cycle drops cannot accumulate over many cycles.
    'baseline': {'name': 'seed85_curated_yolo26m',
                 'weights_sha256': '8682baa083ba5148d378762ddba5c5176cafaedd72b0c2eb369d3ea992924f16',
                 'MEAN_5': 0.8221, 'mAP50_95': 0.4803,
                 'ap50': {'insulator_broken': 0.8336, 'insulator_flashover': 0.796, 'bird_nest': 0.9358,
                          'foreign_object': 0.7781, 'broken_strand': 0.7668}},
    'auto_deploy': True,
    'modal_app': 'powerline-yolo26m-benchmark',
    'modal_env': 'main',
    'smoke_images': 3,
    'smoke_max_ms': 3000,
    'smoke_wait_s': 300,               # time for old containers to drain after a deploy
    'stash_root': 'stash://bunpmc/projects/finding_model',
    'poll_s': 60,
    'timeout_s': 6 * 3600,
}


class CycleError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str):
    print(f'[{datetime.now():%H:%M:%S}] {msg}', flush=True)


# ---------- external tools ----------

def kaggle(*args: str, check: bool = True) -> str:
    p = subprocess.run(['kaggle', *args], capture_output=True, text=True)
    out = '\n'.join(l for l in (p.stdout + p.stderr).splitlines() if 'outdated' not in l.lower())
    if check and p.returncode != 0:
        raise CycleError(f'kaggle {" ".join(args)} failed ({p.returncode}): {out[-800:]}')
    return out


def camber(*args: str, check: bool = True, attempts: int = 4) -> str:
    """Stash calls are idempotent (cp overwrites); retry transient network errors with backoff."""
    for i in range(attempts):
        p = subprocess.run(['camber', 'stash', *args], capture_output=True, text=True)
        out = '\n'.join(l for l in (p.stdout + p.stderr).splitlines()
                        if l.strip() and not any(w in l.lower() for w in ('out of date', 'camber update', 'latest version', 'stay up to date')))
        if p.returncode == 0 or not check:
            return out
        if i + 1 < attempts:
            log(f'camber stash {args[0]} failed (attempt {i + 1}/{attempts}), retrying')
            time.sleep(RETRY_BASE_S * 2 ** i)
    err = out.split('Usage:')[0].strip()
    raise CycleError(f'camber stash {" ".join(args)} failed: {err[-500:]}')


def stash_put(local: Path, remote: str):
    camber('cp', str(local), remote)


def stash_get(remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(['camber', 'stash', 'cp', remote, str(local)], capture_output=True, text=True)
    return p.returncode == 0 and local.exists()


def modal_bin() -> str:
    found = shutil.which('modal') or str(Path(sys.executable).parent / 'modal')
    if not Path(found).exists():
        raise CycleError('modal CLI not found; run with the Python environment that has the modal package')
    return found


def modal_deploy(cfg):
    log('modal deploy')
    p = subprocess.run([modal_bin(), 'deploy', '--env', cfg['modal_env'], 'app.py'], cwd=MODAL_DIR,
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise CycleError(f'modal deploy failed: {(p.stdout + p.stderr)[-1500:]}')


def smoke_test(cfg, expected_model: str, expected_thresholds: str | None, images: list[Path]) -> dict:
    """The deployed endpoint must serve the expected files and return valid detections quickly."""
    import modal
    detector = modal.Cls.from_name(cfg['modal_app'], 'Detector', environment_name=cfg['modal_env'])()
    deadline, info = time.time() + cfg['smoke_wait_s'], {}
    while time.time() < deadline:
        info = detector.model_info.remote()
        if info['model_sha256'] == expected_model and info['thresholds_sha256'] == expected_thresholds:
            break
        time.sleep(15)   # an old container may still be draining
    else:
        raise CycleError(f'endpoint serves {info.get("model_sha256", "?")[:12]}, expected {expected_model[:12]}')
    if not images:
        raise CycleError('no smoke images')
    times, predict_ms = [], []
    # the first inference on a fresh container includes CUDA warm-up (~3-4 s on T4): validated, not timed
    for i, img in enumerate([images[0], *images]):
        t0 = time.time()
        r = detector.predict_jpeg.remote(img.read_bytes())
        dets = r.get('detections')
        if not isinstance(dets, list) or r.get('model') != info['name']:
            raise CycleError(f'invalid response for {img.name}: {str(r)[:300]}')
        for d in dets:
            if d.get('class_name') not in CLASSES or not 0 <= float(d.get('confidence', -1)) <= 1:
                raise CycleError(f'invalid detection for {img.name}: {d}')
        if i == 0:
            continue
        times.append((time.time() - t0) * 1000)
        predict_ms.append(float(r.get('predict_ms', 0)))
    worst = max(predict_ms)
    if worst > cfg['smoke_max_ms']:
        raise CycleError(f'warm predict {worst:.0f} ms > {cfg["smoke_max_ms"]} ms')
    return {'model': info, 'images': len(images), 'predict_ms': [round(t) for t in predict_ms],
            'round_trip_ms': [round(t) for t in times]}


def kaggle_user(cfg) -> str:
    if cfg['kaggle_user']:
        return cfg['kaggle_user']
    for line in kaggle('config', 'view').splitlines():
        if 'username' in line:
            return line.split(':', 1)[1].strip()
    raise CycleError('cannot determine the Kaggle username (kaggle config view)')


# ---------- feedback, gate ----------

def validate_feedback(d: Path) -> list[Path]:
    """Every image has a label file with valid YOLO rows (class 0-5, coordinates in [0, 1], positive size)."""
    imgs = sorted(p for p in (d / 'images').iterdir() if p.suffix.lower() in IMAGE_EXTS) if (d / 'images').is_dir() else []
    if not imgs:
        raise CycleError(f'no images in {d / "images"}')
    errors = []
    for img in imgs:
        lbl = d / 'labels' / (img.stem + '.txt')
        if not lbl.exists():
            errors.append(f'{img.name}: missing label file (use an empty .txt for no objects)')
            continue
        for n, line in enumerate(lbl.read_text().splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            try:
                c, x, y, w, h = int(parts[0]), *map(float, parts[1:5])
                ok = len(parts) == 5 and 0 <= c < len(CLASSES) and all(0 <= v <= 1 for v in (x, y, w, h)) and w > 0 and h > 0
            except (ValueError, IndexError):
                ok = False
            if not ok:
                errors.append(f'{lbl.name}:{n}: invalid YOLO row {line!r}')
    extra = {p.stem for p in (d / 'labels').glob('*.txt')} - {p.stem for p in imgs} if (d / 'labels').is_dir() else set()
    if extra:
        errors.append(f'{len(extra)} label files without an image, e.g. {sorted(extra)[:3]}')
    if errors:
        raise CycleError('invalid feedback:\n  ' + '\n  '.join(errors[:30]))
    return imgs


def gate(metrics: dict, rules: dict, baseline: dict | None = None) -> dict:
    """Challenger vs the serving champion and vs the fixed baseline (seed85), same tolerances for both.
    Also imported by the Kaggle kernel (shipped in the bundle) to skip threshold tuning for a failing challenger."""
    ch, cc = metrics['champion'], metrics['challenger']
    checks = {
        'mean5': cc['MEAN_5'] - ch['MEAN_5'] >= rules['min_mean5_gain'],
        'map50_95': ch['mAP50_95'] - cc['mAP50_95'] <= rules['max_map50_95_drop'],
    }
    drops = {c: round(ch['per_class'][c]['ap50'] - cc['per_class'][c]['ap50'], 4) for c in DEFECTS}
    checks['defect_classes'] = all(v <= rules['max_defect_ap50_drop'] for v in drops.values())
    floor = None
    if baseline:
        floor = {'MEAN_5': round(cc['MEAN_5'] - baseline['MEAN_5'], 4),
                 'mAP50_95': round(cc['mAP50_95'] - baseline['mAP50_95'], 4),
                 'ap50': {c: round(cc['per_class'][c]['ap50'] - baseline['ap50'][c], 4) for c in DEFECTS}}
        checks['baseline_mean5'] = floor['MEAN_5'] >= rules['min_mean5_gain']
        checks['baseline_map50_95'] = -floor['mAP50_95'] <= rules['max_map50_95_drop']
        checks['baseline_defect_classes'] = all(-v <= rules['max_defect_ap50_drop'] for v in floor['ap50'].values())
    return {
        'passed': all(checks.values()),
        'checks': checks,
        'delta': {'MEAN_5': round(cc['MEAN_5'] - ch['MEAN_5'], 4), 'mAP50': round(cc['mAP50'] - ch['mAP50'], 4),
                  'mAP50_95': round(cc['mAP50_95'] - ch['mAP50_95'], 4)},
        'defect_ap50_drop': drops,
        'vs_baseline': floor,
        'rules': rules,
    }


def write_gate_report(result: Path, metrics: dict, g: dict) -> str:
    ch, cc = metrics['champion'], metrics['challenger']
    rows = [f'| {k} | {ch[k]} | {cc[k]} | {cc[k] - ch[k]:+.4f} |' for k in ('MEAN_5', 'mAP50', 'mAP50_95', 'precision', 'recall')]
    rows += [f'| AP50 {c} | {ch["per_class"][c]["ap50"]} | {cc["per_class"][c]["ap50"]} | '
             f'{cc["per_class"][c]["ap50"] - ch["per_class"][c]["ap50"]:+.4f} |' for c in CLASSES]
    md = (f'# Gate report: cycle {metrics["cycle_id"]}\n\n'
          f'**Result: {"PASSED" if g["passed"] else "FAILED"}** ({", ".join(k for k, v in g["checks"].items() if not v) or "all checks met"})\n\n'
          f'Gate set: {metrics["gate_set"]["name"]} ({metrics["gate_set"]["images"]} images). '
          f'Train: {metrics["train_set"]["feedback_images"]} feedback + {metrics["train_set"]["replay_images"]} replay images.\n\n'
          '| Metric | Champion | Challenger | Δ |\n|---|---:|---:|---:|\n' + '\n'.join(rows) + '\n\n'
          + (f'Challenger vs baseline (seed85): MEAN_5 {g["vs_baseline"]["MEAN_5"]:+.4f}, '
             f'mAP50_95 {g["vs_baseline"]["mAP50_95"]:+.4f}, defect AP50 {json.dumps(g["vs_baseline"]["ap50"])}\n\n'
             if g.get('vs_baseline') else '') +
          f'Rules: {json.dumps(g["rules"])}\n\nChampion sha256 `{ch["weights_sha256"]}`, challenger sha256 `{cc["weights_sha256"]}`.\n')
    (result / 'gate.json').write_text(json.dumps(g, indent=2))
    (result / 'gate_report.md').write_text(md)
    return md


# ---------- registry and model files ----------

def load_registry(cfg) -> dict:
    local = WORK / 'registry.json'
    if stash_get(f'{cfg["stash_root"]}/registry.json', local):
        return json.loads(local.read_text())
    return {'current': None, 'previous': None, 'cycles': []}


def save_registry(cfg, reg: dict):
    local = WORK / 'registry.json'
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(reg, indent=2))
    stash_put(local, f'{cfg["stash_root"]}/registry.json')


def current_entry() -> dict:
    m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    return {'name': m.get('name') or 'seed85_curated_yolo26m', 'model_sha256': sha256(CHAMPION),
            'thresholds_sha256': sha256(THRESHOLDS) if THRESHOLDS.exists() else None}


def archive_current(cfg) -> Path:
    """Keep the serving model, manifest and thresholds locally and on Camber before anything changes."""
    entry = current_entry()
    keep = HISTORY / entry['model_sha256'][:16]
    keep.mkdir(parents=True, exist_ok=True)
    for src, name in ((CHAMPION, 'best.pt'), (MANIFEST, 'manifest.json'), (THRESHOLDS, 'thresholds.json')):
        if src.exists():
            shutil.copy2(src, keep / name)
            stash_put(keep / name, f'{cfg["stash_root"]}/models/{keep.name}/{name}')
    (keep / 'entry.json').write_text(json.dumps(entry, indent=2))
    return keep


def install(model: Path, thresholds: Path | None, name: str, source: str):
    shutil.copy2(model, CHAMPION)
    MANIFEST.write_text(json.dumps({'name': name, 'source': source, 'sha256': sha256(CHAMPION), 'names': CLASSES}, indent=2))
    if thresholds is not None:
        shutil.copy2(thresholds, THRESHOLDS)


def restore(keep: Path):
    shutil.copy2(keep / 'best.pt', CHAMPION)
    for name, dst in (('manifest.json', MANIFEST), ('thresholds.json', THRESHOLDS)):
        if (keep / name).exists():
            shutil.copy2(keep / name, dst)


def smoke_images(cycle_id: str) -> list[Path]:
    return sorted((WORK / cycle_id / 'smoke').glob('*'))


def deploy_and_verify(cfg, cycle_id: str) -> dict:
    expected = current_entry()
    modal_deploy(cfg)
    return smoke_test(cfg, expected['model_sha256'], expected['thresholds_sha256'], smoke_images(cycle_id))


def promote(cfg, result: Path, metrics: dict, cycle_id: str) -> dict:
    """Swap in the challenger; on any deploy or smoke failure restore and redeploy the previous champion."""
    name = f'al_{cycle_id}_yolo26m'
    new_dir = f'{cfg["stash_root"]}/models/{metrics["challenger"]["weights_sha256"][:16]}'
    try:   # archive and uploads first: a failure here leaves the serving files untouched
        keep = archive_current(cfg)
        old = json.loads((keep / 'entry.json').read_text())
        stash_put(result / 'challenger.pt', f'{new_dir}/best.pt')
        stash_put(result / 'thresholds.json', f'{new_dir}/thresholds.json')
    except Exception as exc:
        log(f'promotion aborted before the swap ({exc}); serving model unchanged')
        return {'status': 'promote_failed', 'error': str(exc), 'model': current_entry()}
    try:
        install(result / 'challenger.pt', result / 'thresholds.json', name, f'{cfg["stash_root"]}/cycles/{cycle_id}')
        new = current_entry()
        stash_put(MANIFEST, f'{new_dir}/manifest.json')
        smoke = deploy_and_verify(cfg, cycle_id)
        log(f'deployed {name} ({new["model_sha256"][:12]}); smoke ok, warm predict {smoke["predict_ms"]} ms')
        return {'status': 'deployed', 'model': new, 'previous': old, 'smoke': smoke}
    except Exception as exc:   # anything after the swap: put the previous champion back
        log(f'deploy/smoke failed ({exc}); rolling back to {old["model_sha256"][:12]}')
        restore(keep)
        try:
            smoke = deploy_and_verify(cfg, cycle_id)
        except Exception as exc2:
            return {'status': 'rollback_failed', 'error': str(exc), 'rollback_error': str(exc2), 'model': old}
        return {'status': 'rolled_back', 'error': str(exc), 'model': old, 'smoke': smoke}


# ---------- Kaggle ----------

def build_bundle(feedback: Path, imgs: list[Path], cfg: dict, user: str, cycle_id: str) -> Path:
    bundle = WORK / cycle_id / 'bundle'
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / 'feedback' / 'images').mkdir(parents=True)
    (bundle / 'feedback' / 'labels').mkdir(parents=True)
    for img in imgs:
        shutil.copy2(img, bundle / 'feedback' / 'images' / img.name)
        shutil.copy2(feedback / 'labels' / (img.stem + '.txt'), bundle / 'feedback' / 'labels' / (img.stem + '.txt'))
    shutil.copy2(CHAMPION, bundle / 'champion.pt')
    code = bundle / 'code'   # serving post-processing + tuner, so thresholds are tuned with the deployed logic
    shutil.copytree(MODAL_DIR / 'powerline_ai', code / 'powerline_ai',
                    ignore=shutil.ignore_patterns('__pycache__', 'thresholds.json'))
    shutil.copy2(MODAL_DIR / 'tune_thresholds.py', code / 'tune_thresholds.py')
    shutil.copy2(Path(__file__).resolve(), code / 'al_cycle.py')   # gate() is shared with the kernel
    (bundle / 'cycle.json').write_text(json.dumps({
        'cycle_id': cycle_id, 'champion_sha256': sha256(CHAMPION), 'seed': cfg['seed'],
        'replay_images': cfg['replay_images'], 'ultralytics': cfg['ultralytics'], 'train': cfg['train'],
        'gate': cfg['gate'], 'baseline': cfg.get('baseline'),
        'feedback_images': len(imgs)}, indent=2))
    (bundle / 'dataset-metadata.json').write_text(json.dumps({
        'title': 'UAV-PMS active learning cycle', 'id': f'{user}/{cfg["dataset_slug"]}', 'licenses': [{'name': 'other'}]}))
    smoke = WORK / cycle_id / 'smoke'
    smoke.mkdir(parents=True, exist_ok=True)
    for img in imgs[:cfg['smoke_images']]:
        shutil.copy2(img, smoke / img.name)
    return bundle


def push_dataset(bundle: Path, cfg: dict, user: str, cycle_id: str) -> str:
    ref = f'{user}/{cfg["dataset_slug"]}'
    exists = kaggle('datasets', 'status', ref, check=False).lower()
    if 'ready' in exists or 'pending' in exists:
        kaggle('datasets', 'version', '-p', str(bundle), '-m', f'cycle {cycle_id}', '-r', 'zip')
    else:
        kaggle('datasets', 'create', '-p', str(bundle), '-r', 'zip')   # private by default
    deadline = time.time() + 1800
    while time.time() < deadline:   # the kernel must see the new version
        if 'ready' in kaggle('datasets', 'status', ref, check=False).lower():
            return ref
        time.sleep(15)
    raise CycleError(f'dataset {ref} not ready after 30 min')


def push_kernel(cfg: dict, user: str, dataset_ref: str, cycle_id: str, accel_index: int) -> str:
    kdir = WORK / cycle_id / 'kernel'
    kdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(KERNEL_SRC, kdir / KERNEL_SRC.name)
    ref = f'{user}/{cfg["kernel_slug"]}'
    acc = cfg['accelerators'][accel_index]
    (kdir / 'kernel-metadata.json').write_text(json.dumps({
        'id': ref, 'title': cfg['kernel_slug'], 'code_file': KERNEL_SRC.name, 'language': 'python',
        'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_internet': True,
        'machine_shape': acc, 'dataset_sources': [dataset_ref, NEMO_DATASET],
        'competition_sources': [], 'kernel_sources': []}, indent=2))
    kaggle('kernels', 'push', '-p', str(kdir), '--accelerator', acc)
    (WORK / cycle_id / 'state.json').write_text(json.dumps({'accel_index': accel_index, 'pushed_at': time.time(),
                                                            'dataset': dataset_ref}))
    log(f'kernel {ref} pushed on {acc}')
    return ref


def wait_kernel(ref: str, cfg: dict, pushed_at: float, can_switch: bool = True) -> str:
    deadline = time.time() + cfg['timeout_s']
    last = ''
    while time.time() < deadline:
        out = kaggle('kernels', 'status', ref, check=False).lower()
        state = next((s for s in ('complete', 'error', 'cancel', 'running', 'queued') if s in out), out[-80:])
        if state != last:
            log(f'kernel {ref}: {state}')
            last = state
        if state in ('complete', 'error', 'cancel'):
            return state
        if can_switch and state == 'queued' and time.time() - pushed_at > cfg['queue_timeout_s']:
            return 'queued_timeout'
        time.sleep(cfg['poll_s'])
    raise CycleError(f'kernel {ref} did not finish within {cfg["timeout_s"]} s')


# ---------- commands ----------

class Lock:
    def __init__(self):
        self.path = WORK / 'cycle.lock'

    def __enter__(self):
        WORK.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and time.time() - self.path.stat().st_mtime < 12 * 3600:
            raise CycleError(f'another cycle is running ({self.path.read_text().strip()}); remove {self.path} if it is stale')
        self.path.write_text(f'pid {os.getpid()} since {now()}')
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def cmd_cycle(a, cfg):
    imgs = validate_feedback(a.feedback)
    if len(imgs) < cfg['min_feedback_images'] and not a.force:
        raise CycleError(f'only {len(imgs)} feedback images (< {cfg["min_feedback_images"]}); nothing to do')
    if not CHAMPION.exists():
        raise CycleError(f'champion not found: {CHAMPION}')
    with Lock():
        user = kaggle_user(cfg)
        cycle_id = a.cycle_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        log(f'cycle {cycle_id}: {len(imgs)} feedback images, champion {sha256(CHAMPION)[:12]}, Kaggle user {user}')
        bundle = build_bundle(a.feedback, imgs, cfg, user, cycle_id)
        dataset_ref = push_dataset(bundle, cfg, user, cycle_id)
        push_kernel(cfg, user, dataset_ref, cycle_id, 0)
        shutil.rmtree(bundle)   # the bundle is on Kaggle now
        a.cycle_id = cycle_id
        return _fetch(a, cfg)


def cmd_fetch(a, cfg):
    with Lock():
        return _fetch(a, cfg)


def _fetch(a, cfg):
    """Wait for the kernel (switching accelerator if queued too long), download, gate, then deploy or record."""
    user = kaggle_user(cfg)
    ref = f'{user}/{cfg["kernel_slug"]}'
    cycle_id = a.cycle_id
    state_file = WORK / cycle_id / 'state.json'
    st = json.loads(state_file.read_text()) if state_file.exists() else {'accel_index': 0, 'pushed_at': time.time()}
    while True:
        can_switch = st['accel_index'] + 1 < len(cfg['accelerators'])   # on the last one, just keep waiting
        state = wait_kernel(ref, cfg, st['pushed_at'], can_switch)
        if state != 'queued_timeout':
            break
        log(f'queued > {cfg["queue_timeout_s"]} s on {cfg["accelerators"][st["accel_index"]]}; trying the next accelerator')
        push_kernel(cfg, user, st.get('dataset', f'{user}/{cfg["dataset_slug"]}'), cycle_id, st['accel_index'] + 1)
        st = json.loads(state_file.read_text())
    result = WORK / cycle_id / 'result'
    result.mkdir(parents=True, exist_ok=True)
    kaggle('kernels', 'output', ref, '-p', str(result), '-o', check=False)
    wanted = ('metrics.json', 'challenger.pt', 'thresholds.json', 'kernel.log', 'train_results.csv')
    found = {p.name: p for p in result.rglob('*') if p.is_file() and p.name in wanted}
    for name, p in found.items():
        if p.parent != result:
            shutil.move(str(p), result / name)
    for p in result.iterdir():   # drop anything else the kernel output contained
        if p.name not in wanted:
            shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()
    if state != 'complete' or not (result / 'metrics.json').exists():
        tail = (result / 'kernel.log').read_text()[-1500:] if (result / 'kernel.log').exists() else ''
        if not tail:
            tail = kaggle('kernels', 'logs', ref, check=False)[-3000:] or '(no log)'
        safe_record(cfg, cycle_id, None, None, {'status': 'kernel_failed', 'error': f'state {state}'}, result)
        raise CycleError(f'kernel finished with state {state}; log:\n{tail}')
    a.result = result
    return _gate_and_deploy(a, cfg, cycle_id)


def _gate_and_deploy(a, cfg, cycle_id):
    result = a.result
    metrics = json.loads((result / 'metrics.json').read_text())
    if sha256(result / 'challenger.pt') != metrics['challenger']['weights_sha256']:
        raise CycleError('challenger.pt does not match metrics.json')
    if metrics['champion']['weights_sha256'] != sha256(CHAMPION):
        raise CycleError('the serving champion changed since this cycle started; not deploying a stale comparison')
    if sha256(result / 'thresholds.json') != metrics.get('challenger_thresholds_sha256'):
        raise CycleError('thresholds.json does not match metrics.json')
    g = gate(metrics, cfg['gate'], cfg.get('baseline'))
    print(write_gate_report(result, metrics, g))
    if g['passed'] and not metrics.get('thresholds_tuned', True):
        raise CycleError('challenger passed the gate here but its thresholds were not tuned in the kernel')
    if not g['passed']:
        outcome = {'status': 'gate_failed', 'model': current_entry()}
    elif not cfg['auto_deploy']:
        outcome = {'status': 'gate_passed_not_deployed', 'model': current_entry()}
    else:
        outcome = promote(cfg, result, metrics, cycle_id)
    safe_record(cfg, cycle_id, metrics, g, outcome, result)
    log(f'cycle {cycle_id}: {outcome["status"]}')
    return {'deployed': 0, 'gate_failed': 3, 'gate_passed_not_deployed': 0, 'rolled_back': 4,
            'promote_failed': 2}.get(outcome['status'], 5)


def safe_record(cfg, cycle_id, metrics, g, outcome, result: Path):
    """Recording is best effort: production state is already decided; keep a local copy and warn loudly."""
    (result / 'outcome.json').parent.mkdir(parents=True, exist_ok=True)
    (result / 'outcome.json').write_text(json.dumps(outcome, indent=2, default=str))
    try:
        record(cfg, cycle_id, metrics, g, outcome, result)
    except Exception as exc:
        log(f'WARNING: could not record cycle {cycle_id} on Camber ({exc}); local copy in {result}')


def record(cfg, cycle_id, metrics, g, outcome, result: Path):
    """Upload the cycle artifacts and append it to the registry on Camber (and locally)."""
    remote = f'{cfg["stash_root"]}/cycles/{cycle_id}'
    for name in ('metrics.json', 'gate.json', 'gate_report.md', 'kernel.log', 'train_results.csv', 'thresholds.json'):
        if (result / name).exists():
            stash_put(result / name, f'{remote}/{name}')
    reg = load_registry(cfg)
    entry = {'cycle_id': cycle_id, 'finished_at': now(), 'status': outcome['status'],
             'gate_passed': g['passed'] if g else None, 'delta': g['delta'] if g else None,
             'champion_sha256': metrics['champion']['weights_sha256'] if metrics else None,
             'challenger_sha256': metrics['challenger']['weights_sha256'] if metrics else None,
             'train_set': metrics['train_set'] if metrics else None,
             'error': outcome.get('error'), 'artifacts': remote}
    reg['cycles'] = [c for c in reg['cycles'] if c['cycle_id'] != cycle_id] + [entry]
    if outcome['status'] == 'deployed':
        reg['previous'], reg['current'] = outcome['previous'], {**outcome['model'], 'deployed_at': now(), 'cycle_id': cycle_id}
    elif reg['current'] is None:
        reg['current'] = {**current_entry(), 'deployed_at': None, 'cycle_id': None}
    save_registry(cfg, reg)


def cmd_init(a, cfg):
    """Create finding_model/ on Camber with the serving champion and an initial registry (idempotent)."""
    reg = load_registry(cfg)
    if reg.get('current'):
        log(f'registry already exists; current = {reg["current"]["name"]} {reg["current"]["model_sha256"][:12]}')
        return 0
    keep = archive_current(cfg)
    entry = json.loads((keep / 'entry.json').read_text())
    readme = WORK / 'README.md'
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text('# finding_model\n\nActive-learning model registry for the UAV-PMS AI service (written by '
                      'uavpms-server/modal/al_cycle.py).\n\n- `registry.json`: current and previous serving model, every cycle\n'
                      '- `models/<sha16>/`: best.pt, thresholds.json, manifest.json of every model that served or was promoted\n'
                      '- `cycles/<cycle_id>/`: metrics.json, gate.json, gate_report.md, kernel.log, train_results.csv, thresholds.json\n')
    stash_put(readme, f'{cfg["stash_root"]}/README.md')
    save_registry(cfg, {'current': {**entry, 'deployed_at': None, 'cycle_id': None,
                                    'note': 'champion at registry creation'}, 'previous': None, 'cycles': []})
    log(f'initialised {cfg["stash_root"]} with {entry["name"]} {entry["model_sha256"][:12]}')
    return 0


def cmd_gate(a, cfg):
    metrics = json.loads((a.result / 'metrics.json').read_text())
    g = gate(metrics, cfg['gate'], cfg.get('baseline'))
    print(write_gate_report(a.result, metrics, g))
    return 0 if g['passed'] else 3


def cmd_rollback(a, cfg):
    with Lock():
        reg = load_registry(cfg)
        prev = reg.get('previous')
        if not prev:
            raise CycleError('no previous champion in the registry')
        keep = HISTORY / prev['model_sha256'][:16]
        if not (keep / 'best.pt').exists():
            for name in ('best.pt', 'manifest.json', 'thresholds.json'):
                stash_get(f'{cfg["stash_root"]}/models/{keep.name}/{name}', keep / name)
        if not (keep / 'best.pt').exists() or sha256(keep / 'best.pt') != prev['model_sha256']:
            raise CycleError(f'previous champion {keep.name} not available or corrupted')
        current = archive_current(cfg)
        restore(keep)
        rid = 'rollback_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        smoke = WORK / rid / 'smoke'
        smoke.mkdir(parents=True, exist_ok=True)
        for img in (a.smoke_images or [])[:cfg['smoke_images']]:
            shutil.copy2(img, smoke / img.name)
        try:
            deploy_and_verify(cfg, rid)
        except Exception as exc:
            restore(current)
            deploy_and_verify(cfg, rid)
            raise CycleError(f'rollback deploy failed ({exc}); kept the current model')
        reg['current'], reg['previous'] = {**current_entry(), 'deployed_at': now(), 'cycle_id': rid}, \
            json.loads((current / 'entry.json').read_text())
        reg['cycles'].append({'cycle_id': rid, 'finished_at': now(), 'status': 'manual_rollback',
                              'model_sha256': prev['model_sha256']})
        save_registry(cfg, reg)
        log(f'rolled back to {prev["name"]} ({prev["model_sha256"][:12]})')
        return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, help='JSON file overriding DEFAULTS')
    p.add_argument('--env-file', type=Path, help='KEY=VALUE file with CAMBER_API_KEY etc. (values are not printed)')
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('cycle')
    c.add_argument('--feedback', type=Path, required=True)
    c.add_argument('--cycle-id')
    c.add_argument('--force', action='store_true', help='run below min_feedback_images')
    fe = sub.add_parser('fetch', help='resume: wait for the pushed kernel, download, gate, deploy')
    fe.add_argument('--cycle-id', required=True)
    g = sub.add_parser('gate')
    g.add_argument('--result', type=Path, required=True)
    sub.add_parser('init', help='create finding_model/ on Camber with the current champion')
    rb = sub.add_parser('rollback')
    rb.add_argument('--smoke-images', type=Path, nargs='*', help='images for the post-rollback smoke test')
    a = p.parse_args(argv)
    if a.env_file:
        for line in a.env_file.read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"\''))
    cfg = json.loads(json.dumps(DEFAULTS))
    if a.config:
        for k, v in json.loads(a.config.read_text()).items():
            cfg[k] = {**cfg[k], **v} if isinstance(v, dict) and isinstance(cfg.get(k), dict) else v
    try:
        return {'cycle': cmd_cycle, 'fetch': cmd_fetch, 'gate': cmd_gate, 'rollback': cmd_rollback, 'init': cmd_init}[a.cmd](a, cfg)
    except CycleError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
