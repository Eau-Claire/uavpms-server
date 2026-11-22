"""MF04 AI test suite against the deployed Modal detector (Detector.predict_jpeg via the Modal SDK).

Every test has an ID, a criterion fixed before running, measured values and a verdict.
Writes results/<run>/report.md, results.json and calls.csv (one row per request).

Scope: the AI service (decode -> preprocess -> inference -> NMS -> filter) and the client
round trip. Database/queue stages of the MF04 backend (T5, T6), end-to-end job latency and
worker restart (AI-PERF-08) need the backend and are not measured here.

  python perf/mf04_ai_tests.py --dataset <finding_nemo_human_reviewed_6cls> [--soak-minutes 30]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from powerline_ai.postprocess import CLASSES, overlap  # noqa: E402

APP, ENV = 'powerline-yolo26m-benchmark', 'main'
SERVER_AI_LIMIT_MS = 3000   # Report 5 scope: AI processing < 3 s (T1-T4 inside the service)
TOTAL_LIMIT_MS = 5000       # Report 5 §2.1.4: mean <= 5 s per image (client round trip)
IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}
CALLS, LOCK = [], threading.Lock()


# ------------------------------------------------------------------ helpers
def pct(values, p):
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def stats(values):
    values = [x for x in values if x is not None]
    if not values:
        return {}
    return dict(n=len(values), mean=statistics.fmean(values), p50=pct(values, 50), p95=pct(values, 95),
                min=min(values), max=max(values))


def jpeg(image: Image.Image, quality=92) -> bytes:
    buf = io.BytesIO()
    image.convert('RGB').save(buf, 'JPEG', quality=quality)
    return buf.getvalue()


def server_ai_ms(r):
    t = r.get('timing') or {}
    return (t.get('decode_ms') or 0) + r['predict_ms']


class Endpoint:
    def __init__(self):
        import modal
        self.cls = modal.Cls.from_name(APP, 'Detector', environment_name=ENV)()

    def call(self, test_id, data, label=''):
        t0 = time.perf_counter()
        row = dict(test=test_id, label=label, bytes=len(data), started=time.time())
        try:
            r = self.cls.predict_jpeg.remote(data)
            row.update(ok=True, roundtrip_ms=(time.perf_counter() - t0) * 1000, server_ai_ms=server_ai_ms(r),
                       detections=len(r['detections']), **{k: v for k, v in (r.get('timing') or {}).items()})
            out = (r, None)
        except Exception as exc:  # recorded; robustness tests expect some of these
            row.update(ok=False, roundtrip_ms=(time.perf_counter() - t0) * 1000, error=f'{type(exc).__name__}: {exc}'[:300])
            out = (None, row['error'])
        with LOCK:
            CALLS.append(row)
        return out

    def resources(self):
        return self.cls.resource_stats.remote()


def read_gt(label_path: Path, w, h):
    gt = []
    if label_path.is_file():
        for line in label_path.read_text(encoding='utf-8-sig').splitlines():
            p = line.split()
            if len(p) == 5:
                c, x, y, bw, bh = int(float(p[0])), *map(float, p[1:])
                gt.append((c, [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
    return gt


def match(gt, dets, thr=0.5):
    """Class-aware greedy matching by confidence. Returns (tp_flags_per_gt, fp_per_class)."""
    used, fp = set(), [0] * len(CLASSES)
    for d in sorted(dets, key=lambda x: -x['confidence']):
        best, bi = 0.0, None
        for gi, (c, box) in enumerate(gt):
            if c == d['class_id'] and gi not in used:
                v = overlap(box, d['bbox_xyxy'])[0]
                if v > best:
                    best, bi = v, gi
        if bi is not None and best >= thr:
            used.add(bi)
        else:
            fp[d['class_id']] += 1
    return [gi in used for gi in range(len(gt))], fp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0, tp=tp, fp=fp, fn=fn)


def accuracy(results):
    """results: list of (gt, dets). Per-class and micro P/R/F1 at IoU 0.5."""
    n = len(CLASSES)
    tp, fp, fn = [0] * n, [0] * n, [0] * n
    for gt, dets in results:
        hit, f = match(gt, dets)
        for (c, _), h in zip(gt, hit):
            tp[c] += h
            fn[c] += not h
        for c in range(n):
            fp[c] += f[c]
    per = {CLASSES[i]: prf(tp[i], fp[i], fn[i]) for i in range(n)}
    micro = prf(sum(tp), sum(fp), sum(fn))
    micro['macro_f1'] = statistics.fmean(v['f1'] for v in per.values())
    return dict(per_class=per, micro=micro)


VALID_FIELDS = {'class_id', 'class_name', 'confidence', 'bbox_xyxy', 'merged'}


def output_problems(r):
    probs = []
    w, h = r['width'], r['height']
    for d in r['detections']:
        if set(d) != VALID_FIELDS:
            probs.append(f'fields {sorted(d)}')
        if not (0 <= d['class_id'] < len(CLASSES) and CLASSES[d['class_id']] == d['class_name']):
            probs.append(f'class {d["class_id"]}/{d["class_name"]}')
        if not 0 <= d['confidence'] <= 1:
            probs.append(f'confidence {d["confidence"]}')
        x1, y1, x2, y2 = d['bbox_xyxy']
        if not (0 <= x1 < x2 <= w + 0.5 and 0 <= y1 < y2 <= h + 0.5):
            probs.append(f'bbox {d["bbox_xyxy"]} in {w}x{h}')
    return probs


# ------------------------------------------------------------------ tests
def run(a):
    ep = Endpoint()
    rng = random.Random(85)
    images_dir = lambda s: a.dataset / 'images' / s
    labels_dir = lambda s: a.dataset / 'labels' / s
    val = sorted(p for p in images_dir('val').iterdir() if p.suffix.lower() in IMAGE_EXTS)
    tests, out_checked, out_bad = [], 0, []

    def add(test_id, title, criterion, measured, passed, notes=''):
        verdict = 'Passed' if passed is True else 'Failed' if passed is False else 'Measured'
        tests.append(dict(id=test_id, title=title, criterion=criterion, measured=measured, verdict=verdict, notes=notes))
        print(f'[{verdict:8}] {test_id} {title}: {json.dumps(measured, default=str)[:300]}', flush=True)

    def check_output(r):
        nonlocal out_checked
        out_checked += 1
        out_bad.extend(output_problems(r))

    # Cold start: idle past scaledown_window (60 s) so the first call boots a container.
    cold = {}
    if a.cold_wait:
        print(f'Waiting {a.cold_wait}s for the container to scale to zero...', flush=True)
        time.sleep(a.cold_wait)
        r, err = ep.call('COLD', jpeg(Image.open(val[0])), 'cold start')
        res0 = ep.resources()
        # Valid only if this container booted for this request: it has served exactly that one call.
        genuine = res0['served'] == 1 and res0['uptime_s'] < a.cold_wait
        cold = dict(roundtrip_ms=CALLS[-1]['roundtrip_ms'], predict_ms=r and r['predict_ms'], error=err,
                    model_load_s=res0['model_load_s'], container_uptime_s=res0['uptime_s'],
                    container_served=res0['served'], genuine_cold_start=genuine)
        add('AI-PERF-00', 'Cold start (container khởi động + nạp model + lần predict đầu)',
            'Chỉ đo, báo riêng; hợp lệ khi container phục vụ đúng 1 request (request này)', cold, None,
            'Modal scale-to-zero sau khi rảnh' + ('' if genuine else
            '; KHÔNG HỢP LỆ: container chưa tắt nên đây không phải cold start, cần chờ lâu hơn'))
    else:
        ep.call('WARMUP', jpeg(Image.open(val[0])), 'warm-up')

    res_start = ep.resources()

    # AI-PERF-01..03: one image at three resolutions, sequential, warm container.
    base = Image.open(val[1]).convert('RGB')
    for test_id, size, n in (('AI-PERF-01', (640, 640), 30), ('AI-PERF-02', (1920, 1080), 20),
                             ('AI-PERF-03', (3840, 2160), 20)):
        data = jpeg(base.resize(size, Image.BILINEAR))
        rows = []
        for _ in range(n):
            r, err = ep.call(test_id, data, f'{size[0]}x{size[1]}')
            if r:
                check_output(r)
            rows.append(CALLS[-1])
        ok = [x for x in rows if x['ok']]
        m = dict(size=f'{size[0]}x{size[1]}', jpeg_kib=round(len(data) / 1024), errors=n - len(ok),
                 server_ai_ms=stats([x['server_ai_ms'] for x in ok]),
                 roundtrip_ms=stats([x['roundtrip_ms'] for x in ok]),
                 stages_mean_ms={k: statistics.fmean(x[k] for x in ok)
                                 for k in ('decode_ms', 'preprocess_ms', 'inference_ms', 'nms_ms', 'filter_ms')})
        passed = (len(ok) == n and m['server_ai_ms']['p95'] < SERVER_AI_LIMIT_MS
                  and m['roundtrip_ms']['mean'] <= TOTAL_LIMIT_MS)
        add(test_id, f'1 ảnh {size[0]}×{size[1]}, {n} lần liên tiếp',
            f'0 lỗi; xử lý AI p95 < {SERVER_AI_LIMIT_MS} ms; round-trip trung bình ≤ {TOTAL_LIMIT_MS} ms', m, passed)

    # AI-PERF-04: 10 distinct images sequential.
    t0 = time.perf_counter()
    for p in val[2:12]:
        r, _ = ep.call('AI-PERF-04', p.read_bytes(), p.name)
        if r:
            check_output(r)
    dur = time.perf_counter() - t0
    rows = [x for x in CALLS if x['test'] == 'AI-PERF-04']
    m = dict(images=10, total_s=dur, images_per_s=10 / dur, errors=sum(not x['ok'] for x in rows),
             roundtrip_ms=stats([x['roundtrip_ms'] for x in rows if x['ok']]))
    add('AI-PERF-04', '10 ảnh khác nhau gửi liên tiếp', f'0 lỗi; round-trip trung bình ≤ {TOTAL_LIMIT_MS} ms', m,
        m['errors'] == 0 and m['roundtrip_ms']['mean'] <= TOTAL_LIMIT_MS)

    # AI-PERF-05: 50 concurrent requests (one container, max_containers=1).
    batch = [p.read_bytes() for p in val[12:62]]
    seq_overhead = statistics.median(x['roundtrip_ms'] - x['server_ai_ms'] for x in CALLS
                                     if x['test'] == 'AI-PERF-04' and x['ok'])
    t0 = time.perf_counter()
    with ThreadPoolExecutor(50) as pool:
        results = list(pool.map(lambda d: ep.call('AI-PERF-05', d, 'concurrent'), batch))
    dur = time.perf_counter() - t0
    for r, _ in results:
        if r:
            check_output(r)
    rows = [x for x in CALLS if x['test'] == 'AI-PERF-05']
    ok = [x for x in rows if x['ok']]
    waits = [x['roundtrip_ms'] - x['server_ai_ms'] - seq_overhead for x in ok]
    m = dict(requests=50, errors=len(rows) - len(ok), total_s=dur, throughput_img_s=len(ok) / dur,
             roundtrip_ms=stats([x['roundtrip_ms'] for x in ok]), est_queue_wait_ms=stats(waits),
             server_ai_ms=stats([x['server_ai_ms'] for x in ok]))
    add('AI-PERF-05', '50 ảnh gửi đồng thời', '0 lỗi; round-trip p95 ≤ ' + f'{TOTAL_LIMIT_MS} ms', m,
        m['errors'] == 0 and m['roundtrip_ms']['p95'] <= TOTAL_LIMIT_MS,
        'Một container GPU xử lý tuần tự; thời gian chờ hàng đợi ước lượng = round-trip − xử lý AI − overhead mạng đo ở AI-PERF-04')

    # AI-PERF-06: 100 distinct images sequential, resources before/after.
    r_before = ep.resources()
    t0 = time.perf_counter()
    for p in val[62:162]:
        r, _ = ep.call('AI-PERF-06', p.read_bytes(), p.name)
        if r:
            check_output(r)
    dur = time.perf_counter() - t0
    r_after = ep.resources()
    rows = [x for x in CALLS if x['test'] == 'AI-PERF-06']
    ok = [x for x in rows if x['ok']]
    first, last = [x['server_ai_ms'] for x in ok[:10]], [x['server_ai_ms'] for x in ok[-10:]]
    m = dict(images=100, errors=len(rows) - len(ok), total_s=dur, images_per_s=len(ok) / dur,
             server_ai_ms=stats([x['server_ai_ms'] for x in ok]),
             first10_mean_ms=statistics.fmean(first), last10_mean_ms=statistics.fmean(last),
             rss_mib=(r_before['rss_mib'], r_after['rss_mib']),
             gpu_mem_used_mib=(r_before['gpu_mem_used_mib'], r_after['gpu_mem_used_mib']),
             torch_max_allocated_mib=r_after['torch_max_allocated_mib'])
    add('AI-PERF-06', '100 ảnh liên tục', '0 lỗi; RSS tăng ≤ 10%; latency 10 ảnh cuối ≤ 1,2× 10 ảnh đầu', m,
        m['errors'] == 0 and r_after['rss_mib'] <= 1.10 * r_before['rss_mib']
        and m['last10_mean_ms'] <= 1.2 * m['first10_mean_ms'] + 5)

    # Full val and test through the deployed endpoint: accuracy, small objects, background, output validity.
    split_results = {}
    for split in a.splits:
        imgs = sorted(p for p in images_dir(split).iterdir() if p.suffix.lower() in IMAGE_EXTS)[:a.limit or None]

        def one(p, split=split):
            r, err = ep.call(f'ACC-{split}', p.read_bytes(), p.name)
            return p, r, err
        t0 = time.perf_counter()
        with ThreadPoolExecutor(8) as pool:
            done = list(pool.map(one, imgs))
        dur = time.perf_counter() - t0
        pairs, errors, background = [], 0, []
        for p, r, err in done:
            if not r:
                errors += 1
                continue
            check_output(r)
            gt = read_gt(labels_dir(split) / f'{p.stem}.txt', r['width'], r['height'])
            pairs.append((gt, r['detections'], r['width'], r['height']))
            if not gt:
                background.append(len(r['detections']))
        acc = accuracy([(g, d) for g, d, _, _ in pairs])
        split_results[split] = dict(images=len(imgs), errors=errors, duration_s=dur, acc=acc, pairs=pairs,
                                    background=background)
        add(f'AI-ACC-{split.upper()}', f'Độ chính xác endpoint đã deploy trên {split} ({len(imgs)} ảnh)',
            'Chỉ đo (P/R/F1 @IoU 0.5, sau hậu xử lý); so với số offline trong thresholds.json',
            dict(errors=errors, micro=acc['micro'], per_class_f1={k: round(v['f1'], 4) for k, v in acc['per_class'].items()},
                 images_per_s=len(imgs) / dur), None)

    # AI-ROB-01/02: photometric degradation on labelled val images (labels stay valid).
    sample = rng.sample([p for p in val if read_gt(labels_dir('val') / f'{p.stem}.txt', 1, 1)], a.rob_images)
    variants = {
        'original': lambda im: im,
        'dark': lambda im: ImageEnhance.Brightness(im).enhance(0.35),
        'motion_blur': lambda im: im.filter(ImageFilter.Kernel((5, 5), [0] * 10 + [1] * 5 + [0] * 10, 5)),
    }
    rob = {}
    for name, fn in variants.items():
        pairs = []
        for p in sample:
            im = Image.open(p).convert('RGB')
            r, err = ep.call(f'ROB-{name}', jpeg(fn(im)), p.name)
            if r:
                check_output(r)
                pairs.append((read_gt(labels_dir('val') / f'{p.stem}.txt', r['width'], r['height']), r['detections']))
        rob[name] = accuracy(pairs)['micro']
    for test_id, name, title in (('AI-ROB-01', 'dark', 'Ảnh thiếu sáng (độ sáng × 0,35)'),
                                 ('AI-ROB-02', 'motion_blur', 'Ảnh mờ chuyển động (kernel ngang 5 px)')):
        m = dict(images=len(sample), original=rob['original'], degraded=rob[name],
                 delta_recall=rob[name]['recall'] - rob['original']['recall'],
                 delta_precision=rob[name]['precision'] - rob['original']['precision'])
        add(test_id, title, 'Chỉ đo mức suy giảm so với ảnh gốc (có ground truth)', m, None)
    add('AI-ROB-03', 'Background phức tạp', 'Không thực hiện', {}, None,
        'Không có tập ảnh nền phức tạp được gán nhãn riêng; không tự tính mAP khi không có ground truth')

    # AI-ROB-04: recall by GT size (short side after letterbox to 640) for the small-defect classes.
    if 'val' in split_results:
        buckets = {}
        for gt, dets, w, h in split_results['val']['pairs']:
            hit, _ = match(gt, dets)
            for (c, box), ok_ in zip(gt, hit):
                if CLASSES[c] not in ('broken_strand', 'insulator_broken', 'insulator_flashover', 'foreign_object'):
                    continue
                side = min(box[2] - box[0], box[3] - box[1]) * 640 / max(w, h)
                b = '<16px' if side < 16 else '16-32px' if side < 32 else '32-64px' if side < 64 else '>=64px'
                k = (CLASSES[c], b)
                buckets.setdefault(k, [0, 0])
                buckets[k][0] += ok_
                buckets[k][1] += 1
        m = {f'{c} {b}': dict(recall=round(t / n, 3), n=n) for (c, b), (t, n) in sorted(buckets.items())}
        add('AI-ROB-04', 'Vật thể nhỏ: recall theo kích thước box (cạnh ngắn ở 640 px), val',
            'Chỉ đo', m, None)
        bg = split_results['val']['background']
        m = dict(background_images=len(bg), images_with_detection=sum(x > 0 for x in bg),
                 false_boxes=sum(bg), rate=sum(x > 0 for x in bg) / max(len(bg), 1))
        add('AI-ROB-05', 'Ảnh không có đối tượng thuộc 6 class (ảnh nền trong val)', 'Chỉ đo tỉ lệ ảnh có box sai', m, None)

    # AI-ROB-06: invalid input must be rejected with a clear error and the worker must keep serving.
    from powerline_ai.settings import MAX_BYTES, MAX_PIXELS
    side = int(MAX_PIXELS ** 0.5) + 16
    over_pixels = jpeg(Image.new('RGB', (side, side), (90, 90, 90)), 60)
    valid = val[0].read_bytes()
    gif = io.BytesIO(); Image.new('RGB', (64, 64)).save(gif, 'GIF')
    over_bytes = b'\xff\xd8\xff' + b'\0' * (MAX_BYTES + 1)
    mpo = io.BytesIO(); Image.open(val[0]).convert('RGB').save(mpo, 'MPO')
    png = io.BytesIO(); Image.open(val[0]).save(png, 'PNG')
    cases = [('truncated JPEG', valid[:len(valid) // 3], False), ('text file', b'not an image\n' * 100, False),
             ('empty body', b'', False), ('GIF', gif.getvalue(), False),
             (f'JPEG > {MAX_PIXELS // 1_000_000} MP', over_pixels, False),
             (f'body > {MAX_BYTES // 2**20} MiB', over_bytes, False),
             ('valid PNG', png.getvalue(), True), ('valid MPO (DJI)', mpo.getvalue(), True)]
    before = ep.resources()
    rows = []
    for name, data, should_pass in cases:
        r, err = ep.call('AI-ROB-06', data, name)
        alive, _ = ep.call('AI-ROB-06-alive', valid, f'after {name}')
        rows.append(dict(input=name, bytes=len(data), accepted=r is not None, expected_accept=should_pass,
                         error=(err or '')[:160], worker_alive=alive is not None))
    after = ep.resources()
    same_container = after['model_load_s'] == before['model_load_s'] and after['uptime_s'] > before['uptime_s']
    passed = all(x['accepted'] == x['expected_accept'] and x['worker_alive'] for x in rows) and same_container
    add('AI-ROB-06', 'Ảnh hỏng / sai định dạng / quá lớn',
        'File không hợp lệ bị từ chối có thông báo lỗi; request hợp lệ ngay sau vẫn thành công; container không khởi động lại',
        dict(cases=rows, same_container=same_container), passed)

    # AI-PERF-07: soak test.
    if a.soak_minutes:
        pool_imgs = val[162:]
        end = time.time() + a.soak_minutes * 60
        samples, i, next_sample = [], 0, time.time()
        while time.time() < end:
            if time.time() >= next_sample:
                s = ep.resources(); s['t_min'] = (time.time() - (end - a.soak_minutes * 60)) / 60
                samples.append(s); next_sample += 60
            p = pool_imgs[i % len(pool_imgs)]; i += 1
            r, _ = ep.call('AI-PERF-07', p.read_bytes(), p.name)
            if r:
                check_output(r)
        samples.append(ep.resources())
        rows = [x for x in CALLS if x['test'] == 'AI-PERF-07']
        ok = [x for x in rows if x['ok']]
        n_min = max(1, int(a.soak_minutes))
        t_start = rows[0]['started']
        per_min = [[x['server_ai_ms'] for x in ok if int((x['started'] - t_start) // 60) == k] for k in range(n_min)]
        p95_first, p95_last = pct(per_min[0], 95), pct(per_min[-1], 95)
        rss_2, rss_end = samples[min(2, len(samples) - 1)]['rss_mib'], samples[-1]['rss_mib']
        restarts = len({round(s['model_load_s'], 3) for s in samples}) - 1
        m = dict(minutes=a.soak_minutes, requests=len(rows), errors=len(rows) - len(ok),
                 error_rate=(len(rows) - len(ok)) / max(len(rows), 1), images_per_s=len(ok) / (a.soak_minutes * 60),
                 server_ai_ms=stats([x['server_ai_ms'] for x in ok]), p95_first_min=p95_first, p95_last_min=p95_last,
                 rss_mib_min2=rss_2, rss_mib_end=rss_end, gpu_mem_used_mib_end=samples[-1]['gpu_mem_used_mib'],
                 gpu_util_pct_samples=[s['gpu_util_pct'] for s in samples], container_restarts=restarts,
                 resource_samples=samples)
        add('AI-PERF-07', f'Chạy liên tục {a.soak_minutes} phút',
            'Tỉ lệ lỗi < 1%; RSS cuối ≤ 1,1× RSS phút 2; p95 phút cuối ≤ 1,2× phút đầu; không khởi động lại container',
            m, m['error_rate'] < 0.01 and rss_end <= 1.1 * rss_2 and p95_last <= 1.2 * p95_first + 5 and restarts == 0)

    add('AI-PERF-08', 'Restart AI worker khi còn job', 'Không thực hiện ở đây', {}, None,
        'Cần bridge + RabbitMQ trên VPS (job store SQLite, retry, dedup EventId)')
    add('AI-OUT-01', 'Kiểm tra output: class thuộc 6 class, confidence ∈ [0,1], bbox hợp lệ trong ảnh',
        '0 detection không hợp lệ trên mọi response thành công',
        dict(responses_checked=out_checked, problems=len(out_bad), examples=out_bad[:5]), not out_bad)
    return tests, res_start, split_results


# ------------------------------------------------------------------ report
def rounded(v):
    if isinstance(v, float):
        return round(v, 4) if abs(v) < 10 else round(v, 1)
    if isinstance(v, dict):
        return {k: rounded(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [rounded(x) for x in v]
    return v


def write_report(a, tests, env, out):
    lines = [f'# Kiểm thử AI MF04 — {env["run_at"]}', '',
             '## Môi trường', '',
             *[f'- **{k}**: {v}' for k, v in env.items()], '',
             '## Ngưỡng chấp nhận (định trước)', '',
             f'- Xử lý AI phía server (T1 decode → T4 hậu xử lý): p95 < {SERVER_AI_LIMIT_MS} ms (Report 5, Scope).',
             f'- Thời gian một ảnh phía client (round-trip, không tính cold start): trung bình ≤ {TOTAL_LIMIT_MS} ms (Report 5 §2.1.4).',
             '- Cold start được đo và báo riêng. T5 (DB/storage) và T6 (hàng đợi MF04) không thuộc phạm vi lần đo này.', '',
             '## Tổng hợp', '', '| ID | Hạng mục | Kết quả |', '|---|---|---|',
             *[f'| {t["id"]} | {t["title"]} | **{t["verdict"]}** |' for t in tests], '']
    for t in tests:
        lines += [f'## {t["id"]} — {t["title"]}', '', f'- Tiêu chí: {t["criterion"]}', f'- Kết quả: **{t["verdict"]}**']
        if t['notes']:
            lines.append(f'- Ghi chú: {t["notes"]}')
        m = {k: v for k, v in t['measured'].items() if k != 'resource_samples'}
        if m:
            lines += ['', '```json', json.dumps(rounded(m), indent=2, ensure_ascii=False, default=str), '```']
        lines.append('')
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--splits', nargs='+', default=['val', 'test'])
    p.add_argument('--rob-images', type=int, default=150)
    p.add_argument('--limit', type=int, default=0, help='cap images per accuracy split (0 = all; for dry runs)')
    p.add_argument('--soak-minutes', type=float, default=30)
    p.add_argument('--cold-wait', type=int, default=300, help='idle seconds before the cold-start call (0 = skip)')
    p.add_argument('--out', type=Path, default=ROOT / 'perf' / 'results')
    a = p.parse_args()
    run_at = datetime.now().strftime('%Y-%m-%d_%H%M')
    out = a.out / run_at
    out.mkdir(parents=True)
    import modal, PIL
    thresholds = ROOT / 'powerline_ai' / 'thresholds.json'
    env = dict(run_at=run_at, endpoint=f'Modal app {APP} / env {ENV} / Detector.predict_jpeg (SDK)',
               gpu='Tesla T4 (Modal), max_containers=1, scaledown_window=60 s',
               model='seed85_curated_yolo26m', checkpoint_sha256=json.loads(
                   (ROOT / 'checkpoint_source.json').read_text())['sha256'],
               thresholds_sha256=hashlib.sha256(thresholds.read_bytes()).hexdigest()[:16],
               dataset=a.dataset.name, client=f'{platform.system()} {platform.machine()}, Python {platform.python_version()}, '
                                              f'modal {modal.__version__}, Pillow {PIL.__version__}')
    tests, res_start, splits = run(a)
    env['container_resources_at_start'] = {k: round(v, 1) if isinstance(v, float) else v for k, v in res_start.items()}
    with (out / 'calls.csv').open('w', newline='', encoding='utf-8') as f:
        keys = sorted({k for r in CALLS for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(CALLS)
    (out / 'results.json').write_text(json.dumps(dict(environment=env, tests=tests), indent=2, ensure_ascii=False,
                                                 default=str), encoding='utf-8')
    write_report(a, tests, env, out)
    print('\nReport:', out / 'report.md')


if __name__ == '__main__':
    main()
