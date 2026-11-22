"""MF04 end-to-end AI test on the deployed stack (run on the VPS, next to the containers).

Path under test:
  Inspector POST /api/v1/inspections/upload -> InspectionMediaUploadedEvent -> AIAnalysisRuns (Pending)
  -> RabbitMQ ai.analysis.server.image.requested -> ai-bridge -> Modal detector
  -> ai.analysis.result -> ProcessAiAnalysisResult -> DetectedAnomalies + gRPC severity
  -> Analyst GET /api/v1/missions/{id}/ai-analysis/detections

Subcommands
  preflight   MockAI off, exactly one consumer (the bridge) on the server image queue, bridge running.
  run         Upload images and verify every run: completed once, real model (not MockAI), detections
              persisted with valid boxes and a gRPC severity, visible to the Analyst; time each stage.
              Options turn the same run into the remaining MF04 cases:
                --concurrency N             concurrent jobs
                --restart-bridge-after S    restart the bridge S seconds into the run (AI-PERF-08)
                --stop-bridge-for S         bridge down while uploading, started again after S seconds
                                            (jobs must wait in RabbitMQ, not be lost)
                --duplicate-check           re-upload the first file to the same mission: same media, no new run
                --review                    Analyst rejects one detection (MF04.3 review path)

Read-only towards the backend code: it only calls the public API, reads the database and
restarts/stops the ai-bridge container. Needs: pip install httpx 'psycopg[binary]'.
Setup: a roles JSON with "inspector" (assigned to the mission) and "analyst";
TARGET_API=http://localhost:<gateway port>; DB_CONNECTION (Npgsql or libpq format).
The mission must be Executing or Completed. Use a test mission: uploads and reviews are real writes.

  python perf/mf04_e2e.py preflight
  python perf/mf04_e2e.py run --roles roles.json --mission <id> --images <dir> --count 10 --label e2e
  python perf/mf04_e2e.py run --roles roles.json --mission <id> --images <dir> --count 50 --concurrency 10 --label concurrent
  python perf/mf04_e2e.py run --roles roles.json --mission <id> --images <dir> --count 20 --restart-bridge-after 15 --label recovery
  python perf/mf04_e2e.py run --roles roles.json --mission <id> --images <dir> --count 10 --stop-bridge-for 60 --label outage
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx
import psycopg

HERE = Path(__file__).resolve().parent

QUEUE = 'ai.analysis.server.image.requested'
IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}
SEVERITY = re.compile(r'AI evaluation: severity=(\w+), risk=(\w+), score=([\d.]+)')
TERMINAL = {'Completed', 'Failed'}


def sh(*cmd, check=False):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f'{" ".join(cmd)} failed: {r.stderr.strip()}')
    return r.stdout.strip()


def bridge_container(name=None):
    if name:
        return name
    found = sh('docker', 'ps', '-a', '--filter', 'name=ai-bridge', '--format', '{{.Names}}').splitlines()
    if len(found) != 1:
        raise SystemExit(f'Cannot pick the bridge container ({found}); pass --bridge-container')
    return found[0]


def pct(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))] if v else None


def stats(values):
    values = [x for x in values if x is not None]
    if not values:
        return None
    return dict(n=len(values), mean=round(statistics.mean(values), 1), p50=round(pct(values, 50), 1),
                p95=round(pct(values, 95), 1), max=round(max(values), 1))


# ---------------------------------------------------------------- preflight

def preflight(a):
    problems, info = [], {}
    mock = sh('docker', 'exec', a.ai_container, 'printenv', 'MockAI__Enabled').lower()
    info['MockAI__Enabled'] = mock or '(unset)'
    if mock != 'false':
        problems.append(f'MockAI__Enabled={mock or "unset"}: the mock consumer competes with the bridge. '
                        'Set MOCK_AI_ENABLED=false and recreate uav-ai-inspection-service.')
    queues = sh('docker', 'exec', a.rabbit_container, 'rabbitmqctl', '-q', 'list_queues',
                'name', 'consumers', 'messages', check=True)
    for line in queues.splitlines():
        parts = line.split()
        if parts and parts[0].startswith('ai.analysis'):
            info[parts[0]] = dict(consumers=int(parts[1]), messages=int(parts[2]))
    q = info.get(QUEUE)
    if not q:
        problems.append(f'queue {QUEUE} not found')
    elif q['consumers'] != 1:
        problems.append(f'{QUEUE} has {q["consumers"]} consumers; expected exactly 1 (the bridge)')
    bridge = bridge_container(a.bridge_container)
    info['bridge'] = dict(name=bridge, state=sh('docker', 'inspect', '-f', '{{.State.Status}}', bridge))
    if info['bridge']['state'] != 'running':
        problems.append(f'bridge {bridge} is {info["bridge"]["state"]}')
    print(json.dumps(info, indent=2))
    print('PREFLIGHT', 'PASSED' if not problems else 'FAILED')
    for p in problems:
        print(' -', p)
    return not problems, info


# ---------------------------------------------------------------- run

def normalize_connection_string(value: str) -> str:
    """Npgsql style 'Host=..;Port=..;Database=..;Username=..;Password=..' -> libpq conninfo."""
    if ';' not in value or '=' not in value:
        return value
    keys = {'host': 'host', 'port': 'port', 'database': 'dbname', 'username': 'user', 'user id': 'user',
            'password': 'password'}
    pairs = dict(p.split('=', 1) for p in value.strip().strip(';').split(';') if '=' in p)
    return ' '.join(f"{keys[k.strip().lower()]}='{v.strip()}'" for k, v in pairs.items()
                    if k.strip().lower() in keys)


def _token(body):
    data = body.get('data') or {}
    return (body.get('accessToken') or body.get('token') or data.get('accessToken') or data.get('token')
            or (data.get('authResult') or {}).get('accessToken'))


def login(roles_file, role, otp=None):
    """Log in through the gateway: trusted-device token first; if the backend still asks for an OTP,
    verify with the given test OTP (the staging backend accepts a fixed mock OTP for test accounts)."""
    spec = json.loads(Path(roles_file).read_text(encoding='utf-8'))[role]
    if spec.get('token'):
        return spec['token']
    email = spec['email'].strip().lower()
    headers = {'X-Device-Trust-Token': spec['deviceTrustToken']} if spec.get('deviceTrustToken') else {}
    with httpx.Client(base_url=(os.getenv('AUTH_API') or os.environ['TARGET_API']).rstrip('/'), timeout=30) as c:
        r = c.post('/api/v1/auth/login', json={'email': email, 'password': spec['password']}, headers=headers)
        body = r.json() if r.headers.get('content-type', '').startswith('application/json') else {}
        token = _token(body)
        if r.status_code == 200 and not token and 'otp' in str(body.get('message', '')).lower() and otp:
            r = c.post('/api/v1/auth/otp/verify', json={'email': email, 'otp': otp, 'purpose': 'Login'})
            body = r.json() if r.headers.get('content-type', '').startswith('application/json') else {}
            token = _token(body)
    if r.status_code != 200 or not token:
        raise SystemExit(f'login as {role} failed ({r.status_code}): {r.text[:200]}. '
                         'Register the deviceTrustToken, put a token in the roles file, or pass --otp.')
    return token


def unique_bytes(path: Path) -> bytes:
    """Same image, new checksum, still a valid file (the backend rejects trailing bytes):
    JPEG gets a COM segment right after SOI, PNG a tEXt chunk right after IHDR."""
    data, tag = path.read_bytes(), f'mf04-e2e-{uuid.uuid4()}'.encode()
    if data[:2] == b'\xff\xd8':
        return data[:2] + b'\xff\xfe' + (len(tag) + 2).to_bytes(2, 'big') + tag + data[2:]
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        import zlib
        body = b'tEXt' + b'Comment\x00' + tag
        chunk = (len(body) - 4).to_bytes(4, 'big') + body + zlib.crc32(body).to_bytes(4, 'big')
        return data[:33] + chunk + data[33:]
    raise SystemExit(f'{path.name}: only JPEG and PNG are supported')


def upload(client, token, mission, path, data):
    started = time.time()
    r = client.post('/api/v1/inspections/upload', headers={'Authorization': f'Bearer {token}'},
                    data={'missionId': mission, 'capturedAt': datetime.now(timezone.utc).isoformat()},
                    files={'file': (path.name, data, 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg')})
    row = dict(file=path.name, bytes=len(data), upload_started=started, upload_ms=(time.time() - started) * 1000,
               status=r.status_code)
    if r.status_code == 200:
        row['media_id'] = r.json()['data']['mediaId']
    else:
        row['error'] = r.text[:300]
    return row


def db_state(conn, media_ids):
    with conn.cursor() as cur:
        cur.execute('''SELECT r."Id", r."MediaId", r."Status", r."ModelName", r."ModelVersion", r."ErrorMessage",
                              r."CreatedAt", r."StartedAt", r."CompletedAt"
                       FROM "AIAnalysisRuns" r WHERE r."MediaId" = ANY(%s::uuid[])''', (media_ids,))
        runs = cur.fetchall()
        cur.execute('''SELECT a."MediaId", a."AnalysisRunId", c."CategoryCode", a."ConfidenceScore", a."BoundingBox",
                              a."AnalystNotes", a."ValidationStatus", m."WidthPx", m."HeightPx"
                       FROM "DetectedAnomalies" a JOIN "DefectCategories" c ON c."Id" = a."CategoryId"
                       JOIN "InspectionMedia" m ON m."Id" = a."MediaId"
                       WHERE a."MediaId" = ANY(%s::uuid[]) AND a."SupersededById" IS NULL''', (media_ids,))
        anomalies = cur.fetchall()
    return runs, anomalies


def bbox_ok(raw, w, h):
    b = raw if isinstance(raw, dict) else json.loads(raw)
    x, y, bw, bh = (float(b[k]) for k in ('x', 'y', 'width', 'height'))
    if bw <= 0 or bh <= 0 or x < -1 or y < -1:
        return False
    if max(x + bw, y + bh) <= 1.0001:  # normalized coordinates
        return True
    return not (w and h) or (x + bw <= w + 1 and y + bh <= h + 1)


def run(a):
    ok, pre = preflight(a)
    if not ok and not a.ignore_preflight:
        raise SystemExit('Preflight failed; fix it or pass --ignore-preflight (results then do not count).')
    bridge = bridge_container(a.bridge_container)
    files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in IMAGE_EXTS)[:a.count]
    if not files:
        raise SystemExit('no images')
    inspector, analyst = login(a.roles, a.uploader_role, a.otp), login(a.roles, 'analyst', a.otp)
    out = Path(a.out) / f'{datetime.now():%Y-%m-%d_%H%M}_{a.label}'
    out.mkdir(parents=True, exist_ok=True)
    events = []
    t0 = time.time()

    def bridge_action(kind, delay, duration=None):
        time.sleep(delay)
        if kind == 'restart':
            events.append(dict(at_s=round(time.time() - t0, 1), action='docker restart bridge'))
            sh('docker', 'restart', bridge, check=True)
        else:
            time.sleep(duration)
            events.append(dict(at_s=round(time.time() - t0, 1), action='docker start bridge'))
            sh('docker', 'start', bridge, check=True)

    if a.stop_bridge_for:
        events.append(dict(at_s=0.0, action='docker stop bridge'))
        sh('docker', 'stop', bridge, check=True)
        threading.Thread(target=bridge_action, args=('start', 0, a.stop_bridge_for), daemon=True).start()
    if a.restart_bridge_after:
        threading.Thread(target=bridge_action, args=('restart', a.restart_bridge_after), daemon=True).start()

    with httpx.Client(base_url=os.environ['TARGET_API'].rstrip('/'), timeout=120) as client:
        payloads = [(p, unique_bytes(p)) for p in files]
        with ThreadPoolExecutor(a.concurrency) as pool:
            uploads = list(pool.map(lambda x: upload(client, inspector, a.mission, *x), payloads))
        media = {u['media_id']: u for u in uploads if 'media_id' in u}
        print(f'uploaded {len(media)}/{len(uploads)}')

        duplicate = None
        if a.duplicate_check and media:
            p, data = payloads[0]
            r = upload(client, inspector, a.mission, p, data)
            # Same mission: the backend returns the existing media (idempotent); another mission: DUPLICATE_MEDIA.
            duplicate = dict(status=r['status'], returned_media=r.get('media_id'), original_media=uploads[0].get('media_id'),
                             passed=r['status'] == 200 and r.get('media_id') == uploads[0].get('media_id'),
                             body=r.get('error'))

        conn = psycopg.connect(normalize_connection_string(os.environ['DB_CONNECTION']), autocommit=True)
        ids = list(media)
        done_at = {}
        deadline = time.time() + a.timeout
        while time.time() < deadline:
            runs, _ = db_state(conn, ids)
            for r in runs:
                if r[2] in TERMINAL and str(r[1]) not in done_at:
                    done_at[str(r[1])] = time.time()
            if len(done_at) >= len(ids):
                break
            time.sleep(1)
        time.sleep(3)  # let duplicate deliveries (if any) land before counting
        runs, anomalies = db_state(conn, ids)

        visible = None
        r = client.get(f'/api/v1/missions/{a.mission}/ai-analysis/detections',
                       headers={'Authorization': f'Bearer {analyst}'})
        if r.status_code == 200:
            visible = {d['mediaId']: d['detectionCount'] for d in r.json()['data']}
        analyst_seen_at = time.time()

        review = None
        if a.review and anomalies:
            det_list = r.json()['data'] if r.status_code == 200 else []
            target = next((d for m in det_list if m['mediaId'] in media for d in m['detections']), None)
            if target:
                rr = client.put(f'/api/v1/missions/{a.mission}/ai-analysis/detections/{target["id"]}/review',
                                headers={'Authorization': f'Bearer {analyst}'},
                                json={'decision': 'Rejected', 'notes': 'mf04 e2e test upload'})
                # Rejected: test images carry no tower, and Accepted requires one (TOWER_REQUIRED).
                review = dict(status=rr.status_code, detection=target['id'], decision='Rejected', body=rr.text[:300])

    # ---------------- checks
    runs_by_media = {}
    for r in runs:
        runs_by_media.setdefault(str(r[1]), []).append(r)
    an_by_media = {}
    for x in anomalies:
        an_by_media.setdefault(str(x[0]), []).append(x)

    rows, checks = [], {}
    for mid, u in media.items():
        rs = runs_by_media.get(mid, [])
        r = rs[0] if rs else None
        ans = an_by_media.get(mid, [])
        sev = [SEVERITY.match(x[5] or '') for x in ans]
        rows.append(dict(
            file=u['file'], media_id=mid, upload_ms=round(u['upload_ms']),
            runs=len(rs), status=r[2] if r else None, model=r[3] if r else None, model_version=r[4] if r else None,
            error=r[5] if r else None,
            # client clock: upload start -> run seen terminal in DB (polled every 1 s)
            e2e_ms=round((done_at[mid] - u['upload_started']) * 1000) if mid in done_at else None,
            # VPS clock. CreatedAt = run queued by BE; CompletedAt = result built by the bridge (after Modal);
            # StartedAt = BE received the result (set on first result message, not when work started).
            bridge_done_ms=round((r[8] - r[6]).total_seconds() * 1000) if r and r[8] else None,
            result_delivery_ms=round((r[7] - r[8]).total_seconds() * 1000) if r and r[7] and r[8] else None,
            detections=len(ans), classes=sorted({x[2] for x in ans}),
            bbox_invalid=sum(not bbox_ok(x[4], x[7], x[8]) for x in ans),
            no_severity=sum(m is None for m in sev),
            severities=sorted({m.group(1) for m in sev if m}),
            analyst_sees=None if visible is None else visible.get(mid, 0)))

    n = len(rows)
    checks['all_uploads_accepted'] = bool(uploads) and len(media) == len(uploads)
    if not rows:  # nothing reached the AI path: every other check would pass vacuously
        checks['at_least_one_job'] = False
    checks['all_runs_completed'] = all(x['status'] == 'Completed' for x in rows)
    checks['exactly_one_run_per_media'] = all(x['runs'] == 1 for x in rows)
    checks['real_model_not_mock'] = all(x['model'] == a.expect_model for x in rows)
    checks['bbox_valid'] = all(x['bbox_invalid'] == 0 for x in rows)
    checks['grpc_severity_on_every_detection'] = all(x['no_severity'] == 0 for x in rows)
    checks['analyst_sees_all_detections'] = visible is not None and all(
        x['analyst_sees'] == x['detections'] for x in rows)
    if duplicate:
        checks['duplicate_upload_idempotent'] = duplicate['passed']
    if review:
        checks['analyst_review_saved'] = review['status'] == 200
    e2e = [x['e2e_ms'] for x in rows]
    if a.e2e_limit_ms:
        checks[f'e2e_p95_le_{a.e2e_limit_ms}ms'] = (stats(e2e) or {}).get('p95', 1e12) <= a.e2e_limit_ms

    severity_by_class = {}
    for x in anomalies:
        m = SEVERITY.match(x[5] or '')
        severity_by_class.setdefault(x[2], {}).setdefault(m.group(1) if m else 'none', 0)
        severity_by_class[x[2]][m.group(1) if m else 'none'] += 1

    since = datetime.fromtimestamp(t0 - 5, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    def container_log(name):  # docker logs forwards the container stdout and stderr separately
        r = subprocess.run(['docker', 'logs', '--since', since, name], capture_output=True, text=True)
        return r.stdout + r.stderr

    ai_log, bridge_text = container_log(a.ai_container), container_log(bridge)
    logs = dict(
        unknown_category_skipped=len(re.findall(r'DefectCategory not found', ai_log)),
        duplicate_result_skipped=len(re.findall(r'already in terminal state', ai_log)),
        bridge_processing_failed=bridge_text.count('"processing_failed"'),
        bridge_publish_retry=bridge_text.count('"publish_retry"'),
        bridge_attempts_exhausted=bridge_text.count('WORKER_ATTEMPTS_EXHAUSTED'))
    checks['no_detection_dropped_for_unknown_category'] = logs['unknown_category_skipped'] == 0

    summary = dict(
        label=a.label, run_at=datetime.now().isoformat(timespec='seconds'), mission=a.mission, images=n,
        upload_errors=[dict(file=u['file'], status=u['status'], error=u.get('error')) for u in uploads if 'media_id' not in u],
        concurrency=a.concurrency, preflight=pre, bridge_events=events, duplicate=duplicate, review=review,
        timing_ms=dict(upload=stats([x['upload_ms'] for x in rows]), e2e_upload_to_db_completed=stats(e2e),
                       queued_to_bridge_done=stats([x['bridge_done_ms'] for x in rows]),
                       bridge_done_to_be_received=stats([x['result_delivery_ms'] for x in rows])),
        wall_s=round(analyst_seen_at - t0, 1), detections=sum(x['detections'] for x in rows),
        severity_by_class=severity_by_class, logs=logs, checks=checks,
        result='Passed' if all(checks.values()) else 'Failed')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    with (out / 'jobs.jsonl').open('w', encoding='utf-8') as f:
        for x in rows:
            f.write(json.dumps(x, default=str) + '\n')
    print(json.dumps({k: summary[k] for k in ('label', 'images', 'upload_errors', 'timing_ms', 'severity_by_class', 'logs',
                                              'checks', 'result')}, indent=2, default=str))
    print('written to', out)
    return summary['result'] == 'Passed'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('preflight', 'run'))
    p.add_argument('--ai-container', default='uav-ai-inspection-service')
    p.add_argument('--rabbit-container', default='uav-rabbitmq')
    p.add_argument('--bridge-container')
    p.add_argument('--mission')
    p.add_argument('--images')
    p.add_argument('--count', type=int, default=10)
    p.add_argument('--concurrency', type=int, default=1)
    p.add_argument('--label', default='e2e')
    p.add_argument('--timeout', type=int, default=900, help='seconds to wait for all runs')
    p.add_argument('--expect-model', default='seed85_curated_yolo26m')
    p.add_argument('--e2e-limit-ms', type=int, help='optional p95 limit for upload -> completed')
    p.add_argument('--restart-bridge-after', type=float)
    p.add_argument('--stop-bridge-for', type=float)
    p.add_argument('--duplicate-check', action='store_true')
    p.add_argument('--review', action='store_true')
    p.add_argument('--ignore-preflight', action='store_true')
    p.add_argument('--roles', help="JSON with 'inspector' and 'analyst': email, password, deviceTrustToken "
                   "(same format as the backend api-test roles.json)")
    p.add_argument('--uploader-role', default='inspector',
                   help='roles-file key used to upload (inspector assigned to the mission, or systemAdmin)')
    p.add_argument('--otp', default=os.getenv('E2E_OTP'), help='test OTP used when the backend asks for one')
    p.add_argument('--out', default=str(HERE / 'results' / 'mf04_e2e'))
    a = p.parse_args()
    if a.command == 'preflight':
        sys.exit(0 if preflight(a)[0] else 1)
    if not (a.mission and a.images and a.roles):
        p.error('run needs --mission, --images and --roles')
    sys.exit(0 if run(a) else 1)


if __name__ == '__main__':
    main()
