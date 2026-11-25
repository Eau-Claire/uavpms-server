"""Active-learning worker (Phase B + C).

Every AL_POLL_SECONDS: when at least AL_MIN_IMAGES label sets were submitted (TrainingStatus = Queued) since the last
batch, it creates a TrainingBatches row, marks all queued label sets Batched, exports them as a YOLO feedback folder
(images from MinIO; video frames cut at FrameIndex), runs `modal/al_cycle.py cycle` (Kaggle fine-tune -> gate on Finding
Nemo val -> Modal deploy + smoke test, automatic rollback) and records the outcome:
  deployed      -> label sets become Trained
  anything else -> label sets go back to Queued; they join the next batch, which needs AL_MIN_IMAGES *new* images.
One batch at a time (unique index on running batches + single process). After `rollback_failed` the worker pauses
until someone fixes production and marks that batch (it never retries blindly).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

CLASSES = ['insulator', 'insulator_broken', 'insulator_flashover', 'bird_nest', 'foreign_object', 'broken_strand']
FINAL_STATUSES = {'deployed', 'gate_passed_not_deployed', 'gate_failed', 'rolled_back', 'rollback_failed'}

log = logging.getLogger('al_worker')


# ---------- pure helpers (unit tested) ----------

def parse_npgsql(conn: str) -> dict:
    """Npgsql connection string (Host=..;Port=..;Database=..;Username=..;Password=..) -> psycopg kwargs."""
    keys = {'host': 'host', 'server': 'host', 'port': 'port', 'database': 'dbname', 'username': 'user',
            'user id': 'user', 'userid': 'user', 'password': 'password', 'ssl mode': 'sslmode', 'sslmode': 'sslmode'}
    out = {}
    for part in conn.split(';'):
        if '=' not in part:
            continue
        k, v = part.split('=', 1)
        k = keys.get(k.strip().lower())
        if k:
            out[k] = v.strip().lower() if k == 'sslmode' else v.strip()
    if 'sslmode' in out:
        out['sslmode'] = {'disable': 'disable', 'require': 'require', 'prefer': 'prefer'}.get(out['sslmode'], 'prefer')
    return out


def storage_key(file_url: str, default_bucket: str) -> tuple[str, str]:
    """'bucket/object' (current format) or an old full URL -> (bucket, object)."""
    key = file_url.strip()
    if '://' in key:
        key = key.split('://', 1)[1].split('/', 1)[1] if '/' in key.split('://', 1)[1] else ''
    key = key.lstrip('/')
    bucket, _, obj = key.partition('/')
    if not obj:
        return default_bucket, bucket
    return bucket, obj


def to_yolo(boxes: list[dict]) -> list[str]:
    """Top-left normalized boxes from the label editor -> YOLO rows 'cls cx cy w h' (clipped to the image)."""
    rows = []
    for b in boxes:
        cls = b.get('className')
        if cls not in CLASSES:
            raise ValueError(f'unknown class {cls!r}')
        x, y = max(0.0, float(b['x'])), max(0.0, float(b['y']))
        w, h = min(float(b['width']), 1 - x), min(float(b['height']), 1 - y)
        if w <= 0 or h <= 0:
            raise ValueError(f'empty box {b}')
        rows.append(f'{CLASSES.index(cls)} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}')
    return rows


def video_frame_jpeg(video: Path, frame_index: int) -> bytes:
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f'cannot read frame {frame_index}')
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError('cannot encode frame')
        return buf.tobytes()
    finally:
        cap.release()


def export_feedback(rows: list[dict], out: Path, fetch) -> tuple[int, list[str]]:
    """Write images/<id>.jpg + labels/<id>.txt for each label set; fetch(bucket, object, dest) downloads a file.
    Rows that cannot be exported are skipped and reported (they return to the queue with the batch)."""
    (out / 'images').mkdir(parents=True, exist_ok=True)
    (out / 'labels').mkdir(parents=True, exist_ok=True)
    tmp = out / '_download'
    tmp.mkdir(exist_ok=True)
    done, errors = 0, []
    for r in rows:
        sid = str(r['id'])
        try:
            labels = to_yolo(json.loads(r['boxes']) if isinstance(r['boxes'], str) else r['boxes'])
            bucket, obj = r['bucket'], r['object']
            ext = Path(obj).suffix.lower() or '.jpg'
            src = tmp / f'{sid}{ext}'
            fetch(bucket, obj, src)
            if r['frame_index'] is not None:
                (out / 'images' / f'{sid}.jpg').write_bytes(video_frame_jpeg(src, int(r['frame_index'])))
            elif ext in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                src.rename(out / 'images' / f'{sid}{".jpg" if ext == ".jpeg" else ext}')
            else:
                raise ValueError(f'unsupported image type {ext}')
            (out / 'labels' / f'{sid}.txt').write_text('\n'.join(labels) + ('\n' if labels else ''))
            done += 1
        except Exception as exc:  # one bad image must not sink the batch
            errors.append(f'{sid}: {exc}')
        finally:
            for p in tmp.glob(f'{sid}*'):
                p.unlink(missing_ok=True)
    tmp.rmdir()
    return done, errors


def db_status(cycle_status: str | None) -> str:
    return cycle_status if cycle_status in FINAL_STATUSES else 'error'


# ---------- worker ----------

class Worker:
    def __init__(self, env=os.environ):
        self.enabled = env.get('AL_ENABLED', 'false').lower() == 'true'
        self.min_images = int(env.get('AL_MIN_IMAGES', '30'))
        self.poll_s = int(env.get('AL_POLL_SECONDS', '600'))
        self.modal_dir = Path(env.get('AL_MODAL_DIR', '/data/modal'))
        self.cycle_config = env.get('AL_CYCLE_CONFIG') or None
        self.db = parse_npgsql(env['DB_CONNECTION'])
        self.minio = dict(endpoint=env.get('MINIO_ENDPOINT', ''), access_key=env.get('MINIO_ACCESS_KEY', ''),
                          secret_key=env.get('MINIO_SECRET_KEY', ''), secure=env.get('MINIO_USE_SSL', 'false').lower() == 'true')
        self.bucket = env.get('MINIO_BUCKET', 'uav-media')

    def connect(self):
        import psycopg
        from psycopg.rows import dict_row
        return psycopg.connect(**self.db, row_factory=dict_row, autocommit=False)

    def fetch(self, bucket: str, obj: str, dest: Path):
        from minio import Minio
        Minio(**self.minio).fget_object(bucket, obj, str(dest))

    # --- state transitions ---

    def paused_reason(self, cur) -> str | None:
        cur.execute('SELECT "Id", "Status" FROM "TrainingBatches" ORDER BY "CreatedAt" DESC LIMIT 1')
        last = cur.fetchone()
        if last and last['Status'] == 'rollback_failed':
            return f'last batch {last["Id"]} ended with rollback_failed; fix production and set its Status to error'
        return None

    def new_queued_count(self, cur) -> int:
        cur.execute('''SELECT count(*) AS n FROM "ImageLabelSets" s
                       WHERE s."TrainingStatus" = 'Queued' AND s."LabelStatus" = 'FullyReviewed'
                         AND s."SubmittedAt" > COALESCE((SELECT max("CreatedAt") FROM "TrainingBatches"), '-infinity')''')
        return cur.fetchone()['n']

    def create_batch(self, conn) -> tuple[uuid.UUID, str, list[dict]] | None:
        """Atomically create the running batch and take every queued label set."""
        with conn.cursor() as cur:
            cur.execute('SELECT 1 FROM "TrainingBatches" WHERE "Status" = \'Running\'')
            if cur.fetchone():
                return None
            batch_id = uuid.uuid4()
            cycle_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            cur.execute('''INSERT INTO "TrainingBatches" ("Id","Status","ImageCount","CycleId","CreatedAt")
                           VALUES (%s,'Running',0,%s,now())''', (batch_id, cycle_id))
            cur.execute('''UPDATE "ImageLabelSets" SET "TrainingStatus"='Batched', "TrainingBatchId"=%s, "UpdatedAt"=now()
                           WHERE "TrainingStatus"='Queued' AND "LabelStatus"='FullyReviewed'
                           RETURNING "Id" AS id, "MediaId" AS media_id, "FrameIndex" AS frame_index, "Boxes"::text AS boxes''',
                        (batch_id,))
            rows = cur.fetchall()
            cur.execute('UPDATE "TrainingBatches" SET "ImageCount"=%s WHERE "Id"=%s', (len(rows), batch_id))
            ids = [r['media_id'] for r in rows]
            cur.execute('SELECT "Id" AS id, "FileUrl" AS url FROM "InspectionMedia" WHERE "Id" = ANY(%s)', (ids,))
            urls = {m['id']: m['url'] for m in cur.fetchall()}
        conn.commit()
        for r in rows:
            r['bucket'], r['object'] = storage_key(urls.get(r['media_id'], ''), self.bucket)
        return batch_id, cycle_id, rows

    def finish_batch(self, conn, batch_id, status: str, result: dict, model: dict | None):
        trained = status == 'deployed'
        with conn.cursor() as cur:
            cur.execute('''UPDATE "TrainingBatches" SET "Status"=%s, "FinishedAt"=now(), "Result"=%s::jsonb,
                           "ModelName"=%s, "ModelSha256"=%s WHERE "Id"=%s''',
                        (status, json.dumps(result, default=str), (model or {}).get('name'),
                         (model or {}).get('model_sha256'), batch_id))
            cur.execute('''UPDATE "ImageLabelSets" SET "TrainingStatus"=%s, "UpdatedAt"=now()
                           WHERE "TrainingBatchId"=%s AND "TrainingStatus"='Batched' ''',
                        ('Trained' if trained else 'Queued', batch_id))
        conn.commit()
        log.info('batch %s finished: %s (label sets -> %s)', batch_id, status, 'Trained' if trained else 'Queued')

    # --- one cycle ---

    def run_cycle(self, args: list[str], cycle_id: str) -> tuple[str, dict]:
        cmd = [sys.executable, str(self.modal_dir / 'al_cycle.py')]
        if self.cycle_config:
            cmd += ['--config', self.cycle_config]
        p = subprocess.run(cmd + args, cwd=self.modal_dir, capture_output=True, text=True)
        tail = (p.stdout + p.stderr)[-4000:]
        log.info('al_cycle exit %s\n%s', p.returncode, tail)
        outcome_file = self.modal_dir / '.al_work' / cycle_id / 'result' / 'outcome.json'
        outcome = json.loads(outcome_file.read_text()) if outcome_file.exists() else {}
        gate_file = outcome_file.parent / 'gate.json'
        result = {'exit_code': p.returncode, 'outcome': outcome,
                  'gate': json.loads(gate_file.read_text()) if gate_file.exists() else None,
                  'log_tail': tail[-1500:]}
        return db_status(outcome.get('status')), result

    def process(self, conn, batch_id, cycle_id, rows):
        feedback = self.modal_dir / '.al_work' / f'feedback_{cycle_id}'
        try:
            done, errors = export_feedback(rows, feedback, self.fetch)
            log.info('batch %s: exported %s/%s label sets%s', batch_id, done, len(rows), f'; errors: {errors[:5]}' if errors else '')
            if done == 0:
                self.finish_batch(conn, batch_id, 'error', {'export_errors': errors}, None)
                return
            status, result = self.run_cycle(['cycle', '--feedback', str(feedback), '--cycle-id', cycle_id, '--force'], cycle_id)
            result['export'] = {'exported': done, 'errors': errors}
            self.finish_batch(conn, batch_id, status, result, result['outcome'].get('model'))
        except Exception as exc:
            log.exception('batch %s failed', batch_id)
            conn.rollback()
            self.finish_batch(conn, batch_id, 'error', {'error': str(exc)}, None)

    def resume_running(self, conn):
        """A batch left Running by a restart: resume waiting for its Kaggle kernel, or give its images back."""
        with conn.cursor() as cur:
            cur.execute('SELECT "Id" AS id, "CycleId" AS cycle FROM "TrainingBatches" WHERE "Status"=\'Running\'')
            row = cur.fetchone()
        conn.commit()
        if not row:
            return
        (self.modal_dir / '.al_work' / 'cycle.lock').unlink(missing_ok=True)   # single worker: the lock is stale
        if (self.modal_dir / '.al_work' / row['cycle'] / 'state.json').exists():
            log.info('resuming batch %s (cycle %s)', row['id'], row['cycle'])
            status, result = self.run_cycle(['fetch', '--cycle-id', row['cycle']], row['cycle'])
            self.finish_batch(conn, row['id'], status, result, result['outcome'].get('model'))
        else:
            self.finish_batch(conn, row['id'], 'error', {'error': 'worker restarted before the Kaggle kernel was pushed'}, None)

    def tick(self, conn):
        if not self.enabled:
            return 'disabled'
        with conn.cursor() as cur:
            reason = self.paused_reason(cur)
            n = None if reason else self.new_queued_count(cur)
        conn.commit()
        if reason:
            log.warning('paused: %s', reason)
            return 'paused'
        if n < self.min_images:
            log.info('%s new queued label sets (< %s); waiting', n, self.min_images)
            return 'waiting'
        taken = self.create_batch(conn)
        if taken is None:
            return 'busy'
        batch_id, cycle_id, rows = taken
        log.info('batch %s (cycle %s): %s label sets', batch_id, cycle_id, len(rows))
        self.process(conn, batch_id, cycle_id, rows)
        return 'ran'

    def loop(self):
        log.info('active-learning worker: enabled=%s, min_images=%s, poll=%ss, modal_dir=%s',
                 self.enabled, self.min_images, self.poll_s, self.modal_dir)
        if not (self.modal_dir / 'models' / 'best.pt').exists():
            raise SystemExit(f'champion not found: {self.modal_dir / "models" / "best.pt"}')
        with self.connect() as conn:
            self.resume_running(conn)
        while True:
            try:
                with self.connect() as conn:
                    self.tick(conn)
            except Exception:
                log.exception('tick failed')
            time.sleep(self.poll_s)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s', stream=sys.stdout)
    if len(sys.argv) > 1 and sys.argv[1] == 'once':   # one tick, for tests and manual runs
        w = Worker()
        with w.connect() as c:
            w.resume_running(c)
            print(w.tick(c))
    else:
        Worker().loop()
