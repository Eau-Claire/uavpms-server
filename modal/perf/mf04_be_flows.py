"""MF04 backend flow tests after AI detection (run on the VPS, black-box through the gateway + DB reads).

Tests (each uses fresh InReview detections of the test mission; all writes are real):
  TKT-1  Analyst Review screen route  PUT /anomalies/{id}/validate {status: Confirmed}  -> ticket?
  TKT-2  AI review route              PUT /missions/{m}/ai-analysis/detections/{id}/review {decision: Accepted} -> ticket?
  TKT-3  Mission screen route         POST /missions/{m}/detections/{id}/review {status: Confirmed}, sent twice -> 1 ticket
  TKT-4  Mission screen route, 5 identical requests at once -> still 1 ticket
  TKT-5  Analyst Review route, 5 identical Confirm requests at once -> exactly 1 accepted
  RE-1   Reanalysis of one media: second request while pending -> 409; after completion, detections not duplicated
  RPT-1  Defect report for the mission: excel/pdf export, counts vs database (Rejected excluded)

  python perf/mf04_be_flows.py --roles roles.json --mission <id> --otp 123456 --out ~/mf04_e2e_results
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import httpx
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mf04_e2e import login, normalize_connection_string  # noqa: E402


class Ctx:
    def __init__(self, a):
        self.a = a
        self.http = httpx.Client(base_url=os.environ['TARGET_API'].rstrip('/'), timeout=120)
        self.db = psycopg.connect(normalize_connection_string(os.environ['DB_CONNECTION']), autocommit=True)
        self.tok = {r: login(a.roles, r, a.otp) for r in ('analyst', 'systemAdmin')}
        self.used = set()

    def q(self, sql, *args):
        with self.db.cursor() as cur:
            cur.execute(sql, args)
            return cur.fetchall() if cur.description else None

    def call(self, method, path, role='analyst', **kw):
        r = self.http.request(method, path, headers={'Authorization': f'Bearer {self.tok[role]}'}, **kw)
        try:
            body = r.json()
        except ValueError:
            body = r.text[:300]
        return r.status_code, body

    def fresh_anomaly(self):
        """An InReview, not superseded, defect detection of the test mission not used yet in this run."""
        rows = self.q('''SELECT a."Id" FROM "DetectedAnomalies" a JOIN "InspectionMedia" m ON m."Id" = a."MediaId"
                         WHERE m."MissionId" = %s AND a."ValidationStatus" = 'InReview' AND a."SupersededById" IS NULL
                           AND NOT a."IsDeleted" AND NOT EXISTS (SELECT 1 FROM "MaintenanceTickets" t WHERE t."AnomalyId" = a."Id")
                         ORDER BY a."CreatedAt" DESC LIMIT 50''', self.a.mission)
        aid = next(str(r[0]) for r in rows if str(r[0]) not in self.used)
        self.used.add(aid)
        return aid

    def tower(self):
        return str(self.q('''SELECT "TowerId" FROM "MissionTargets" WHERE "MissionId" = %s AND NOT "IsDeleted"
                             ORDER BY "Sequence" LIMIT 1''', self.a.mission)[0][0])

    def link_tower(self, aid):
        return self.call('PUT', f'/api/v1/anomalies/{aid}/tower', json={'towerId': self.tower()})

    def tickets(self, aid):
        return self.q('SELECT count(*) FROM "MaintenanceTickets" WHERE "AnomalyId" = %s AND NOT "IsDeleted"', aid)[0][0]

    def status(self, aid):
        return self.q('SELECT "ValidationStatus" FROM "DetectedAnomalies" WHERE "Id" = %s', aid)[0][0]


def short(body):
    if isinstance(body, dict):
        return {k: body.get(k) for k in ('success', 'message', 'errorCode') if k in body}
    return body


def wait_tickets(c, aid, seconds=20):
    end = time.time() + seconds
    while time.time() < end:
        if c.tickets(aid):
            break
        time.sleep(1)
    return c.tickets(aid)


def tkt_ai_validate(c):
    aid = c.fresh_anomaly()
    link = c.link_tower(aid)
    first = c.call('PUT', f'/api/v1/anomalies/{aid}/validate', json={'status': 'Confirmed', 'analystNotes': 'mf04 test'})
    n = wait_tickets(c, aid)
    second = c.call('PUT', f'/api/v1/anomalies/{aid}/validate', json={'status': 'Confirmed', 'analystNotes': 'mf04 test'})
    return dict(anomaly=aid, link_tower=link[0], confirm=(first[0], short(first[1])), status_after=c.status(aid),
                second_confirm=(second[0], short(second[1])), tickets=n,
                passed=first[0] == 200 and n == 1 and second[0] == 400)


def tkt_ai_review(c, since):
    aid = c.fresh_anomaly()
    link = c.link_tower(aid)
    r = c.call('PUT', f'/api/v1/missions/{c.a.mission}/ai-analysis/detections/{aid}/review',
               json={'decision': 'Accepted', 'notes': 'mf04 test'})
    n = wait_tickets(c, aid)
    time.sleep(5)
    log = subprocess.run(['docker', 'logs', '--since', since, 'uav-operations-service'], capture_output=True, text=True)
    errors = [l[:220] for l in (log.stdout + log.stderr).splitlines()
              if ('ERR' in l or 'Exception' in l or 'fail' in l.lower()) and not l.startswith('   at ')]
    return dict(anomaly=aid, link_tower=link[0], review=(r[0], short(r[1])), status_after=c.status(aid), tickets=n,
                ops_log_errors=len(errors), ops_log_sample=errors[:4],
                passed=r[0] == 200 and n == 1)


def ops_review(c, aid):
    return c.call('POST', f'/api/v1/missions/{c.a.mission}/detections/{aid}/review',
                  json={'status': 'Confirmed', 'reviewNotes': 'mf04 test'})


def tkt_ops_sequential(c):
    aid = c.fresh_anomaly()
    c.link_tower(aid)
    first, second = ops_review(c, aid), ops_review(c, aid)
    n = c.tickets(aid)
    return dict(anomaly=aid, first=(first[0], short(first[1])), second=(second[0], short(second[1])),
                status_after=c.status(aid), tickets=n, passed=first[0] == 200 and n == 1)


def tkt_ops_concurrent(c, k=5):
    aid = c.fresh_anomaly()
    c.link_tower(aid)
    with ThreadPoolExecutor(k) as pool:
        res = list(pool.map(lambda _: ops_review(c, aid), range(k)))
    n = c.tickets(aid)
    return dict(anomaly=aid, statuses=[r[0] for r in res], tickets=n, passed=n == 1)


def tkt_validate_concurrent(c, k=5):
    aid = c.fresh_anomaly()
    c.link_tower(aid)
    with ThreadPoolExecutor(k) as pool:
        res = list(pool.map(lambda _: c.call('PUT', f'/api/v1/anomalies/{aid}/validate',
                                             json={'status': 'Confirmed', 'analystNotes': 'mf04 test'}), range(k)))
    ok = sum(r[0] == 200 for r in res)
    return dict(anomaly=aid, statuses=[r[0] for r in res], accepted=ok, status_after=c.status(aid), passed=ok == 1)


def reanalysis(c):
    media = str(c.q('''SELECT m."Id" FROM "InspectionMedia" m
                       WHERE m."MissionId" = %s AND EXISTS (SELECT 1 FROM "DetectedAnomalies" a WHERE a."MediaId" = m."Id"
                             AND a."SupersededById" IS NULL AND a."ValidationStatus" = 'InReview')
                         AND (SELECT count(*) FROM "AIAnalysisRuns" r WHERE r."MediaId" = m."Id") = 1
                       ORDER BY m."CreatedAt" DESC LIMIT 1''', c.a.mission)[0][0])

    def counts():
        return c.q('''SELECT (SELECT count(*) FROM "AIAnalysisRuns" WHERE "MediaId" = %s),
                             (SELECT count(*) FROM "DetectedAnomalies" WHERE "MediaId" = %s AND "SupersededById" IS NULL AND NOT "IsDeleted"),
                             (SELECT count(*) FROM "DetectedAnomalies" WHERE "MediaId" = %s)''', media, media, media)[0]
    before = counts()
    path = f'/api/v1/missions/{c.a.mission}/ai-analysis/from-media/{media}'
    first = c.call('POST', path)
    second = c.call('POST', path)
    end = time.time() + 180
    while time.time() < end:
        done = c.q('''SELECT count(*) FROM "AIAnalysisRuns" WHERE "MediaId" = %s AND "Status" IN ('Completed','Failed')''', media)[0][0]
        if done >= 2:
            break
        time.sleep(2)
    time.sleep(3)
    after = counts()
    per_run = c.q('''SELECT r."Status", r."CreatedAt", count(a."Id") FROM "AIAnalysisRuns" r
                     LEFT JOIN "DetectedAnomalies" a ON a."AnalysisRunId" = r."Id"
                     WHERE r."MediaId" = %s GROUP BY 1, 2 ORDER BY 2''', media)
    return dict(media=media, first=(first[0], short(first[1])), second_while_pending=(second[0], short(second[1])),
                runs_before=before[0], runs_after=after[0], active_detections_before=before[1],
                active_detections_after=after[1], per_run=[(s, str(t), n) for s, t, n in per_run],
                checks=dict(new_run_created=after[0] == before[0] + 1,
                            duplicate_request_409=second[0] == 409,
                            detections_not_duplicated=after[1] == before[1]),
                passed=after[0] == before[0] + 1 and second[0] == 409 and after[1] == before[1])


def report(c):
    expected = c.q('''SELECT count(*) FILTER (WHERE a."ValidationStatus" <> 'Rejected'),
                             count(*) FILTER (WHERE a."ValidationStatus" = 'Rejected'),
                             count(*) FILTER (WHERE a."ValidationStatus" = 'Confirmed')
                      FROM "DetectedAnomalies" a JOIN "InspectionMedia" m ON m."Id" = a."MediaId"
                      WHERE m."MissionId" = %s AND a."SupersededById" IS NULL AND NOT a."IsDeleted"''', c.a.mission)[0]
    payload = {'title': f'MF04 test {datetime.now():%H%M%S}', 'type': 'Defect', 'missionIds': [c.a.mission],
               'description': 'mf04 e2e test report'}
    st, body = c.call('POST', '/api/v1/reports', json=payload)
    if st == 400:  # enum may be numeric only
        st, body = c.call('POST', '/api/v1/reports', json=dict(payload, type=0))
    rid = (body.get('data') or {}).get('id') if isinstance(body, dict) else None
    out = dict(create=(st, short(body)), report_id=rid, db_in_scope=expected[0], db_rejected=expected[1],
               db_confirmed=expected[2])
    if not rid:
        out['passed'] = False
        return out
    st, detail = c.call('GET', f'/api/v1/reports/{rid}')
    out['detail_keys'] = list((detail.get('data') or {}).keys()) if isinstance(detail, dict) else None
    out['detail'] = {k: v for k, v in (detail.get('data') or {}).items()
                     if isinstance(v, (int, float)) or 'count' in k.lower() or 'total' in k.lower()} if isinstance(detail, dict) else None
    exports = {}
    for fmt in ('excel', 'pdf'):
        r = c.http.get(f'/api/v1/reports/{rid}/download', params={'format': fmt},
                       headers={'Authorization': f'Bearer {c.tok["analyst"]}'})
        e = dict(status=r.status_code, bytes=len(r.content), type=r.headers.get('content-type'))
        if fmt == 'excel' and r.status_code == 200:
            try:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                sheets = [n for n in z.namelist() if n.startswith('xl/worksheets/sheet')]
                e['sheets'] = {n: z.read(n).decode('utf-8', 'ignore').count('<row ') for n in sheets}
                strings = z.read('xl/sharedStrings.xml').decode('utf-8', 'ignore') if 'xl/sharedStrings.xml' in z.namelist() else ''
                e['mentions_rejected'] = 'Rejected' in strings
            except zipfile.BadZipFile:
                e['note'] = 'not an xlsx'
        if fmt == 'pdf':
            e['is_pdf'] = r.content[:4] == b'%PDF'
        exports[fmt] = e
    out['exports'] = exports
    out['delete_draft'] = c.call('DELETE', f'/api/v1/reports/{rid}')[0]
    out['passed'] = None  # counts in the export are compared by reading the output, see summary
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--roles', required=True)
    p.add_argument('--mission', required=True)
    p.add_argument('--otp', default=os.getenv('E2E_OTP'))
    p.add_argument('--only', nargs='*')
    p.add_argument('--out', default=str(Path(__file__).resolve().parent / 'results' / 'mf04_e2e'))
    a = p.parse_args()
    c = Ctx(a)
    since = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    tests = {'TKT-1': lambda: tkt_ai_validate(c), 'TKT-2': lambda: tkt_ai_review(c, since),
             'TKT-3': lambda: tkt_ops_sequential(c), 'TKT-4': lambda: tkt_ops_concurrent(c),
             'TKT-5': lambda: tkt_validate_concurrent(c), 'RE-1': lambda: reanalysis(c), 'RPT-1': lambda: report(c)}
    results = {}
    for name, fn in tests.items():
        if a.only and name not in a.only:
            continue
        try:
            results[name] = fn()
        except Exception as exc:
            results[name] = dict(error=f'{type(exc).__name__}: {exc}', passed=False)
        print(name, json.dumps(results[name], default=str))
    out = Path(a.out).expanduser() / f'{datetime.now():%Y-%m-%d_%H%M}_be_flows'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'be_flows.json').write_text(json.dumps(results, indent=2, default=str), encoding='utf-8')
    print('written to', out)


if __name__ == '__main__':
    main()
