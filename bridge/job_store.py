"""Single-worker durable inbox/results. Persist before RabbitMQ ACK."""
import hashlib
import json
import sqlite3

class JobStore:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, hash TEXT, request TEXT, attempts INTEGER DEFAULT 0, result TEXT, published INTEGER DEFAULT 0)')
        self.db.commit()

    def accept(self, request):
        body = json.dumps(request, sort_keys=True, separators=(',', ':'), allow_nan=False)
        digest = hashlib.sha256(body.encode()).hexdigest()
        old = self.db.execute('SELECT hash FROM jobs WHERE id=?', (request['RequestId'],)).fetchone()
        if old and old[0] != digest:
            raise ValueError('RequestId reused with different payload')
        self.db.execute('INSERT OR IGNORE INTO jobs(id,hash,request) VALUES(?,?,?)', (request['RequestId'], digest, body))
        self.db.commit()

    def pending(self):
        return self.db.execute('SELECT id,request,attempts,result FROM jobs WHERE published=0 ORDER BY rowid LIMIT 1').fetchone()

    def attempt(self, identifier):
        self.db.execute('UPDATE jobs SET attempts=attempts+1 WHERE id=?', (identifier,)); self.db.commit()

    def save_result(self, identifier, result):
        self.db.execute('UPDATE jobs SET result=? WHERE id=?', (json.dumps(result, allow_nan=False), identifier)); self.db.commit()

    def published(self, identifier):
        self.db.execute('UPDATE jobs SET published=1 WHERE id=?', (identifier,)); self.db.commit()
