"""Bounded downloads from explicitly configured trusted storage hosts; no redirects."""
import time
from urllib.parse import urlparse
import requests

def download(url, path, allowed_hosts, max_bytes):
    parsed=urlparse(url)
    if parsed.scheme not in ('http','https') or parsed.hostname not in allowed_hosts or parsed.username or parsed.password:
        raise ValueError('Media URL is not an approved storage origin')
    started=time.monotonic(); total=0
    with requests.get(url,stream=True,timeout=(10,30),allow_redirects=False) as response:
        if response.status_code != 200: raise ValueError('Media download failed')
        with open(path,'wb') as target:
            for chunk in response.iter_content(65536):
                total+=len(chunk)
                if total>max_bytes or time.monotonic()-started>120: raise ValueError('Media limit exceeded')
                target.write(chunk)
