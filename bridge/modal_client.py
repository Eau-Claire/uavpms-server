"""Modal credentials never sent to storage hosts."""
import threading
import requests

MAX_SEND_BYTES = 50 * 1024 * 1024  # same limit as the Modal endpoint (powerline_ai.settings.MAX_BYTES)
_local = threading.local()


def _session():
    # One keep-alive session per worker thread: avoids a new TLS handshake to Modal on every frame.
    if not hasattr(_local, 'session'):
        _local.session = requests.Session()
    return _local.session


def predict(url, key, secret, image):
    if len(image) > MAX_SEND_BYTES: raise ValueError('Frame exceeds Modal byte limit')
    response=_session().post(url,data=image,headers={'Content-Type':'image/jpeg','Modal-Key':key,'Modal-Secret':secret},
                             timeout=(10,120),allow_redirects=False)
    if response.status_code != 200: raise RuntimeError('Modal inference request failed')
    return response.json()
