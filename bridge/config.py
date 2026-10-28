import json
import os
from pathlib import Path
from urllib.parse import urlparse

def load():
    mapping=json.loads(Path(os.environ.get('CATEGORY_MAP','categories.json')).read_text())
    expected={'insulator','insulator_broken','insulator_flashover','bird_nest','foreign_object','broken_strand'}
    if set(mapping)!=expected or any(v is not None and (not v or v.startswith('REPLACE')) for v in mapping.values()):
        raise ValueError('Configure all six class mappings; null explicitly excludes a class')
    url=os.environ['MODAL_URL']
    if urlparse(url).scheme!='https': raise ValueError('Modal URL must be HTTPS')
    hosts=set(filter(None,os.environ['MEDIA_ALLOWED_HOSTS'].split(',')))
    if not hosts: raise ValueError('Configure trusted storage hosts')
    return dict(mapping=mapping,hosts=hosts,modal_url=url,modal_key=os.environ['MODAL_KEY'],
                modal_secret=os.environ['MODAL_SECRET'],max_bytes=200*1024*1024,
                max_seconds=120,max_frames=120,sample_fps=1)
