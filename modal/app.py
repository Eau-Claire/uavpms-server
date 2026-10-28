"""Modal deployment wiring only. Run modal deploy --env main app.py."""
from pathlib import Path
import modal
from fastapi import Request

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'models/best.pt'
if not MODEL.is_file():
    raise FileNotFoundError(f'Run prepare_model.py first: {MODEL}')
app = modal.App('powerline-yolo26m-benchmark')
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('libgl1', 'libglib2.0-0')
         .pip_install('ultralytics==8.4.172', 'fastapi', 'pillow')
         .add_local_dir(ROOT / 'powerline_ai', '/root/powerline_ai')
         .add_local_file(MODEL, '/models/best.pt'))

@app.cls(image=image, gpu='T4', max_containers=1, scaledown_window=60, timeout=120)
class Detector:
    @modal.enter()
    def load(self):
        from powerline_ai.inference import YoloDetector
        self.detector = YoloDetector()

    @modal.fastapi_endpoint(method='POST', requires_proxy_auth=True)
    async def detect(self, request: Request):
        from powerline_ai.media import read_image
        return self.detector.predict(await read_image(request))
