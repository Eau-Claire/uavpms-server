"""Modal deployment wiring only. Run modal deploy --env main app.py."""
from pathlib import Path
import time
import modal
from fastapi import Request

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'models/best.pt'
if modal.is_local() and not MODEL.is_file():
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
        self.started_at = time.time()
        self.detector = YoloDetector()
        self.load_s = time.time() - self.started_at

    def _run(self, data):
        from powerline_ai.media import decode_image
        t0 = time.perf_counter()
        frame = decode_image(data)
        return self.detector.predict(frame, decode_ms=(time.perf_counter() - t0) * 1000)

    @modal.fastapi_endpoint(method='POST', requires_proxy_auth=True)
    async def detect(self, request: Request):
        from powerline_ai.media import read_body
        return self._run(await read_body(request))

    @modal.method()
    def predict_jpeg(self, data: bytes):
        """Authenticated SDK path for image and sampled-video tests; uses Modal profile."""
        return self._run(data)

    @modal.method()
    def resource_stats(self):
        """Container resource snapshot for performance tests; no inference."""
        import resource, subprocess, torch
        smi = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.strip()
        name, util, used, total = [x.strip() for x in smi.split(',')] if smi else ('', '', '', '')
        rss_kb = next((int(line.split()[1]) for line in open('/proc/self/status') if line.startswith('VmRSS:')), 0)
        return dict(gpu=name, gpu_util_pct=float(util or 0), gpu_mem_used_mib=float(used or 0),
                    gpu_mem_total_mib=float(total or 0),
                    torch_allocated_mib=torch.cuda.memory_allocated() / 2**20,
                    torch_max_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                    rss_mib=rss_kb / 1024, max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                    served=self.detector.served, uptime_s=time.time() - self.started_at, model_load_s=self.load_s)
