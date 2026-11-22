# Persistent Docker inference

CPU Docker alternative to the Modal GPU endpoint. It loads the same seed85 YOLO26m six-class checkpoint and imports the same `modal/powerline_ai` package as Modal, so thresholds and post-processing (`postprocess.py`, `thresholds.json`) are identical on both endpoints. It exposes a direct API for a trusted backend. Modal and the RabbitMQ bridge remain supported.

Place the verified checkpoint at `modal/models/best.pt`, then:

```bash
cd inference
cp .env.example .env
openssl rand -hex 32  # place this value in DETECT_API_TOKEN
docker compose build
docker compose up -d
curl http://127.0.0.1:8000/ready
```

Send encoded JPEG/PNG bytes, not multipart:

```bash
curl --fail-with-body http://127.0.0.1:8000/detect \
  -H "Authorization: Bearer $DETECT_API_TOKEN" \
  -H 'Content-Type: image/jpeg' --data-binary @sample.jpg
```

The response matches the model portion of the Modal contract: `model`, source `width`/`height`, `detections`, `raw_count` (boxes before post-processing) and `predict_ms`; each detection contains `class_id`, `class_name`, `confidence`, `bbox_xyxy` in source pixels and `merged` (number of model boxes merged into it).

The build context is the repository root (`compose.yaml` sets it), so the image can copy `modal/powerline_ai`. The root `.dockerignore` is an allowlist: only `inference/requirements.txt`, `inference/server.py` and `modal/powerline_ai/` enter the build. Rebuild the image after changing `thresholds.json`.

The service binds loopback by default. Put HTTPS Nginx/Caddy in front of it when called across hosts. Keep one Uvicorn worker because every worker loads another model copy. For GPU deployment, use a CUDA base image and NVIDIA Container Toolkit rather than changing only `DEVICE`.
