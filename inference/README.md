# Persistent Docker inference

CPU Docker alternative to the Modal GPU endpoint. It loads the same curated YOLO26m six-class checkpoint and exposes a direct API for a trusted backend. Modal and the RabbitMQ bridge remain supported.

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

The response matches the model portion of the Modal contract: `model`, source `width`/`height`, `detections`, and `predict_ms`; each detection contains `class_id`, `class_name`, `confidence`, and `bbox_xyxy` in source pixels.

The service binds loopback by default. Put HTTPS Nginx/Caddy in front of it when called across hosts. Keep one Uvicorn worker because every worker loads another model copy. For GPU deployment, use a CUDA base image and NVIDIA Container Toolkit rather than changing only `DEVICE`.
