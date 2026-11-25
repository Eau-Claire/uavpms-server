# Build context: the uavpms-server repo root.  docker build -f al_worker/Dockerfile .
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY al_worker/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
# Serving code + AL cycle (seeded into the /data/modal volume by the entrypoint)
COPY modal/app.py modal/al_cycle.py modal/al_train_kernel.py modal/tune_thresholds.py /opt/modal-src/
COPY modal/powerline_ai /opt/modal-src/powerline_ai
COPY al_worker/worker.py al_worker/entrypoint.sh /app/
# camber CLI: mounted from the host at /usr/local/bin/camber (static binary)
ENTRYPOINT ["/app/entrypoint.sh"]
