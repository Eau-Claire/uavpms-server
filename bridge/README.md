# RabbitMQ → Modal bridge

Chạy cạnh BE trên Linux/Docker, không chạy consumer trong Modal. Hỗ trợ ảnh và video lấy mẫu 1 fps (tối đa 120 s/120 frame/200 MiB); không tracking.
Video: detection cùng lớp, box chồng nhau (IoU ≥ 0,3) ở các frame lấy mẫu gần nhau được coi là cùng một lỗi; mỗi lỗi chỉ giữ detection có confidence cao nhất, frame đó (sạch, không vẽ box) được lưu lên MinIO `<bucket>/frames/<MediaId>/<RequestId>/<FrameIndex>.jpg` và gửi về BE trong `ImageUrl` (frames.py). Cần các biến `MINIO_*` (cùng tên với .env của BE); thiếu thì không lưu frame. Ảnh tĩnh không đổi. Giới hạn nằm trong config.py. Fail toàn job nếu vượt giới hạn, không trả kết quả đầy đủ giả từ video bị cắt ngắn.

```bash
cp .env.example .env
cp categories.example.json categories.json
# Điền thông tin local; category map phải là mã DB thật.
# BE_DOCKER_NETWORK phải là mạng Docker đang có broker và storage.
BE_DOCKER_NETWORK=uavpms_default docker compose up -d --build
```

`.env` chứa AMQP URL, Modal URL, Modal-Key/Secret; chỉ nằm trên host. Quyền broker: đọc 2 queue request, đọc metadata queue/exchange và publish result. Topology do BE tạo; worker kiểm tra queue/exchange tồn tại, không tự sửa declaration.

Module:
- worker.py: consumer, ACK, publish confirms/mandatory, orchestration.
- job_store.py: durable SQLite inbox + result outbox, dedup RequestId/payload hash.
- media.py: tải từ allowlist host, không follow redirect, giới hạn byte/thời gian.
- modal_client.py: HTTPS inference, proxy auth.
- processor.py: ảnh/video sampling.
- frames.py: gom detection theo lỗi trong video, lưu frame tốt nhất lên MinIO.
- contracts.py: normalization, category map, deterministic detection IDs.

ACK request ngay sau khi commit SQLite. Volume `bridge-state` là bắt buộc: mất volume sẽ mất job đã ACK. Chỉ chạy một replica; file lock chặn hai worker dùng cùng volume. Không scale ngang với hai volume riêng. Disk chứa job và signed media URL, cần bảo vệ và theo dõi dung lượng; chưa có retention/cleanup tự động hoặc giới hạn backlog.

Mỗi job tối đa 3 lần xử lý, attempts giữ qua restart. Sau đó lưu/publish Failed. Kết quả Completed/Failed được lưu trước publish; retry publish giữ nguyên EventId và body. Nếu crash sau broker confirm trước đánh dấu published thì có thể publish lại: BE phải dedup. Crash giữa inference và lưu result có thể tính GPU lại; chưa có checkpoint từng frame. Lỗi payload/RequestId conflict bị reject vào DLX; BE phải cấu hình queue DLQ và theo dõi.

Heartbeat broker không bị chặn bởi inference: xử lý chạy thread riêng. Worker xử lý song song tối đa `WORKERS` job (mặc định 3, 1..8) để chồng thời gian mạng tới storage và Modal; một job không bao giờ chạy hai lần cùng lúc. Video dài chiếm một slot. Frame trên 20 MP được thu nhỏ trước khi gửi (không từ chối); box gửi BE chuẩn hóa 0..1 nên vẫn đúng với ảnh gốc. Kết nối HTTPS tới Modal được giữ (keep-alive) theo từng thread. Storage allowlist chỉ dành cho host do nhóm kiểm soát; không cho arbitrary public host/DNS. Download không kèm auth header; cần FileUrl truy cập nội bộ hoặc signed URL còn hạn. Worker không tự xin lại signed URL hết hạn.

Đã kiểm thử offline các contract và durable state. Chưa chạy end-to-end RabbitMQ/Modal/BE thật vì thiếu credentials và category map. Pi device-key contract ở ../BE_HANDOFF.md. Modal auth: https://modal.com/docs/guide/webhook-proxy-auth ; AMQP client: https://docs.aio-pika.com/quick-start.html .
