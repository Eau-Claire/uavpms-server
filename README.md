# uavpms-server

AI server deployment, độc lập với repo training và [Pi edge](https://github.com/Eau-Claire/uavpms-edge).

- [modal/](modal/README.md): GPU endpoint curated YOLO26m, proxy authentication.
- [bridge/](bridge/README.md): worker cạnh BE, RabbitMQ → Modal → RabbitMQ result; Docker Compose và durable job store.
- [BE_HANDOFF.md](BE_HANDOFF.md): contract/mapping và yêu cầu idempotency phía BE.

Clone repo, chép checkpoint riêng vào `modal/models/best.pt`, theo README Modal để deploy. Sau đó điền URL/token vào `bridge/.env`, map category DB, chọn mạng Docker BE và chạy Compose. Không có checkpoint/secret trong Git. Chưa chạy end-to-end môi trường thật; các test ở đây là offline.

```bash
python3 -m unittest discover -s bridge/tests -v
python3 -m unittest discover -s modal/tests -v
```

BE vẫn nằm repo UavPms_Org. Không mở public RabbitMQ để Modal truy cập; bridge gọi outbound HTTPS.
