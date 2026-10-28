# Contract bàn giao BE — chưa triển khai endpoint device-key

## Pi: endpoint mới

`POST /api/v1/edge/detections`, HTTPS; headers `X-Device-Key` và `Idempotency-Key` (UUID).
JSON giữ các field VisionDetectionDto: ClassName (mã DB), Confidence [0,1], Timestamp UTC, Latitude/Longitude (0,0 nếu chưa có GPS), BoundingBox pixel xyxy, TrackId=0 (chưa tracking). Không gửi DroneId ở chế độ device-key.

BE cần implement:
1. Authentication scheme riêng, không mở anonymous: SHA-256 key UTF-8, tìm thiết bị đang active, lấy UAV từ DB. Không dùng DroneId trong body làm danh tính. Rate limit theo thiết bị; không log credential.
2. Bảng device: Id, UavId, KeyHash (unique), IsActive, CreatedAt, RevokedAt. Key đủ ngẫu nhiên (script tạo 32 byte); hash không cần salt cho secret entropy cao. Chỉ lưu key gốc trên Pi, hash trong BE.
3. Bảng receipt: DeviceId + IdempotencyKey UNIQUE, PayloadHash, Status, Response. Cùng key/nội dung trả response cũ; khác nội dung HTTP 409. Không chỉ check-then-insert. Lưu receipt, anomaly và notification outbox trong cùng transaction; duplicate không phát notification lần hai.
4. Validate category tồn tại, confidence/bbox/time hợp lệ. Từ UAV tìm đúng mission đang Executing (không có hoặc có nhiều mission phù hợp phải báo lỗi rõ). IsEmergencyClass ở DB quyết định khẩn cấp. Không fallback category 1 cho mã sai.
5. Thành công HTTP 200/201 JSON `{"success":true,"recordId":"..."}` kể cả replay. 401/403 khi key sai/thu hồi; 429 rate limit. Sender giữ hàng đợi khi chưa được xác nhận. Không được trả success trước khi đã lưu bền vững.
6. Route gateway `/api/v{version}/edge/{everything}` tới AIInspectionService. Giữ JWT cho các API người dùng hiện tại.

Pi không tự refresh key. Thu hồi/cấp mới qua thao tác quản trị, không nhúng key vào Git. Pending alerts giữ EventId khi đổi key cùng device; receipt cần scope theo DeviceId ổn định, không theo hash key.

## Server bridge

Dùng các queue server.image/server.video và result hiện có, không cần public RabbitMQ. Worker chạy cùng mạng Docker BE, gọi Modal qua HTTPS bằng proxy token. Xem bridge/README.md.

BE result consumer phải dedup EventId bằng unique constraint và cập nhật run theo AnalysisId=RequestId; ghi trạng thái/detection/notification outbox trong transaction. Retry kết quả giữ EventId và detection Id. Lưu ý đây là at-least-once, không tự nhiên exactly-once.

Contract đã đọc từ checkout hiện tại: AIAnalysisResultEvent KHÔNG có AssetId; TowerId lấy từ request.TowerId hoặc legacy AssetId và đặt trên detection. InspectionId dùng Guid.Empty vì request không cung cấp inspection. Xác nhận handler bản server dùng AnalysisId/MediaId/MissionId như checkout trước khi bật thật.

## Cần điền khi nối

- CategoryCode thực tế cho 5 lỗi; IsEmergencyClass và confidence policy do nhóm chốt.
- Device/UAV + mission; key hash trong BE và file secret trên Pi.
- AMQP URL/vhost, mạng Docker, media host allowlist, Modal URL và proxy credentials.
- Kiểm thử duplicate, timeout sau commit, key bị revoke, sai category, storage lỗi, broker restart.

Không thay đổi BE repo hay deploy trong lần chuẩn bị này. Endpoint `/vision/detections/json` cũ vẫn JWT; endpoint `/edge/detections` ở trên là contract dự kiến, chưa tồn tại trên server.
