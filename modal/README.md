# Modal — curated YOLO26m

Thư mục triển khai độc lập. Model **curated_yolo26m/weights/best.pt**, test mAP50 .8172. App mới `powerline-yolo26m-benchmark`, không thay app YOLO11s cũ. Không phụ thuộc thư mục Pi.

## Chuẩn bị trên desktop

Chạy từ thư mục `modal`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-prepare.txt
python prepare_model.py --checkpoint /path/curated_yolo26m/weights/best.pt
```

Chỉ dùng checkpoint tin cậy. Script kiểm tra 6 lớp và lưu models/best.pt, models/manifest.json kèm SHA-256. Chưa tự kiểm tra kiến trúc từ filename, cần chọn đúng checkpoint. models/best.pt được đóng kèm khi deploy.

## Deploy

```bash
modal setup
modal profile list
modal deploy --env main app.py
```

Dùng profile workspace **bunpmc**; --env main chỉ chọn environment. CLI trả endpoint URL. Dashboard https://modal.com/apps/bunpmc/main không phải endpoint inference. GPU T4, max1 container, scale-to-zero, scaledown60s; có chi phí build/storage/inference theo Modal.

Endpoint có proxy authentication. Tạo Proxy Token cho workspace/environment tương ứng, đặt MODAL_KEY/MODAL_SECRET và POWERLINE_CLOUD_URL trong môi trường; không commit secrets.

```bash
curl --fail -X POST "$POWERLINE_CLOUD_URL" \
  -H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET" \
  -H 'Content-Type: image/jpeg' --data-binary @sample.jpg
```

Request body JPEG/PNG bytes (không multipart), tối đa10MiB/20MP. Inference640/conf.25/FP32. Response có class_id,class_name,confidence,bbox_xyxy(pixel), width,height,predict_ms. Thời gian predict không bao gồm cold start/network, không thay benchmark RTX4060. Chạy độc lập bằng curl hoặc client bất kỳ; Pi là client tùy chọn.

Trạng thái: đã deploy lên bunpmc/main và gọi SDK inference thành công trên ảnh tổng hợp 640×640. Endpoint HTTP không credential trả 401. Chưa kiểm thử camera iPhone do URL LAN chưa kết nối được. Xem deployment_status.json; model vẫn cấp riêng, không nằm trong Git.

Tài liệu: https://modal.com/docs/guide/webhook-proxy-auth

## Checkpoint Camber

Từ thư mục modal, dùng Camber CLI đã đăng nhập:

```bash
mkdir -p models
camber stash cp stash://bunpmc/projects/curated_yolo26m/weights/best.pt models/best.pt
python prepare_model.py --checkpoint models/best.pt
```

## BE / RabbitMQ integration

Xem [BE_INTEGRATION.md](BE_INTEGRATION.md) và [rabbitmq.example.json](rabbitmq.example.json). Khuyến nghị bridge worker ở mạng BE gọi Modal qua HTTPS; không mở RabbitMQ lên Internet. Endpoint hiện chỉ xử lý ảnh, chưa có consumer RabbitMQ/video worker. app.py đã tách inference, media validation và settings vào powerline_ai/.

## Camera iPhone qua mạng LAN

Chạy client trên máy cùng Wi-Fi với iPhone; Modal không tự truy cập được IP LAN của điện thoại. Client lấy JPEG rồi gọi SDK Modal đã xác thực bằng profile máy này, không cần mở endpoint public hay phát cảnh báo BE.

```bash
pip install -r requirements.txt opencv-python
python -m modal setup  # chỉ khi máy chưa đăng nhập
python camera_test.py --camera-url http://IPHONE-IP:PORT/video --frames 10 --show
```

Bỏ `--show` trên máy headless; thêm `--output-dir camera-output` nếu muốn lưu ảnh có bbox và JSON (mặc định không lưu). `--interval 1` là thời gian nghỉ sau mỗi lần inference, không bảo đảm 1 FPS. Client mặc định chỉ gọi 10 frame; lần đầu có cold start. `q` đóng cửa sổ khi đang hiển thị hoặc Ctrl+C dừng. Với ảnh tĩnh dùng `--image /path/test.jpg`. SDK method `predict_jpeg` yêu cầu tài khoản Modal có quyền app, không phải API anonymous. Không cấp token quản trị Modal này cho Pi; Pi production dùng luồng riêng.
