# Modal — seed85 curated YOLO26m

Thư mục triển khai độc lập. Model **seed85 curated YOLO26m** (`stash://bunpmc/projects/finding_seed/seed85/curated_yolo26m/weights/best.pt`): val mAP50 .8406 / mAP50-95 .4797, test mAP50 .8118 / mAP50-95 .4573. Model cũ là `curated_yolo26m` seed42, xem `checkpoint_source.json`. App `powerline-yolo26m-benchmark`, không thay app YOLO11s cũ. Không phụ thuộc thư mục Pi.

## Hậu xử lý detection

Kết quả YOLO thô đi qua [powerline_ai/postprocess.py](powerline_ai/postprocess.py) trước khi trả về:

1. **Ngưỡng riêng từng class** thay cho một `conf=0.25` chung, để giảm box rác.
2. **Gộp box cùng class** khi IoU ≥ 0.45 hoặc box nhỏ nằm ≥ 70% trong box lớn: một lỗi chỉ còn một box (bao ngoài, giữ confidence cao nhất, field `merged` = số box đã gộp).
3. **Xử lý khác class**: NMS của YOLO chỉ lọc trong cùng class. Hai box lỗi khác class chồng nhau (IoU ≥ 0.55) thì giữ box tin cậy hơn. Box `insulator` gần như trùng box lỗi (IoU ≥ 0.8) thì bỏ `insulator`. Box lỗi nằm bên trong box `insulator` thì giữ cả hai.

Ngưỡng mặc định nằm trong `PostprocessConfig`. Nên tune trên **val** (không dùng test) bằng checkpoint đang deploy. Script ghi `powerline_ai/thresholds.json`, file này được đóng gói khi deploy:

```bash
python tune_thresholds.py --weights models/best.pt \
  --images /path/finding_nemo_human_reviewed_6cls/images/val \
  --labels /path/finding_nemo_human_reviewed_6cls/labels/val
```

Script in bảng P/R/F1 theo class cho cấu hình trước (conf .25, không hậu xử lý) và sau khi tune. Response có thêm `raw_count` (số box trước hậu xử lý); các field cũ giữ nguyên.

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

Request body JPEG/PNG bytes (không multipart), tối đa10MiB/20MP. Inference 640/FP32, NMS IoU .6, ngưỡng theo class và hậu xử lý như trên. Response có class_id,class_name,confidence,bbox_xyxy(pixel), width,height,predict_ms. Thời gian predict không bao gồm cold start/network, không thay benchmark RTX4060. Chạy độc lập bằng curl hoặc client bất kỳ; Pi là client tùy chọn.

Trạng thái: đã deploy lên bunpmc/main và gọi SDK inference thành công trên ảnh tổng hợp 640×640. Endpoint HTTP không credential trả 401. Xem deployment_status.json; model vẫn cấp riêng, không nằm trong Git.

Tài liệu: https://modal.com/docs/guide/webhook-proxy-auth

## Checkpoint Camber

Từ thư mục modal, dùng Camber CLI đã đăng nhập:

```bash
mkdir -p models
camber stash cp stash://bunpmc/projects/finding_seed/seed85/curated_yolo26m/weights/best.pt models/best.pt
python prepare_model.py --checkpoint models/best.pt
```

## BE / RabbitMQ integration

Xem [BE_INTEGRATION.md](BE_INTEGRATION.md) và [rabbitmq.example.json](rabbitmq.example.json). Khuyến nghị bridge worker ở mạng BE gọi Modal qua HTTPS; không mở RabbitMQ lên Internet. Endpoint hiện chỉ xử lý ảnh, chưa có consumer RabbitMQ/video worker. app.py đã tách inference, media validation và settings vào powerline_ai/.

## Giao diện web test ảnh/video

Từ thư mục `modal`, tạo venv nếu chưa có (không cài pip vào Python hệ thống):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-ui.txt
python -m modal setup # bỏ qua nếu đã đăng nhập đúng workspace
python test_ui/server.py
```

Mở http://127.0.0.1:8765 . Chọn ảnh JPEG/PNG hoặc video, bấm Gửi lên AI. Video mặc định 1 frame/s, tối đa 10 frame đầu (có thể chọn 1–30); hiển thị từng frame có bbox và tải JSON. Có nút dừng sau request hiện tại. Đây là UI test local dùng SDK Modal/profile của máy, không phải UI public/BE; token không đưa vào JavaScript. Preview file là local, chỉ frame được chọn mới gửi GPU. Mỗi lần chỉ xử lý một file; không tự chạy lại khi đổi lựa chọn. Giới hạn upload 100 MiB, mỗi frame 20 MP/10 MiB JPEG; không kiểm tra toàn video khi giới hạn số frame thấp.

File upload tạm được xóa sau xử lý; kết quả giữ trong RAM tới job tiếp theo hoặc restart. Không có durable queue, multi-user, tự retry hay đồng bộ BE. Chỉ bind loopback, không expose service này ra Internet. Đóng tab không tự hủy job; bấm Dừng hoặc dừng process server.
