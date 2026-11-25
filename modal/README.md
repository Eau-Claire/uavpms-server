# Modal — seed85 curated YOLO26m

Thư mục triển khai độc lập. Model **seed85 curated YOLO26m** (`stash://bunpmc/projects/finding_seed/seed85/curated_yolo26m/weights/best.pt`): val mAP50 .8406 / mAP50-95 .4797, test mAP50 .8118 / mAP50-95 .4573. Model cũ là `curated_yolo26m` seed42, xem `checkpoint_source.json`. App `powerline-yolo26m-benchmark`, không thay app YOLO11s cũ. Không phụ thuộc thư mục Pi.

## Hậu xử lý detection

Kết quả YOLO thô đi qua [powerline_ai/postprocess.py](powerline_ai/postprocess.py) trước khi trả về:

1. **Ngưỡng riêng từng class** thay cho một `conf=0.25` chung, để giảm box rác.
2. **Gộp box cùng class** khi IoU ≥ 0.45 hoặc box nhỏ nằm ≥ 70% trong box lớn: một lỗi chỉ còn một box (bao ngoài, giữ confidence cao nhất, field `merged` = số box đã gộp).
3. **Xử lý khác class**: NMS của YOLO chỉ lọc trong cùng class. Hai box lỗi khác class chồng nhau (IoU ≥ 0.55) thì giữ box tin cậy hơn. Box `insulator` gần như trùng box lỗi (IoU ≥ 0.8) thì bỏ `insulator`. Box lỗi nằm bên trong box `insulator` thì giữ cả hai.

Ngưỡng và việc bật/tắt bước 2, 3 được **chọn từ dữ liệu**: `tune_thresholds.py` chạy checkpoint một lần trên val, thử 4 tổ hợp (có/không gộp × có/không xử lý khác class), mỗi tổ hợp tune ngưỡng từng class theo F1 (IoU ≥ 0.5), rồi giữ tổ hợp có macro-F1 cao nhất. Kết quả ghi vào `powerline_ai/thresholds.json`, được đóng gói khi deploy (Modal và Docker CPU dùng chung file này):

```bash
python tune_thresholds.py --weights models/best.pt \
  --images /path/finding_nemo_human_reviewed_6cls/images/val \
  --labels /path/finding_nemo_human_reviewed_6cls/labels/val --device 0
# xác nhận một lần trên test, không tune lại theo test
python tune_thresholds.py --weights models/best.pt \
  --images /path/finding_nemo_human_reviewed_6cls/images/test \
  --labels /path/finding_nemo_human_reviewed_6cls/labels/test --evaluate powerline_ai/thresholds.json
```

Dự đoán thô được cache trong `.cache/` (gitignored), khóa theo SHA-256 checkpoint. Thiếu `thresholds.json` thì dùng mặc định trong `PostprocessConfig`. Đổi checkpoint thì phải tune lại.

**Cấu hình hiện tại** (seed85, tune trên val, xác nhận một lần trên test, IoU ≥ 0.5): ngưỡng insulator .35, insulator_broken .25, insulator_flashover .25, bird_nest .35, foreign_object .40, broken_strand .20; gộp box **bật**; xử lý khác class **tắt** (trên val không có cặp box lỗi khác class nào chồng IoU ≥ .55 nên bước này không đổi kết quả).

| Split | Cấu hình | Precision | Recall | Box sai (FP) | Box sót (FN) | Macro-F1 |
|---|---|---|---|---|---|---|
| val | conf .25, không hậu xử lý | .858 | .855 | 452 | 466 | .817 |
| val | thresholds.json | .892 | .840 | 327 | 512 | .824 |
| test | conf .25, không hậu xử lý | .862 | .844 | 385 | 444 | .790 |
| test | thresholds.json | .892 | .821 | 284 | 510 | .796 |

Hậu xử lý giảm khoảng 27% box sai, đổi lại sót thêm khoảng 2% vật thể. Nó không sửa được box đúng vị trí nhưng sai class (ví dụ foreign_object bị đoán insulator_broken); lỗi đó cần cải thiện model.

Script in bảng P/R/F1 theo class cho cấu hình trước (conf .25, không hậu xử lý) và từng tổ hợp sau khi tune. Response có thêm `raw_count` (số box trước hậu xử lý) và `merged` trong mỗi detection; các field cũ giữ nguyên.

## Chuẩn bị trên desktop

Chạy từ thư mục `modal`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-prepare.txt
python prepare_model.py --checkpoint /path/seed85/curated_yolo26m/weights/best.pt
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

## Giao diện Streamlit test ảnh

Gọi đúng endpoint đã deploy (`Detector.predict_jpeg`) bằng Modal profile của máy, không đặt token trong trang. Từ thư mục `modal`:

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-streamlit.txt
streamlit run test_ui/streamlit_app.py --server.address 127.0.0.1
```

Mở http://127.0.0.1:8501 . Upload ảnh JPEG/PNG rồi bấm **Gửi lên AI**, hoặc nhập thư mục dataset (chứa `images/` và `labels/`) để lấy ảnh mẫu val/test ngẫu nhiên: box màu là dự đoán, box xanh lá là label đúng. Có lọc theo class, lọc thêm confidence (chỉ phía hiển thị), bảng detection và tải JSON. Mỗi ảnh là một lần gọi GPU có tính phí; lần đầu chờ cold start khoảng 15 giây.

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

## Active learning (tự động train lại, gate, deploy, rollback)

Hai file: `al_cycle.py` (chạy ở máy/VPS điều phối) và `al_train_kernel.py` (chạy trên Kaggle GPU). Test: `tests/test_al_cycle.py`.

```bash
PY=/path/to/python-co-modal     # vd yolov11/.venv/bin/python; cần thêm `kaggle` và `camber` trên PATH
$PY al_cycle.py --env-file <file có CAMBER_API_KEY> init                          # một lần: tạo stash finding_model/
$PY al_cycle.py --env-file <...> cycle --feedback <thư mục images/ + labels/>     # một vòng đầy đủ
$PY al_cycle.py --env-file <...> fetch --cycle-id <id>                            # chạy tiếp nếu bị ngắt giữa chừng
$PY al_cycle.py --env-file <...> rollback                                         # về model trước
```

Một vòng `cycle`:
1. Kiểm tra feedback (YOLO 6 lớp, mỗi ảnh có file nhãn) và ngưỡng `min_feedback_images` (mặc định 30).
2. Đẩy dataset private `<user>/uavpms-al-cycle` (feedback + champion `models/best.pt` + `powerline_ai/` + `tune_thresholds.py`), push kernel private `<user>/uavpms-al-train` trên T4; xếp hàng quá 20 phút thì thử P100, GPU cuối thì chờ tiếp.
3. Kernel: dừng nếu ảnh feedback trùng val/test; fine-tune từ champion trên feedback + 2.000 ảnh replay (seed cố định), tune ngưỡng trên val bằng đúng code hậu xử lý đang chạy, lấy epoch cuối (lịch train cố định, không chọn checkpoint), chấm champion và challenger trên tập gate cố định `finding_nemo_human_reviewed_6cls/val`. Tập test không dùng ở đây để giữ sạch cho báo cáo.
4. Gate: MEAN_5 không giảm, mAP50-95 giảm ≤ 0,005, không lớp lỗi nào giảm AP50 > 0,02; kiểm tra champion trên Modal vẫn là champion lúc bắt đầu vòng.
5. Qua gate (và `auto_deploy`): lưu champion lên Camber → thay `models/best.pt`, `models/manifest.json` (tên `al_<cycle>_yolo26m`), `powerline_ai/thresholds.json` → `modal deploy` → smoke: endpoint phải báo đúng sha model + ngưỡng và trả detection hợp lệ < 3 s. Lỗi bất kỳ sau bước thay file → tự khôi phục champion cũ và deploy lại.
6. Ghi `stash://bunpmc/projects/finding_model/`: `registry.json` (current, previous, mọi vòng), `models/<sha16>/`, `cycles/<id>/` (metrics, gate report, log).

Mã thoát: 0 deployed (hoặc qua gate khi `auto_deploy=false`), 2 lỗi cấu hình/Kaggle, 3 trượt gate, 4 đã rollback, 5 rollback thất bại (cần người xử lý). Chỉ một vòng chạy tại một thời điểm (`.al_work/cycle.lock`). Không có gì chạy giữa các vòng: BE quyết định khi nào gọi.

Endpoint thêm `Detector.model_info()` (tên, sha model, sha ngưỡng). Tên model lấy từ `models/manifest.json` nếu có; bridge gửi BE đúng tên model mà endpoint trả về. Sau khi đổi model, chạy E2E với `perf/mf04_e2e.py --expect-model <tên mới>`.
