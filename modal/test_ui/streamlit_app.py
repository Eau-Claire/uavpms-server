"""Local Streamlit test UI for the deployed Modal detector.

Calls Detector.predict_jpeg through the Modal SDK with this machine's Modal profile;
no token is placed in the page. Run from the modal/ folder:

  streamlit run test_ui/streamlit_app.py --server.address 127.0.0.1
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import time
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

APP_NAME = 'powerline-yolo26m-benchmark'
ENVIRONMENT = 'main'
CLASSES = ('insulator', 'insulator_broken', 'insulator_flashover',
           'bird_nest', 'foreign_object', 'broken_strand')
COLORS = {'insulator': (66, 135, 245), 'insulator_broken': (235, 64, 52),
          'insulator_flashover': (245, 166, 35), 'bird_nest': (139, 87, 42),
          'foreign_object': (200, 50, 220), 'broken_strand': (20, 190, 190)}
GT_COLOR = (40, 200, 40)
MAX_BYTES = 50 * 1024 * 1024  # keep in sync with powerline_ai.settings.MAX_BYTES
IMAGE_EXTS = ('.jpg', '.jpeg', '.png')

st.set_page_config(page_title='UAVPMS detector test', layout='wide')


@st.cache_resource(show_spinner=False)
def detector():
    import modal
    return modal.Cls.from_name(APP_NAME, 'Detector', environment_name=ENVIRONMENT)()


def call_endpoint(data: bytes) -> dict:
    started = time.perf_counter()
    result = detector().predict_jpeg.remote(data)
    result['roundtrip_ms'] = (time.perf_counter() - started) * 1000
    return result


def read_gt(label_path: Path, width: int, height: int):
    boxes = []
    if label_path.is_file():
        for line in label_path.read_text(encoding='utf-8-sig').splitlines():
            p = line.split()
            if len(p) == 5:
                c, x, y, w, h = int(float(p[0])), *map(float, p[1:])
                boxes.append(dict(class_name=CLASSES[c], bbox_xyxy=[(x - w / 2) * width, (y - h / 2) * height,
                                                                    (x + w / 2) * width, (y + h / 2) * height]))
    return boxes


def has_class(label_path: Path, class_id: int) -> bool:
    if not label_path.is_file():
        return False
    return any(line.split()[:1] == [str(class_id)] for line in label_path.read_text(encoding='utf-8-sig').splitlines())


def draw(image: Image.Image, detections, gt=None) -> Image.Image:
    canvas = image.convert('RGB').copy()
    d = ImageDraw.Draw(canvas)
    lw = max(2, canvas.width // 400)
    font = ImageFont.load_default(size=max(14, canvas.width // 60))
    for g in gt or []:
        d.rectangle(g['bbox_xyxy'], outline=GT_COLOR, width=lw)
    for det in detections:
        color = COLORS[det['class_name']]
        x1, y1, x2, y2 = det['bbox_xyxy']
        d.rectangle([x1, y1, x2, y2], outline=color, width=lw * 2)
        label = f"{det['class_name']} {det['confidence']:.2f}" + (f" x{det['merged']}" if det.get('merged', 1) > 1 else '')
        tx, ty = x1, max(0, y1 - font.size - 6)
        d.rectangle(d.textbbox((tx, ty), label, font=font), fill=color)
        d.text((tx, ty), label, fill=(255, 255, 255), font=font)
    return canvas


def to_payload(image_bytes: bytes) -> bytes:
    """The endpoint accepts JPEG/MPO/PNG up to MAX_BYTES; re-encode anything larger as JPEG."""
    if len(image_bytes) <= MAX_BYTES:
        return image_bytes
    with Image.open(io.BytesIO(image_bytes)) as im:
        buf = io.BytesIO()
        im.convert('RGB').save(buf, format='JPEG', quality=90)
        return buf.getvalue()


# ---------------------------------------------------------------- sidebar
st.sidebar.header('Endpoint')
st.sidebar.caption(f'Modal app `{APP_NAME}` / env `{ENVIRONMENT}`, gọi bằng Modal profile của máy này.')
show = st.sidebar.multiselect('Class hiển thị', CLASSES, default=list(CLASSES))
extra_conf = st.sidebar.slider('Lọc thêm confidence (chỉ để xem, server đã lọc theo ngưỡng riêng)', 0.0, 1.0, 0.0, 0.05)
st.sidebar.header('Ảnh mẫu có label')
dataset = st.sidebar.text_input('Thư mục dataset (chứa images/ và labels/)', '')
split = st.sidebar.selectbox('Split', ('val', 'test'))
n_samples = st.sidebar.number_input('Số ảnh mẫu', 1, 20, 4)
only_class = st.sidebar.selectbox('Chỉ lấy ảnh có class', ('(tất cả)',) + CLASSES)

if 'results' not in st.session_state:
    st.session_state.results = {}  # sha256 -> (name, bytes, result, label_path or None)

st.title('Test AI phát hiện lỗi đường dây')
st.write('Chọn ảnh rồi bấm **Gửi lên AI**. Box màu là dự đoán của model; box **xanh lá** là label đúng (chỉ có với ảnh mẫu từ dataset).')

col_up, col_sample = st.columns(2)
with col_up:
    uploads = st.file_uploader('Ảnh JPEG/MPO/PNG', type=['jpg', 'jpeg', 'png', 'mpo'], accept_multiple_files=True)
    send_uploads = st.button('Gửi lên AI', type='primary', disabled=not uploads)
with col_sample:
    st.write('Ảnh mẫu từ dataset (có label để so)')
    send_samples = st.button('Lấy ảnh mẫu ngẫu nhiên và gửi', disabled=not dataset)
    if st.button('Xóa kết quả'):
        st.session_state.results = {}

jobs = []
if send_uploads:
    jobs = [(f.name, f.getvalue(), None) for f in uploads]
if send_samples:
    images_dir, labels_dir = Path(dataset) / 'images' / split, Path(dataset) / 'labels' / split
    if not images_dir.is_dir():
        st.error(f'Không thấy {images_dir}')
    else:
        pool = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if only_class != '(tất cả)':
            pool = [p for p in pool if has_class(labels_dir / f'{p.stem}.txt', CLASSES.index(only_class))]
        for p in random.sample(pool, min(int(n_samples), len(pool))):
            jobs.append((p.name, p.read_bytes(), labels_dir / f'{p.stem}.txt'))
        if not pool:
            st.warning('Không có ảnh phù hợp.')

if jobs:
    bar = st.progress(0.0, text='Đang gửi... lần đầu có thể mất ~15 giây vì GPU khởi động lạnh')
    for i, (name, data, label) in enumerate(jobs, 1):
        key = hashlib.sha256(data).hexdigest()
        try:
            result = call_endpoint(to_payload(data))
            st.session_state.results[key] = (name, data, result, label)
        except Exception as exc:  # show the failure for this image and continue with the rest
            st.error(f'{name}: {type(exc).__name__}: {exc}')
        bar.progress(i / len(jobs), text=f'{i}/{len(jobs)} ảnh')
    bar.empty()

# ---------------------------------------------------------------- results
for key, (name, data, result, label) in reversed(list(st.session_state.results.items())):
    image = Image.open(io.BytesIO(data))
    dets = [d for d in result['detections'] if d['class_name'] in show and d['confidence'] >= extra_conf]
    gt = read_gt(label, image.width, image.height) if label else None
    st.divider()
    st.subheader(name)
    left, right = st.columns([3, 2])
    with left:
        st.image(draw(image, dets, gt), width='stretch')
    with right:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Box trả về', len(result['detections']))
        m2.metric('Box thô', result.get('raw_count', '—'))
        m3.metric('GPU predict', f"{result['predict_ms']:.0f} ms")
        m4.metric('Tổng (mạng)', f"{result['roundtrip_ms']:.0f} ms")
        if gt is not None:
            st.caption('Label đúng: ' + (', '.join(sorted(g['class_name'] for g in gt)) or 'không có box'))
        if dets:
            st.dataframe([{'class': d['class_name'], 'conf': round(d['confidence'], 3),
                           'gộp': d.get('merged', 1),
                           'box (x1,y1,x2,y2)': [round(v) for v in d['bbox_xyxy']]} for d in dets],
                         width='stretch', hide_index=True)
        else:
            st.info('Không có box nào (sau bộ lọc hiển thị).')
        st.download_button('Tải JSON', json.dumps(result, indent=2), file_name=f'{Path(name).stem}.json',
                           mime='application/json', key=f'dl-{key}')
        st.caption(f"Model: `{result['model']}` · ảnh {result['width']}×{result['height']}")
