"""Loopback-only manual test UI. Modal credentials stay on this machine."""
import asyncio
import base64
import json
import math
import tempfile
import threading
import time
from pathlib import Path
from uuid import uuid4

import cv2
import modal
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

app = FastAPI()
state = {'job': None, 'busy': False}
guard = threading.Lock()
MAX_BYTES = 100 * 1024 * 1024

@app.middleware('http')
async def local_only(request, call_next):
    # Reject cross-origin mutations; no CORS or public binding.
    if request.headers.get('host') not in ('127.0.0.1:8765', 'localhost:8765'):
        return HTMLResponse('Invalid host', status_code=403)
    if request.method == 'POST' and request.headers.get('origin') not in ('http://127.0.0.1:8765', 'http://localhost:8765'):
        return HTMLResponse('Invalid origin', status_code=403)
    return await call_next(request)

@app.get('/')
def index():
    return HTMLResponse(Path(__file__).with_name('index.html').read_text())

@app.get('/api/job/{identifier}')
def status(identifier: str):
    with guard:
        job=state['job']
        if not job or job['id']!=identifier: raise HTTPException(404,'Job not found')
        return dict(job, results=list(job['results']))

@app.post('/api/job/{identifier}/stop')
def stop(identifier: str):
    with guard:
        if not state['job'] or state['job']['id']!=identifier: raise HTTPException(404,'Job not found')
        state['job']['stop']=True
    return {'ok':True}

def annotated(frame, detections):
    height,width=frame.shape[:2]
    scale=min(1,960/max(width,height))
    canvas=cv2.resize(frame,(round(width*scale),round(height*scale)))
    for d in detections:
        x1,y1,x2,y2=[round(v*scale) for v in d['bbox_xyxy']]
        cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,210,255),2)
        cv2.putText(canvas,f"{d['class_name']} {d['confidence']:.2f}",(x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,210,255),2)
    ok,encoded=cv2.imencode('.jpg',canvas,[cv2.IMWRITE_JPEG_QUALITY,80])
    if not ok: raise ValueError('Cannot encode preview')
    return base64.b64encode(encoded).decode()

def process(job,path,kind,fps,limit):
    cap=None
    try:
        detector=modal.Cls.from_name('powerline-yolo26m-benchmark','Detector',environment_name='main')()
        if kind=='image':
            frames=[(0,None)]
        else:
            cap=cv2.VideoCapture(str(path));native=cap.get(cv2.CAP_PROP_FPS);total=cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if not math.isfinite(native) or not math.isfinite(total) or native<=0 or total<=0: raise ValueError('Video không đọc được FPS hoặc duration')
            step=max(1,round(native/fps))
            indices=range(0,int(total),step)
            frames=[(i,i/native) for i in list(indices[:limit])]
            job['note']=f'Video {total/native:.1f}s; lấy mẫu {len(frames)} frame đầu, không tracking và không đánh giá mọi frame.'
        job['total']=len(frames)
        for index,seconds in frames:
            if job['stop']:break
            if cap is None:frame=cv2.imread(str(path))
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
                if not ok:raise ValueError('Không đọc được frame video')
            if frame is None:raise ValueError('Không đọc được ảnh')
            if frame.shape[0]*frame.shape[1]>20_000_000:raise ValueError('Ảnh/frame vượt 20 megapixel')
            ok,encoded=cv2.imencode('.jpg',frame)
            if not ok or len(encoded)>10*1024*1024:raise ValueError('Frame JPEG vượt giới hạn 10 MiB')
            start=time.perf_counter()
            result=detector.predict_jpeg.remote(encoded.tobytes())
            result.update(frame_index=index,timestamp_seconds=seconds,roundtrip_ms=round((time.perf_counter()-start)*1000,1))
            preview=annotated(frame,result['detections'])
            with guard:job['results'].append({'prediction':result,'preview':preview})
        job['status']='stopped' if job['stop'] else 'done'
    except Exception as exc:
        # Do not return SDK traceback, local credentials or remote internal paths.
        job['error']=f'Xử lý thất bại ({type(exc).__name__}). Kiểm tra định dạng file, Modal profile và log server.'
        job['status']='error'
    finally:
        if cap is not None:cap.release()
        path.unlink(missing_ok=True)
        with guard:state['busy']=False

@app.post('/api/analyze')
async def analyze(request:Request,kind:str='image',fps:float=1,limit:int=10):
    if kind not in ('image','video') or not math.isfinite(fps) or not .1<=fps<=5 or not 1<=limit<=30:
        raise HTTPException(400,'Invalid options')
    with guard:
        if state['busy']:raise HTTPException(409,'Đang xử lý một file; hãy chờ hoặc bấm Dừng')
        state['busy']=True
    path=None
    try:
        with tempfile.NamedTemporaryFile(delete=False,suffix='.media') as handle:
            path=Path(handle.name);size=0
            async for chunk in request.stream():
                size+=len(chunk)
                if size>MAX_BYTES:raise HTTPException(413,'File vượt 100 MiB')
                handle.write(chunk)
        if not size:raise HTTPException(400,'File rỗng')
        job=dict(id=str(uuid4()),status='processing',results=[],total=1 if kind=='image' else limit,stop=False,note='')
        with guard:state['job']=job
        threading.Thread(target=process,args=(job,path,kind,fps,limit),daemon=True).start()
        return {'id':job['id']}
    except BaseException:
        if path:path.unlink(missing_ok=True)
        with guard:state['busy']=False
        raise

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8765)
