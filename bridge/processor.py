import tempfile
import time
from pathlib import Path
import cv2
from contracts import detections, event
from media import download
from modal_client import predict

def process(request, config):
    started=time.monotonic(); results=[]; metadata=None
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'media'
        download(request['FileUrl'],path,config['hosts'],config['max_bytes'])
        def infer(frame, index=None, timestamp=None):
            if frame is None: raise ValueError('Cannot decode media')
            if frame.shape[0]*frame.shape[1]>20_000_000: raise ValueError('Frame too large')
            ok,encoded=cv2.imencode('.jpg',frame)
            if not ok: raise ValueError('Cannot encode frame')
            output=predict(config['modal_url'],config['modal_key'],config['modal_secret'],encoded.tobytes())
            results.extend(detections(request,output,config['mapping'],index,timestamp))
        if request['MediaType'].lower()=='image':
            infer(cv2.imread(str(path)))
        else:
            cap=cv2.VideoCapture(str(path))
            try:
                fps=cap.get(cv2.CAP_PROP_FPS); count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if fps<=0 or count<=0 or count/fps>config['max_seconds']: raise ValueError('Invalid or too long video')
                metadata=dict(Duration=count/fps,Fps=fps,Width=int(cap.get(3)),Height=int(cap.get(4)))
                step=max(1,round(fps/config['sample_fps'])); sampled=0
                for index in range(0,int(count),step):
                    if sampled>=config['max_frames']: raise ValueError('Video sample limit exceeded')
                    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
                    if not ok: raise ValueError('Cannot decode sampled video frame')
                    infer(frame,index,index/fps);sampled+=1
            finally: cap.release()
    result=event(request,results,elapsed=(time.monotonic()-started)*1000,metadata=metadata)
    result['RawResult']={'VideoSamplingFps':config['sample_fps'] if metadata else None,'Tracking':False}
    return result
