import tempfile
import time
from pathlib import Path
import cv2
from contracts import detections, event
from frames import best_per_defect, save_best_frames
from media import download
from modal_client import predict

MAX_FRAME_PIXELS = 20_000_000  # larger UAV frames are downscaled, not rejected


def fit(frame, max_pixels=MAX_FRAME_PIXELS):
    """Downscale frames above max_pixels. The model letterboxes to 640 px anyway, and boxes are sent to the
    backend normalized to the frame Modal received, so they stay valid for the original image."""
    h, w = frame.shape[:2]
    if h * w <= max_pixels:
        return frame
    scale = (max_pixels / (h * w)) ** 0.5
    return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def encode(frame):
    ok, encoded = cv2.imencode('.jpg', fit(frame))
    if not ok: raise ValueError('Cannot encode frame')
    return encoded.tobytes()


def process(request, config):
    started=time.monotonic(); results=[]; metadata=None; served=[]   # model name reported by the endpoint
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'media'
        download(request['FileUrl'],path,config['hosts'],config['max_bytes'])
        def infer(frame, index=None, timestamp=None):
            if frame is None: raise ValueError('Cannot decode media')
            output=predict(config['modal_url'],config['modal_key'],config['modal_secret'],encode(frame))
            if output.get('model'): served.append(output['model'])
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
                frames_with_defects=[]  # (sample number, detections of that frame)
                for index in range(0,int(count),step):
                    if sampled>=config['max_frames']: raise ValueError('Video sample limit exceeded')
                    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
                    if not ok: raise ValueError('Cannot decode sampled video frame')
                    before=len(results)
                    infer(frame,index,index/fps)
                    if len(results)>before:
                        # Same encoding as the frame sent to Modal, so the saved frame matches the boxes.
                        (Path(folder)/f'{index}.jpg').write_bytes(encode(frame))
                        frames_with_defects.append((sampled,results[before:]))
                    sampled+=1
            finally: cap.release()
            # One detection per defect: the highest-confidence one, with its frame saved as ImageUrl.
            results=best_per_defect(frames_with_defects)
            if config['frame_store']:
                save_best_frames(results,request,Path(folder),config['frame_store'])
    result=event(request,results,elapsed=(time.monotonic()-started)*1000,metadata=metadata,
                 model=served[-1] if served else None)
    result['RawResult']={'VideoSamplingFps':config['sample_fps'] if metadata else None,'Tracking':False}
    return result
