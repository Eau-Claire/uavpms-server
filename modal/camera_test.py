"""Sample a local IP camera and run authenticated Modal inference (no BE alerts)."""
import argparse
import json
import time
from pathlib import Path
import cv2
import modal


def capture(url):
    # Open a fresh connection per sample so GPU latency does not build stale frames.
    cap=cv2.VideoCapture(url,cv2.CAP_FFMPEG,[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,5000,cv2.CAP_PROP_READ_TIMEOUT_MSEC,5000])
    try:
        if not cap.isOpened(): raise RuntimeError('Cannot connect to camera; check Wi-Fi, URL and app server')
        ok,frame=cap.read()
        if not ok: raise RuntimeError('Camera returned no decodable frame')
        return frame
    finally:cap.release()


def main():
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--camera-url')
    source.add_argument('--image',type=Path)
    parser.add_argument('--frames',type=int,default=10)
    parser.add_argument('--interval',type=float,default=1)
    parser.add_argument('--show',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if args.frames<1 or args.interval<0:parser.error('Invalid frames/interval')
    detector=modal.Cls.from_name('powerline-yolo26m-benchmark','Detector',environment_name='main')()
    if args.output_dir:args.output_dir.mkdir(parents=True,exist_ok=True)
    try:
        for index in range(1 if args.image else args.frames):
            frame=cv2.imread(str(args.image)) if args.image else capture(args.camera_url)
            if frame is None:raise RuntimeError('Cannot decode image')
            ok,encoded=cv2.imencode('.jpg',frame)
            if not ok:raise RuntimeError('Cannot encode frame')
            started=time.perf_counter()
            result=detector.predict_jpeg.remote(encoded.tobytes())
            result['roundtrip_ms']=(time.perf_counter()-started)*1000
            print(json.dumps(result),flush=True)
            for detection in result['detections']:
                x1,y1,x2,y2=map(round,detection['bbox_xyxy'])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
                cv2.putText(frame,f"{detection['class_name']} {detection['confidence']:.2f}",(x1,max(15,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            if args.output_dir:
                cv2.imwrite(str(args.output_dir/f'frame-{index:04d}.jpg'),frame)
                (args.output_dir/f'frame-{index:04d}.json').write_text(json.dumps(result,indent=2))
            if args.show:
                cv2.imshow('iPhone -> Modal YOLO26m',frame)
                if cv2.waitKey(1)&255==ord('q'):break
            if not args.image:time.sleep(args.interval)
    finally:
        if args.show:cv2.destroyAllWindows()

if __name__=='__main__':main()
