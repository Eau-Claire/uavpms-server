"""Stage a trusted six-class Modal checkpoint on desktop."""
import argparse, hashlib, json, shutil
from pathlib import Path
NAMES=['insulator','insulator_broken','insulator_flashover','bird_nest','foreign_object','broken_strand']
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    a=p.parse_args()
    if not a.checkpoint.is_file():p.error(f'Missing checkpoint: {a.checkpoint}')
    from ultralytics import YOLO
    model=YOLO(str(a.checkpoint))
    names=[model.names[i] for i in range(len(model.names))]
    if names!=NAMES:raise ValueError(f'Expected six benchmark classes, got {names}')
    root=Path(__file__).resolve().parent/'models';root.mkdir(exist_ok=True)
    dst=root/'best.pt'
    if a.checkpoint.resolve()!=dst.resolve():shutil.copy2(a.checkpoint,dst)
    manifest={'source':str(a.checkpoint.resolve()),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'names':names}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Prepared model in',root)
if __name__=='__main__':main()
