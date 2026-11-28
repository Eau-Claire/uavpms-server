import sys
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from frames import best_per_defect, iou


def det(cls, x, conf, frame):
    return dict(Class=cls, Confidence=conf, FrameIndex=frame, BoundingBox=dict(X=x, Y=.1, Width=.2, Height=.2))


class GroupingTests(unittest.TestCase):
    def test_iou(self):
        a=dict(X=0,Y=0,Width=.2,Height=.2)
        self.assertAlmostEqual(iou(a,a),1)
        self.assertEqual(iou(a,dict(X=.5,Y=.5,Width=.1,Height=.1)),0)

    def test_same_defect_across_frames_keeps_best(self):
        frames=[(0,[det('bird_nest',.10,.6,0)]),(1,[det('bird_nest',.12,.9,30)]),(2,[det('bird_nest',.14,.7,60)])]
        best=best_per_defect(frames)
        self.assertEqual(len(best),1); self.assertEqual(best[0]['FrameIndex'],30)

    def test_different_class_or_place_stays_separate(self):
        frames=[(0,[det('bird_nest',.1,.6,0),det('foreign_object',.1,.8,0),det('bird_nest',.7,.5,0)])]
        self.assertEqual(len(best_per_defect(frames)),3)

    def test_gap_and_reappearance(self):
        # one missed sample keeps the group; a long gap starts a new defect
        short=[(0,[det('bird_nest',.1,.6,0)]),(2,[det('bird_nest',.1,.8,60)])]
        self.assertEqual(len(best_per_defect(short)),1)
        long=[(0,[det('bird_nest',.1,.6,0)]),(5,[det('bird_nest',.1,.8,150)])]
        self.assertEqual(len(best_per_defect(long)),2)

    def test_two_defects_in_one_frame_do_not_merge(self):
        frames=[(0,[det('bird_nest',.1,.6,0),det('bird_nest',.25,.7,0)]),(1,[det('bird_nest',.1,.9,30),det('bird_nest',.25,.5,30)])]
        best=sorted(best_per_defect(frames),key=lambda d:d['BoundingBox']['X'])
        self.assertEqual([d['Confidence'] for d in best],[.9,.7])


class ProcessTests(unittest.TestCase):
    """process() with a fake Modal endpoint, fake download and fake MinIO."""

    def setUp(self):
        try:
            import numpy, cv2  # noqa: F401
        except ImportError:
            self.skipTest('numpy/opencv not installed')
        import processor
        self.processor = processor
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.sent = []      # JPEGs sent to Modal, in order
        self.saved = {}     # frame index -> JPEG uploaded to MinIO
        self.confidence = []  # bird_nest confidence returned for each call; 0 = nothing detected

    def fake_predict(self, url, key, secret, jpeg):
        self.sent.append(jpeg)
        confidence = self.confidence[len(self.sent) - 1]
        found = [] if confidence == 0 else [dict(class_name='bird_nest', confidence=confidence, bbox_xyxy=[16, 12, 48, 36])]
        return dict(model='m', width=160, height=120, detections=found)

    def run_process(self, media_type, media_file, frame_store):
        test = self

        class FakeStore:
            def save(self, request, frame_index, jpeg):
                test.saved[frame_index] = jpeg
                return f'https://s3.test/uav-media/frames/{frame_index}.jpg'

        self.processor.download = lambda url, path, hosts, limit: Path(path).write_bytes(media_file.read_bytes())
        self.processor.predict = self.fake_predict
        request = dict(RequestId=str(uuid4()), MediaId=str(uuid4()), MissionId=str(uuid4()),
                       MediaType=media_type, FileUrl='https://s3.test/media')
        config = dict(hosts=set(), max_bytes=1 << 30, modal_url='', modal_key='', modal_secret='',
                      mapping={'bird_nest': 'NEST'}, max_seconds=120, max_frames=120, sample_fps=1,
                      frame_store=FakeStore() if frame_store else None)
        return self.processor.process(request, config)

    def make_video(self):
        import cv2, numpy as np
        video = Path(self.folder.name) / 'v.avi'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (160, 120))
        for i in range(50):  # 5 s at 10 fps -> 5 samples at 1 fps (frames 0, 10, 20, 30, 40)
            writer.write(np.full((120, 160, 3), i * 5, np.uint8))
        writer.release()
        return video

    def test_video_keeps_best_detection_and_saves_its_frame(self):
        self.confidence = [.5, .9, .6, 0, 0]  # same bird nest in frames 0, 10, 20
        result = self.run_process('Video', self.make_video(), frame_store=True)
        self.assertEqual(len(self.sent), 5)
        self.assertEqual(len(result['Results']), 1)
        best = result['Results'][0]
        self.assertEqual((best['FrameIndex'], best['Confidence']), (10, .9))
        self.assertEqual(best['ImageUrl'], 'https://s3.test/uav-media/frames/10.jpg')
        self.assertEqual(list(self.saved), [10])
        self.assertEqual(self.saved[10], self.sent[1])  # saved frame is exactly the frame Modal saw

    def test_video_without_minio_still_groups_but_saves_nothing(self):
        self.confidence = [.5, .9, .6, 0, 0]
        result = self.run_process('Video', self.make_video(), frame_store=False)
        self.assertEqual(len(result['Results']), 1)
        self.assertNotIn('ImageUrl', result['Results'][0])
        self.assertEqual(self.saved, {})

    def test_image_flow_unchanged(self):
        import cv2, numpy as np
        image = Path(self.folder.name) / 'i.jpg'
        cv2.imwrite(str(image), np.zeros((120, 160, 3), np.uint8))
        self.confidence = [.7]
        result = self.run_process('Image', image, frame_store=True)
        self.assertEqual(len(result['Results']), 1)
        self.assertNotIn('ImageUrl', result['Results'][0])  # BE keeps using the uploaded image
        self.assertIsNone(result['Results'][0]['FrameIndex'])
        self.assertEqual(self.saved, {})


if __name__=='__main__':unittest.main()
