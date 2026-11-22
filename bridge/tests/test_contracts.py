import json
import sys
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from contracts import validate, detections, event
from job_store import JobStore

class BridgeTests(unittest.TestCase):
    def request(self):
        return dict(RequestId=str(uuid4()),MediaId=str(uuid4()),MissionId=str(uuid4()),MediaType='Image',FileUrl='https://storage.test/image')

    def test_durable_handoff_replay_conflict_and_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'jobs.db';store=JobStore(path);request=self.request()
            validate(request);store.accept(request);store.accept(request)
            self.assertEqual(store.db.execute('SELECT count(*) FROM jobs').fetchone()[0],1)
            with self.assertRaises(ValueError):store.accept(dict(request,FileUrl='different'))
            result=event(request);store.save_result(request['RequestId'],result);store.db.close()
            store=JobStore(path)
            self.assertEqual(json.loads(store.pending()[3])['EventId'],result['EventId'])
            store.published(request['RequestId']);self.assertIsNone(store.pending());store.db.close()

    def test_normalized_boxes_and_stable_ids(self):
        request=self.request();output=dict(width=100,height=50,detections=[dict(class_name='broken_strand',confidence=.9,bbox_xyxy=[10,5,60,30])])
        result=detections(request,output,{'broken_strand':'STRAND'})
        self.assertEqual(result[0]['BoundingBox'],dict(X=.1,Y=.1,Width=.5,Height=.5))
        self.assertEqual(result,detections(request,output,{'broken_strand':'STRAND'}))
        self.assertEqual(detections(request,output,{'broken_strand':None}),[])

    def test_pending_batch_skips_inflight_and_published(self):
        with tempfile.TemporaryDirectory() as directory:
            store=JobStore(Path(directory)/'jobs.db');ids=[]
            for _ in range(4):
                r=self.request();store.accept(r);ids.append(r['RequestId'])
            store.published(ids[0])
            self.assertEqual([row[0] for row in store.pending_batch([],10)],ids[1:])
            self.assertEqual([row[0] for row in store.pending_batch([ids[1]],1)],[ids[2]])
            self.assertEqual(store.pending_batch(ids[1:],5),[]);store.db.close()


class FrameTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy, cv2  # noqa: F401
        except ImportError:
            self.skipTest('numpy/opencv not installed')

    def test_large_frame_downscaled_not_rejected(self):
        import numpy as np
        from processor import fit, encode
        frame=np.zeros((5000,8000,3),np.uint8)  # 40 MP UAV frame
        small=fit(frame,20_000_000)
        self.assertLessEqual(small.shape[0]*small.shape[1],20_000_000)
        self.assertAlmostEqual(small.shape[1]/small.shape[0],8000/5000,places=2)
        self.assertGreater(len(encode(frame)),0)
        same=np.zeros((1080,1920,3),np.uint8)
        self.assertIs(fit(same,20_000_000),same)

if __name__=='__main__':unittest.main()
