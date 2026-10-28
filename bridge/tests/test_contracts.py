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

if __name__=='__main__':unittest.main()
