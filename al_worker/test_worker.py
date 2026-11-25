"""Unit tests for the pure parts of the worker (DB flow is covered by the end-to-end run on a throwaway Postgres)."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import worker as w


class HelperTests(unittest.TestCase):
    def test_parse_npgsql(self):
        kw = w.parse_npgsql('Host=uavpms-db;Port=5432;Database=uavpms;Username=u;Password=p=1;SSL Mode=Disable;Pooling=true')
        self.assertEqual(kw, {'host': 'uavpms-db', 'port': '5432', 'dbname': 'uavpms', 'user': 'u', 'password': 'p=1',
                              'sslmode': 'disable'})

    def test_storage_key(self):
        self.assertEqual(w.storage_key('uav-media/2026/11/a.jpg', 'x'), ('uav-media', '2026/11/a.jpg'))
        self.assertEqual(w.storage_key('https://uav-minio.duckdns.org/uav-media/2026/11/a.jpg', 'x'),
                         ('uav-media', '2026/11/a.jpg'))
        self.assertEqual(w.storage_key('a.jpg', 'uav-media'), ('uav-media', 'a.jpg'))

    def test_to_yolo_converts_top_left_to_center(self):
        rows = w.to_yolo([{'className': 'insulator_broken', 'x': 0.15, 'y': 0.2, 'width': 0.25, 'height': 0.3}])
        self.assertEqual(rows, ['1 0.275000 0.350000 0.250000 0.300000'])
        self.assertEqual(w.to_yolo([]), [])

    def test_to_yolo_clips_and_rejects(self):
        self.assertEqual(w.to_yolo([{'className': 'insulator', 'x': 0.9, 'y': 0.0, 'width': 0.2, 'height': 0.5}]),
                         ['0 0.950000 0.250000 0.100000 0.500000'])
        with self.assertRaises(ValueError):
            w.to_yolo([{'className': 'cat', 'x': 0, 'y': 0, 'width': 0.1, 'height': 0.1}])
        with self.assertRaises(ValueError):
            w.to_yolo([{'className': 'insulator', 'x': 1.0, 'y': 0, 'width': 0.1, 'height': 0.1}])

    def test_db_status(self):
        self.assertEqual(w.db_status('deployed'), 'deployed')
        self.assertEqual(w.db_status('kernel_failed'), 'error')
        self.assertEqual(w.db_status(None), 'error')


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        import cv2
        import numpy as np
        self.src = self.tmp / 'src'
        self.src.mkdir()
        img = np.full((40, 60, 3), 128, np.uint8)
        cv2.imwrite(str(self.src / 'a.jpg'), img)
        cv2.imwrite(str(self.src / 'b.png'), img)
        vw = cv2.VideoWriter(str(self.src / 'v.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (60, 40))
        for i in range(20):
            vw.write(np.full((40, 60, 3), i * 10, np.uint8))
        vw.release()

    def fetch(self, bucket, obj, dest):
        shutil.copy(self.src / obj, dest)

    def row(self, i, obj, frame=None, boxes=None):
        return {'id': f'id{i}', 'bucket': 'uav-media', 'object': obj, 'frame_index': frame,
                'boxes': json.dumps(boxes if boxes is not None else
                                    [{'className': 'bird_nest', 'x': 0.1, 'y': 0.1, 'width': 0.2, 'height': 0.2,
                                      'source': 'ai', 'confidence': 90}])}

    def test_exports_images_frames_and_negatives(self):
        out = self.tmp / 'fb'
        rows = [self.row(1, 'a.jpg'), self.row(2, 'b.png', boxes=[]), self.row(3, 'v.mp4', frame=12),
                self.row(4, 'missing.jpg'), self.row(5, 'a.jpg', boxes=[{'className': 'cat', 'x': 0, 'y': 0, 'width': 0.1, 'height': 0.1}])]
        done, errors = w.export_feedback(rows, out, self.fetch)
        self.assertEqual(done, 3)
        self.assertEqual(len(errors), 2)
        self.assertEqual(sorted(p.name for p in (out / 'images').iterdir()), ['id1.jpg', 'id2.png', 'id3.jpg'])
        self.assertEqual(sorted(p.name for p in (out / 'labels').iterdir()), ['id1.txt', 'id2.txt', 'id3.txt'])
        self.assertEqual((out / 'labels' / 'id2.txt').read_text(), '')          # negative sample
        self.assertTrue((out / 'labels' / 'id1.txt').read_text().startswith('3 '))
        import cv2
        frame = cv2.imread(str(out / 'images' / 'id3.jpg'))
        self.assertEqual(frame.shape[:2], (40, 60))
        self.assertAlmostEqual(float(frame.mean()), 120, delta=12)               # frame 12 has value 120
        self.assertFalse((out / '_download').exists())


if __name__ == '__main__':
    unittest.main()
