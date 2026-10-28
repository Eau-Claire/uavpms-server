import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from powerline_ai.be_contract import normalized_box, detection_id
class ContractTests(unittest.TestCase):
    def test_pixels_to_normalized_top_left(self):
        self.assertEqual(normalized_box([100,50,300,150],1000,500),dict(X=.1,Y=.1,Width=.19999999999999998,Height=.19999999999999998))
    def test_clip(self):
        self.assertEqual(normalized_box([-1,-2,110,120],100,100),dict(X=0.,Y=0.,Width=1.,Height=1.))
    def test_invalid(self):
        for box in [[2,0,1,1],[0,0,float('nan'),1]]:
            with self.assertRaises(ValueError):normalized_box(box,100,100)
    def test_id_stable(self):
        rid='01a167fe-1b73-494a-a0af-efde73065eaf'
        self.assertEqual(detection_id(rid,2,0),detection_id(rid,2,0))
        self.assertNotEqual(detection_id(rid,2,0),detection_id(rid,3,0))
if __name__=='__main__':unittest.main()
