import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from powerline_ai.postprocess import CLASSES, PostprocessConfig, postprocess
from powerline_ai.settings import load_postprocess_config

LOW = PostprocessConfig(class_confidence={c: 0.1 for c in CLASSES})


def det(cls, conf, box):
    return dict(class_id=CLASSES.index(cls), class_name=cls, confidence=conf, bbox_xyxy=list(box))


class PostprocessTests(unittest.TestCase):
    def test_per_class_threshold(self):
        cfg = PostprocessConfig(class_confidence={**LOW.class_confidence, 'insulator': 0.5})
        out = postprocess([det('insulator', 0.4, (0, 0, 10, 10)), det('bird_nest', 0.4, (50, 50, 60, 60))], cfg)
        self.assertEqual([d['class_name'] for d in out], ['bird_nest'])

    def test_merge_two_boxes_on_one_defect(self):
        out = postprocess([det('foreign_object', 0.8, (0, 0, 100, 100)),
                           det('foreign_object', 0.6, (40, 0, 140, 100))], LOW)  # IoU 0.43, contain 0.6
        self.assertEqual(len(out), 2)  # below both merge limits: two separate objects
        out = postprocess([det('foreign_object', 0.8, (0, 0, 100, 100)),
                           det('foreign_object', 0.6, (20, 0, 120, 100))], LOW)  # IoU 0.67
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]['bbox_xyxy'], [0, 0, 120, 100])
        self.assertEqual(out[0]['confidence'], 0.8)
        self.assertEqual(out[0]['merged'], 2)

    def test_merge_nested_box(self):
        out = postprocess([det('broken_strand', 0.5, (0, 0, 200, 40)),
                           det('broken_strand', 0.7, (20, 5, 60, 35))], LOW)  # small box inside big one
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]['bbox_xyxy'], [0, 0, 200, 40])

    def test_cross_class_defects_keep_most_confident(self):
        out = postprocess([det('insulator_broken', 0.55, (0, 0, 100, 100)),
                           det('foreign_object', 0.70, (5, 5, 100, 100))], LOW)
        self.assertEqual([d['class_name'] for d in out], ['foreign_object'])

    def test_defect_inside_insulator_is_kept(self):
        out = postprocess([det('insulator', 0.9, (0, 0, 100, 300)),
                           det('insulator_broken', 0.6, (20, 100, 80, 160))], LOW)
        self.assertEqual(sorted(d['class_name'] for d in out), ['insulator', 'insulator_broken'])

    def test_near_identical_insulator_dropped_for_defect(self):
        for ins_conf in (0.9, 0.3):  # whichever is more confident, the defect label wins
            out = postprocess([det('insulator', ins_conf, (0, 0, 100, 100)),
                               det('insulator_broken', 0.6, (2, 2, 100, 100))], LOW)
            self.assertEqual([d['class_name'] for d in out], ['insulator_broken'])

    def test_disabled_only_thresholds(self):
        cfg = PostprocessConfig(class_confidence=LOW.class_confidence, merge_same_class=False, resolve_cross_class=False)
        out = postprocess([det('foreign_object', 0.8, (0, 0, 100, 100)),
                           det('foreign_object', 0.6, (20, 0, 120, 100)),
                           det('bird_nest', 0.05, (0, 0, 5, 5))], cfg)
        self.assertEqual(len(out), 2)

    def test_input_not_mutated(self):
        a = det('foreign_object', 0.8, (0, 0, 100, 100))
        postprocess([a, det('foreign_object', 0.6, (20, 0, 120, 100))], LOW)
        self.assertEqual(a['bbox_xyxy'], [0, 0, 100, 100])


class SettingsTests(unittest.TestCase):
    def test_thresholds_file_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'thresholds.json'
            p.write_text(json.dumps({'_source': {}, 'class_confidence': {'foreign_object': 0.6}}))
            cfg = load_postprocess_config(p)
            self.assertEqual(cfg.threshold('foreign_object'), 0.6)
            self.assertEqual(cfg.threshold('insulator'), PostprocessConfig().threshold('insulator'))

    def test_rejects_unknown_class(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'thresholds.json'
            p.write_text(json.dumps({'class_confidence': {'fo': 0.6}}))
            with self.assertRaises(ValueError):
                load_postprocess_config(p)

    def test_missing_file_uses_defaults(self):
        self.assertEqual(load_postprocess_config(Path('/nonexistent/x.json')), PostprocessConfig())


if __name__ == '__main__':
    unittest.main()
