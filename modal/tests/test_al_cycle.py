"""Active-learning cycle: Kaggle, Camber and Modal are mocked; the local model files are real temp files."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # modal/
import al_cycle as al  # noqa: E402

PC = {c: {'ap50': 0.8, 'ap50_95': 0.5} for c in al.CLASSES}


def metrics(ch_mean5=0.80, cc_mean5=0.81, ch_m=0.47, cc_m=0.47, cc_drop=None):
    cc_pc = json.loads(json.dumps(PC))
    if cc_drop:
        cc_pc[cc_drop[0]]['ap50'] -= cc_drop[1]
    base = {'precision': 0.85, 'recall': 0.78, 'mAP50': 0.83}
    return {'cycle_id': 't', 'gate_set': {'name': 'g', 'images': 10}, 'train_set': {'feedback_images': 30, 'replay_images': 100},
            'champion': {**base, 'MEAN_5': ch_mean5, 'mAP50_95': ch_m, 'per_class': PC, 'weights_sha256': 'a'},
            'challenger': {**base, 'MEAN_5': cc_mean5, 'mAP50_95': cc_m, 'per_class': cc_pc, 'weights_sha256': 'b'}}


def make_feedback(root: Path, n=3, bad=None):
    (root / 'images').mkdir(parents=True)
    (root / 'labels').mkdir()
    for i in range(n):
        (root / 'images' / f'img{i}.jpg').write_bytes(b'\xff\xd8fake' + bytes([i]))
        (root / 'labels' / f'img{i}.txt').write_text('1 0.5 0.5 0.2 0.2\n' if i else '')
    if bad:
        (root / 'labels' / 'img0.txt').write_text(bad)
    return root


class GateTests(unittest.TestCase):
    R = al.DEFAULTS['gate']

    def test_pass_when_not_worse(self):
        self.assertTrue(al.gate(metrics(), self.R)['passed'])

    def test_fail_on_mean5_drop(self):
        g = al.gate(metrics(cc_mean5=0.79), self.R)
        self.assertFalse(g['passed']); self.assertFalse(g['checks']['mean5'])

    def test_fail_on_map_drop(self):
        self.assertFalse(al.gate(metrics(cc_m=0.46), self.R)['checks']['map50_95'])

    def test_fail_on_single_defect_class_drop(self):
        g = al.gate(metrics(cc_drop=('broken_strand', 0.03)), self.R)
        self.assertFalse(g['passed']); self.assertEqual(g['defect_ap50_drop']['broken_strand'], 0.03)

    def test_insulator_drop_does_not_block(self):
        self.assertTrue(al.gate(metrics(cc_drop=('insulator', 0.10)), self.R)['passed'])

    BASE = {'MEAN_5': 0.80, 'mAP50_95': 0.47, 'ap50': {c: 0.8 for c in al.DEFECTS}}

    def test_floor_blocks_slow_drift(self):
        # champion already drifted below the seed85 floor; a challenger equal to it passes vs champion but not vs floor
        m = metrics(ch_mean5=0.79, cc_mean5=0.79)
        self.assertTrue(al.gate(m, self.R)['passed'])
        g = al.gate(m, self.R, self.BASE)
        self.assertFalse(g['passed']); self.assertFalse(g['checks']['baseline_mean5'])
        self.assertEqual(g['vs_baseline']['MEAN_5'], -0.01)

    def test_floor_uses_same_tolerances(self):
        m = metrics(ch_m=0.466, cc_m=0.466)                       # 0.004 below floor: within 0.005
        self.assertTrue(al.gate(m, self.R, self.BASE)['checks']['baseline_map50_95'])
        m = metrics(cc_drop=('foreign_object', 0.03))             # 0.03 below floor for one defect class
        self.assertFalse(al.gate(m, self.R, self.BASE)['checks']['baseline_defect_classes'])

    def test_floor_passes_when_at_or_above(self):
        self.assertTrue(al.gate(metrics(ch_mean5=0.80, cc_mean5=0.81), self.R, self.BASE)['passed'])

    def test_default_baseline_is_the_serving_seed85(self):
        b = al.DEFAULTS['baseline']
        self.assertEqual(b['weights_sha256'][:12], '8682baa083ba')
        self.assertEqual(set(b['ap50']), set(al.DEFECTS))


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_valid(self):
        self.assertEqual(len(al.validate_feedback(make_feedback(self.tmp / 'fb'))), 3)

    def test_invalid_rows(self):
        for bad in ('7 0.5 0.5 0.1 0.1', '1 1.5 0.5 0.1 0.1', '1 0.5 0.5 0 0.1', 'x y z', '1 0.5 0.5 0.1'):
            with self.assertRaises(al.CycleError, msg=bad):
                al.validate_feedback(make_feedback(self.tmp / bad.replace(' ', '_'), bad=bad))

    def test_missing_label_and_orphan_label(self):
        d = make_feedback(self.tmp / 'fb')
        (d / 'labels' / 'img1.txt').unlink()
        (d / 'labels' / 'ghost.txt').write_text('')
        with self.assertRaises(al.CycleError) as e:
            al.validate_feedback(d)
        self.assertIn('missing label', str(e.exception)); self.assertIn('without an image', str(e.exception))


class CycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        m = self.tmp / 'modal'
        (m / 'models').mkdir(parents=True)
        (m / 'powerline_ai').mkdir()
        (m / 'powerline_ai' / 'postprocess.py').write_text('# code')
        (m / 'tune_thresholds.py').write_text('# tuner')
        (m / 'models' / 'best.pt').write_bytes(b'champion-weights')
        (m / 'models' / 'manifest.json').write_text('{"sha256": "old"}')
        (m / 'powerline_ai' / 'thresholds.json').write_text('{"t": 1}')
        self.modal = m
        self.stash = {}
        self.deployed = []
        self.fail_smoke_for = set()
        patches = [mock.patch.object(al, 'MODAL_DIR', m), mock.patch.object(al, 'CHAMPION', m / 'models' / 'best.pt'),
                   mock.patch.object(al, 'MANIFEST', m / 'models' / 'manifest.json'),
                   mock.patch.object(al, 'THRESHOLDS', m / 'powerline_ai' / 'thresholds.json'),
                   mock.patch.object(al, 'HISTORY', m / 'models' / 'history'), mock.patch.object(al, 'WORK', self.tmp / 'work'),
                   mock.patch.object(al.time, 'sleep', lambda s: None),
                   mock.patch.object(al, 'stash_put', self.fake_put), mock.patch.object(al, 'stash_get', self.fake_get),
                   mock.patch.object(al, 'modal_deploy', lambda cfg: self.deployed.append(al.sha256(al.CHAMPION))),
                   mock.patch.object(al, 'smoke_test', self.fake_smoke)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.calls = []

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def fake_put(self, local, remote):
        self.stash[remote] = Path(local).read_bytes()

    def fake_get(self, remote, local):
        if remote not in self.stash:
            return False
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_bytes(self.stash[remote])
        return True

    def fake_smoke(self, cfg, expected_model, expected_thr, images):
        if expected_model in self.fail_smoke_for:
            raise al.CycleError('endpoint answered garbage')
        return {'model': {'model_sha256': expected_model}, 'images': len(images), 'predict_ms': [40], 'round_trip_ms': [500]}

    def fake_kaggle(self, m, finals=('complete',), queued_first=0):
        seq = iter(['queued'] * queued_first + ['running'] + list(finals))

        def run(*args, check=True):
            self.calls.append(args)
            if args[:2] == ('config', 'view'):
                return '- username: tester'
            if args[:2] == ('datasets', 'status'):
                return 'ready' if any(c[:2] in (('datasets', 'create'), ('datasets', 'version')) for c in self.calls) else '404'
            if args[:2] == ('datasets', 'create'):
                b = Path(args[args.index('-p') + 1])
                self.sent = {'cycle': json.loads((b / 'cycle.json').read_text()),
                             'code': sorted(p.name for p in (b / 'code').rglob('*')),
                             'feedback': sorted(p.name for p in (b / 'feedback' / 'images').iterdir())}
            if args[:2] == ('kernels', 'push'):
                self.kmeta = json.loads((Path(args[args.index('-p') + 1]) / 'kernel-metadata.json').read_text())
            if args[:2] == ('kernels', 'status'):
                return f'status "{next(seq)}"'
            if args[:2] == ('kernels', 'output'):
                out = Path(args[args.index('-p') + 1]) / 'result'
                out.mkdir(parents=True, exist_ok=True)
                (out / 'challenger.pt').write_bytes(b'challenger-weights')
                (out / 'thresholds.json').write_text('{"t": 2}')
                mm = json.loads(json.dumps(m))
                mm['challenger']['weights_sha256'] = al.sha256(out / 'challenger.pt')
                mm['champion']['weights_sha256'] = al.sha256(self.modal / 'models' / 'best.pt')
                mm['challenger_thresholds_sha256'] = al.sha256(out / 'thresholds.json')
                (out / 'metrics.json').write_text(json.dumps(mm))
                (out / 'kernel.log').write_text('ok')
            return ''
        return run

    def args(self, **kw):
        fb = self.tmp / 'fb'
        if not fb.exists():
            make_feedback(fb, n=3)
        ns = mock.Mock(feedback=fb, cycle_id='c1', force=True, result=None, smoke_images=[])
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    def cfg(self, **kw):
        c = json.loads(json.dumps(al.DEFAULTS))
        c['baseline'] = None   # these tests exercise the deploy flow; the floor has its own tests (GateTests)
        c.update(kw)
        return c

    def registry(self):
        return json.loads(self.stash[f'{al.DEFAULTS["stash_root"]}/registry.json'])

    def test_pass_deploys_and_records(self):
        with mock.patch.object(al, 'kaggle', side_effect=self.fake_kaggle(metrics())):
            self.assertEqual(al.cmd_cycle(self.args(), self.cfg()), 0)
        self.assertEqual(self.sent['feedback'], ['img0.jpg', 'img1.jpg', 'img2.jpg'])
        self.assertIn('tune_thresholds.py', self.sent['code']); self.assertNotIn('thresholds.json', self.sent['code'])
        self.assertEqual(self.kmeta['dataset_sources'], ['tester/uavpms-al-cycle', al.NEMO_DATASET])
        self.assertEqual(self.kmeta['machine_shape'], 'NvidiaTeslaT4')
        self.assertEqual((self.modal / 'models' / 'best.pt').read_bytes(), b'challenger-weights')
        self.assertEqual(json.loads((self.modal / 'powerline_ai' / 'thresholds.json').read_text()), {'t': 2})
        self.assertEqual(json.loads((self.modal / 'models' / 'manifest.json').read_text())['name'], 'al_c1_yolo26m')
        self.assertEqual(len(self.deployed), 1)
        reg = self.registry()
        self.assertEqual(reg['current']['model_sha256'], al.sha256(self.modal / 'models' / 'best.pt'))
        self.assertEqual(reg['previous']['model_sha256'], al.hashlib.sha256(b'champion-weights').hexdigest())
        self.assertEqual(reg['cycles'][-1]['status'], 'deployed')
        old = reg['previous']['model_sha256'][:16]
        self.assertEqual(self.stash[f'{al.DEFAULTS["stash_root"]}/models/{old}/best.pt'], b'champion-weights')
        self.assertIn(f'{al.DEFAULTS["stash_root"]}/cycles/c1/gate_report.md', self.stash)
        self.assertFalse((self.tmp / 'work' / 'cycle.lock').exists())

    def test_untuned_thresholds_are_never_deployed(self):
        m = metrics(); m['thresholds_tuned'] = False
        with mock.patch.object(al, 'kaggle', side_effect=self.fake_kaggle(m)):
            with self.assertRaisesRegex(al.CycleError, 'not tuned'):
                al.cmd_cycle(self.args(), self.cfg())
        self.assertEqual((self.modal / 'models' / 'best.pt').read_bytes(), b'champion-weights')
        self.assertEqual(self.deployed, [])

    def test_gate_fail_changes_nothing(self):
        with mock.patch.object(al, 'kaggle', side_effect=self.fake_kaggle(metrics(cc_mean5=0.70))):
            self.assertEqual(al.cmd_cycle(self.args(), self.cfg()), 3)
        self.assertEqual((self.modal / 'models' / 'best.pt').read_bytes(), b'champion-weights')
        self.assertEqual(self.deployed, [])
        self.assertEqual(self.registry()['cycles'][-1]['status'], 'gate_failed')

    def test_smoke_failure_rolls_back(self):
        self.fail_smoke_for = {al.hashlib.sha256(b'challenger-weights').hexdigest()}
        with mock.patch.object(al, 'kaggle', side_effect=self.fake_kaggle(metrics())):
            self.assertEqual(al.cmd_cycle(self.args(), self.cfg()), 4)
        self.assertEqual((self.modal / 'models' / 'best.pt').read_bytes(), b'champion-weights')
        self.assertEqual(json.loads((self.modal / 'powerline_ai' / 'thresholds.json').read_text()), {'t': 1})
        self.assertEqual(len(self.deployed), 2)   # challenger, then champion again
        self.assertEqual(self.deployed[-1], al.hashlib.sha256(b'champion-weights').hexdigest())
        self.assertEqual(self.registry()['cycles'][-1]['status'], 'rolled_back')

    def test_long_queue_switches_accelerator(self):
        with mock.patch.object(al, 'kaggle', side_effect=self.fake_kaggle(metrics(), queued_first=2)):
            self.assertEqual(al.cmd_cycle(self.args(), self.cfg(queue_timeout_s=-1)), 0)
        pushes = [c for c in self.calls if c[:2] == ('kernels', 'push')]
        self.assertEqual([c[-1] for c in pushes], ['NvidiaTeslaT4', 'NvidiaTeslaP100'])

    def test_stale_champion_is_not_deployed(self):
        def kaggle(*args, check=True):
            out = run(*args, check=check)
            if args[:2] == ('kernels', 'output'):   # someone changed the serving model meanwhile
                (self.modal / 'models' / 'best.pt').write_bytes(b'other-model')
            return out
        run = self.fake_kaggle(metrics())
        with mock.patch.object(al, 'kaggle', side_effect=kaggle):
            with self.assertRaises(al.CycleError) as e:
                al.cmd_cycle(self.args(), self.cfg())
        self.assertIn('champion changed', str(e.exception))
        self.assertEqual(self.deployed, [])

    def test_kernel_error_is_recorded(self):
        with mock.patch.object(al, 'kaggle', side_effect=self.fake_kaggle(metrics(), finals=('error',))):
            with self.assertRaises(al.CycleError) as e:
                al.cmd_cycle(self.args(), self.cfg())
        self.assertIn('state error', str(e.exception))
        self.assertEqual(self.registry()['cycles'][-1]['status'], 'kernel_failed')

    def test_lock_prevents_parallel_cycles(self):
        (self.tmp / 'work').mkdir(parents=True)
        (self.tmp / 'work' / 'cycle.lock').write_text('pid 1')
        with mock.patch.object(al, 'kaggle') as k:
            with self.assertRaises(al.CycleError):
                al.cmd_cycle(self.args(), self.cfg())
        k.assert_not_called()

    def test_below_threshold_does_nothing(self):
        with mock.patch.object(al, 'kaggle') as k:
            with self.assertRaises(al.CycleError):
                al.cmd_cycle(self.args(force=False), self.cfg())
        k.assert_not_called()

    def test_stash_failure_before_swap_keeps_serving_files(self):
        def put(local, remote):
            if remote.endswith('/best.pt') and Path(local).read_bytes() == b'challenger-weights':
                raise al.CycleError('stash down')
            self.fake_put(local, remote)
        with mock.patch.object(al, 'stash_put', put), \
                mock.patch.object(al, 'kaggle', side_effect=self.fake_kaggle(metrics())):
            self.assertEqual(al.cmd_cycle(self.args(), self.cfg()), 2)
        self.assertEqual((self.modal / 'models' / 'best.pt').read_bytes(), b'champion-weights')
        self.assertEqual(self.deployed, [])
        self.assertEqual(self.registry()['cycles'][-1]['status'], 'promote_failed')   # still recorded
        self.assertIn('stash down', self.registry()['cycles'][-1]['error'])

    def test_archive_failure_before_swap_is_recorded(self):
        def put(local, remote):
            if '/models/' in remote:
                raise al.CycleError('i/o timeout')
            self.fake_put(local, remote)
        with mock.patch.object(al, 'stash_put', put), \
                mock.patch.object(al, 'kaggle', side_effect=self.fake_kaggle(metrics())):
            self.assertEqual(al.cmd_cycle(self.args(), self.cfg()), 2)
        self.assertEqual((self.modal / 'models' / 'best.pt').read_bytes(), b'champion-weights')
        self.assertEqual(self.deployed, [])
        self.assertEqual(self.registry()['cycles'][-1]['status'], 'promote_failed')
        self.assertFalse((self.tmp / 'work' / 'cycle.lock').exists())

    def test_record_failure_does_not_undo_a_good_deploy(self):
        with mock.patch.object(al, 'record', side_effect=al.CycleError('stash down')), \
                mock.patch.object(al, 'kaggle', side_effect=self.fake_kaggle(metrics())):
            self.assertEqual(al.cmd_cycle(self.args(), self.cfg()), 0)
        self.assertEqual((self.modal / 'models' / 'best.pt').read_bytes(), b'challenger-weights')
        self.assertEqual(json.loads((self.tmp / 'work' / 'c1' / 'result' / 'outcome.json').read_text())['status'], 'deployed')

    def test_init_is_idempotent(self):
        self.assertEqual(al.cmd_init(self.args(), self.cfg()), 0)
        reg = self.registry()
        self.assertEqual(reg['current']['model_sha256'], al.hashlib.sha256(b'champion-weights').hexdigest())
        self.assertIn(f'{al.DEFAULTS["stash_root"]}/README.md', self.stash)
        self.assertEqual(al.cmd_init(self.args(), self.cfg()), 0)
        self.assertEqual(self.registry(), reg)

    def test_manual_rollback_restores_previous(self):
        with mock.patch.object(al, 'kaggle', side_effect=self.fake_kaggle(metrics())):
            al.cmd_cycle(self.args(), self.cfg())
        shutil.rmtree(self.modal / 'models' / 'history')   # force the restore from Camber
        self.assertEqual(al.cmd_rollback(self.args(), self.cfg()), 0)
        self.assertEqual((self.modal / 'models' / 'best.pt').read_bytes(), b'champion-weights')
        self.assertEqual(self.registry()['cycles'][-1]['status'], 'manual_rollback')
        self.assertEqual(self.registry()['current']['model_sha256'], al.hashlib.sha256(b'champion-weights').hexdigest())




class SmokeTestTests(unittest.TestCase):
    """smoke_test against a fake Modal Detector: cold first call, model hash check, detection validation."""
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.imgs = []
        for i in range(3):
            (self.tmp / f'{i}.jpg').write_bytes(b'jpg')
            self.imgs.append(self.tmp / f'{i}.jpg')
        self.cfg = {**al.DEFAULTS, 'smoke_wait_s': 60}

    def run_smoke(self, predict_ms, served='new', dets=None):
        calls = iter(predict_ms)
        info = {'name': 'm', 'model_sha256': served, 'thresholds_sha256': 't'}
        det = mock.Mock()
        det.model_info.remote.side_effect = lambda: info
        det.predict_jpeg.remote.side_effect = lambda b: {
            'model': 'm', 'predict_ms': next(calls),
            'detections': dets if dets is not None else [{'class_name': 'bird_nest', 'confidence': 0.9}]}
        fake_modal = mock.Mock()
        fake_modal.Cls.from_name.return_value = mock.Mock(return_value=det)
        clock = iter(range(0, 10 ** 6, 20))
        with mock.patch.dict(sys.modules, {'modal': fake_modal}), mock.patch.object(al.time, 'sleep'), \
                mock.patch.object(al.time, 'time', lambda: next(clock)):
            return al.smoke_test(self.cfg, 'new', 't', self.imgs), det

    def test_cold_first_inference_is_not_timed(self):
        out, det = self.run_smoke([3800, 45, 41, 41])
        self.assertEqual(out['predict_ms'], [45, 41, 41])
        self.assertEqual(det.predict_jpeg.remote.call_count, 4)   # warm-up + 3 timed

    def test_slow_warm_inference_fails(self):
        with self.assertRaisesRegex(al.CycleError, 'warm predict'):
            self.run_smoke([3800, 45, 3500, 41])

    def test_old_model_still_served_fails(self):
        with self.assertRaisesRegex(al.CycleError, 'endpoint serves'):
            self.run_smoke([40] * 4, served='old')

    def test_invalid_detection_fails(self):
        with self.assertRaisesRegex(al.CycleError, 'invalid detection'):
            self.run_smoke([40] * 4, dets=[{'class_name': 'cat', 'confidence': 0.9}])


class CamberRetryTests(unittest.TestCase):
    def run_with(self, codes):
        results = iter(mock.Mock(returncode=c, stdout='', stderr='dial tcp: i/o timeout\nUsage:\n  camber stash cp') for c in codes)
        with mock.patch.object(al.subprocess, 'run', side_effect=lambda *a, **k: next(results)) as run, \
                mock.patch.object(al.time, 'sleep'):
            try:
                return al.camber('cp', 'a', 'b'), run.call_count
            except al.CycleError as exc:
                return exc, run.call_count

    def test_transient_error_is_retried(self):
        out, n = self.run_with([1, 1, 0])
        self.assertEqual((out, n), ('dial tcp: i/o timeout\nUsage:\n  camber stash cp', 3))

    def test_persistent_error_raises_without_usage_text(self):
        exc, n = self.run_with([1, 1, 1, 1])
        self.assertEqual(n, 4)
        self.assertIsInstance(exc, al.CycleError)
        self.assertIn('i/o timeout', str(exc)); self.assertNotIn('Usage', str(exc))


if __name__ == '__main__':
    unittest.main()
