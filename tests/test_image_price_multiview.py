"""فحوص نزاهة تجربة الصور المتعددة، دون الوصول إلى test."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from image_price_multiview import sample_weights, bootstrap, kernel, gate
from image_price_multiview_runtime import ImagePriceRuntime
from image_price_multiview_finish import exact_cached_head


class MultiViewTests(unittest.TestCase):
    def test_each_listing_has_equal_total_loss_weight(self):
        rows = [{'listing_uid': 'one'}] + [{'listing_uid': 'five'} for _ in range(5)]
        w = sample_weights(rows, np.arange(6))
        self.assertAlmostEqual(w[0], 1)
        self.assertAlmostEqual(w[1:].sum(), 1)
        # Do not normalize to image-count mean: C must measure the same ad-level loss.
        self.assertAlmostEqual(w.sum(), 2)

    def test_paired_bootstrap_identical_models_has_zero_difference(self):
        y = np.array([100., 200., 300.]); p = np.array([110., 240., 270.])
        out = bootstrap(y, p, p, ['a', 'a', 'b'], 100)
        self.assertEqual(out['new_minus_reference_MAE_95CI'], [0., 0.])
        self.assertEqual(out['n_groups'], 2)

    def test_kernel_factorization_matches_explicit_descriptors(self):
        rng = np.random.default_rng(12); raw = rng.normal(size=(6, 154))
        mean = raw[:3].mean(0, keepdims=True); std = np.maximum(raw[:3].std(0, keepdims=True), 1e-6)
        bases = {k: rng.normal(size=(6, 14)) for k in ('crop', 'full', 'x53')}
        bases = {k: x / np.linalg.norm(x, axis=1, keepdims=True) for k, x in bases.items()}
        grams = {k: x @ x.T for k, x in bases.items()}
        v = np.clip((raw - mean) / std, -5, 5); v /= np.linalg.norm(v, axis=1, keepdims=True)
        sw = .1 / 1.95
        explicit = {k: np.hstack([np.sqrt(1 - sw) * x, np.sqrt(sw) * v]) if k != 'full' else x for k, x in bases.items()}
        a = np.array([0, 2, 4]); b = np.array([1, 3, 5]); s = np.clip(explicit['x53'][a] @ explicit['x53'][b].T, -1, 1)
        expected = .55 * (.72 * explicit['crop'][a] @ explicit['crop'][b].T + .28 * explicit['full'][a] @ explicit['full'][b].T) + .25 * s**2 + .2 * np.exp(-1.5 * np.maximum(2 - 2 * s, 0))
        np.testing.assert_allclose(kernel(grams, raw, a, b, mean, std), expected, atol=1e-12)

    def test_practical_gate_checks_expensive_band_and_percentage_error(self):
        y = np.array([100., 200., 1000.]); old = np.array([110., 220., 1100.])
        worse_expensive = np.array([100., 200., 1110.])
        verdict = gate(y, worse_expensive, old, 500)
        self.assertFalse(verdict['expensive_MAE_not_worse_by_over_5_percent'])
        self.assertTrue(all(gate(y, y, old, 500).values()))

    def test_exported_runtime_checks_integrity_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            import json
            p = Path(folder); (p / 'head.npz').write_bytes(b'changed')
            (p / 'LOCK.json').write_text(json.dumps({'sha256': {'head.npz': 'not-the-hash'}}))
            with self.assertRaisesRegex(ValueError, 'بصمة'):
                ImagePriceRuntime(p)

    def test_abstention_never_exposes_raw_as_user_price(self):
        from types import SimpleNamespace
        rt = ImagePriceRuntime.__new__(ImagePriceRuntime)
        rt.gem = SimpleNamespace(domain_threshold=1.2637)
        rt.predict_batch_paths = lambda paths: (np.array([133504.91]), np.array([1.8587]))
        result = rt.predict('unused-in-this-test')
        self.assertEqual(result['status'], 'abstained')
        self.assertIsNone(result['predicted_asking_price_sar'])
        self.assertEqual(result['raw_predicted_asking_price_sar'], 133504.91)

    def test_invalid_numeric_prediction_cannot_be_displayed(self):
        from types import SimpleNamespace
        rt = ImagePriceRuntime.__new__(ImagePriceRuntime)
        rt.gem = SimpleNamespace(domain_threshold=1.)
        rt.predict_batch_paths = lambda paths: (np.array([float('nan')]), np.array([.1]))
        with self.assertRaises(ValueError): rt.predict('unused')

    def test_double_cached_reference_matches_independent_dual_kernel(self):
        import json
        rng = np.random.default_rng(18)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder); (p / 'cache').mkdir()
            raw = rng.normal(size=(5, 154)); mean = raw[:2].mean(0, keepdims=True)
            std = np.maximum(raw[:2].std(0, keepdims=True), 1e-6)
            v = np.clip((raw - mean) / std, -5, 5)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            normalized = {}
            for key, width in [('crop', 3), ('x53', 4), ('full', 5)]:
                base = rng.normal(size=(5, width)); base /= np.linalg.norm(base, axis=1, keepdims=True)
                stored = base if key == 'full' else np.hstack([base, np.zeros((5, 154))])
                np.save(p / 'cache' / (key + '.npy'), stored)
                normalized[key] = base if key == 'full' else np.hstack([np.sqrt(1 - .1 / 1.95) * base, np.sqrt(.1 / 1.95) * v])
            np.save(p / 'cache/raw.npy', raw)
            coef = np.array([.8, -.6]); support = normalized['x53'][:2].astype(np.float32)
            np.savez(p / 'head.npz', vs_mean=mean, vs_std=std,
                w_crop=.55 * .72 * normalized['crop'][:2].T @ coef,
                w_full=.55 * .28 * normalized['full'][:2].T @ coef,
                support53=support, poly_coef=.25 * coef, rbf_coef=.2 * coef, intercept=[10.])
            (p / 'policy.json').write_text(json.dumps({'bounds': [1., 20.]}))
            sim = np.clip(normalized['x53'][2:] @ support.astype(np.float64).T, -1, 1)
            dual = .55 * (.72 * normalized['crop'][2:] @ normalized['crop'][:2].T + .28 * normalized['full'][2:] @ normalized['full'][:2].T) + .25 * sim**2 + .2 * np.exp(-1.5 * (2 - 2 * sim))
            expected = np.exp(10. + dual @ coef)
            np.testing.assert_allclose(exact_cached_head(p, np.arange(2, 5)), expected, rtol=1e-12)


if __name__ == '__main__': unittest.main()
