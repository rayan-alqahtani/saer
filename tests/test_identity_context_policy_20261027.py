import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from identity_context_runtime_20261027 import apply_policy


class IdentityPolicyTests(unittest.TestCase):
    def setUp(self):
        self.classes = {'make': ['A', 'B'], 'model': ['A|One', 'A|Two', 'B|Three'], 'body': ['sedan', 'suv']}
        self.policy = {'temperatures': {'make': 1., 'model': 1., 'body': 1.},
                       'choice': {'make_weight': .5, 'body_weight': 0.},
                       'body_compatibility': [[1, 0], [0, 1], [.5, .5]]}
        self.probs = {'make': np.array([[.9, .1]]), 'model': np.array([[.3, .2, .5]])}

    def test_zero_weight_preserves_model(self):
        self.policy['choice']['make_weight'] = 0.
        np.testing.assert_allclose(apply_policy(self.probs, self.classes, self.policy), self.probs['model'], atol=1e-12)

    def test_uniform_make_has_no_effect(self):
        self.probs['make'][:] = .5
        np.testing.assert_allclose(apply_policy(self.probs, self.classes, self.policy), self.probs['model'], atol=1e-12)

    def test_soft_evidence_cannot_delete_a_class(self):
        p = apply_policy(self.probs, self.classes, self.policy)
        self.assertTrue(np.all(p > 0))
        np.testing.assert_allclose(p.sum(axis=1), 1.)

    def test_strong_model_can_outweigh_wrong_make(self):
        self.probs['model'][:] = [.000001, .000001, .999998]
        self.probs['make'][:] = [.999, .001]
        self.assertEqual(int(apply_policy(self.probs, self.classes, self.policy).argmax()), 2)

    def test_unused_body_does_not_change_output(self):
        before = apply_policy(self.probs, self.classes, self.policy)
        self.probs['body'] = np.array([[.999, .001]])
        np.testing.assert_allclose(before, apply_policy(self.probs, self.classes, self.policy))


if __name__ == '__main__':
    unittest.main()
