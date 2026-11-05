import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from robotics_landscape import EpisodeFeatures, LandscapeBuilder, EpisodeLandscape


SOURCE = dict(kind='policy_hidden', source_layer='network.3', dimension=4, pooling='mean decisions')


def episode(rows):
    collector = EpisodeFeatures(max_samples=3, seed=12)
    for row in rows:
        collector.add(np.asarray(row, dtype=np.float32))
    return collector.finish()


class EpisodeLandscapeTests(unittest.TestCase):
    def test_bounded_fresh_feature_pooling_does_not_touch_global_rng(self):
        np.random.seed(19); expected = np.random.get_state()
        collector = EpisodeFeatures(max_samples=3, seed=12)
        for index in range(100): collector.add([index, 1, 2, 3])
        result = collector.finish()
        self.assertEqual(result['feature_samples'], 100)
        self.assertEqual(result['samples'].shape, (3, 4))
        mean = np.array([49.5, 1, 2, 3]); mean /= np.linalg.norm(mean)
        np.testing.assert_allclose(result['vector'], mean, rtol=1e-6)
        actual = np.random.get_state()
        np.testing.assert_equal(expected, actual)
        with self.assertRaises(ValueError): collector.add([1, 2])
        with self.assertRaises(ValueError): collector.add([1, 2, float('nan'), 4])
        self.assertIsNone(EpisodeFeatures().finish())

    def test_single_episode_uses_decision_features_for_frozen_basis(self):
        summary = episode([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        builder = LandscapeBuilder(max_fit_samples=8)
        builder.add(dict(seed=42, success=True, steps=3, image='baseline-seed-42.jpg'), summary, SOURCE)
        landscape = builder.finish()
        self.assertGreater(landscape.public['projection']['rank'], 0)
        point = landscape.public['points'][0]
        self.assertEqual((point['index'], point['seed']), (0, 42))
        np.testing.assert_allclose(landscape.project(42, summary)['position'], point['position'])
        before = json.dumps(landscape.public)
        changed = episode([[0,0,0,1],[1,0,0,1]])
        shifted = landscape.project(42, changed)
        self.assertFalse(np.allclose(shifted['position'], point['position']))
        self.assertEqual(before, json.dumps(landscape.public))
        self.assertEqual(shifted['feature_samples'], 2)
        with tempfile.TemporaryDirectory() as tmp:
            landscape.save(tmp)
            restored = EpisodeLandscape.load(tmp)
            np.testing.assert_array_equal(restored.states, landscape.states)
            self.assertEqual(restored.public, landscape.public)
            self.assertEqual(restored.project(42, changed), shifted)
            self.assertEqual(restored.seeds, [42])

    def test_zero_features_are_not_fabricated_and_dimension_changes_fail(self):
        builder = LandscapeBuilder(max_fit_samples=4)
        summary = episode([[0,0,0,0]])
        builder.add(dict(seed=4,success=False,steps=1), summary, SOURCE)
        landscape = builder.finish()
        self.assertEqual(landscape.public['projection']['rank'], 0)
        self.assertEqual(landscape.public['points'][0]['position'], [0,0,0])
        self.assertIsNone(landscape.public['points'][0]['self_similarity'])
        self.assertIsNone(landscape.project(4, summary)['similarity'])
        with self.assertRaises(ValueError):
            builder.add(dict(seed=5,success=True), episode([[1,2,3]]), dict(SOURCE, dimension=3))
        self.assertIsNone(LandscapeBuilder().finish())

    def test_global_fit_sample_bound_and_exact_seed_pairing(self):
        builder = LandscapeBuilder(max_fit_samples=7)
        for seed in range(20):
            builder.add(dict(seed=seed,success=True,steps=1), episode([[1,seed,0,1],[2,seed,1,1]]), SOURCE)
        landscape = builder.finish()
        self.assertEqual(len(landscape.states), 20)
        self.assertEqual(landscape.public['projection']['fit_samples'], 7)
        self.assertEqual(landscape.public['points'][19]['seed'], 19)
        with self.assertRaises(ValueError): landscape.project(25, episode([[1,2,3,4]]))


if __name__ == '__main__': unittest.main()
