"""Bounded episode feature pooling and a frozen baseline PCA coordinate system."""
import copy
import json
from pathlib import Path
import numpy as np


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    return (vector / norm if norm else vector).astype(np.float32)


class EpisodeFeatures:
    """Pool fresh decisions; keep a small local-RNG reservoir for fitting PCA."""
    def __init__(self, max_samples=32, seed=0):
        self.maximum = max_samples
        self.rng = np.random.default_rng(seed)
        self.count = 0
        self.total = None
        self.samples = []

    def add(self, vector):
        row = np.asarray(vector, dtype=np.float32)
        if row.ndim != 1 or not row.size or not np.isfinite(row).all():
            raise ValueError('Policy features must be a finite, nonempty vector.')
        if self.total is None: self.total = np.zeros(row.size, dtype=np.float64)
        if self.total.shape != row.shape: raise ValueError('Policy feature dimensions changed during the episode.')
        self.total += row
        self.count += 1
        if len(self.samples) < self.maximum: self.samples.append(_unit(row))
        else:
            index = int(self.rng.integers(self.count))
            if index < self.maximum: self.samples[index] = _unit(row)

    def finish(self):
        if not self.count: return None
        return dict(vector=_unit(self.total / self.count), samples=np.asarray(self.samples), feature_samples=self.count)


class LandscapeBuilder:
    def __init__(self, max_fit_samples=512):
        self.maximum = max_fit_samples
        self.rng = np.random.default_rng(0)
        self.rows = []
        self.seen = 0
        self.entries = []
        self.source = None

    def add(self, record, features, source):
        if features is None or type(record.get('success')) is not bool or record.get('error'): return
        source = copy.deepcopy(source)
        dimension = int(features['vector'].size)
        source['dimension'] = dimension
        if self.source is not None and any(self.source.get(key) != source.get(key) for key in ('kind','source_layer','dimension','pooling')):
            raise ValueError('Policy feature source changed between episodes. Analyze again with a consistent policy adapter.')
        self.source = source
        if any(row[0]['seed'] == record['seed'] for row in self.entries): raise ValueError('Duplicate baseline seed.')
        self.entries.append((copy.deepcopy(record), features['vector'].copy(), features['feature_samples']))
        for row in features['samples']:
            self.seen += 1
            if len(self.rows) < self.maximum: self.rows.append(row.copy())
            else:
                index = int(self.rng.integers(self.seen))
                if index < self.maximum: self.rows[index] = row.copy()

    def finish(self):
        if not self.entries: return None
        from embeddings import _pca
        mean, components, projection = _pca(np.asarray(self.rows))
        projection.update(method='PCA fitted to sampled baseline decision features', fit_samples=len(self.rows),
                          episode_pooling='Mean of fresh decision features, then L2 normalization')
        states = np.stack([entry[1] for entry in self.entries])
        points = []
        for index, (record, vector, count) in enumerate(self.entries):
            point = {key: record[key] for key in ('seed','success','steps','image','video') if key in record}
            point.update(index=index, name=f'Episode {index+1} · seed {record["seed"]}',
                         position=((vector-mean) @ components.T).tolist(), feature_samples=count,
                         self_similarity=1. if np.any(vector) else None)
            points.append(point)
        public = dict(version=1, source=self.source, projection=projection, points=points)
        return EpisodeLandscape(states, mean, components, public)


class EpisodeLandscape:
    def __init__(self, states, mean, components, public):
        self.states = np.asarray(states, dtype=np.float32)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.components = np.asarray(components, dtype=np.float64)
        self.public = public
        self.seeds = [point['seed'] for point in public['points']]
        if (self.states.ndim != 2 or not self.states.size or len(self.states) != len(self.seeds)
            or self.mean.shape != (self.states.shape[1],) or self.components.shape != (3,self.states.shape[1])
            or any(not np.isfinite(array).all() for array in (self.states,self.mean,self.components))
            or len(set(self.seeds)) != len(self.seeds)):
            raise ValueError('Invalid saved episode features. Analyze again.')
        self.indexes = {seed:index for index,seed in enumerate(self.seeds)}

    def project(self, seed, features):
        if seed not in self.indexes: raise ValueError('The episode seed has no baseline embedding.')
        vector = features['vector']
        if vector.shape != self.mean.shape: raise ValueError('Policy feature dimension changed. Analyze again.')
        baseline = self.states[self.indexes[seed]]
        similarity = float(np.clip(vector @ baseline, -1, 1)) if np.any(vector) and np.any(baseline) else None
        return dict(position=((vector-self.mean) @ self.components.T).tolist(), similarity=similarity,
                    feature_samples=features['feature_samples'])

    def save(self, directory):
        directory = Path(directory)
        temporary = directory/'episode-features.tmp.npz'
        np.savez_compressed(temporary, states=self.states, mean=self.mean, components=self.components)
        temporary.replace(directory/'episode-features.npz')
        temporary = directory/'episode-embedding.tmp.json'
        temporary.write_text(json.dumps(self.public, allow_nan=False))
        temporary.replace(directory/'episode-embedding.json')

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        try:
            public = json.loads((directory/'episode-embedding.json').read_text())
            with np.load(directory/'episode-features.npz', allow_pickle=False) as arrays:
                return cls(arrays['states'], arrays['mean'], arrays['components'], public)
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError('Saved episode features are unavailable. Run Analyze again before DQN discovery.') from exc
