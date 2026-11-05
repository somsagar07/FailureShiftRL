from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from robotics_worker import Rollouts


class EpisodeMediaTests(unittest.TestCase):
    def test_aborted_episode_has_no_completed_feature_projection(self):
        viewer=object.__new__(Rollouts)
        viewer.config={'environment':{'horizon':2},'record_video':False}
        viewer.completed=0; viewer.total=1; viewer.cameras=[]; viewer.last_features=None
        viewer.env=SimpleNamespace(_get_observations=lambda **kw:{'state':np.ones(2)})
        viewer.policy=SimpleNamespace(reset=lambda:None,act=lambda obs:np.zeros(1),
            take_features=lambda:np.ones(2),feature_info=lambda:{})
        viewer.frame=lambda:np.zeros((4,4,3),dtype=np.uint8)
        viewer.preview=lambda frame:None
        viewer.landscape=SimpleNamespace(indexes={42:0},public={'source':{}},project=lambda *args:{'position':[1,2,3]})
        def fail_step(action): raise RuntimeError('Simulator step failed')
        viewer.env.step=fail_step
        with patch('robotics_env.restore_snapshot'),patch('robotics_env.apply_variation'),patch('robotics_worker.emit'):
            record=viewer.episode(0,{},42,Path('/unused'),1)
        self.assertIsNone(record['success'])
        self.assertIn('Simulator step failed',record['error'])
        self.assertNotIn('embedding',record)
        self.assertIsNone(viewer.last_features)

    def test_each_seed_retains_its_own_frame_while_videos_are_bounded(self):
        viewer=object.__new__(Rollouts)
        viewer.config={'record_video':False}; viewer.media={}
        with tempfile.TemporaryDirectory() as tmp:
            viewer.directory=Path(tmp)
            for success,seed in [(True,42),(True,43),(False,44),(False,45)]:
                media=viewer.media_for('baseline',seed,success,[],np.full((4,4,3),seed,dtype=np.uint8),1)
                self.assertEqual(media['image'],f'baseline-seed-{seed}.jpg')
                self.assertTrue((Path(tmp)/media['image']).is_file())
                self.assertNotIn('video',media)
            action=viewer.media_for(3,43,False,[],np.zeros((4,4,3),dtype=np.uint8),1)
            self.assertEqual(action['image'],'action-3-seed-43.jpg')


if __name__=='__main__': unittest.main()
