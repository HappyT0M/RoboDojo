import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from utils.rope_task_metrics import update_obstacle_route
from scripts import create_rope_assets as assets


class SingleObstacleTests(unittest.TestCase):
    def step(self, state, point, grasp=False):
        return update_obstacle_route(state, point, (0, 0, .84053), grasp,
                                     side_x_bounds=(.075, .20), z_bounds=(.779, .87), near_distance=.08)

    def test_right_route_after_grasp(self):
        state = {}
        self.assertFalse(self.step(state, (0, -.20, .80), True))
        self.assertFalse(self.step(state, (.12, -.03, .80)))
        self.assertTrue(self.step(state, (.12, .03, .80)))
        self.assertTrue(self.step(state, (0, .24, .80)))

    def test_no_grasp_and_direct_endpoint_do_not_count(self):
        state = {}
        self.step(state, (.12, -.20, .80))
        self.assertFalse(self.step(state, (.12, .03, .80)))
        self.assertFalse(self.step({}, (0, .24, .80), True))

    def test_wrong_side_over_top_and_straight_through_rejected(self):
        for x, z in ((-.12, .80), (.12, 1.0), (0, .80)):
            state = {}
            self.step(state, (x, -.20, z), True)
            self.assertFalse(self.step(state, (x, .03, z)))

    def test_updater_only_changes_requested_task_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            other = root / 'pass_rope_through_ring_0.json'
            other.write_text('{"custom": true}')
            with patch.object(assets, 'LAYOUTS', root):
                assets.update_posts_layout()
                target = root / 'route_rope_around_posts_0.json'
                before = target.read_text()
                assets.update_posts_layout()
                self.assertEqual(before, target.read_text())
            self.assertEqual(other.read_text(), '{"custom": true}')
            layout = json.loads(before)
            self.assertEqual(set(layout['Geometry']), {'vase', 'mousemat'})
            self.assertEqual(layout['Geometry']['vase'][0]['category_idx'], 3)
