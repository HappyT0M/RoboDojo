"""Pure geometry regression checks for the stock-frame rope task."""
import math
import unittest
from utils.rope_task_metrics import crosses_frame_aperture


class FrameApertureTests(unittest.TestCase):
    def crossing(self, previous, current):
        return crosses_frame_aperture(
            previous, current, (-0.15, 0.15, 0.915),
            (math.sqrt(0.5), math.sqrt(0.5), 0, 0), (0.1755, 0.141), 0.024,
        )

    def test_forward_and_reverse(self):
        self.assertTrue(self.crossing((-0.15, 0.10, 0.915), (-0.15, 0.20, 0.915)))
        self.assertFalse(self.crossing((-0.15, 0.20, 0.915), (-0.15, 0.10, 0.915)))

    def test_ball_clearance_at_side_and_top(self):
        self.assertFalse(self.crossing((0.01, 0.10, 0.915), (0.01, 0.20, 0.915)))
        self.assertFalse(self.crossing((-0.15, 0.10, 1.045), (-0.15, 0.20, 1.045)))

    def test_no_crossing_or_path_around_edge(self):
        self.assertFalse(self.crossing((-0.15, 0.10, 0.915), (-0.15, 0.12, 0.915)))
        self.assertFalse(self.crossing((-0.35, 0.10, 0.915), (0.05, 0.20, 0.915)))

    def test_invalid_clearance(self):
        with self.assertRaises(ValueError):
            crosses_frame_aperture((0,0,1), (0,0,-1), (0,0,0), (1,0,0,0), (.1,.1), .2)
