import unittest

import torch
import numpy as np

from tensor_splines import SpiroBatch


class SpiroTest(unittest.TestCase):
    def test_make_euler_spiral_by_length(self):
        spiro = SpiroBatch.make_euler_spiral_by_length(
            starts=torch.tensor([[0.0, 0.0],
                                 [0.0, 0.0],
                                 [0.0, 0.0]]),
            chord_theta=torch.tensor([0.5, 0.1, -0.4]),
            theta_in=torch.tensor([0.0, -0.2, 0.2]),
            average_curvature=torch.tensor([0.01, -0.02, 0.2]),
            lengths=torch.tensor([10.0, 10.0, 10.0]))
        self.assertTrue(torch.isclose(torch.tensor([10.0, 10.0, 10.0]),
                                      spiro.lengths).all())
        self.assertTrue(torch.isclose(torch.tensor([0.5, 0.1, -0.4]),
                                      spiro._chord_theta).all())
        s_in = torch.tensor([0.0, 0.0, 0.0])
        self.assertTrue(torch.isclose(
            torch.tensor([0.5, 0.1 - 0.2, -0.4 + 0.2]),
            spiro.theta(s_in)).all())
        s_out = torch.tensor([10.0, 10.0, 10.0])
        self.assertTrue(torch.isclose(
            torch.tensor([0.5 + 0.01 * 10, 0.1 - 0.2 - 0.02 * 10, -0.4 + 0.2 + 0.2 * 10]),
            spiro.theta(s_out)).all())

        
        
            
            
