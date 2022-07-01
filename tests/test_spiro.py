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


    def test_conversion_to_and_from_tensor(self):
        spiro = SpiroBatch.make_euler_spiral_by_length(
            starts=torch.tensor([[0.0, 0.0],
                                 [0.0, 0.0],
                                 [0.0, 0.0]]),
            chord_theta=torch.tensor([0.5, 0.1, -0.4]),
            theta_in=torch.tensor([0.0, -0.2, 0.2]),
            average_curvature=torch.tensor([0.01, -0.02, 0.2]),
            lengths=torch.tensor([10.0, 10.0, 10.0]))

        x = spiro.to_tensor()
        self.assertEqual((3, 10), x.shape)

        spiro1 = SpiroBatch.from_tensor(x)
        self.assertTrue(torch.isclose(spiro1.starts, spiro.starts).all())
        self.assertTrue(torch.isclose(spiro1.ends, spiro.ends).all())
        self.assertTrue(torch.isclose(spiro1.k_parameters, spiro.k_parameters).all())
        self.assertTrue(torch.isclose(spiro1.lengths, spiro.lengths).all())
        self.assertTrue(torch.isclose(spiro1._chord_theta, spiro._chord_theta).all())
        self.assertTrue(torch.isclose(spiro1._angle_offsets, spiro._angle_offsets).all())
        
    def test_make_euler_spiral_by_jerk(self):
        spiro = SpiroBatch.make_euler_spiral_by_jerk(
            starts=torch.tensor([[0.0, 0.0],
                                 [0.0, 0.0],
                                 [0.0, 0.0]]),
            theta_start=torch.tensor([0.1, -0.2, 0.0]),
            average_curvature=torch.tensor([0.01, -0.02, 0.2]),
            lengths=torch.tensor([10.0, 10.0, 10.0]),
            theta_jerk=torch.tensor([0.001, -0.005, -0.02]))
        self.assertTrue(torch.isclose(torch.tensor([10.0, 10.0, 10.0]),
                                      spiro.lengths).all())
        self.assertTrue(torch.isclose(torch.tensor([0.1, -0.2, 0.0]),
                                      spiro.theta(torch.zeros(3)),
                                      rtol=1e-5, atol=1e-6).all())
        self.assertTrue(torch.isclose(torch.tensor([0.2, -0.4, 2.0]),
                                      spiro.theta(torch.full((3,), fill_value=10.0))).all())
        self.assertTrue(torch.isclose(torch.tensor([0.1, -0.5, -2.0]),
                                      spiro.k_parameters[..., 1]).all())
        
