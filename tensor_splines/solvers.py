import numpy as np
import torch

from .integral import integrate_eular_spiral


def solve_euler_spiral(theta_in: torch.Tensor, theta_out: torch.Tensor):
    """

    All thetas must be within [-pi, pi].
    """

    assert theta_in.shape == theta_out.shape

    error_old = theta_out - theta_in

    k_parameters = torch.zeros(*theta_in.shape, 4, device=theta_in.device)
    k_parameters[..., 0] = theta_in + theta_out
    k_parameters[..., 1] = (6 * (1. - np.power((.5 / np.pi), 3))) * error_old

    k1_old = torch.zeros_like(theta_in)

    converged = False
    for i in range(10):
        u, v = integrate_eular_spiral(k_parameters)
        k1 = k_parameters[..., 1]
        chord_theta = torch.atan2(v, u)
        error = (theta_out - theta_in) - (0.25 * k1 - 2 * chord_theta)
        if (error.abs() < 1e-9).all():
            converged = True
            break
        new_k1 = k1 + (k1_old - k1) * error / (error - error_old)
        k1_old = k1.clone()
        error_old = error
        k_parameters[..., 1] = new_k1

    assert converged, "Cannot solve Euler Spiral"

    chord = torch.hypot(u, v)
    return k_parameters, chord, chord_theta
