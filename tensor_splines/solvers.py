import numpy as np
import torch

from .integral import integrate_eular_spiral


def solve_euler_spiral(theta_in: torch.Tensor, theta_out: torch.Tensor):
    """

    All thetas must be within [-pi, pi].
    """
    assert theta_in.shape == theta_out.shape

    B = theta_in.shape
    theta_in = theta_in.to(torch.float64)
    theta_out = theta_out.to(torch.float64)

    error_old = theta_out - theta_in

    k_parameters = torch.zeros(*B, 4, device=theta_in.device, dtype=theta_in.dtype)
    k_parameters[..., 0] = theta_in + theta_out
    k_parameters[..., 1] = (6 * (1. - np.power((.5 / np.pi), 3))) * error_old

    k1_old = torch.zeros_like(theta_in)

    converged = torch.zeros_like(theta_in, dtype=torch.bool)
    u = torch.zeros_like(theta_in)
    v = torch.zeros_like(theta_in)
    for i in range(10):
        new_u, new_v = integrate_eular_spiral(k_parameters)
        u = torch.where(converged, u, new_u)
        v = torch.where(converged, v, new_v)
        k1 = k_parameters[..., 1]
        chord_theta = torch.atan2(v, u)
        error = (theta_out - theta_in) - (0.25 * k1 - 2 * chord_theta)
        converged = torch.logical_or(converged, error.abs() < 1e-9)
        if converged.all():
            break
        error_diff = error - error_old
        new_k1 = torch.where(converged,
                             k1,
                             k1 + (k1_old - k1) * error / error_diff)
        k1_old = k1.clone()
        error_old = error
        k_parameters[..., 1] = new_k1

    assert converged.all(), f"Cannot solve Euler Spiral, maximum error: {error.abs().max()}"

    chord = torch.hypot(u, v)
    return k_parameters, chord, chord_theta
