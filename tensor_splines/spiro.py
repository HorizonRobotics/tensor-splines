from __future__ import annotations
import torch
import numpy as np

from .solvers import solve_euler_spiral
from .integral import integrate_eular_spiral

class SpiroBatch(object):
    """A SpiroBatch holds a batch of spiro curves.

    Spiro curve is a specific kind of splines. Mathematically, it is described
    by the following:

        theta(s) = k0 * s + 1/2 k1 * s^2 + 1/6 k2 * s^3 + 1/24 k3 * s^4

    where s goes from -0.5 to 0.5, with -0.5 corresponds to the start point and
    0.5 corresponds to the end point.

    The main factory methods are used to create the batch of spiro with tensors
    and their batch operations.

    """
    def __init__(self, k_parameters: torch.Tensor,
                 starts: torch.Tensor,
                 ends: torch.Tensor,
                 lengths: torch.Tensor):
        """General constructor.

        Normally, you should not directly call this, and should rather call the
        factory methods below.

        Args:

            k_parameters: the k0, k1, k2 and k3 for each of the curves in the
                bacth, [*B, 4]
            starts: the batch of start points for each of the curves, [*B, 2]
            ends: the batch of end points for each of the curves, [*B, 2]
            lengths: the actual length of each of the curves, [*B]

        """
        B = k_parameters.shape[:-1]
        assert k_parameters.shape == (*B, 4)
        assert starts.shape == ends.shape == (*B, 2)
        assert lengths.shape == B

        self._k_parameters = k_parameters
        self._lengths = lengths
        self._starts = starts
        self._ends = ends

    @staticmethod
    def make_euler_spiral(starts: torch.Tensor,
                          ends: torch.Tensor,
                          theta_in: torch.Tensor,
                          theta_out: torch.Tensor) -> SpiroBatch:
        """Factory method to create a batch of Euler Spiral curves.

        An Euler spiral is a special case of spiro where both k2 and k3 are 0.

        This method will create the whole batch of spiros simultaneously.

        Args:

            starts: the batch of start points
            ends: the batch of end points
            theta_in: the relative angle (in radian) w.r.t. the chord at the
                start point of each curve
            theta_out: the relative angle (in radian) w.r.t. the chord at the
                end point of each curve

        Returns:

            The constructed batch of spiro (Euler spiral) curves.

        """
        k_parameters, chord, chord_beta = solve_euler_spiral(theta_in, theta_out)
        k_parameters = k_parameters.to(theta_in.dtype)
        chord = chord.to(theta_in.dtype)
        chord_beta = chord_beta.to(theta_in.dtype)
        return SpiroBatch(-k_parameters,
                          starts, ends,
                          lengths=(ends - starts).norm(dim=-1) / chord)

    @property
    def k_parameters(self) -> torch.Tensor:
        """Access the k0, k1, k2 and k3 for each one in the batch.
        """
        return self._k_parameters

    @property
    def batch_shape(self):
        return self._k_parameters.shape[:-1]

    @property
    def lengths(self) -> torch.Tensor:
        return self._lengths

    def curvature(self, s: torch.Tensor) -> torch.Tensor:
        """Compute the curvature (i.e. d(theta)/ds) at the specified s.

        Note that here the s is along the actual arc length (as opposite to the
        prototype s which is within the range [-0.5, 0.5]). To compute the
        curvature, a substitution of the variable is conducted.

        """
        s_proto = s / self._lengths - 0.5
        return (self._k_parameters[..., 0] +
                self._k_parameters[..., 1] * s_proto +
                self._k_parameters[..., 2] * torch.pow(s_proto, 2) * 0.5 +
                self._k_parameters[..., 3] * torch.pow(s_proto, 3) / 6.0) / self._lengths

    def render_single(self, b) -> torch.Tensor:
        """Sample the points on the specified single curve in the batch.

        Returns:

           A tensor (polyline) of shape [n, 2], where n is the number of points
           sampled. The caller does not decide which points are sampled. The
           underlying algorithm will make the decision, so that the returned
           polyline is smooth enough.

        """
        points = []

        def _render_rec(ks, x0, y0, x1, y1, depth):
            bend = np.abs(ks[0]) + np.abs(.5 * ks[1]) + np.abs(.125 * ks[2]) + np.abs((1./48) * ks[3])
            segCh = np.hypot(x1 - x0, y1 - y0)
            segTh = np.arctan2(y1 - y0, x1 - x0)
            xy_u, xy_v = integrate_eular_spiral(torch.tensor(ks, dtype=torch.float64))
            xy_u = xy_u.numpy()
            xy_v = xy_v.numpy()
            ch = np.hypot(xy_u, xy_v)
            th = np.arctan2(xy_v, xy_u)
            scale = segCh / ch
            rot = segTh - th
            if depth > 5 or bend < 0.1:
                thEven = (1.0/384) * ks[3] + (1/8) * ks[1] + rot
                thOdd = (1.0/48) * ks[2] + 0.5 * ks[0]
                ul = (scale * (1/3)) * np.cos(thEven - thOdd)
                vl = (scale * (1/3)) * np.sin(thEven - thOdd)
                ur = (scale * (1/3)) * np.cos(thEven + thOdd)
                vr = (scale * (1/3)) * np.sin(thEven + thOdd)
                points.append((x0 + ul, y0 + vl))
                points.append((x1 - ur, y1 - vr))
            else:
                # Recursively subdivide
                ksub = np.array([
                    .5 * ks[0] - .125 * ks[1] + (1./64) * ks[2] - (1./768) * ks[3],
                    .25 * ks[1] - (1./16) * ks[2] + (1./128) * ks[3],
                    .125 * ks[2] - (1./32) * ks[3],
                    (1./16) * ks[3]
                ], dtype=np.float32)
                thsub = rot - .25 * ks[0] + (1./32) * ks[1] - (1./384) * ks[2] + (1./6144) * ks[3]
                cth = 0.5 * scale * np.cos(thsub)
                sth = 0.5 * scale * np.sin(thsub)
                xysub_u, xysub_v = integrate_eular_spiral(torch.tensor(ksub, dtype=torch.float64))
                xysub_u = xysub_u.numpy()
                xysub_v = xysub_v.numpy()
                xmid = x0 + cth * xysub_u - sth * xysub_v
                ymid = y0 + cth * xysub_v + sth * xysub_u
                _render_rec(ksub, x0, y0, xmid, ymid, depth + 1)
                points.append((xmid, ymid))
                ksub[0] += 0.25 * ks[1] + (1/384) * ks[3]
                ksub[1] += 0.125 * ks[2]
                ksub[2] += (1/16) * ks[3]
                _render_rec(ksub, xmid, ymid, x1, y1, depth + 1)

        _render_rec(self._k_parameters[b].cpu().numpy(),
                    self._starts[b, 0].cpu().numpy(),
                    self._starts[b, 1].cpu().numpy(),
                    self._ends[b, 0].cpu().numpy(),
                    self._ends[b, 1].cpu().numpy(), 0)
        return torch.tensor(points)

    def plot_single(self, b, ax, color='b'):
        """Sample the points of the specified curve in the batch and draw it.

        """
        points = self.render_single(b)
        ax.plot(points[:, 0].numpy(), points[:, 1].numpy(), color=color)
