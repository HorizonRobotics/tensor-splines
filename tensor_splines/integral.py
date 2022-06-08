import torch

def _integrate_spiro_12th_degree(k_parameters: torch.Tensor):
    # The following tensors are all of shape [*B]
    t1_1 = k_parameters[..., 0]
    t1_2 = .5 * k_parameters[..., 1]
    t1_3 = (1./6) * k_parameters[..., 2]
    t1_4 = (1./24) * k_parameters[..., 3]

    t2_2 = t1_1 * t1_1
    t2_3 = 2 * (t1_1 * t1_2)
    t2_4 = 2 * (t1_1 * t1_3) + t1_2 * t1_2
    t2_5 = 2 * (t1_1 * t1_4 + t1_2 * t1_3)
    t2_6 = 2 * (t1_2 * t1_4) + t1_3 * t1_3
    t2_7 = 2 * (t1_3 * t1_4)
    t2_8 = t1_4 * t1_4

    t3_4 = t2_2 * t1_2 + t2_3 * t1_1
    t3_6 = t2_2 * t1_4 + t2_3 * t1_3 + t2_4 * t1_2 + t2_5 * t1_1
    t3_8 = t2_4 * t1_4 + t2_5 * t1_3 + t2_6 * t1_2 + t2_7 * t1_1
    t3_10 = t2_6 * t1_4 + t2_7 * t1_3 + t2_8 * t1_2

    t4_4 = t2_2 * t2_2
    t4_5 = 2 * (t2_2 * t2_3)
    t4_6 = 2 * (t2_2 * t2_4) + t2_3 * t2_3
    t4_7 = 2 * (t2_2 * t2_5 + t2_3 * t2_4)
    t4_8 = 2 * (t2_2 * t2_6 + t2_3 * t2_5) + t2_4 * t2_4
    t4_9 = 2 * (t2_2 * t2_7 + t2_3 * t2_6 + t2_4 * t2_5)
    t4_10 = 2 * (t2_2 * t2_8 + t2_3 * t2_7 + t2_4 * t2_6) + t2_5 * t2_5

    t5_6 = t4_4 * t1_2 + t4_5 * t1_1
    t5_8 = t4_4 * t1_4 + t4_5 * t1_3 + t4_6 * t1_2 + t4_7 * t1_1
    t5_10 = t4_6 * t1_4 + t4_7 * t1_3 + t4_8 * t1_2 + t4_9 * t1_1

    t6_6 = t4_4 * t2_2
    t6_7 = t4_4 * t2_3 + t4_5 * t2_2
    t6_8 = t4_4 * t2_4 + t4_5 * t2_3 + t4_6 * t2_2
    t6_9 = t4_4 * t2_5 + t4_5 * t2_4 + t4_6 * t2_3 + t4_7 * t2_2
    t6_10 = t4_4 * t2_6 + t4_5 * t2_5 + t4_6 * t2_4 + t4_7 * t2_3 + t4_8 * t2_2

    t7_8 = t6_6 * t1_2 + t6_7 * t1_1
    t7_10 = t6_6 * t1_4 + t6_7 * t1_3 + t6_8 * t1_2 + t6_9 * t1_1

    t8_8 = t6_6 * t2_2
    t8_9 = t6_6 * t2_3 + t6_7 * t2_2
    t8_10 = t6_6 * t2_4 + t6_7 * t2_3 + t6_8 * t2_2

    t9_10 = t8_8 * t1_2 + t8_9 * t1_1

    t10_10 = t8_8 * t2_2

    u = torch.ones_like(k_parameters[..., 0])
    u -= (1./24) * t2_2 + (1./160) * t2_4 + (1./896) * t2_6 + (1./4608) * t2_8
    u += (1./1920) * t4_4 + (1./10752) * t4_6 + (1./55296) * t4_8 + (1./270336) * t4_10
    u -= (1./322560) * t6_6 + (1./1658880) * t6_8 + (1./8110080) * t6_10
    u += (1./92897280) * t8_8 + (1./454164480) * t8_10
    u -= 2.4464949595157930e-11 * t10_10
    v = (1./12) * t1_2 + (1./80) * t1_4
    v -= (1./480) * t3_4 + (1./2688) * t3_6 + (1./13824) * t3_8 + (1./67584) * t3_10
    v += (1./53760) * t5_6 + (1./276480) * t5_8 + (1./1351680) * t5_10
    v -= (1./11612160) * t7_8 + (1./56770560) * t7_10
    v += 2.4464949595157932e-10 * t9_10

    return u, v


def _integrate_spiro_12th_degree_n_parts(k_parameters: torch.Tensor, n: int):
    th1 = k_parameters[..., 0]
    th2 = .5 * k_parameters[..., 1]
    th3 = (1./6) * k_parameters[..., 2]
    th4 = (1./24) * k_parameters[..., 3]

    ds = 1. / n
    ds2 = ds * ds
    ds3 = ds2 * ds
    k = k_parameters * ds

    x = torch.zeros_like(th1)
    y = torch.zeros_like(th1)

    s = .5 * ds - .5
    for i in range(n):
        s2 = s * s
        s3 = s2 * s
        W = torch.tensor([[1., s, .5 * s2, (1./6) * s3],
                          [0., ds, ds * s, .5 * ds * s2],
                          [0., 0., ds2, s * ds2],
                          [0., 0., 0., ds3]], device=k_parameters.device)
        km = torch.einsum('...ij,...j->...i', W, k)
        u, v = _integrate_spiro_12th_degree(km)

        th = (((th4 * s + th3) * s + th2) * s + th1) * s
        cos = torch.cos(th)
        sin = torch.sin(th)

        x += cos * u - sin * v
        y += cos * v + sin * u
        s += ds
    return x * ds, y * ds


def integrate_eular_spiral(k_parameters: torch.Tensor):
    estimated_error_raw = .2 * k_parameters[..., 0] * k_parameters[..., 0] + torch.abs(
        k_parameters[..., 1])
    if (estimated_error_raw < 1.0).all():
        return _integrate_spiro_12th_degree(k_parameters)
    return _integrate_spiro_12th_degree_n_parts(k_parameters, n=4)
