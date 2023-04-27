# A two-stage heuristic
#   assuming the dual solutions give the assignment,
#   in this case, use DP to solve the problem with "known cities"
import collections

import numpy as np


def adjust_cost(sa, Ak, d, route):
    indicator = np.ones(shape=Ak.shape[0], dtype=np.bool)
    sar = np.array(sa, dtype=np.int)
    # unselected
    indicator[sar - 1] = False
    # unselected has infinity cost
    _, edges = Ak[indicator, :].nonzero()
    dt = d.copy()
    dt = dt - dt.max() * 2
    dt[edges] = 1e6

    # xk = route.solve_primal_by_dp(dt, verbose=True, debugging=True)
    return dt


def main(xk, A, d, _vAx, route):
    """
    :param xk: dual solution
    :param route: route object to call DP.
    :return:
    """
    # covers
    vA = np.hstack(_vAx.values())
    cc = collections.defaultdict(float)
    assign = collections.defaultdict(list)

    for idx in range(vA.shape[0]):
        for k in vA[idx].argsort():
            if route.vrp.c[idx + 1] + cc[k] <= route.vrp.C:
                assign[k].append(idx + 1)
                cc[k] += route.vrp.c[idx + 1]
                break

    for k, sa in assign.items():
        co = adjust_cost(sa, A[k], d[k], route)
