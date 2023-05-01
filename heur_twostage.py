# A two-stage heuristic
#   assuming the dual solutions give the assignment,
#   in this case, use DP to solve the problem with "known cities"
import collections

import numpy as np


def solve(sa, Ak, d, route):
    indicator = np.ones(shape=Ak.shape[0], dtype=np.bool)
    sar = np.array(sa, dtype=np.int)
    # selected
    indicator[sar - 1] = True
    # selected has infinity cost
    _, edges = Ak[indicator, :].nonzero()

    xk = route.solve_primal_by_dp(
        d - d.max() * 2,
        select=(np.array([0, *sa], dtype=np.int), edges),
        verbose=True,
        # debugging=True
    )
    return xk


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
    xkh = {}
    for k, sa in assign.items():
        xkh[k] = solve(sa, A[k], d[k], route)
    return xkh
