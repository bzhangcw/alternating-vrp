# A two-stage heuristic
#   assuming the dual solutions give the assignment,
#   in this case, use DP to solve the problem with "known cities"
import collections

import numpy as np


def solve(sa, edges, d, route, verbose=False):
    _d = d
    xk, _route = route.solve_primal_by_dp(
        _d - _d.max() * 2,
        select=(np.array([0, *sa], dtype=np.int), edges),
        dump=verbose,
        # debugging=True
    )
    return xk, _route


def main(xk, A, d, d_real, _vAx, route, verbose=True):
    """
    :param xk: dual solution
    :param route: route object to call DP.
    :return:
    """
    # covers
    vA = np.hstack(_vAx.values())
    cc = collections.defaultdict(float)
    assign = collections.defaultdict(list)

    # for idx in range(vA.shape[0]):
    #     for k in vA[idx].argsort():
    #         if route.vrp.c[idx + 1] + cc[k] <= route.vrp.C:
    #             assign[k].append(idx + 1)
    #             cc[k] += route.vrp.c[idx + 1]
    #             break
    # xkh = {}
    # for k, sa in assign.items():
    #     xkh[k] = solve(sa, A[k], d[k], route)
    # generate a primal feasible solution till every node is used.
    scope = set(np.array(range(vA.shape[0])) + 1)
    k = 0
    cost = 0
    xkh = []
    edges = [(s, t) for s in route.vrp.V for t in route.vrp.V if s != t]
    em = {k: s for k, (s, t) in enumerate(edges)}
    while True:
        idx = min(k, vA.shape[1] - 1)
        if verbose:
            print(k, scope)
        xx, r = solve(scope, edges, d[idx].flatten(), route, verbose)
        xkh.append(xx)
        scope = scope.difference(r)
        cost += (d_real[0] @ xx)[0]
        if verbose:
            print(k, r, scope)
        k += 1
        if len(scope) == 0:
            break

    return xkh, cost
