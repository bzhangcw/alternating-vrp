"""
functional interface module for bcd
% consider the model:
%   min c'x
%     s.t. Ax<=b, Bx<=d, x \in {0,1}
%       - A: binding part
%       - B: block diagonal decomposed part
% ALM:
%   min c'x+rho*\|max{Ax-b+lambda/rho,0}\|^2
%     s.t. Bx<=d, x \in {0,1}
% implement the BCD to solve ALM (inc. indefinite proximal version),
% - ordinary linearized proximal BCD
% - indefinite proximal BCD which includes an extrapolation step.
% - restart utilities
"""
import functools
from typing import Dict
import time
import numpy as np
import scipy
import scipy.sparse.linalg as ssl
import tqdm
from gurobipy import *

from route import Route

from enum import IntEnum


class Primal(IntEnum):
    Null = 0  # there is no primal algorithm


class Dual(IntEnum):
    # proximal linearized BCD
    ProxLinear = 0  # binding only
    ProxLinearCapacity = 1  # binding + capacity
    ProxLinearCapacityTW = 2  # binding + capacity + TW
    NonlinearProxLinearCapacityTW = 3  # binding + capacity + TW, but we use bilinear TW constraint


class DualSubproblem(IntEnum):
    Route = 1
    CapaRoute = 2
    CapaWindowRoute = 3


# BCD params
ALGORITHM_TYPE = {
    # solve proximal linear sub-problem
    # relax capacity and time-window
    "prox-I": (
        Dual.ProxLinearCapacity, DualSubproblem.Route, Primal.Null
    ),
    "prox-II": (
        Dual.ProxLinearCapacityTW, DualSubproblem.Route, Primal.Null
    )
}


class BCDParams(object):

    def __init__(self):
        self.kappa = 0.2
        self.alpha = 1.0
        self.beta = 1
        self.gamma = 0.1  # parameter for argmin x
        self.changed = 0
        self.num_stuck = 0
        self.eps_num_stuck = 3
        self.iter = 0
        self.lb = 1e-6
        self.lb_arr = []
        self.ub_arr = []
        self.gap = 1
        self.dual_method = 0
        self.dual_update = 0
        self.primal_method = 0
        self.max_number = 1
        self.norms = ([], [], [])
        self.multipliers = ([], [], [])
        self.itermax = 10000
        self.dual_linearize_max = 10
        self.dual_linearize = True

        self.parse_environ()

    def parse_environ(self):
        import os

        self.dual_update, self.dual_method, self.primal_method \
            = ALGORITHM_TYPE[os.environ.get('method', "prox-I")]

    def update_bound(self, lb):
        if lb >= self.lb:
            self.lb = lb
            self.changed = 1
            self.num_stuck = 0
        else:
            self.changed = 0
            self.num_stuck += 1

        if self.num_stuck >= self.eps_num_stuck:
            self.kappa *= 0.5
            self.num_stuck = 0
        self.lb_arr.append(lb)

    def update_incumbent(self, ub):
        self.ub_arr.append(ub)

    def update_gap(self):
        _best_ub = min(self.ub_arr)
        _best_lb = max(self.lb_arr)
        self.gap = (_best_ub - _best_lb) / (abs(_best_lb) + 1e-3)


def _Ax(block, x):
    return block['A'] @ x


@np.vectorize
def _nonnegative(x):
    return max(x, 0)


def show_log_header(bcdpar: BCDParams):
    headers = ["k", "t", "c'x", "lobj", "|Ax - b|", "error", "rho", "tau", "iter"]
    slots = ["{:^3s}", "{:^7s}", "{:^9s}", "{:^9s}", "{:^10s}", "{:^10s}", "{:^9s}", "{:^9s}", "{:4s}"]
    _log_header = " ".join(slots).format(*headers)
    lt = _log_header.__len__()
    print("*" * lt)
    print(("{:^" + f"{lt}" + "}").format("BCD for MILP"))
    print(("{:^" + f"{lt}" + "}").format("(c) Chuwen Zhang, Shanwen Pu, Rui Wang"))
    print(("{:^" + f"{lt}" + "}").format("2023"))
    print("*" * lt)
    print("Algorithm details:")
    print((f" :dual_update           : {bcdpar.dual_update.name}"))
    print((f" :dual_subproblem       : {bcdpar.dual_method.name}"))
    print((f" :dual_linearize        : {bcdpar.dual_linearize}"))
    print((f" :dual_linearize_inner  : {bcdpar.dual_linearize_max}"))
    print((f" :primal_method         : {bcdpar.primal_method.name}"))
    print("*" * lt)
    print(_log_header)
    print("*" * lt)


def optimize(bcdpar: BCDParams, block_data: Dict, route: Route):
    """

    Args:
        bcdpar: BCDParam
        block_data:  matlab dict storing bcd-styled block vrp instance
            self.block_data["A"] = []  # couple A
            self.block_data["b"] = np.ones((len(V) - 1, 1))
            self.block_data["B"] = []  # sub A
            self.block_data["q"] = []  # sub b
            self.block_data["c"] = []  # demand
            self.block_data["C"] = []  # capacity
            self.block_data["d"] = []  # obj coeff
    Note:
        % basic model:
            self.block_data["B"] = []  # sub A
            self.block_data["q"] = []  # sub b
        % capacity:
            self.block_data["c"] = []  # demand
            self.block_data["C"] = []  # capacity
        % time window:
            self.block_data['M'], self.block_data['T'],
            self.block_data['a'], self.block_data['b']
    Returns:

    """
    # data
    start = time.time()
    A, b, B, q = block_data['A'], block_data['b'], block_data['B'], block_data['q']
    c, C, d = block_data['c'], block_data['C'], block_data['d']
    P, T, l, u = block_data['P'], block_data['T'], block_data['l'], block_data['u']
    M = 1e3  # todo => block_data['M']
    q = - T.reshape((-1, 1)) + M

    # query model size
    A1 = A[0]
    m, n = A1.shape
    nblock = len(A)
    Anorm = 20  # scipy.sparse.linalg.norm(A) / 10

    # alias
    rho = 100
    tau = 1 / 30
    sigma = 1.1
    # primal variables
    xk = [np.ones((n, 1)) for _ in A]
    wk = [np.ones((m + 1, 1)) for _ in A]
    # projection operator for w
    _proj = lambda x, θ: -np.linalg.solve(P.T @ P, P.T @ (M * x - q + θ / rho))
    # dual variables
    lbd = rho * np.ones((m, 1))
    mu = [rho for _ in A]
    theta = [rho * np.ones((n, 1)) for _ in A]

    # logger

    show_log_header(bcdpar)

    # - k: outer iteration num
    # - it: inner iteration num
    # - idx: 1-n block idx
    #       it may not be the block no
    # A_k x_k
    _vAx = {idx: _A @ xk[idx] for idx, _A in enumerate(A)}
    _vBx = {idx: 0 for idx, _A in enumerate(A)}  # violation of capacity
    _vWx = {idx: 0 for idx, _A in enumerate(A)}  # violation of time-window
    # c_k x_k
    _vcx = {idx: (_c @ xk[idx]).trace() for idx, _c in enumerate(c)}  # original obj
    _vcxl = {idx: (_c @ xk[idx]).trace() for idx, _c in enumerate(c)}  # lagrangian obj for each block
    # x_k - x_k* (fixed point error)
    _eps_fix_point = {idx: 0 for idx, _ in enumerate(A)}

    for k in range(bcdpar.itermax):
        ############################################
        # dual update (BCD/ALM/ADMM)
        ############################################
        # update auxilliary `w` time window if relaxed
        if bcdpar.dual_update in {Dual.ProxLinearCapacityTW, Dual.NonlinearProxLinearCapacityTW}:
            for idx in range(nblock):
                wk[idx] = _w = _proj(xk[idx], theta[idx])

        # inner iteration
        for it in range(bcdpar.dual_linearize_max if bcdpar.dual_linearize else 1):
            # idx: A[idx]@x[idx]
            for idx in range(nblock):
                ############################################
                # update gradient
                ############################################
                Ak = A[idx]
                _Ax = sum(_vAx.values())
                ############################################
                # create dual cost
                ############################################
                if bcdpar.dual_update == Dual.ProxLinear:
                    _d = d[idx].reshape((-1, 1)) + Ak.T @ lbd + rho * Ak.T @ (_Ax - b) \
                         + (-xk[idx] / tau + 0.5 / tau)
                elif bcdpar.dual_update == Dual.ProxLinearCapacity:
                    _d = d[idx].reshape((-1, 1)) + Ak.T @ lbd + rho * Ak.T @ (_Ax - b) \
                         + rho * (c[idx] @ _nonnegative(xk[idx] - C[idx][0] + mu[idx] / rho)).trace() \
                         + (-xk[idx] / tau + 0.5 / tau)
                elif bcdpar.dual_update == Dual.ProxLinearCapacityTW:
                    _d = d[idx].reshape((-1, 1)) + Ak.T @ lbd + rho * Ak.T @ (_Ax - b) \
                         + rho * (c[idx] @ _nonnegative(xk[idx] - C[idx][0] + mu[idx] / rho)).trace() \
                         + rho * M * _nonnegative(P @ wk[idx] + M * xk[idx] - q + theta[idx] / rho) \
                         + (-xk[idx] / tau + 0.5 / tau)
                elif bcdpar.dual_update == Dual.NonlinearProxLinearCapacityTW:
                    # todo for WR
                    # add update for nonlinear (bilinear timewindow)
                    pass
                else:
                    pass
                ############################################
                # solve dual subproblem
                ############################################
                if bcdpar.dual_method == DualSubproblem.Route:
                    _x = route.solve_primal_by_mip(_d.flatten())
                elif bcdpar.dual_method == DualSubproblem.CapaRoute:
                    # _x = route.solve_primal_by_mip(_d.flatten())
                    raise ValueError("not implemented")
                elif bcdpar.dual_method == DualSubproblem.CapaWindowRoute:
                    # IN THIS MODE, YOU ALSO HAVE w,
                    # otherwise, you update in the after bcd for x
                    #   if it is a VRPTW
                    # _x = route.solve_primal_by_mip(_d.flatten())
                    # wk[idx] = _w = _proj(xk[idx], theta[idx])
                    raise ValueError("not implemented")
                else:
                    raise ValueError("not implemented")

                ############################################
                # summarize
                ############################################
                _v_sp = (_d.T @ _x).trace()
                #
                _eps_fix_point[idx] = np.linalg.norm(xk[idx] - _x)

                # update this block
                xk[idx] = _x
                _vAx[idx] = Ak @ _x
                _vcx[idx] = _cx = d[idx] @ _x.flatten()
                _vcxl[idx] = _cx
                # calculate violation of capacity if relaxed
                if bcdpar.dual_update in {Dual.ProxLinear}:
                    _vBx[idx] = 0
                else:
                    _vBx[idx] = _nonnegative((c[idx] @ _x).trace() - C[idx][0])
                # calculate violation of time window if relaxed
                if bcdpar.dual_update in {Dual.ProxLinearCapacityTW, Dual.NonlinearProxLinearCapacityTW}:
                    _vWx[idx] = _nonnegative(P @ wk[idx] + M * xk[idx] - q)
                else:
                    _vWx[idx] = np.zeros_like(theta[idx])

            # fixed-point eps
            if sum(_eps_fix_point.values()) < 1e-4:
                break

        ############################################
        # primal update: some heuristic
        ############################################
        # todo for PSW
        # ADD A PRIMAL METHOD FOR FEASIBLE SOLUTION

        _iter_time = time.time() - start
        _Ax = sum(_vAx.values())

        eps_pfeas = \
            np.linalg.norm(_Ax - b, np.inf) \
            + sum(_vBx.values()) \
            + sum(np.linalg.norm(_, np.inf) for _ in _vWx.values())
        cx = sum(_vcx.values())
        # todo for WR, how to calculate lower bound?
        lobj = sum(_vcxl.values()) + (lbd.T @ (_Ax - b)).trace()
        eps_fp = sum(_eps_fix_point.values())
        _log_line = "{:03d} {:.1e} {:+.2e} {:+.2e} {:+.3e} {:+.3e} {:+.3e} {:.2e} {:04d}".format(
            k, _iter_time, cx, lobj, eps_pfeas, eps_fp, rho, tau, it + 1
        )
        print(_log_line)
        if eps_pfeas == 0 and eps_fp < 1e-4:
            break

        ############################################
        # update dual variables
        ############################################
        rho *= sigma
        lbd += rho * (_Ax - b)
        for idx in range(nblock):
            mu[idx] = _nonnegative(((c[idx] @ _x).trace() - C[idx][0]) * rho + mu[idx])
            theta[idx] = _nonnegative((P @ wk[idx] + M * xk[idx] - q) * rho + theta[idx])

        bcdpar.iter += 1

    return xk
