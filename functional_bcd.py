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
import argparse
import functools
from typing import Dict, Tuple
import time
import numpy as np
import scipy
import scipy.sparse.linalg as ssl
import tqdm
from gurobipy import *
import networkx as nx

from route import Route

from enum import IntEnum

from heur_seq import heur_seq
from vrp import VRP


class Primal(IntEnum):
    Null = 0  # there is no primal algorithm
    SetPar = 1  # set partitioning


class Dual(IntEnum):
    # proximal linearized BCD
    ProxLinear = 0  # binding only
    ProxLinearCapacity = 1  # binding + capacity
    ProxLinearCapacityTW = 2  # binding + capacity + TW
    NonlinearProxLinearCapacityTW = (
        3  # binding + capacity + TW, but we use bilinear TW constraint
    )
    ProxLinearCapacity_nal = 4
    linearProxLinearCapacityTW_cx = 5
    linearProxLinearCapacityTW_nx = 6
    # NonlinearProxLinearCapacityTW = 7
    # NonlinearProxLinearCapacityTW_nal=8


class DualSubproblem(IntEnum):
    Route = 1
    CapaRoute = 2
    CapaWindowRoute = 3
    Assignment = 4
    CapaAssignment = 5


# BCD params
ALGORITHM_TYPE = {
    # solve proximal linear sub-problem
    # relax capacity and time-window
    "prox-I": (Dual.ProxLinearCapacity, DualSubproblem.Route, Primal.Null),
    "prox-II": (Dual.ProxLinearCapacityTW, DualSubproblem.Route, Primal.Null),
    "prox-III": (Dual.ProxLinear, DualSubproblem.CapaRoute, Primal.Null),
    "prox-III-tw": (Dual.ProxLinear, DualSubproblem.CapaWindowRoute, Primal.Null),
    "prox-III-h": (Dual.ProxLinear, DualSubproblem.CapaRoute, Primal.SetPar),
    "prox-IV": (Dual.ProxLinear, DualSubproblem.Route, Primal.Null),
    "prox-V": (Dual.ProxLinearCapacity_nal, DualSubproblem.Route, Primal.Null),
    "prox-VI": (Dual.linearProxLinearCapacityTW_cx, DualSubproblem.Route, Primal.Null),
    "prox-VII": (Dual.linearProxLinearCapacityTW_nx, DualSubproblem.Route, Primal.Null),
    "prox-I-ap-no-primal": (Dual.ProxLinearCapacity, DualSubproblem.Assignment, Primal.Null),
    "prox-I-ap": (Dual.ProxLinearCapacity, DualSubproblem.Assignment, Primal.SetPar),
    "prox-I-apc-no-primal": (Dual.ProxLinear, DualSubproblem.CapaAssignment, Primal.Null),
    "prox-I-apc": (Dual.ProxLinear, DualSubproblem.CapaAssignment, Primal.SetPar),
    # "prox-VIII": (Dual.NonlinearProxLinearCapacityTW, DualSubproblem.Route, Primal.Null),
    # "prox-IX": (Dual.NonlinearProxLinearCapacityTW_nal, DualSubproblem.Route, Primal.Null),
}


class BCDParams(object):
    parser = argparse.ArgumentParser(
        "Alternating Method for VRP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        type=str,
        default="prox-III-tw",
        choices=[*list(ALGORITHM_TYPE.keys())],
        help="""
        Choose algorithm
        """
    )
    parser.add_argument(
        "--dual_linearize_max",
        type=int,
        default=20,
        help="""
        maximum inner iteration (linearization)
        """
    )
    parser.add_argument(
        "--iter_max",
        type=int,
        default=200,
        help="""
        maximum outer iteration
        """
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=2000,
        help="""
        maximum time limit
        """
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=2,
        help="""
        if verbosity > 1, print inner iteration info          
        """
    )
    parser.add_argument(
        "--fp",
        type=str,
        default="dataset/data/SolomonDataset_v2/r101-25",
        help="""data path"""
    )
    parser.add_argument(
        "--n_vehicles",
        type=int,
        default=12,
        help="""
        number of vehicles used
        """
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.1,
        help="""
        sigma
        """
    )
    parser.add_argument(
        "--tsig",
        type=float,
        default=1.1,
        help="""
        tau_sig
        """
    )
    parser.add_argument(
        "--rho0",
        type=float,
        default=1,
        help="""
        rho0
        """
    )


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
        self.iter_max = 10000
        self.seq_heur_iter_max = 3
        self.dual_linearize_max = 10
        self.dual_linearize = True
        self.args = None
        self.verbosity = 2
        self.time_limit = 0
        self.sigma = 1
        self.tsig = 1
        self.rho0 = 1
        self.parse_environ()

    def parse_environ(self):
        import os
        self.args = self.parser.parse_args()
        self.dual_update, self.dual_method, self.primal_method = ALGORITHM_TYPE[
            self.args.method
        ]
        self.dual_linearize_max = self.args.dual_linearize_max
        self.iter_max = self.args.iter_max
        self.verbosity = self.args.verbosity
        self.fp = self.args.fp
        self.time_limit = self.args.time_limit
        self.n_vehicles = self.args.n_vehicles
        self.sigma = self.args.sigma
        self.tsig = self.args.tsig
        self.rho0 = self.args.rho0

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
    return block["A"] @ x


def has_conflict(j, s_j, i, s_i, var_map):
    city_j = set()
    city_i = set()

    x_j = np.fromstring(s_j)
    y_i = np.fromstring(s_i)

    for s in range(len(x_j)):
        if x_j[s] > 0.5:
            var_name = var_map[j][s]
            city_j.union(get_cities(var_name))

    for s in range(len(y_i)):
        if y_i[s] > 0.5:
            var_name = var_map[i][s]
            city_j.union(get_cities(var_name))

    return len(city_j.intersection(city_i)) > 0


def get_cities(var_name):
    assert var_name.startswith("x")
    return var_name[2:].split("]")[0].split(",")[:2]


def detect_conflict(G, _s, idx, block_nodes, var_map):
    for node in block_nodes[idx]:
        if node == _s:
            continue
        G.add_edge(_s, node)

    for i in range(len(block_nodes)):
        if i == idx:
            continue
        for node in block_nodes[i]:
            if has_conflict(idx, _s, i, np.fromstring(node), var_map):
                G.add_edge(_s, node)


def mis_heur(G: nx.Graph, xk, params, c):
    nblock, A, b, k, n, d = params
    circs = nx.maximal_independent_set(G)
    # circs = nx.algorithms.approximation.max_independent_set(G)
    print("MIS Heur: num of feas vehicle = {}".format(len(circs)))

    # A_k x_k
    _vAx = {idx: _A @ xk[idx] for idx, _A in enumerate(A)}
    _vBx = {idx: 0 for idx, _A in enumerate(A)}  # violation of capacity
    _vWx = {idx: 0 for idx, _A in enumerate(A)}  # violation of time-window
    # c_k x_k
    _vcx = {idx: (_c @ xk[idx]).trace() for idx, _c in enumerate(c)}  # original obj
    _vcxl = {
        idx: (_c @ xk[idx]).trace() for idx, _c in enumerate(c)
    }  # lagrangian obj for each block
    # x_k - x_k* (fixed point error)
    _eps_fix_point = {idx: 0 for idx, _ in enumerate(A)}
    for idx in range(nblock):
        Ak = A[idx]
        _x = xk[idx]
        ############################################
        # summarize
        ############################################
        #
        _eps_fix_point[idx] = np.linalg.norm(xk[idx] - _x)

        # update this block
        xk[idx] = _x
        _vAx[idx] = Ak @ _x
        _vcx[idx] = _cx = d[idx] @ _x.flatten()
        # calculate violation of capacity if relaxed
    _heur_time = time.time()
    xk = [np.fromstring(_circ, dtype=np.float64).reshape((n, 1)) for _circ in circs]
    _vAx = {idx: _A @ xk[idx] for idx, _A in enumerate(A)}
    _Ax = sum(_vAx.values())

    eps_pfeas = (
            np.linalg.norm(_Ax - b, np.inf)
            + sum(_vBx.values())
            + sum(np.linalg.norm(_, np.inf) for _ in _vWx.values())
    )
    cx = sum(_vcx.values())

    eps_fp = sum(_eps_fix_point.values())
    _log_line = (
        "{:03d} {:.1e} {:+.2e} {:+.2e} {:+.3e} {:+.3e} {:+.3e} {:.2e} {:04d}H".format(
            k, 0, cx, 0, eps_pfeas, eps_fp, 0, 0, 0
        )
    )
    print(_log_line)


def convert_xk_vec(xk, var_map, N):
    x = np.zeros(N)
    indice = np.where(xk.reshape((N, N - 1)) == 1)

    cities = indice[0].tolist()
    cities2 = indice[1].tolist()
    for i in cities + cities2:
        x[i] = 1

    return x


def set_par_heur(list_xk, c, var_map, N):
    m = Model("heur")
    m.setParam("LogToConsole", 0)
    x = tupledict()
    circs = tupledict()

    select_constrs = tupledict()
    for idx in range(len(list_xk)):
        for i in range(len(list_xk[idx])):
            x[idx, i] = m.addVar(vtype=GRB.BINARY)

            _x = list_xk[idx][i]
            circs[idx, i] = convert_xk_vec(_x, var_map, N)

        f_ps = x.select(idx, "*")

        if len(f_ps) > 0:
            select_constrs[idx] = m.addConstr(quicksum(f_ps) == 1, name=f"One[{idx}]")
        else:
            print("no circ for train:", idx)

    Exprs = [0] * N
    partition_constrs = tupledict()
    for j in range(N):
        Exprs[j] = quicksum(
            circs[idx, i][j] * x[idx, i]
            for idx in range(len(list_xk))
            for i in range(len(list_xk[idx]))
        )
        partition_constrs[j] = m.addConstr(Exprs[j] == 1)

    m.setObjective(
        quicksum(
            c[idx][i] * x[idx, i]
            for idx in range(len(list_xk))
            for i in range(len(list_xk[idx]))
        )
    )
    m.optimize()

    if m.status == GRB.INFEASIBLE:
        print("can't find feasible solution!")
        m.remove(partition_constrs)
        m.remove(select_constrs)

        for idx in range(len(list_xk)):
            m.addConstr(x.sum(idx, "*") <= 1, name=f"One[{idx}]")
        for j in range(N):
            partition_constrs[j] = m.addConstr(Exprs[j] <= 1)
        m.setObjective(quicksum(Exprs), sense=GRB.MAXIMIZE)
        m.optimize()
        print(
            "maximum coverage: ",
            m.objval / N,
            f"with {x.sum('*', '*').getValue()} cars",
        )


@np.vectorize
def _nonnegative(x):
    return max(x, 0)


def show_log_header(bcdpar: BCDParams):
    headers = ["k", "t", "c'x", "c'x (H)", "lobj", "|Ax - b|", "|cx-C|", "error", "rhol", "rhom", "tau", "iter"]
    # headers = ["k", "t", "c'x", "lobj", "|Ax - b|", "|cx-C|", "error", "rhol", "rhom", "tau", "iter"]
    slots = [
        "{:^5s}",
        "{:^7s}",
        "{:^8s}",
        "{:^9s}",
        "{:^8s}",
        "{:^10s}",
        "{:^10s}",
        "{:^10s}",
        "{:^9s}",
        "{:^9s}",
        "{:^9s}",
        "{:4s}",
    ]
    _log_header = " ".join(slots).format(*headers)
    lt = _log_header.__len__()
    print("*" * lt)
    print(("{:^" + f"{lt}" + "}").format("BCD for MILP"))
    print(("{:^" + f"{lt}" + "}").format("(c) Chuwen Zhang, Shanwen Pu, Rui Wang"))
    print(("{:^" + f"{lt}" + "}").format("2023"))
    print("*" * lt)
    print("Algorithm details:")
    print((f" :file_name             : {bcdpar.fp}"))
    print((f" :dual_update           : {bcdpar.dual_update.name}"))
    print((f" :dual_subproblem       : {bcdpar.dual_method.name}"))
    print((f" :dual_linearize        : {bcdpar.dual_linearize}"))
    print((f" :dual_linearize_inner  : {bcdpar.dual_linearize_max}"))
    print((f" :primal_method         : {bcdpar.primal_method.name}"))
    print("*" * lt)
    print(_log_header)
    print("*" * lt)


def optimize(bcdpar: BCDParams, vrps: Tuple[VRP, VRP], route: Route):
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
    start_time = time.time()
    vrp, vrp_clone = vrps
    block_data = vrp.block_data
    A, b, B, q = block_data["A"], block_data["b"], block_data["B"], block_data["q"]
    c, C, d = block_data["c"], block_data["C"], block_data["d"]
    P, T, l, u = block_data["P"], block_data["T"], block_data["l"], block_data["u"]
    M = 1e5  # todo => block_data['M']
    q = -T.reshape((-1, 1)) + M
    var_map = dict()
    for k, v in block_data["ind"].items():
        var_map[k] = dict()
        start = min(v.keys())
        for j, val in v.items():
            var_map[k][j - start] = val
    V = block_data["V"]
    N = len(V)
    block_nodes = [[] for _ in A]
    for j in range(len(A)):
        ks = list(var_map[j].keys())
        assert all(ks[i] <= ks[i + 1] for i in range(len(ks) - 1))
    list_xk = [[] for _ in A]

    G = nx.Graph()

    # query model size
    A1 = A[0]
    m, n = A1.shape
    nblock = len(A)
    Anorm = 20  # scipy.sparse.linalg.norm(A) / 10

    # alias
    rho0 = bcdpar.rho0
    rhom = 2
    rhol = rho = rho0
    # tau = 2000 / (scipy.sparse.linalg.norm(A1) * rhol + scipy.sparse.linalg.norm(c[0]) * rhom)
    # nA={idx: 0 for idx, _A in enumerate(A)}
    # for idx in range(nblock):
    #     nA[idx] = scipy.sparse.linalg.norm(A[idx])
    tau = 6.05 / ( scipy.sparse.linalg.norm(A1) * rhol)
    sigma  = bcdpar.sigma
    tsig = bcdpar.tsig
    rhofact1 = 0.9
    ctol = gtol = 1e-6
    rhofact2 = 0.6
    maxrho = 1e16
    eta = etac = etax = 1 / rho0
    subtolx = subtolc = 1 / rho0
    # primal variables
    xk = [np.ones((n, 1)) for _ in A]
    wk = [np.zeros((m + 1, 1)) for _ in A]
    # projection operator for w
    _proj = lambda x, θ: -np.linalg.solve(P.T @ P, P.T @ (M * x - q + θ / rho))
    # dual variables
    lbd = rho * np.zeros((m, 1))
    mu = [0 for _ in A]
    theta = [rho * np.zeros((n, 1)) for _ in A]

    # logger
    _LOG_FORMAT = "{:03d} {:4.1e} {:+8.2f} {:+9.2f} {:+9.2f} {:+.3e} {:+.3e} {:+.3e} {:+.3e} {:+.3e} {:.2e} {:04d}"
    show_log_header(bcdpar)

    # - k: outer iteration num
    # - it: inner iteration num
    # - idx: 1-n block idx
    #       it may not be the block no
    # A_k x_k
    _xnorm = {idx: 0 for idx, _ in enumerate(A)}
    _vAx = {idx: _A @ xk[idx] for idx, _A in enumerate(A)}
    _vBx = {idx: 0 for idx, _A in enumerate(A)}  # violation of capacity
    _vWx = {idx: 0 for idx, _A in enumerate(A)}  # violation of time-window
    # c_k x_k
    _vcx = {idx: (_c @ xk[idx]).trace() for idx, _c in enumerate(c)}  # original obj
    _vcxl = {
        idx: (_c @ xk[idx]).trace() for idx, _c in enumerate(c)
    }  # lagrangian obj for each block
    # x_k - x_k* (fixed point error)
    _eps_fix_point = {idx: 0 for idx, _ in enumerate(A)}
    _grad = {idx: 0 for idx, _ in enumerate(A)}
    ub_bst = np.inf
    for k in range(bcdpar.iter_max):

        _Ax_old = sum(_vAx.values())
        ############################################
        # dual update (BCD/ALM/ADMM)
        ############################################
        # update auxilliary `w` time window if relaxed
        if bcdpar.dual_update in {
            Dual.ProxLinearCapacityTW,
            Dual.NonlinearProxLinearCapacityTW,
        }:
            for idx in range(nblock):
                wk[idx] = _w = _proj(xk[idx], theta[idx])

        _d_k = {}

        _Ax = sum(_vAx.values())
        al_func_new = sum(_vcx.values()) + (lbd.T @ (_Ax - b)).trace() \
                      + rhol * (np.linalg.norm(_Ax - b, 2) ** 2) / 2 + \
                      rhom * sum(
            _nonnegative(c[idx] @ xk[idx] - C[idx][0] + mu[idx] / rhom) ** 2 for idx in range(nblock)) / 2
        # inner iteration
        for it in range(bcdpar.dual_linearize_max if bcdpar.dual_linearize else 1):
            if (time.time() - start_time) > bcdpar.time_limit: exit()  # todo
            _d_it = []

            for tauk in range(20):

                al_func = al_func_new
                for idx in range(nblock):
                    if (time.time() - start_time) > bcdpar.time_limit: exit()  # todo
                    ############################################
                    # update gradient
                    ############################################
                    Ak = A[idx]
                    _Ax = sum(_vAx.values())
                    ############################################
                    # create dual cost
                    ############################################
                    if bcdpar.dual_update == Dual.ProxLinear:
                        # _d = d[idx].reshape((-1, 1)) + Ak.T @ lbd + rhol * Ak.T @ (_Ax- Ak @ xk[idx]-0.5)
                        _d = d[idx].reshape((-1, 1)) + Ak.T @ lbd + rhol * Ak.T @ (_Ax - b) \
                             + (-xk[idx] / tau + 0.5 / tau)
                    elif bcdpar.dual_update == Dual.ProxLinearCapacity:
                        _d = d[idx].reshape((-1, 1)) + Ak.T @ lbd + rhol * Ak.T @ (_Ax - Ak @ xk[idx] - 0.5) \
                             + rhom * c[idx].T * (_nonnegative(c[idx] @ xk[idx] - C[idx][0]) + mu[idx] / rhom) \
                             + (-xk[idx] / tau + 0.5 / tau)
                        # _d = d[idx].reshape((-1, 1)) + Ak.T @ lbd + rhol * Ak.T @ (_Ax - b) \
                        #      + rhom * c[idx].T * (_nonnegative(c[idx] @ xk[idx] - C[idx][0])+mu[idx] / rhom) \
                        #      + (-xk[idx] / tau + 0.5 / tau)
                    elif bcdpar.dual_update == Dual.ProxLinearCapacity_nal:
                        _d = d[idx].reshape((-1, 1)) + Ak.T @ lbd + rhol * Ak.T @ (_Ax - b) \
                             + rhom * (c[idx].T * (_nonnegative(c[idx] @ xk[idx] - C[idx][0] + mu[idx] / rhom))) \
                             + (-xk[idx] / tau + 0.5 / tau)
                    elif bcdpar.dual_update == Dual.ProxLinearCapacityTW:
                        _d = d[idx].reshape((-1, 1)) + Ak.T @ lbd + rho * Ak.T @ (_Ax - b) \
                             + rho * (c[idx].T * _nonnegative(c[idx] @ xk[idx] - C[idx][0] + mu[idx] / rho).trace()) \
                             + rho * M * _nonnegative(P @ wk[idx] + M * xk[idx] - q + theta[idx] / rho) \
                             + (-xk[idx] / tau + 0.5 / tau)
                    elif bcdpar.dual_update == Dual.linearProxLinearCapacityTW_cx:
                        # todo for WR, corresponds to (23)
                        P_bar = np.hstack((P, np.eye(len(xk[idx]))))
                        PbTPb = np.vstack((np.hstack((P.T @ P, P.T)), P_bar))
                        fx = - np.linalg.inv(PbTPb) @ (P_bar.T @ (M * xk[idx] - q + theta[idx] / rho))
                        pfx = np.clip(fx, np.vstack((np.ravel(l).reshape((-1, 1)), np.zeros((n, 1)))),
                                      np.vstack((np.ravel(u).reshape((-1, 1)), 999999 * np.ones((n, 1)))))
                        df = -np.sign(abs(pfx - fx)) + 1
                        _d = d[idx].reshape((-1, 1)) + Ak.T @ lbd + rhol * Ak.T @ (_Ax - b) \
                             + rhom * (c[idx].T * (_nonnegative((c[idx] @ xk[idx]) - C[idx][0] + mu[idx] / rho))) \
                             + rho * M * (
                                     np.eye(n) - P_bar @ np.diag(df.flatten()) @ (np.linalg.inv(PbTPb)) @ (P_bar.T)) @ (
                                     P_bar @ pfx + M * xk[idx] - q + theta[idx] / rho) \
                             + (-xk[idx] / tau + 0.5 / tau)
                    elif bcdpar.dual_update == Dual.linearProxLinearCapacityTW_nx:
                        # todo for WR, corresponds to (27)
                        hx = - np.linalg.inv(P.T @ P) @ (P.T @ (M * xk[idx] - q + theta[idx] / rho))
                        phx = np.clip(hx, np.ravel(l).reshape((-1, 1)), np.ravel(l).reshape((-1, 1)))
                        dh = -np.sign(abs(phx - hx)) + 1
                        _d = d[idx].reshape((-1, 1)) + Ak.T @ lbd + rhol * Ak.T @ (_Ax - b) \
                             + rhom * c[idx].T * (_nonnegative(c[idx] @ xk[idx] - C[idx][0]) + mu[idx] / rhom) \
                             + M * (np.eye(n) - P @ np.diag(dh.flatten()) @ (np.linalg.inv(P.T @ P)) @ P.T) @ (
                                     theta[idx] + rho * _nonnegative(P @ phx + M * xk[idx] - q)) \
                             + (-xk[idx] / tau + 0.5 / tau)
                    elif bcdpar.dual_update == Dual.NonlinearProxLinearCapacityTW:
                        # todo for WR, corresponds to (31)
                        _d = d[idx].reshape((-1, 1)) + Ak.T @ lbd + rho * Ak.T @ (_Ax - b) \
                             + rho * (c[idx].T * _nonnegative(c[idx] @ xk[idx] - C[idx][0] + mu[idx] / rho)) \
                             + rho * (P @ wk[idx] + T) * _nonnegative((P @ wk[idx] + T) * xk[idx] + theta[idx] / rho) \
                             + (-xk[idx] / tau + 0.5 / tau)
                    elif bcdpar.dual_update == Dual.NonlinearProxLinearCapacityTW_nal:
                        # todo for WR, corresponds to (33)
                        _d = d[idx].reshape((-1, 1)) + Ak.T @ lbd + rho * Ak.T @ (_Ax - b) \
                             + rho * (c[idx].T * _nonnegative(c[idx] @ xk[idx] - C[idx][0]) + mu[idx] / rho) \
                             + (P @ wk[idx] + T) * (theta[idx] + rho * _nonnegative((P @ wk[idx] + T) * xk[idx])) \
                             + (-xk[idx] / tau + 0.5 / tau)
                    else:
                        raise ValueError("Dual update unrecognized")
                    ############################################
                    # solve dual subproblem
                    ############################################
                    if bcdpar.dual_method == DualSubproblem.Route:
                        _x = route.solve_primal_by_tsp(_d.flatten(), mode=0)
                    elif bcdpar.dual_method == DualSubproblem.CapaRoute:
                        # _x = route.solve_primal_by_mip(_d.flatten())
                        _x = route.solve_primal_by_tsp(_d.flatten(), mode=1)
                    elif bcdpar.dual_method == DualSubproblem.CapaWindowRoute:
                        # IN THIS MODE, YOU ALSO HAVE w,
                        # otherwise, you update in the after bcd for x
                        #   if it is a VRPTW
                        _x = route.solve_primal_by_tsp(_d.flatten(), mode=2)
                        # wk[idx] = _w = _proj(xk[idx], theta[idx])
                        # raise ValueError("not implemented")
                    elif bcdpar.dual_method == DualSubproblem.Assignment:
                        _x = route.solve_primal_by_assignment(_d.flatten(), mode=0)
                    elif bcdpar.dual_method == DualSubproblem.CapaAssignment:
                        _x = route.solve_primal_by_assignment(_d.flatten(), mode=1)
                    else:
                        raise ValueError("not implemented")

                    ############################################
                    # summarize
                    ############################################
                    _v_sp = (_d.T @ _x).trace()
                    #
                    _eps_fix_point[idx] = np.linalg.norm(xk[idx] - _x)
                    _xnorm[idx] = np.linalg.norm(xk[idx]) ** 2
                    _grad[idx] = np.linalg.norm(_d) ** 2

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
                    if bcdpar.dual_update in {
                        Dual.ProxLinearCapacityTW,
                        Dual.NonlinearProxLinearCapacityTW,
                    }:
                        _vWx[idx] = _nonnegative(P @ wk[idx] + M * xk[idx] - q)
                    else:
                        _vWx[idx] = np.zeros_like(theta[idx])

                    # save circle
                    if bcdpar.primal_method not in {Primal.Null}:
                        _s = _x.tostring()
                        list_xk[idx].append(_x)
                        block_nodes[idx].append(_s)
                        G.add_node(_s)
                        detect_conflict(G, _s, idx, block_nodes, var_map)
                    _d_it.append(_d)  # save each d in inner iter

                al_func_new = sum(_vcx.values()) + (lbd.T @ (_Ax - b)).trace() \
                              + rhol * (np.linalg.norm(_Ax - b, 2) ** 2) / 2 + \
                              rhom * sum(
                    _nonnegative(c[idx] @ xk[idx] - C[idx][0] + mu[idx] / rhom) ** 2 for idx in range(nblock)) / 2

                if al_func_new - al_func <= tau * sum(_eps_fix_point.values()):
                    break
                else:
                    tau /= tsig
                    # tau = 60 / (scipy.sparse.linalg.norm(A1) * rhol)

                _d_k[it] = _d_it  # save each iter's d's

            relerr = sum(_eps_fix_point.values()) / max(1, sum(_xnorm[idx] for idx in range(nblock)))
            if bcdpar.verbosity > 1:
                print("{:01d} cx: {:.1e} al_func:{:+.3e} grad_func:{:+.3e} relerr:{:+.3e} int:{:01d}".format(
                    it, sum(_vcx.values()), al_func_new[0][0], sum(_grad[idx] for idx in range(nblock)), relerr, tauk))

            # fixed-point eps
            if sum(_eps_fix_point.values()) < 1e-4:
                break

        _iter_time = (time.time() - start_time)
        _Ax = sum(_vAx.values())

        eps_pfeas_Axb = np.linalg.norm(_Ax - b, 2)
        eps_pfeas_cap = sum(_vBx.values())

        eps_pfeas = (
                np.linalg.norm(_Ax - b, np.inf)
                + sum(_vBx.values())
                + sum(np.linalg.norm(_, np.inf) for _ in _vWx.values())
        )
        cx = sum(_vcx.values())
        lobj = sum(_vcxl.values()) \
               + (lbd.T @ (_Ax - b)).trace() \
               + sum(mu[idx] * ((c[idx] @ xk[idx]).trace() - C[idx][0]) for idx in range(nblock))
        eps_fp = sum(_eps_fix_point.values())

        # if eps_pfeas == 0 and eps_fp < 1e-4 and np.count_nonzero(_nonnegative( b-_Ax))<=1:
        #     print(np.count_nonzero(_Ax-b))
        #     break
        if eps_pfeas == 0 and eps_fp < 1e-4:
            # print(np.count_nonzero(_Ax-b))
            _log_line = \
                _LOG_FORMAT.format(
                k, _iter_time, cx, ub_bst, lobj, eps_pfeas_Axb, eps_pfeas_cap, eps_fp, rhol, rhom, tau, it + 1
            )
            print(_log_line)
            break

        ############################################
        # primal update: some heuristic
        ############################################
        # ADD A PRIMAL METHOD FOR FEASIBLE SOLUTION

        ub_flag = ""
        if bcdpar.primal_method not in {Primal.Null}:
            # mis_heur(G, xk, (nblock, A, b, k, n, d), c)
            # set_par_heur(list_xk, d, var_map, N)
            suc_cnt = 0
            ub_seq = np.inf
            for it, _d_it in _d_k.items():
                ub_seq_new = heur_seq(vrp_clone, _d_it, xk, random_perm=True, bcdpar=bcdpar, opt_first=False)
                if bcdpar.verbosity > 1:
                    print(it, ub_seq_new)
                if ub_seq > ub_seq_new:
                    ub_seq = ub_seq_new
                    os.rename("seq_heur.sol", "seq_heur_curbst.sol")
                    suc_cnt += 1
                    if suc_cnt >= bcdpar.seq_heur_iter_max:
                        break
            if ub_seq < np.inf:
                ub_flag += "S"

            ub_bst = min(ub_bst, ub_seq)

        # if bcdpar.verbosity > 1:
        #     show_log_header(bcdpar)
        # _log_line = "{:03d} {:.1e} {:.1e} {:+.2e} {:+.2e} {:+.3e} {:+.3e} {:+.3e} {:+.3e} {:+.3e} {:.2e} {:04d}".format(
        #     k, _iter_time, ub_bst, cx, lobj, eps_pfeas_Axb, eps_pfeas_cap, eps_fp, rhol, rhom, tau, it + 1
        # )
        # print(_log_line)
        _log_line = _LOG_FORMAT.format(
            k, _iter_time, cx, ub_bst, lobj, eps_pfeas_Axb, eps_pfeas_cap, eps_fp, rhol, rhom, tau, it + 1
        )
        print(_log_line)

        ############################################
        # update dual variables
        ###########################################
        lbd += rhol * (_Ax - b)
        if np.linalg.norm(_Ax - b) <= etax:
            etax = min(max(etax / rhol ** rhofact1, ctol), 1)
            subtolx = min(max(subtolx / rhol, gtol), 1)
        else:
            rhol = min(rhol * sigma, maxrho)
            etax = min(5 * etax, max(1 / rhol ** rhofact2, ctol))
            subtolx = min(max(1 / rhol, ctol), 1)

        if sum(np.linalg.norm(_vBx[idx]) for idx in range(nblock)) <= etac:
            for idx in range(nblock):
                mu[idx] = _nonnegative(((c[idx] @ xk[idx]).trace() - C[idx][0]) * rhom + mu[idx])
                theta[idx] = _nonnegative(
                    (P @ wk[idx] + M * xk[idx] - q) * rho + theta[idx])
            etac = min(max(etac / rhom ** rhofact1, ctol), 1)
            subtolc = min(max(subtolc / rhom, gtol), 1)
        else:
            rhom = min(rhom + sigma, maxrho)
            etac = min(5 * etac, max(1 / rhom ** rhofact2, ctol))
            subtolc = min(max(1 / rhom, ctol), 1)

        # if bcdpar.iter >= 10 and bcdpar.iter % 10 == 0:
        #     # tau = 2 / (scipy.sparse.linalg.norm(A1) * rhol)
        #    tau/=2

        bcdpar.iter += 1

    return xk
