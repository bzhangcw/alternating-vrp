# maximum independent set helper
import collections
import logging

import networkx as nx
import networkx.algorithms as na
import numpy as np
import pandas as pd


root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
_logger = logging.getLogger("ADMM")
_logger.setLevel(logging.INFO)


class MIS(object):
    SERVED_REVENUE = 1e4

    def __init__(self):
        self.graph = nx.Graph()
        self.gc = collections.Counter()
        self.size_new = 0
        self.size = 0

    def update(self, g_ending_state_vector):
        # from these solved paths we create an assignment problem
        # since the vehicles are homogeneous, we can safely reassign
        # ADMM_local_upperbound[i] += g_ending_state_vector[v].m_VSStateVector[0].PrimalLabelCost
        # path_no_seq[i].append(g_ending_state_vector[v].m_VSStateVector[0].m_visit_sequence)
        # path_time_seq[i].append(g_ending_state_vector[v].m_VSStateVector[0].m_visit_time_sequence)
        data = [
            (
                MISnode(
                    vv.m_VSStateVector[0].m_visit_sequence,
                    vv.m_VSStateVector[0].m_visit_time_sequence,
                ),
                dict(
                    weight=-vv.m_VSStateVector[0].PrimalLabelCost
                    + MIS.SERVED_REVENUE
                    * (vv.m_VSStateVector[0].m_visit_sequence.__len__() - 2),
                    cost=vv.m_VSStateVector[0].PrimalLabelCost,
                ),
            )
            for v, vv in enumerate(g_ending_state_vector)
            if vv is not None
        ]
        data = [[f, p] for f, p in data if f.bool_not_null]
        self.size_new = data.__len__()
        # conflicts with existing nodes
        edges = [
            *((n, m) for n in self.graph.nodes for m, _ in data if m.conflict(other=n)),
            *((n, m) for n, _ in data for m, _ in data if m.conflict(other=n)),
        ]
        self.graph.add_nodes_from(data)
        self.graph.add_edges_from(edges)
        self.size = self.graph.nodes.__len__()

    def compute_best_collection(self, scale, mode=1):
        # weight mis heuristic
        # we compute a maximum weight covering.
        if self.graph.size() == 0:
            return None, 0.0, None

        def _ff_from_view(weight_view):
            self.gc.clear()
            visits = dict.fromkeys(scale, 0)
            ff = []
            cost = 0
            weight = 0
            while weight_view.__len__():
                k, w = weight_view.pop(0)
                neighbors = set(self.graph.neighbors(k))
                acc = 0
                for q, _ in ff:
                    if q in neighbors:
                        acc = 1
                        break
                if acc == 1:
                    continue
                cost += w["cost"]
                weight += w["weight"]
                ff.append([k, w["cost"]])
            for k, _ in ff:
                self.gc.update(set(k.seq))
            visits.update(dict(self.gc))
            return (
                ff,
                cost + 500 * len([k for k, v in visits.items() if v == 0]),
                visits,
                weight,
            )

        if mode == 0:
            weight_view = sorted(
                self.graph.nodes.data(), key=lambda x: x[-1]["weight"], reverse=True
            )
            ff, cost, visits, weight = _ff_from_view(weight_view)
        elif mode == 1:
            # randomized method using the best
            rds = 400
            import numpy as np

            ff_arr = sorted(
                (
                    _ff_from_view(
                        np.random.permutation(self.graph.nodes.data()).tolist()
                    )
                    for _ in range(rds)
                ),
                key=lambda x: x[1],
                reverse=False,
            )
            ff, cost, visits, weight = ff_arr[0]

        else:
            raise ValueError("unsupported")

        return ff, cost, visits


class MISnode(object):
    def __init__(self, m_visit_sequence, m_visit_time_sequence):
        self.seq = tuple(m_visit_sequence)
        self.reduced_seq = tuple(m_visit_sequence[:-2])
        if len(self.reduced_seq) > 0:
            self.tm = tuple(m_visit_time_sequence)
            self.agg = [*self.seq, *self.tm]
            self._hash = tuple(self.agg).__hash__()
            self._str = (
                ",".join(map(str, m_visit_sequence))
                + "@"
                + ",".join(map(str, m_visit_time_sequence))
            )
            if self._hash in GLS:
                # duplicate
                self.bool_not_null = False
            else:
                self.bool_not_null = True
                GLS.add(self._hash)
        else:
            self.bool_not_null = False

    def __hash__(self):
        return self._hash

    def __str__(self):
        return self._str

    def __repr__(self):
        return self._str

    def conflict(self, other):
        # self conflicts other
        if self.reduced_seq == other.reduced_seq:
            return False
        if set(self.reduced_seq).intersection(other.reduced_seq).__len__() > 0:
            return True
        return False


# singleton object
mis_helper = MIS()
GLS = set()
