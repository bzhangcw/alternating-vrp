from vrp import *
from gurobipy import *
from itertools import combinations
from util import subtourelim
import scipy


class Route:
    def __init__(self, vrp: VRP):
        self.vrp = vrp
        self.bool_mip_built = False

        self.m = None
        self.x = None
        self.depot_out = None
        self.depot_in = None
        self.flow = None
        self.capa = None
        self.rows = list(map(lambda x: x[0], self.vrp.E))
        self.cols = list(map(lambda x: x[1], self.vrp.E))
        self.cost_mat_size = len(self.vrp.V)
        self.cost_mat = scipy.sparse.coo_matrix(
            (np.zeros(len(self.vrp.E)), (self.rows, self.cols)),
            shape=(self.cost_mat_size, self.cost_mat_size)
        )

    def create_model(self):
        self.m = Model("VRP")
        self.m.setParam("LogToConsole", 0)
        self.m.setParam("LogFile", "cc.log")

    def add_vars(self):
        V = self.vrp.V
        self.x = self.m.addVars(
            ((s, t) for s in V for t in V if s != t), vtype=GRB.BINARY, name="x"
        )

        self.m._vars = self.x
        self.w = self.m.addVars([s for s in V], vtype=CONST.INTEGER,
                                name="w")

    def add_constrs(self, mode=0):

        vrp = self.vrp
        self.depot_out = self.m.addConstr(
            quicksum(self.x[vrp.p, t] for t in vrp.V_0 if (vrp.p, t) in vrp.E) == 1,
            name="depot_out",
        )
        self.depot_in = self.m.addConstr(
            quicksum(self.x[t, vrp.p] for t in vrp.V_0 if (t, vrp.p) in vrp.E) == 1,
            name="depot_in",
        )

        self.flow = self.m.addConstrs(
            (
                quicksum(self.x[s, t] for s in vrp.in_neighbours(t) if s != t)
                == quicksum(self.x[t, s] for s in vrp.out_neighbours(t) if s != t)
                for t in vrp.V
            ),
            **name_prefix("flow")
        )
        self.flow = self.m.addConstrs(
            (
                quicksum(self.x[s, t] for s in vrp.in_neighbours(t) if s != t)
                <= 1
                for t in vrp.V
            ),
            **name_prefix("flow")
        )

        self.tw = None  # FIXME
        if mode == 0:
            # solve route only
            pass
        elif mode == 1:
            # solve capacitated route
            self.capa = self.m.addConstr(
                (
                        quicksum(
                            vrp.c[t] * self.x[s, t]
                            for s in vrp.V
                            for t in vrp.out_neighbours(s)
                            if s != t and t != vrp.p
                        )
                        <= vrp.C
                ),
            )
        elif mode == 2:
            # solve capacitated route with TW
            self.capa = self.m.addConstr(
                (
                        quicksum(
                            vrp.c[t] * self.x[s, t]
                            for s in vrp.V
                            for t in vrp.out_neighbours(s)
                            if s != t and t != vrp.p
                        )
                        <= vrp.C
                ),
            )
            # '''constraint_6: time-windows and also eliminating sub-tours'''
            self.tw_lb = self.m.addConstrs(
                (vrp.a[s] <= self.w[s] for s in vrp.V),
            )
            self.tw_ub = self.m.addConstrs(
                (self.w[s] <= vrp.b[s] for s in vrp.V),
            )

            M = 1e5
            self.tw = self.m.addConstrs(
                (self.w[s] + vrp.T[s, t] + vrp.service_time[s] - M * (1 - self.x[s, t]) <= self.w[t]
                 for s in vrp.V
                 for t in vrp.out_neighbours(s) if t != 0),
            )

        else:
            raise ValueError("unknown mode")

        self.bool_mip_built = True

    def visualize(self, x):
        e = x.nonzero()[0]
        edges = [self.vrp.E[ee] for ee in e]
        edges_dict = dict(edges)
        node_bfs = defaultdict(int)
        i = 0
        nodes = [0]
        while True:
            nx = edges_dict.get(i)
            node_bfs[i] += 1
            nodes.append(nx)
            if nx is None or nx == 0 or node_bfs[i] > 1:
                break
            i = nx
        return nodes[:-1], edges

    def solve_primal_by_tsp(self, c, mode=0):
        """
        solve the primal problem using the cost vector c
        :param c:
        :param mode: mode of subproblem
            0 - TSP
            1 - C-TSP
            2 - C-TSP-TW
        :return:
        """
        if not self.bool_mip_built:
            self.create_model()
            self.add_vars()
            self.add_constrs(mode)

        self.m.setObjective(
            quicksum(c[idx] * self.x[s, t] for idx, (s, t) in enumerate(self.vrp.E)),
            GRB.MINIMIZE,
        )
        self.m.Params.lazyConstraints = 1
        # @note: keep for dbg
        # xx = np.array(
        #             [v for k, v in self.m.getAttr("x", self.x).items()]
        #         ).reshape((-1, 1))
        # e = xx.nonzero()[0]
        # edges = [vrp.E[ee] for ee in e]
        # edges_dict = dict(edges)
        # edges_dict

        self.m.optimize(
            lambda model, where: subtourelim(model, where, self.vrp.p)
        )
        xr = np.array(
            [v for k, v in self.m.getAttr("x", self.x).items()]
        ).reshape((-1, 1))

        # @note, nonlazy mode, for dbg; do not remove
        # bool_terminate = False
        # while not bool_terminate:
        #     self.m.optimize(
        #         # lambda model, where: subtourelim(model, where, self.vrp.p)
        #     )
        #     xr = np.array(
        #         [v for k, v in self.m.getAttr("x", self.x).items()]
        #     ).reshape((-1, 1))
        #     nodes, edges = self.visualize(xr)
        #     if len(nodes) < len(edges):
        #         cc = list(permutations(nodes, 2))
        #         print(cc)
        #         _ = self.m.addConstr(
        #             quicksum(self.x[i, j] for i, j in cc)
        #             <= len(nodes) - 1
        #         )
        #         continue
        return xr

    def solve_primal_by_assignment(self, c, mode=0):
        """
        solve the primal problem using the cost vector c
        :param c:
        :param mode: mode of subproblem
            0 - TSP
            1 - C-TSP
            2 - C-TSP-TW
        :return:
        """
        if not self.bool_mip_built:
            self.create_model()
            self.add_vars()
            self.add_constrs(mode)

        self.m.setObjective(
            quicksum(c[idx] * self.x[s, t] for idx, (s, t) in enumerate(self.vrp.E)),
            GRB.MINIMIZE,
        )
        self.m.optimize()
        return np.array(
            [v for k, v in self.m.getAttr("x", self.x).items()]
        ).reshape(
            (-1, 1)
        )

    def solve_primal_by_lkh(self, c, *args, **kwargs):
        import elkai
        self.cost_mat.data = c
        _mat = self.cost_mat.todense()
        _route = elkai.solve_float_matrix((_mat + _mat.T).tolist())
        _route.append(0)
        _edges = list(zip(_route[:-1], _route[1:]))
        _sol = {k: 1 for k in _edges}
        x = np.fromiter((_sol.get(k, 0) for k in self.vrp.E), dtype=np.int8).reshape(
            (-1, 1)
        )

        pass

    def solve_primal_by_dp(self, c, *args, **kwargs):
        data = {}
        E = np.array(list(self.vrp.d.keys()), np.int)
        data["f"] = (c - 1e3).tolist()
        data["D"] = list(self.vrp.d.values())
        data["I"] = E[:, 0].tolist()
        data["J"] = E[:, 1].tolist()
        data["c"] = self.vrp.c
        data["C"] = self.vrp.C
        data["a"] = self.vrp.a
        data["b"] = self.vrp.b
        data["V"] = self.vrp.V
        data["T"] = list(self.vrp.T.values())
        data["S"] = self.vrp.service_time
        data["m"] = len(c)
        data["n"] = len(self.vrp.V)
        from cpproute.wrapper import solve_by_dp_cc
        solve_by_dp_cc(
            data
        )


    def dump_to_json(self, c, *args, **kwargs):
        data = {}
        E = np.array(list(self.vrp.d.keys()), np.int)
        data["f"] = (c - 1e3).tolist()
        data["D"] = list(self.vrp.d.values())
        data["I"] = E[:, 0].tolist()
        data["J"] = E[:, 1].tolist()
        data["c"] = self.vrp.c
        data["C"] = self.vrp.C
        data["a"] = self.vrp.a
        data["b"] = self.vrp.b
        data["V"] = self.vrp.V
        data["T"] = list(self.vrp.T.values())
        data["S"] = self.vrp.service_time
        data["m"] = len(c)
        data["n"] = len(self.vrp.V)

        p, _ = self.visualize(self.solve_primal_by_tsp(c - 1e3, 2))
        data['p'] = p
        return data


if __name__ == "__main__":
    import json

    capitals_json = json.load(open("capitals.json"))
    capitals = []
    coordinates = {}
    for state in capitals_json:
        if state not in ["AK", "HI"]:
            capital = capitals_json[state]["capital"]
            capitals.append(capital)
            coordinates[capital] = (
                float(capitals_json[state]["lat"]),
                float(capitals_json[state]["long"]),
            )
    capital_map = {c: i for i, c in enumerate(capitals)}
    coordinates = {capital_map[c]: coordinates[c] for c in capitals}
    capitals = range(len(capitals))


    def distance(city1, city2):
        c1 = coordinates[city1]
        c2 = coordinates[city2]
        diff = (c1[0] - c2[0], c1[1] - c2[1])
        return math.sqrt(diff[0] * diff[0] + diff[1] * diff[1])


    dist = {(c1, c2): distance(c1, c2) for c1, c2 in combinations(capitals, 2)}

    V = capitals
    E = [(c1, c2) for c1 in capitals for c2 in capitals if c1 != c2]
    J = [0, 1, 2]
    c = [1] * len(V)
    C = 10
    d = dist
    l = np.random.randint(0, 50, len(V))
    u = l + 4
    T = {k: v / 2 for k, v in d.items()}
    vrp = VRP(V, E, J, c, C, d, l, u, T)
    route = Route(vrp)

    cc = np.ones((len(V), len(V)))
    route.solve_primal_by_tsp(cc)
