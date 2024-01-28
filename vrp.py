from collections import defaultdict
from itertools import permutations

solver_name = "gurobi"
if solver_name == "copt":
    from coptpy import Envr, quicksum, COPT as CONST, tuplelist
else:
    from gurobipy import Model, quicksum, GRB as CONST, tuplelist
import numpy as np
import pandas as pd
from scipy import sparse


class VRP:
    def __init__(self, V, E, J, c, C, d, a, b, T, sl, coordinates):
        # data
        self.V = V  # node
        self.E = E  # arc
        self.J = J  # vehicles
        self.c = c  # demand
        self.C = C  # capacity
        self.d = d  # distance
        self.a = a  # timewindow lower limit
        self.b = b  # timewindow upper limit
        self.T = T  # spend time
        self.service_time = sl  # service time
        self.coordinates = coordinates

        assert len(self.V) == len(self.c)

        self.p = 0  # depot
        print(self.V)
        self.V_0 = [i for i in self.V if i != self.p]  # nodes except depot

        assert self.p in self.V
        assert len(self.V_0) + 1 == len(self.V)
        for i, j in E:
            assert i != j

        # model
        self.m = None

        # variables
        self.x = (
            None  # x[s,t,j] = 1 if arc (s,t) is used on the route of vehicle j, 0 o.w.
        )
        self.w = None

        # constraints
        self.coup = None
        self.depot_out = None
        self.depot_in = None
        self.flow = None
        self.capa = None
        self.tw = None
        self.tw_lb = None
        self.tw_ub = None

        # block data
        self.block_data = dict()

    def create_model(self):
        if solver_name == "copt":
            self.m = Envr().createModel("VRP")
        else:
            self.m = Model("VRP")

    def add_vars(self):
        self.x = self.m.addVars(
            [(s, t, j) for j in self.J for s, t in self.E],
            vtype=CONST.BINARY,
            **name_prefix("x"),
        )
        self.m._x = self.x

        self.w = self.m.addVars(
            [(s, j) for j in self.J for s in self.V],
            vtype=CONST.INTEGER,
            **name_prefix("w"),
        )

    def add_constrs(self):
        self.m._lazy_cons = []
        self.coup = self.m.addConstrs(
            (
                quicksum(
                    self.x[s, t, j]
                    for s in self.in_neighbours(t)
                    if s != t
                    for j in self.J
                )
                == 1
                for t in self.V_0
            ),
            **name_prefix("coup"),
        )

        self.depot_out = self.m.addConstrs(
            (
                quicksum(
                    self.x[self.p, t, j] for t in self.V_0 if (self.p, t) in self.E
                )
                == 1
                for j in self.J
            ),
            **name_prefix("depot_out"),
        )
        self.depot_in = self.m.addConstrs(
            (
                quicksum(
                    self.x[t, self.p, j] for t in self.V_0 if (t, self.p) in self.E
                )
                == 1
                for j in self.J
            ),
            **name_prefix("depot_in"),
        )

        self.flow = self.m.addConstrs(
            (
                quicksum(self.x[s, t, j] for s in self.in_neighbours(t) if s != t)
                == quicksum(self.x[t, s, j] for s in self.out_neighbours(t) if s != t)
                for t in self.V
                for j in self.J
            ),
            **name_prefix("flow"),
        )

        self.capa = self.m.addConstrs(
            (
                quicksum(
                    self.c[t] * self.x[s, t, j]
                    for s in self.V
                    for t in self.out_neighbours(s)
                    if s != t and t != self.p
                )
                <= self.C
                for j in self.J
            ),
            **name_prefix("capa"),
        )

        # self.tw = None  # FIXME

        self.tw_lb = self.m.addConstrs(
            (self.a[s] <= self.w[s, j] for s in self.V for j in self.J),
            **name_prefix("tw_lb"),
        )
        self.tw_ub = self.m.addConstrs(
            (self.w[s, j] <= self.b[s] for s in self.V for j in self.J),
            **name_prefix("tw_ub"),
        )

        M = 1e5
        self.tw = self.m.addConstrs(
            (
                self.w[s, j]
                + self.T[s, t]
                + self.service_time[s]
                - M * (1 - self.x[s, t, j])
                <= self.w[t, j]
                for s, t in self.E
                if t != self.p
                for j in self.J
            ),
            **name_prefix("time_window"),
        )

        #
        # 啊啊啊这块真的要哭了，搞了好久，终于让我知道了bug在哪，这里要注意t != self.p(0)，因为初始点和终点都是w0,但是不能让这个时间为两个数，
        # 所以这块不用考虑回到depot的问题，因为所有的车都可以满足这个回到depot的时间

        # self.tw = self.m.addConstrs((self.w[s, j] >= self.a[s] for s in self.V for j in self.J), **name_prefix("tw_lower"))
        # self.tw.update({self.m.addConstrs((self.w[s, j] <= self.b[s] for s in self.V for j in self.J), **name_prefix("tw_upper"))})
        # self.tw.update({
        #     t: self.m.addConstr(
        #         self.w[s,j] + self.T[s, t] - self.M * (1 - self.x[s, t, j]) <= self.w[t,j],
        #         name=f"tw_b({s},{t},{j})"
        #     )
        #     for s in self.V for t in self.out_neighbours(s) for j in self.J if s != t and t != self.p and self.T[s, t] > 0
        # })

        # self.tw = self.m.addConstrs((quicksum(
        #     self.w[s,j]- self.w[t,j] + T[s,t] - 1e3 * (1-self.x[s, t, j]) for s in self.V for t in self.out_neighbours(s) if s != t and t != self.p)
        #                    <=
        #                    0
        #                    for j in self.J),
        #                   **name_prefix("timewindow"))
        # for s in self.V:
        #     # assumption: service starts at 9:00 AM, 9 == 0 minutes, each hour after 9 is 60 minutes plus previous hours
        #     self.m.addConstr(self.w[s,j] >= self.a[s])  # service should start after the earliest service start time
        #     self.m.addConstr(self.w[s,j] <= self.b[s])  # service can't be started after the latest service start time

        # create 3 type of redundant constraints, these cons should NOT change the optimal solution
        # constraint on minimum local time window, i.e. the time of node i minus the time of node j (j is the predecessor of i)
        T_s = {}
        for s, t in self.E:
            T_s.setdefault(s, list())
            if t != self.p:
                T_s[s].append(self.T[s, t] + self.service_time[s])

        lb = min([min(v) for k, v in T_s.items()])  # implied by the time window constraint， change lb to lb[s, j] to make ortools can't model it
        ub = 3 * max([max(v) for k, v in T_s.items()])  # same as lb, ortools can model it if lb = lb[j], but not lb = lb[s, j]
        # TODO: change lb to larger values to make sure it is not REDUNDANT.
        # local time window
        self.ltw = self.m.addConstrs(
            (
                self.w[s, j]
                + lb
                - M * (1 - self.x[s, t, j])
                <= self.w[t, j]
                for s, t in self.E
                if t != self.p
                for j in self.J
            ),
            **name_prefix("local_time_window_lb"),
        )
        self.ltw |= self.m.addConstrs(
            (
                self.w[s, j]
                + ub
                + M * (1 - self.x[s, t, j])
                >= self.w[t, j]
                for s, t in self.E
                if t != self.p
                for j in self.J
            ),
            **name_prefix("local_time_window_ub"),
        )

        # maximum total travel time for all vehicles, it shouldn't be larger than sum of largest travel time for each node
        # ortools can model it, but based on experiments, we observe that it forbids ortools to find the optimal solution,
        # even though it is redundant.
        # if change the constraints to limit the total travel time of any 2 vehicles, just sum the 2 constraints below.
        # ortools will also hard to
        # TODO: change max_travel_time to smaller values to make sure it is not REDUNDANT.
        T_s_max = {k: max(v) for k, v in T_s.items()}
        n_edges = len(self.V_0) + len(self.J)
        # get max n_edges from T_s
        max_travel_time = sum(sorted(T_s_max.values())[-n_edges:])
        self.mtt = self.m.addConstrs(self.w[s, j] - self.w[self.p, j] <= max_travel_time for s in self.V for j in self.J)

        # tarvel time difference of 2 routes
        # TODO: this should also be redundant, we can only add single pair (j, k) among these constraints, this will easily break ortools
        max_diff = 300
        c_max = {k: max(v) for k, v in T_s.items()}
        c_min = {k: min(v) for k, v in T_s.items()}
        self.w0 = self.m.addVars(self.J, vtype=CONST.INTEGER, **name_prefix("w0"))  # the arrival time of vehicle
        self.m.addConstrs(
            self.w[s, j]
            + self.T[s, t]
            + self.service_time[s]
            - M * (1 - self.x[s, t, j])
            <= self.w0[j]
            for s, t in self.E
            if t == self.p
            for j in self.J
        )
        self.m.addConstrs(
            self.w[s, j]
            + self.T[s, t]
            + self.service_time[s]
            + M * (1 - self.x[s, t, j])
            >= self.w0[j]
            for s, t in self.E
            if t == self.p
            for j in self.J
        )
        self.ttd = self.m.addConstrs(
            (self.w0[j] - self.w[self.p, j]) - (self.w0[k] - self.w[self.p, k])<= max_diff for s in self.V for j in self.J for k in self.J if j < k)

    def add_obj(self):
        self.m.setObjective(
            quicksum(
                self.d[s, t] * self.x[s, t, j]
                for idx, (s, t) in enumerate(self.E)
                for j in self.J
            ),
            CONST.MINIMIZE,
        )

    def in_neighbours(self, t):
        return [s for s in self.V if (s, t) in self.E]

    def out_neighbours(self, t):
        return [s for s in self.V if (t, s) in self.E]

    def init(self, get_block_data=False):
        self.add_vars()
        self.add_constrs()
        self.add_obj()
        self.no_obj_heur()

        def _matrix_size(_size):
            m, n = _size

            def _format_dit(x):
                if x < 1e4:
                    return f"{x:d}"
                return f"{x:.1e}"

            return f"[{_format_dit(m)}, {_format_dit(n)}]"

        if get_block_data:
            self.m.update()

            A = self.m.getA()
            b = np.array(self.m.getAttr("RHS", self.m.getConstrs()))
            c = np.array(self.m.getAttr("Obj", self.m.getVars()))

            var_indice = [[v.index for v in self.x.select("*", "*", j)] for j in self.J]
            var_ind_name_map = {
                j: {v.index: v.varName for v in self.x.select("*", "*", j)}
                for j in self.J
            }
            coup_indice = [c.index for c in self.coup.values()]

            self.block_data["A"] = []  # couple A
            self.block_data["b"] = np.ones((len(self.V) - 1, 1))
            self.block_data["B"] = []  # sub A
            self.block_data["q"] = []  # sub b
            self.block_data["c"] = []  # demand
            self.block_data["C"] = []  # capacity
            self.block_data["d"] = []  # obj coeff
            self.block_data["ind"] = var_ind_name_map
            self.block_data["V"] = self.V
            logs = []
            n_constrs = 0
            for j in self.J:
                A_j = A[:, var_indice[j]]
                c_j = c[var_indice[j]]

                sub_indice = [self.depot_out[j].index, self.depot_in[j].index] + [
                    c.index for c in self.flow.select("*", j)
                ]
                capa_indice = [self.capa[j].index]

                self.block_data["A"].append(A_j[coup_indice, :])
                self.block_data["B"].append(A_j[sub_indice, :])
                self.block_data["q"].append(b[sub_indice])
                self.block_data["c"].append(A_j[capa_indice, :])
                self.block_data["C"].append(b[capa_indice])
                self.block_data["d"].append(c_j)

                n_constrs += len(coup_indice) + len(sub_indice) + len(capa_indice)
                logs.append(
                    dict(
                        zip(
                            ["idx", "Ak", "Bk", "bk", "ck"],
                            [
                                j,
                                _matrix_size(A_j[coup_indice, :].shape),
                                _matrix_size(A_j[sub_indice, :].shape),
                                len(b),
                                _matrix_size(A_j[capa_indice, :].shape),
                            ],
                        )
                    )
                )

            # assert n_constrs == A.shape[0] + (len(self.J) - 1) * len(self.coup)

            # M, T matrix in time-window constraint
            # [l, u] is the time window
            (
                self.block_data["P"],
                self.block_data["T"],
                self.block_data["l"],
                self.block_data["u"],
            ) = self.get_window_matvec()

            df = pd.DataFrame.from_records(logs)
            log_tables = df.to_markdown(tablefmt="grid", index=False)
            lines = log_tables.split("\n")
            print(lines[0])
            print(
                ("|{:^" + f"{lines[0].__len__() - 2}" + "}|").format(
                    "multi-block model info for cvrp"
                )
            )
            print(
                ("|{:^" + f"{lines[0].__len__() - 2}" + "}|").format(
                    "showing first 10 blocks"
                )
            )
            print("\n".join(lines[0:23]))

    def solve(self, callback=True):
        if solver_name == "copt":
            self.m.solve()
        else:
            if callback:
                self.m.optimize(lambda model, where: self.subtourelim(model, where))
            else:
                self.m.optimize()

    def get_window_matvec(self):
        M = np.zeros((len(self.E), len(self.V)))
        T = np.zeros(len(self.E))
        index = 0
        for s, t in self.E:
            ind_s, ind_t = self.V.index(s), self.V.index(t)
            M[index, ind_s] = 1
            M[index, ind_t] = -1
            T[index] = self.T[s, t]
            index += 1
        return M, T, self.a, self.b

    def no_obj_heur(self):
        self.m.Params.lazyConstraints = 1
        sollim = self.m.Params.SolutionLimit
        self.m.Params.SolutionLimit = 1
        # self.m.optimize(lambda model, where: self.subtourelim(model, where))
        self.m.Params.SolutionLimit = sollim

    def subtourelim(self, model, where):
        if where == CONST.Callback.MIPSOL:
            # make a list of edges selected in the solution
            x_vals = model.cbGetSolution(model._x)

            for j0 in self.J:
                V = set()
                selected = tuplelist(
                    (s, t)
                    for s, t, j in model._x.keys()
                    if j == j0 and x_vals[s, t, j] > 0.5
                )
                for s, t in selected:
                    V.add(s)
                    V.add(t)

                # find the shortest cycle in the selected edge list
                tour = self.subtour(selected)
                if len(tour) < len(V):
                    # add subtour elimination constr. for every pair of cities in subtour
                    tmp_con = (
                        quicksum(model._x[s, t, j0] for s, t in permutations(tour, 2))
                        <= len(tour) - 1
                    )
                    model.cbLazy(tmp_con)
                    # self.m._lazy_cons.append(tmp_con)
                    # print(tmp_con)

    # Given a tuplelist of edges, find the shortest subtour not containing depot
    def subtour(self, edges):
        V = list(set([i for i, j in edges] + [j for i, j in edges]))
        unvisited = V[:]
        cycle = V[:]  # Dummy - guaranteed to be replaced
        depot_connected = [j for i, j in edges.select(self.p, "*")]
        unvisited.remove(self.p)
        while depot_connected:
            current = depot_connected.pop()
            unvisited.remove(current)
            neighbors = [
                j for i, j in edges.select(current, "*") if j in unvisited and j != 0
            ]
            depot_connected += neighbors

        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, "*") if j in unvisited]
            if len(thiscycle) <= len(cycle):
                cycle = thiscycle  # New shortest subtour
        return cycle

    def visualize(self, x=None):
        if x is None:
            # get own solution by MIP
            x = self.m.getAttr("x", self.m._x).values()
            x = np.array(x).reshape(len(self.J), len(self.E)).round().astype(np.int8)
        solution = []
        cc = np.array(self.c)
        for xk in x:
            e = xk.nonzero()[0]
            edges = [self.E[ee] for ee in e]
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

            solution.append([nodes, len(nodes), len(edges), cc[nodes].sum()])

        df = pd.DataFrame(solution, columns=["route-r", "|r|", "|Er|", "sum(c)"])
        print(df.to_markdown())
        pass


def name_prefix(name: str):
    global solver_name
    if solver_name == "copt":
        return {"nameprefix": name}
    else:
        return {"name": name}


if __name__ == "__main__":
    V = [0, 1, 2, 3, 4, 5]
    E = [(s, t) for s in V for t in V if s != t]
    J = [0, 1, 2]
    c = [1] * len(V)
    C = 10
    d = {(s, t): abs(s - t) for s, t in E}
    a = np.random.randint(0, 50, len(V))
    b = a + 4
    T = {k: v / 2 for k, v in d.items()}
    vrp = VRP(V, E, J, c, C, d, a, b, T)
    vrp.create_model()
    vrp.init(get_block_data=True)
    print(len(vrp.block_data))
    print(len(vrp.block_data["A"]))
    vrp.m.write("vrp.lp")
    vrp.solve()
    vrp.m.write("vrp.sol")
    print(vrp.m.objVal)
    # for v in vrp.m.getVars():
    #     print(v.varName, v.x)
