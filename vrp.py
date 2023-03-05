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
    def __init__(self, V, E, J, c, C, d, a, b, T):
        # data
        self.V = V  # node
        self.E = E  # arc
        self.J = J  # vehicles
        self.c = c  # demand
        self.C = C  # capacity
        self.d = d  # distance
        self.a = a  # timewindow lower limit
        self.b = b  # timewindow upper limit
        self.T = T

        assert len(self.V) == len(self.c)

        self.p = 0  # depot
        self.V_0 = [i for i in self.V if i != self.p]  # nodes except depot

        assert self.p in self.V
        assert len(self.V_0) + 1 == len(self.V)
        for i, j in E:
            assert i != j

        # model
        self.m = None

        # variables
        self.x = None  # x[s,t,j] = 1 if arc (s,t) is used on the route of vehicle j, 0 o.w.
        self.w = None

        # constraints
        self.coup = None
        self.depot = None
        self.flow = None
        self.capa = None
        self.tw = None

        # block data
        self.block_data = dict()

    def create_model(self):
        if solver_name == "copt":
            self.m = Envr().createModel("VRP")
        else:
            self.m = Model("VRP")

    def add_vars(self):
        self.x = self.m.addVars([(s, t, j) for j in self.J for s, t in self.E], vtype=CONST.BINARY, **name_prefix("x"))
        self.m._vars = self.x

    def add_constrs(self):
        self.coup = self.m.addConstrs(
            (quicksum(self.x[s, t, j] for s in self.in_neighbours(t) if s != t for j in self.J)
             ==
             1
             for t in self.V_0),
            **name_prefix("coup"))

        self.depot = self.m.addConstrs((quicksum(self.x[self.p, t, j] for t in self.V_0 if (self.p, t) in self.E) == 1
                                        for j in self.J),
                                       **name_prefix("depot"))

        self.flow = self.m.addConstrs((quicksum(self.x[s, t, j] for s in self.in_neighbours(t) if s != t)
                                       ==
                                       quicksum(self.x[t, s, j] for s in self.out_neighbours(t) if s != t)
                                       for t in self.V
                                       for j in self.J),
                                      **name_prefix("flow"))

        self.capa = self.m.addConstrs((quicksum(
            self.c[t] * self.x[s, t, j] for s in self.V for t in self.out_neighbours(s) if s != t and t != self.p)
                                       <=
                                       self.C
                                       for j in self.J),
                                      **name_prefix("capa"))

        self.tw = None  # FIXME

    def add_obj(self):
        self.m.setObjective(
            quicksum(self.d[s, t] * self.x[s, t, j] for idx, (s, t) in enumerate(self.E) for j in self.J),
            CONST.MINIMIZE)

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

            var_indice = [[v.index for v in self.x.select('*', '*', j)] for j in self.J]
            var_ind_name_map = {j: {v.index: v.varName for v in self.x.select('*', '*', j)} for j in self.J}
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

                sub_indice = [self.depot[j].index] + [c.index for c in self.flow.select('*', j)]
                capa_indice = [self.capa[j].index]

                self.block_data["A"].append(A_j[coup_indice, :])
                self.block_data["B"].append(A_j[sub_indice, :])
                self.block_data["q"].append(b[sub_indice])
                self.block_data["c"].append(A_j[capa_indice, :])
                self.block_data["C"].append(b[capa_indice])
                self.block_data["d"].append(c_j)

                n_constrs += len(coup_indice) + len(sub_indice) + len(capa_indice)
                logs.append(
                    dict(zip(
                        ["idx", "Ak", "Bk", "bk", "ck"],
                        [j, _matrix_size(A_j[coup_indice, :].shape), _matrix_size(A_j[sub_indice, :].shape), len(b),
                         _matrix_size(A_j[capa_indice, :].shape)]
                    ))
                )

            # assert n_constrs == A.shape[0] + (len(self.J) - 1) * len(self.coup)

            # M, T matrix in time-window constraint
            # [l, u] is the time window
            self.block_data['P'], self.block_data['T'], \
            self.block_data['l'], self.block_data['u'] \
                = self.get_window_matvec()

            df = pd.DataFrame.from_records(logs)
            log_tables = df.to_markdown(tablefmt="grid", index=False)
            lines = log_tables.split('\n')
            print(lines[0])
            print(("|{:^" + f"{lines[0].__len__() - 2}" + "}|").format(
                "multi-block model info for cvrp"))
            print(("|{:^" + f"{lines[0].__len__() - 2}" + "}|").format("showing first 10 blocks"))
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
        self.m.optimize(lambda model, where: self.subtourelim(model, where))
        self.m.Params.SolutionLimit = sollim

    def subtourelim(self, model, where):
        if where == CONST.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)

            for j0 in self.J:
                V = set()
                selected = tuplelist(
                    (s, t) for s, t, j in model._vars.keys() if j == j0 and vals[s, t, j] > 0.5
                )
                for s, t in selected:
                    V.add(s)
                    V.add(t)

                # find the shortest cycle in the selected edge list
                tour = self.subtour(selected)
                if len(tour) < len(V):
                    # add subtour elimination constr. for every pair of cities in subtour
                    model.cbLazy(
                        quicksum(model._vars[s, t, j0] for s, t in permutations(set(tour), 2))
                        <= len(set(tour)) - 1
                    )

    # Given a tuplelist of edges, find the shortest subtour

    def subtour(self, edges):
        V = list(set([i for i, j in edges] + [j for i, j in edges]))
        unvisited = V[:]
        cycle = V[:]  # Dummy - guaranteed to be replaced
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
