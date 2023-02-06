from gurobipy import *
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

        # model
        self.m = None

        # variables
        self.x = None  # x[s,t,j] = 1 if arc (s,t) is used on the route of vehicle j, 0 o.w.
        self.w = None

        # constraints
        self.cc1 = None
        self.c2 = None
        self.c3 = None
        self.c4 = None
        self.cc5 = None

    def create_model(self):
        self.m = Model("VRP")

    def add_vars(self):
        self.x = self.m.addVars(((s, t, j) for j in self.J for s, t in self.E), vtype=GRB.BINARY, name="x")

    def add_constrs(self):
        self.cc1 = self.m.addConstrs((quicksum(self.x[s, t, j] for s in self.in_neighbours(t) if s != t for j in self.J)
                                      ==
                                      1
                                      for t in self.V_0),
                                     name="cc1")

        self.c2 = self.m.addConstrs((quicksum(self.x[self.p, t, j] for t in self.V_0 if (self.p, t) in self.E) == 1
                                     for j in self.J),
                                    name="c2")

        self.c3 = self.m.addConstrs((quicksum(self.x[s, t, j] for s in self.in_neighbours(t) if s != t)
                                     ==
                                     quicksum(self.x[t, s, j] for s in self.out_neighbours(t) if s != t)
                                     for j in self.J for t in self.V),
                                    name="c3")

        self.c4 = self.m.addConstrs(
            (quicksum(self.x[s, t, j] for s in self.V for t in self.out_neighbours(s) if s != t and t != self.p)
             <=
             self.C
             for j in self.J),
            name="c4")

        self.cc5 = None  # FIXME

    def in_neighbours(self, t):
        return [s for s in self.V if (s, t) in self.E]

    def out_neighbours(self, t):
        return [s for s in self.V if (t, s) in self.E]

    def init(self):
        self.add_vars()
        self.add_constrs()

    def getA_b_c(self, model_index, binding_size):
        pass

    def get_x_indices(self):
        self.x_ind = {j: [v.index for v in self.x.select('*', '*', j)] for j in self.J}
        pass

    def get_window_matvec(self):
        M = np.zeros(len(E), len(V))
        T = np.zeros(len(E))
        index = 0
        for s, t in self.E:
            ind_s, ind_t = self.V.index(s), self.V.index(t)
            M[index, ind_s] = 1
            M[index, ind_t] = -1
            T[index] = self.T[s, t]
            index += 1
        return M, T, self.a, self.b

    def get_mat_vec(self):
        self.get_x_indices()
        M, T, a, b = self.get_window_matvec()
        return M, T, a, b

    def generate_matlab_dict(self):
        def _matrix_size(_size):
            m, n = _size

            def _format_dit(x):
                if x < 1e4:
                    return f"{x:d}"
                return f"{x:.1e}"

            return f"[{_format_dit(m)}, {_format_dit(n)}]"

        mat_dict = dict()

        logs = []
        M, T, a, b = self.get_window_matvec()
        mat_dict['M'] = M
        mat_dict['T'] = T
        mat_dict['a'] = a
        mat_dict['b'] = b
        for idx in self.J:
            _, n = B_k.shape
            struct = {
                "idx": idx,
                "n": len(E),
            }
            #
            mat_dict['b'] = b
            mat_dict['trains'].append(struct)
            logs.append(
                dict(zip(
                    ["idx", "Ak", "Bk", "bk", "ck", "#bind"],
                    [idx, _matrix_size(A_k.shape), _matrix_size(B_k.shape), _matrix_size(b_k.shape),
                     _matrix_size(c_k.shape), len(model_index[traNo])]
                ))
            )
        df = pd.DataFrame.from_records(logs)
        log_tables = df.to_markdown(tablefmt="grid", index=False)
        lines = log_tables.split('\n')
        print(lines[0])
        print(("|{:^" + f"{lines[0].__len__() - 2}" + "}|").format(
            "multi-block model size info for railway time tabling"))
        print(("|{:^" + f"{lines[0].__len__() - 2}" + "}|").format("showing first 10 blocks"))
        print("\n".join(lines[0:23]))
        return mat_dict


if __name__ == "__main__":
    V = [0, 1, 2, 3, 4, 5]
    E = [(s, t) for s in V for t in V if s != t]
    J = [0, 1, 2]
    c = [0, 1, 2, 3, 4, 5]
    C = 10
    d = {(0, 1): 1, (0, 2): 2, (0, 3): 3, (0, 4): 4, (0, 5): 5,
         (1, 2): 1, (1, 3): 2, (1, 4): 3, (1, 5): 4,
         (2, 3): 1, (2, 4): 2, (2, 5): 3,
         (3, 4): 1, (3, 5): 2,
         (4, 5): 1}
    d = {**d, **{(j, i): v for (i, j), v in d.items()}}
    a = np.random.randint(0, 50, len(V))
    b = a + 4
    T = {k: v / 2 for k, v in d.items()}
    vrp = VRP(V, E, J, c, C, d, a, b, T)
    vrp.create_model()
    vrp.init()
    vrp.m.update()
    vrp.m.write("vrp.lp")
    vrp.m.optimize()
    vrp.m.write("vrp.sol")
    print(vrp.m.objVal)
    for v in vrp.m.getVars():
        print(v.varName, v.x)

    vrp.get_mat_vec()
