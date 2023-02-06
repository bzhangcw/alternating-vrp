from vrp import *
from gurobipy import *


class Route:

    def __init__(self, vrp: VRP):
        self.vrp = vrp
        self.bool_mip_built = False

    def create_model(self):
        self.m = Model("VRP")

    def add_vars(self):
        self.x = self.m.addVars(((s, t) for s, t in self.vrp.E), vtype=GRB.BINARY, name="x")

    def add_constrs(self, mode=0):

        vrp = self.vrp
        self.depot = self.m.addConstr(quicksum(self.x[vrp.p, t] for t in vrp.V_0 if (vrp.p, t) in vrp.E) == 1, name="depot")

        self.flow = self.m.addConstrs((quicksum(vrp.x[s, t] for s in vrp.in_neighbours(t) if s != t)
                                     ==
                                     quicksum(vrp.x[t, s] for s in vrp.out_neighbours(t) if s != t)
                                     for t in vrp.V),
                                    **name_prefix("flow"))

        self.tw = None  # FIXME
        if mode == 0:
            # solve route only
            pass
        elif mode == 1:
            # solve capacitated route
            self.capa = self.m.addConstrs((quicksum(
                vrp.c[t] * self.x[s, t, j] for s in vrp.V for t in vrp.out_neighbours(s) if s != t and t != vrp.p)
                                         <=
                                         vrp.C
                                         for j in vrp.J),
                                        **name_prefix("capa"))
        elif mode == 2:
            # solve capacitated route
            vrp.capa = vrp.m.addConstrs((quicksum(
                vrp.c[t] * vrp.x[s, t, j] for s in vrp.V for t in vrp.out_neighbours(s) if s != t and t != vrp.p)
                                         <=
                                         vrp.C
                                         for j in vrp.J),
                                        **name_prefix("capa"))
            # todo, time window
        else:
            raise ValueError("unknown mode")

        self.bool_mip_built = True

    def solve_primal_by_mip(self, c, mode=0):
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

