from vrp import VRP
from gurobipy import *


class Route:

    def __init__(self, vrp: VRP):
        self.vrp = vrp

    def create_model(self):
        self.m = Model("VRP")

    def add_vars(self):
        self.x = self.m.addVars(((s, t) for s, t in self.vrp.E), vtype=GRB.BINARY, name="x")

    def add_constrs(self, mode=0):
        self.c3 = self.m.addConstrs((quicksum(self.x[s, t] for s in self.vrp.in_neighbours(t) if s != t)
                                     == quicksum(self.x[t, s] for s in self.vrp.out_neighbours(t) if s != t)
                                     for t in self.vrp.V), name="c3")

        self.c4 = self.m.addConstrs(
            (quicksum(self.x[s, t] for s in self.vrp.V for t in self.vrp.out_neighbours(s)
                      if s != t and t != self.vrp.p) <= self.vrp.C),
            name="c4")
        if mode == 0:
            # solve
        self.bool_mip_built = True

    def solve_primal_by_mip(self):

