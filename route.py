from vrp import *
from gurobipy import *
from itertools import combinations


class Route:
    def __init__(self, vrp: VRP):
        self.vrp = vrp
        self.bool_mip_built = False

        self.m = None
        self.x = None
        self.depot = None
        self.flow = None
        self.capa = None

    def create_model(self):
        self.m = Model("VRP")
        self.m.setParam("LogToConsole", 0)

    def add_vars(self):
        V = self.vrp.V
        self.x = self.m.addVars(
            ((s, t) for s in V for t in V if s != t), vtype=GRB.BINARY, name="x"
        )

        self.m._vars = self.x

    def add_constrs(self, mode=0):

        vrp = self.vrp
        self.depot = self.m.addConstr(
            quicksum(self.x[vrp.p, t] for t in vrp.V_0 if (vrp.p, t) in vrp.E) == 1,
            name="depot",
        )

        self.flow = self.m.addConstrs(
            (
                quicksum(self.x[s, t] for s in vrp.in_neighbours(t) if s != t)
                == quicksum(self.x[t, s] for s in vrp.out_neighbours(t) if s != t)
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
            vrp.capa = vrp.m.addConstr(
                (
                    quicksum(
                        vrp.c[t] * vrp.x[s, t]
                        for s in vrp.V
                        for t in vrp.out_neighbours(s)
                        if s != t and t != vrp.p
                    )
                    <= vrp.C
                ),
            )
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

        self.m.setObjective(
            quicksum(c[idx] * self.x[s, t] for idx, (s, t) in enumerate(self.vrp.E)),
            GRB.MINIMIZE,
        )
        self.m.Params.lazyConstraints = 1
        self.m.optimize(lambda model, where: self.subtourelim(model, where))
        return np.array([v for k, v in self.m.getAttr("x", self.x).items()]).reshape(
            (-1, 1)
        )

    # Callback - use lazy constraints to eliminate sub-tours
    def subtourelim(self, model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = tuplelist(
                (i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5
            )
            # find the shortest cycle in the selected edge list
            tour = self.subtour(selected)
            if len(tour) < len(self.vrp.V):
                # add subtour elimination constr. for every pair of cities in subtour
                model.cbLazy(
                    quicksum(model._vars[i, j] for i, j in combinations(tour, 2))
                    <= len(tour) - 1
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
    route.solve_primal_by_mip(cc)
