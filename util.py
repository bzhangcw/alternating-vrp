"""
utility modules, include utility for subgradient method.
"""
import datetime
import logging
import os
import sys
from collections import defaultdict
from itertools import combinations

from gurobipy import GRB, tuplelist, quicksum

##############################
# DEFAULTS
##############################
NODE_SINK = 't'
NODE_SINK_ARR = '_t'
##############################
# package-wise global variables
##############################
# flattened yv2xa
# (s', t', s, t) arc : value
xa_map = defaultdict(lambda: defaultdict(lambda: {"a": 0, "s": 0, "p": 0}))
# node precedence map in terms of arrival/departure interval
node_prec_map = defaultdict(list)
# original Lagrangian
# multiplier = defaultdict(lambda: {"aa": 0, "ap": 0, "ss": 0, "sp": 0, "pa": 0, "ps": 0, "pp": 0})  # each (station, t)
multiplier = dict()  # each (station, t)
z_vars = dict()
# node multiplier
yv_multiplier = {}  # the multiplier of each v
yvc_multiplier = defaultdict(lambda: {"a": 0, "s": 0, "p": 0})  # the multiplier of each v with type c
safe_int = {}
# category list
category = ["s", "a", "p"]
safe_int_df = None

##############################
# LOGGER
##############################
_grb_logger = logging.getLogger("gurobipy.gurobipy")
_grb_logger.setLevel(logging.ERROR)

logFormatter = logging.Formatter("%(asctime)s: %(message)s")
logger = logging.getLogger("railway")
logger.setLevel(logging.INFO)


# consoleHandler = logging.StreamHandler(sys.stdout)
# consoleHandler.setFormatter(logFormatter)
# logger.addHandler(consoleHandler)


# Callback - use lazy constraints to eliminate sub-tours
def subtourelim(V, model, where):
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._vars)
        selected = tuplelist(
            (i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5
        )
        # find the shortest cycle in the selected edge list
        tour = subtour(selected)
        if len(tour) < len(V):
            # add subtour elimination constr. for every pair of cities in subtour
            model.cbLazy(
                quicksum(model._vars[i, j] for i, j in combinations(tour, 2))
                <= len(tour) - 1
            )


# Given a tuplelist of edges, find the shortest subtour

def subtour(edges):
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


# global graph generation id.
class GraphCounter(object):
    def __init__(self):
        # number of nodes (unique) created
        self.vc = 0
        # number of edges (unique) created (not used yet)
        self.ec = 0
        self.id_nodes = {}
        self.tuple_id = {}


gc = GraphCounter()


class SysParams(object):
    # todo, support ArgumentParser
    DBG = False
    station_size = 0
    train_size = 0
    time_span = 0
    iter_max = 0
    up = 0

    def __init__(self):
        subdir_result = self.subdir_result = datetime.datetime.now().strftime('%y%m%d-%H%M')
        fdir_result = self.fdir_result = f"result/{subdir_result}"
        os.makedirs(fdir_result, exist_ok=True)
        fileHandler = logging.FileHandler("{0}/{1}.log".format(fdir_result, "out"))
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

    def parse_environ(self):
        import os
        self.station_size = int(os.environ.get('station_size', 29))
        self.train_size = int(os.environ.get('train_size', 50))
        self.time_span = int(os.environ.get('time_span', 1080))
        self.iter_max = int(os.environ.get('iter_max', 100))
        self.up = int(os.environ.get('up', 0))
        self.log_problem_size(logger)

    def log_problem_size(self, logger):
        logger.info(
            f"size: #train,#station,#timespan,#iter_max: {self.train_size, self.station_size, self.time_span, self.iter_max}"
        )

