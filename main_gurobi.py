import multiprocessing
import os
import pickle
import time
from itertools import combinations

import cloudpickle
import dill
import pandas as pd
from tqdm import tqdm

import io_solomon
from functional_bcd import *
from vrp import *


def create_toy_instance():
    # V = [0, 1, 2, 3, 4, 5]
    # E = [(s, t) for s in V for t in V if s != t]
    # J = [0, 1, 2]
    # c = [1] * len(V)
    # C = 10
    # d = {(s, t): abs(s - t) for s, t in E}
    # a = np.random.randint(0, 50, len(V))
    # b = a + 4
    # T = {k: v / 2 for k, v in d.items()}
    # vrp = VRP(V, E, J, c, C, d, a, b, T)
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

    dist = {
        (c1, c2): distance(c1, c2) for c1 in capitals for c2 in capitals if c1 != c2
    }

    V = capitals
    E = [(c1, c2) for c1 in capitals for c2 in capitals if c1 != c2]
    J = [0, 1, 2]
    c = [1] * len(V)
    C = 18
    d = dist
    l = np.random.randint(0, 50, len(V))
    u = l + 4
    T = {k: v / 2 for k, v in d.items()}
    vrp = VRP(V, E, J, c, C, d, l, u, T)
    return vrp


def read_solomon(
    fp="dataset/solomon-100-original/c101.txt", n_vehicles=10, n_customers=25
):
    timestamp = int(time.time()).__str__()[:6]
    pkl_fp = fp + ".data_{}-{}.pkl".format(n_vehicles, timestamp)
    try:
        with open(pkl_fp, "rb") as f:
            V, E, J, c, C, d, l, u, T, sl, coordinates = pickle.load(f)
    except FileNotFoundError:
        V, E, J, c, C, d, l, u, T, sl, coordinates = io_solomon.data_loader(
            fp, n_vehicles=n_vehicles, n_customers=n_customers
        )
        with open(fp + ".data_{}-{}.pkl".format(n_vehicles, timestamp), "wb") as f:
            pickle.dump((V, E, J, c, C, d, l, u, T, sl, coordinates), f)

    vrp = VRP(V, E, J, c, C, d, l, u, T, sl, coordinates)
    return vrp


if __name__ == "__main__":
    params_bcd = BCDParams()
    # create vrp instance
    # vrp = create_toy_instance()
    # read solomon results

    solo_res = pd.read_csv("dataset/solomon-results.csv")
    grb_res = pd.DataFrame(columns=["Problem", "NV", "Distance", "Time", "Remark"])
    grb_res_file = "dataset/gurobi_results.csv"
    # if exists, mv to backup
    if os.path.exists(grb_res_file):
        os.rename(grb_res_file, grb_res_file + ".bak")
    # enumerate each row in the dataframe
    for i, row in tqdm(solo_res.iterrows(), total=solo_res.shape[0]):
        try:
            # file name and vehicle number
            filename, n_customer = row["Problem"].split(".")
            n_customer = int(n_customer)
            filename = filename.lower() + ".txt"
            # get the file path
            fp = "dataset/solomon-100-original/" + filename
            # read the solomon instance
            vrp = read_solomon(
                fp=fp, n_vehicles=int(row["NV"]), n_customers=n_customer
            )
            # create the model
            vrp.create_model()
            vrp.init(get_block_data=False)
            # set the time limit
            vrp.m.Params.TimeLimit = params_bcd.time_limit
            # solve the model
            vrp.solve()
            grb_row = pd.DataFrame(
                [[row["Problem"], int(row["NV"]), vrp.m.objVal, vrp.m.Runtime, ""]],
                columns=["Problem", "NV", "Distance", "Time", "Remark"],
            )
        except Exception as e:
            grb_row = pd.DataFrame(
                [[row["Problem"], row["NV"], np.nan, np.nan, e.__str__()]],
                columns=["Problem", "NV", "Distance", "Time", "Remark"],
            )
        finally:
            grb_row.to_csv(grb_res_file, mode="a", header=False, index=False)
