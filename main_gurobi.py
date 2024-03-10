import multiprocessing
import pickle

import io_solomon
from functional_bcd import *
from vrp import *


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


def process_row(args):
    row, params_bcd, lock, solve, write = args
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
        vrp.m.Params.TimeLimit = 1500
        # solve the model
        if solve:
            vrp.solve()
        if write:
            vrp.write(f"{row['Problem']}.{int(row['NV'])}.mps.gz")

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
        pass
        # with lock:
        #     grb_row.to_csv("dataset/gurobi_results.csv", mode="a", header=False, index=False)


if __name__ == "__main__":
    solve = True
    write = False
    parallel = True

    params_bcd = BCDParams()
    solo_res = pd.read_csv("dataset/solomon-results.csv")

    # Create a manager
    manager = multiprocessing.Manager()

    # Use the manager to create a lock
    lock = manager.Lock()

    # cpu_count
    cpu_count = multiprocessing.cpu_count()

    # assert NV is not null
    solo_res = solo_res[solo_res["NV"].notnull()]
    assert solo_res["NV"].notnull().all()
    solo_res = solo_res.sort_values("Problem")

    print(solo_res)

    if parallel:
        # Create a multiprocessing Pool
        with multiprocessing.Pool(cpu_count) as pool:
            # Use the map function to apply process_row to each row in solo_res
            pool.map(process_row, [(row, params_bcd, lock, solve, write) for _, row in solo_res.iterrows()])
    else:
        for _, row in solo_res.iterrows():
            process_row((row, params_bcd, lock))
