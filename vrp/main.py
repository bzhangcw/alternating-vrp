import os
import argparse
from solver.base_solver import Solver
from util.instance_loader import load_instance


if __name__ == '__main__':
    # add command line arguments
    parser = argparse.ArgumentParser(description='Solve VRP instances')
    parser.add_argument('--add_hard_cons', action='store_true', help='Add hard constraints')
    parser.add_argument('--dirname', type=str, default='../dataset/solomon-100-original', help='Directory name')
    arg = parser.parse_args()
    add_hard_cons = arg.add_hard_cons
    dirname = arg.dirname

    filenames = [f for f in os.listdir(dirname) if f.endswith('.txt')]

    data_triplets = [(25, 3), (50, 5), (100, 10)]
    mtt = {(25, 3): 1200, (50, 5): 1200, (100, 10): 1200}
    time_precision_scaler = 1000
    for filename in filenames:
        for n_customers, n_vehicles in data_triplets:
            print(f'Processing {filename} with {n_customers} customers and {n_vehicles} vehicles')
            data = load_instance(os.path.join(dirname, filename), time_precision_scaler, n_customers, n_vehicles)
            solver = Solver(data, time_precision_scaler)
            if add_hard_cons:
                solver.create_model_with_local_tw(set_max_travel_time=True, tub=mtt[(n_customers, n_vehicles)])
            else:
                solver.create_model()

            settings = dict()
            settings['time_limit'] = n_customers * 1
            settings["log_search"] = False

            solver.solve_model(settings)

            solver.print_solution()
