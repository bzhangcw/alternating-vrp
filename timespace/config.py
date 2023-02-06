import argparse
import json
import logging


class Algs(object):
    alg_dual = "classical"
    alg_primal = "classical"

    @classmethod
    def parse(cls, args):
        cls.alg_dual = args.alg_dual
        cls.alg_primal = args.alg_primal


logging.basicConfig(
    format="[%(asctime)s %(name)4s]  %(message)s",
    filemode="w",
    level=logging.INFO,
    datefmt="%y/%m/%d@%H:%M:%S",
)
parser = argparse.ArgumentParser(description="CVRP ADMM")
parser.add_argument(
    "--path",
    default="SolomonDataset_v2/r101-25",
    type=str,
    help="path of the instance",
)
parser.add_argument(
    "--alg_dual",
    default="prox_linear",
    type=str,
    choices=["prox_linear", "classical"],
    help="path of the instance",
)
parser.add_argument(
    "--alg_primal",
    default="mis",
    type=str,
    choices=["mis", "classical"],
    help="path of the instance",
)

args = parser.parse_args()
Algs.parse(args)
logging.info(f"params: \n{json.dumps(args.__dict__, indent=2)}")
fpath = args.path
instance = fpath.split("/")[-1]

# 还是按原代码单独调参，懒得改了
if instance == "r101-25":
    vehicle_fleet_size = 11
    g_number_of_time_intervals = 230
    fixed_cost = 0
    waiting_arc_cost = 0
    service_length = 10
    origin_node = 0
    departure_time_beginning = 0
    destination_node = 26
    arrival_time_ending = 230
    g_number_of_ADMM_iterations = 100


elif instance == "r101-50":
    vehicle_fleet_size = 14
    g_number_of_time_intervals = 230
    fixed_cost = 0
    waiting_arc_cost = 0
    service_length = 10
    origin_node = 0
    departure_time_beginning = 0
    destination_node = 51
    arrival_time_ending = 230

    g_number_of_ADMM_iterations = 100

elif instance == "r101-50":
    vehicle_fleet_size = 21
    g_number_of_time_intervals = 230
    fixed_cost = 0
    waiting_arc_cost = 0
    service_length = 10
    origin_node = 0
    departure_time_beginning = 0
    destination_node = 101
    arrival_time_ending = 230

    g_number_of_ADMM_iterations = 200

elif instance == "rc101-25":
    vehicle_fleet_size = 7
    g_number_of_time_intervals = 240
    fixed_cost = 0
    waiting_arc_cost = 0
    service_length = 10
    origin_node = 0
    departure_time_beginning = 0
    destination_node = 26
    arrival_time_ending = 240

    g_number_of_ADMM_iterations = 200

elif instance == "rc101-50":
    vehicle_fleet_size = 11
    g_number_of_time_intervals = 240
    fixed_cost = 0
    waiting_arc_cost = 0
    service_length = 10
    origin_node = 0
    departure_time_beginning = 0
    destination_node = 51
    arrival_time_ending = 240

    g_number_of_ADMM_iterations = 300

# elif instance == 'r101-100':
else:
    vehicle_fleet_size = 16
    g_number_of_time_intervals = 240
    fixed_cost = 0
    waiting_arc_cost = 0
    service_length = 10
    origin_node = 0
    departure_time_beginning = 0
    destination_node = 101
    arrival_time_ending = 240

    g_number_of_ADMM_iterations = 150
