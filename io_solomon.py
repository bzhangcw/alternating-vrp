import math
import csv
import copy
import xlrd
import time

import scipy
from scipy.optimize import linprog
import numpy as np


class Node:
    def __init__(self):
        self.node_id = 0
        self.x = 0.0
        self.y = 0.0
        self.type = 0
        self.outbound_node_list = []
        self.outbound_node_size = 0
        self.outbound_link_list = []
        self.demand = 0.0
        self.g_activity_node_beginning_time = 0
        self.g_activity_node_ending_time = 0
        self.base_profit_for_searching = 0
        self.base_profit_for_lr_2 = 0
        self.base_profit_for_lr = 0


class Link:
    def __init__(self):
        self.link_id = 0
        self.from_node_id = 0
        self.to_node_id = 0
        self.distance = 0.0
        self.spend_tm = 1.0


class Agent:
    def __init__(self):
        self.agent_id = 0
        self.from_node_id = 0
        self.to_node_id = 0
        self.departure_time_beginning = 0.0
        self.arrival_time_beginning = 0.0
        self.capacity = 0


def g_ReadInputData(path):
    # the parameter need to be changed
    vehicle_fleet_size = 10
    g_number_of_time_intervals = 1236
    fixed_cost = 0
    waiting_arc_cost = 0
    service_length = 90
    origin_node = 0
    departure_time_beginning = 0
    destination_node = 101
    arrival_time_ending = 1236

    g_number_of_ADMM_iterations = 50
    g_number_of_cutting_plane_iterations = 50

    _MAX_LABEL_COST = 100000
    g_node_list = []
    g_agent_list = []
    g_link_list = []

    g_number_of_nodes = 0
    g_number_of_links = 0
    g_number_of_agents = 0
    g_number_of_customers = 0
    g_number_of_vehicles = 0

    base_profit = 0
    rho = 6
    constants = 100

    import scipy
    from scipy.optimize import linprog

    g_ending_state_vector = [None] * g_number_of_vehicles

    path_no_seq = []
    path_time_seq = []
    service_times = []
    record_profit = []

    dp_result = [_MAX_LABEL_COST, _MAX_LABEL_COST] * g_number_of_vehicles
    dp_result_lowerbound = [_MAX_LABEL_COST] * g_number_of_vehicles
    dp_result_upperbound = [_MAX_LABEL_COST] * g_number_of_vehicles
    BestKSize = 100
    global_upperbound = 9999
    global_lowerbound = -9999

    ADMM_local_lowerbound = [0] * g_number_of_ADMM_iterations
    ADMM_local_upperbound = [0] * g_number_of_ADMM_iterations

    # read nodes information
    try:
        book = xlrd.open_workbook(path + "/input_node.xls")  # todo: turn into csv file
    except:
        book = xlrd.open_workbook(path + "/input_node.xlsx")  # todo: turn into csv file
    sh = book.sheet_by_index(0)
    # set the original node
    node = Node()
    node.node_id = 0
    node.type = 1
    node.g_activity_node_beginning_time = 0
    # node.g_activity_node_ending_time = 100
    g_node_list.append(node)
    g_number_of_nodes += 1

    for l in range(1, sh.nrows):  # read each lines
        try:
            node = Node()
            node.node_id = int(sh.cell_value(l, 0))
            node.type = 2
            node.x = float(sh.cell_value(l, 1))
            node.y = float(sh.cell_value(l, 2))
            node.demand = float(sh.cell_value(l, 3))
            node.g_activity_node_beginning_time = int(sh.cell_value(l, 4))
            node.g_activity_node_ending_time = int(sh.cell_value(l, 5))
            node.base_profit_for_searching = base_profit
            node.base_profit_for_lr = base_profit
            node.base_profit_for_lr_2 = base_profit
            g_node_list.append(node)
            g_number_of_nodes += 1
            g_number_of_customers += 1
            if g_number_of_nodes % 100 == 0:
                print('reading {} nodes..' \
                      .format(g_number_of_nodes))
        except:
            print('Bad read. Check file your self')
    print('nodes_number:{}'.format(g_number_of_nodes))
    print('customers_number:{}'.format(g_number_of_customers))

    # set the destination node
    node = Node()
    node.type = 1
    node.node_id = g_number_of_nodes  #
    # node.g_activity_node_beginning_time = 0
    node.g_activity_node_ending_time = g_number_of_time_intervals
    # g_node_list.append(node)  # FIXME: why not append the destination node?
    # g_number_of_nodes += 1

    V = set([node.node_id for node in g_node_list])

    with open(path + '/input_link.csv', 'r') as fl:
        linel = fl.readlines()
        for l in linel[1:]:
            l = l.strip().split(',')

            link = Link()
            link.link_id = int(l[0])
            link.from_node_id = int(l[1])
            link.to_node_id = int(l[2])
            link.distance = float(l[3])
            link.spend_tm = int(l[4])

            if link.from_node_id not in V or link.to_node_id not in V:
                continue
            g_node_list[link.from_node_id].outbound_node_list.append(link.to_node_id)
            g_node_list[link.from_node_id].outbound_node_size = len(g_node_list[link.from_node_id].outbound_node_list)
            g_link_list.append(link)
            g_number_of_links += 1
            # add the outbound_link information of each node
            g_node_list[link.from_node_id].outbound_link_list.append(link)
            if g_number_of_links % 8000 == 0:
                print('reading {} links..' \
                      .format(g_number_of_links))

        print('links_number:{}'.format(g_number_of_links))

    for i in range(vehicle_fleet_size):
        agent = Agent()
        agent.agent_id = i
        agent.from_node_id = 0
        agent.to_node_id = g_number_of_nodes
        agent.departure_time_beginning = 0
        agent.arrival_time_ending = g_number_of_time_intervals
        agent.capacity = 200
        g_agent_list.append(agent)
        g_number_of_vehicles += 1

    print('vehicles_number:{}'.format(g_number_of_vehicles))

    V = sorted([node.node_id for node in g_node_list])
    assert 0 in V
    E = [(link.from_node_id, link.to_node_id) for link in g_link_list]
    J = sorted([agent.agent_id for agent in g_agent_list])
    c = [-1] * len(V)
    for node in g_node_list:
        node_id: int = node.node_id
        c[node_id] = node.demand
    C = g_agent_list[0].capacity
    d = {(link.from_node_id, link.to_node_id): link.distance for link in g_link_list}
    l = [-1e9] * len(V)
    u = [-1e9] * len(V)
    for node in g_node_list:
        node_id: int = node.node_id
        l[node_id] = node.g_activity_node_beginning_time
        u[node_id] = node.g_activity_node_ending_time
        assert l[node_id] > -1e8
        assert u[node_id] > -1e8
    T = {(link.from_node_id, link.to_node_id): link.spend_tm for link in g_link_list}

    return V, E, J, c, C, d, l, u, T


def data_loader(path="dataset/data/SolomonDataset_v2/C101-100"):
    return g_ReadInputData(path)


if __name__ == '__main__':
    V, E, J, c, C, d, l, u, T = data_loader()
    from vrp import VRP
    VRP(V, E, J, c, C, d, l, u, T)
    print('data_loader finished')
