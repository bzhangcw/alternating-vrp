import copy
import logging

import xlrd

import config as cfg

MAX_LABEL_COST = 100000
g_node_list = []
g_agent_list = []
g_link_list = []

g_number_of_nodes = 0
g_number_of_links = 0
g_number_of_agents = 0
g_number_of_customers = 0
g_number_of_vehicles = 0

alg = cfg.Algs
stepsize = 0
base_profit = 0
rho = 1
constants = 10

path_no_seq = []
path_time_seq = []
service_times = []
service_times_lr = []
record_profit = []
repeat_served = []
un_served = []

BestKSize = 100
global_upperbound = 9999
global_lowerbound = -9999

ADMM_local_lowerbound = [global_lowerbound] * cfg.g_number_of_ADMM_iterations
ADMM_local_upperbound = [global_upperbound] * cfg.g_number_of_ADMM_iterations

# will be reset
g_ending_state_vector = [None] * g_number_of_vehicles
dp_result = [MAX_LABEL_COST, MAX_LABEL_COST] * g_number_of_vehicles
dp_result_lowerbound = [MAX_LABEL_COST] * g_number_of_vehicles
dp_result_upperbound = [MAX_LABEL_COST] * g_number_of_vehicles


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


def g_ReadInputData():
    # initialization
    global g_number_of_agents
    global g_number_of_vehicles
    global g_number_of_customers
    global g_number_of_nodes
    global g_number_of_links

    # read nodes information
    try:
        book = xlrd.open_workbook(f"{cfg.fpath}/input_node.xlsx")
    except:
        logging.info("fallback to xls")
        book = xlrd.open_workbook(f"{cfg.fpath}/input_node.xls")
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
            g_node_list.append(node)
            g_number_of_nodes += 1
            g_number_of_customers += 1
            if g_number_of_nodes % 100 == 0:
                logging.info("reading {} nodes..".format(g_number_of_nodes))
        except:
            logging.info("Bad read. Check file your self")
    logging.info("nodes_number:{}".format(g_number_of_nodes))
    logging.info("customers_number:{}".format(g_number_of_customers))

    # set the destination node
    node = Node()
    node.type = 1
    node.node_id = g_number_of_nodes  #
    # node.g_activity_node_beginning_time = 0
    node.g_activity_node_ending_time = cfg.g_number_of_time_intervals
    g_node_list.append(node)
    g_number_of_nodes += 1

    with open(f"{cfg.fpath}/input_link.csv", "r") as fl:
        linel = fl.readlines()
        for l in linel[1:]:
            l = l.strip().split(",")

            link = Link()
            link.link_id = int(l[0])
            link.from_node_id = int(l[1])
            link.to_node_id = int(l[2])
            link.distance = float(l[3])
            link.spend_tm = int(l[4])
            g_node_list[link.from_node_id].outbound_node_list.append(link.to_node_id)
            g_node_list[link.from_node_id].outbound_node_size = len(
                g_node_list[link.from_node_id].outbound_node_list
            )
            g_link_list.append(link)
            g_number_of_links += 1
            # add the outbound_link information of each node
            g_node_list[link.from_node_id].outbound_link_list.append(link)
            if g_number_of_links % 8000 == 0:
                logging.info("reading {} links..".format(g_number_of_links))

        logging.info("links_number:{}".format(g_number_of_links))

    for i in range(cfg.vehicle_fleet_size):
        agent = Agent()
        agent.agent_id = i
        agent.from_node_id = 0
        agent.to_node_id = g_number_of_nodes
        agent.departure_time_beginning = 0
        agent.arrival_time_ending = cfg.g_number_of_time_intervals
        agent.capacity = 200
        g_agent_list.append(agent)
        g_number_of_vehicles += 1

    logging.info("vehicles_number:{}".format(g_number_of_vehicles))

    global g_ending_state_vector
    global dp_result, dp_result_lowerbound, dp_result_upperbound

    g_ending_state_vector = [None] * g_number_of_vehicles
    dp_result = [0, 0] * g_number_of_vehicles
    dp_result_lowerbound = [MAX_LABEL_COST] * g_number_of_vehicles
    dp_result_upperbound = [MAX_LABEL_COST] * g_number_of_vehicles


class CVSState:
    def __init__(self):
        self.current_node_id = 0
        self.passenger_service_state = [0] * g_number_of_nodes
        self.m_visit_sequence = []
        self.m_visit_time_sequence = []
        self.m_vehicle_capacity = 0

        self.LabelCost = 0  # with LR price and rho
        self.LabelCost_for_lr = 0  # with LR price
        self.PrimalLabelCost = 0  # without LR price
        self.m_final_arrival_time = 0
        self.passenger_vehicle_visit_allowed_flag = [
            1 for i in range(g_number_of_nodes)
        ]
        self.total_travel_cost = 0
        self.total_waiting_cost = 0
        self.total_fixed_cost = 0

    def mycopy(self, pElement):
        self.current_node_id = copy.copy(pElement.current_node_id)
        self.passenger_service_state = []
        self.passenger_service_state = copy.copy(pElement.passenger_service_state)
        self.passenger_vehicle_visit_allowed_flag = []
        self.passenger_vehicle_visit_allowed_flag = copy.copy(
            pElement.passenger_vehicle_visit_allowed_flag
        )
        self.m_visit_sequence = []
        self.m_visit_sequence = copy.copy(pElement.m_visit_sequence)
        self.m_visit_time_sequence = []
        self.m_visit_time_sequence = copy.copy(pElement.m_visit_time_sequence)
        self.LabelCost = copy.copy(pElement.LabelCost)
        self.LabelCost_for_lr = copy.copy(pElement.LabelCost_for_lr)
        self.PrimalLabelCost = copy.copy(pElement.PrimalLabelCost)
        self.m_vehicle_capacity = copy.copy(pElement.m_vehicle_capacity)

        self.total_travel_cost = copy.copy(pElement.total_travel_cost)
        self.total_waiting_cost = copy.copy(pElement.total_waiting_cost)
        self.total_fixed_cost = copy.copy(pElement.total_fixed_cost)

    def CalculateLabelCost(self, vehicle_id):
        # LabelCost

        # fixed_cost for each vehicle
        if from_node_id == 0 and to_node_id != g_number_of_nodes - 1:
            self.LabelCost = self.LabelCost + cfg.fixed_cost
            self.LabelCost_for_lr = self.LabelCost_for_lr + cfg.fixed_cost
            self.PrimalLabelCost = self.PrimalLabelCost + cfg.fixed_cost
            self.total_fixed_cost += cfg.fixed_cost

        # transportation_cost
        self.LabelCost = (
            self.LabelCost
            - g_node_list[to_node_id].base_profit_for_searching
            + link_no.distance
        )
        self.LabelCost_for_lr = (
            self.LabelCost_for_lr
            - g_node_list[to_node_id].base_profit_for_lr
            + link_no.distance
        )  # no necessary
        self.PrimalLabelCost = self.PrimalLabelCost + link_no.distance
        self.total_travel_cost += link_no.distance

        # waiting cost
        if from_node_id != 0 and waiting_cost_flag == 1:
            self.LabelCost = (
                self.LabelCost
                + (g_node_list[to_node_id].g_activity_node_beginning_time - next_time)
                * cfg.waiting_arc_cost
            )
            self.LabelCost_for_lr = (
                self.LabelCost_for_lr
                + (g_node_list[to_node_id].g_activity_node_beginning_time - next_time)
                * cfg.waiting_arc_cost
            )
            self.PrimalLabelCost = (
                self.PrimalLabelCost
                + (g_node_list[to_node_id].g_activity_node_beginning_time - next_time)
                * cfg.waiting_arc_cost
            )
            self.total_waiting_cost += (
                g_node_list[to_node_id].g_activity_node_beginning_time - next_time
            ) * cfg.waiting_arc_cost

    """def generate_string_key(self):
        str ='n'
        str = str + "%d"%(self.current_node_id)
       # for i in range(g_number_of_customers):
        #    if self.passenger_service_state[i]==1:
        #        str=str+ "_"+"%d"%(i)+"["+"%d"%(self.passenger_service_state[i])+"]"
        return str    """

    def generate_string_key(self):
        str = self.current_node_id
        return str


class C_time_indexed_state_vector:
    def __init__(self):
        self.current_time = 0
        self.m_VSStateVector = []
        self.m_state_map = []

    def Reset(self):
        self.current_time = 0
        self.m_VSStateVector = []
        self.m_state_map = []

    def m_find_state_index(self, string_key):
        if string_key in self.m_state_map:
            return self.m_state_map.index(string_key)
        else:
            return -1

    def update_state(self, new_element, ULFlag):
        string_key = new_element.generate_string_key()
        state_index = self.m_find_state_index(string_key)
        if state_index == -1:
            self.m_VSStateVector.append(new_element)
            self.m_state_map.append(string_key)
        else:
            if ULFlag == 0:  # ADMM
                if new_element.LabelCost < self.m_VSStateVector[state_index].LabelCost:
                    self.m_VSStateVector[state_index] = new_element
            else:  # LR(ULFlag == 1)
                if (
                    new_element.LabelCost_for_lr
                    < self.m_VSStateVector[state_index].LabelCost_for_lr
                ):
                    self.m_VSStateVector[state_index] = new_element

    def Sort(self, ULFlag):
        if ULFlag == 0:  # ADMM
            self.m_VSStateVector = sorted(
                self.m_VSStateVector, key=lambda x: x.LabelCost
            )
        if ULFlag == 1:  # LR
            self.m_VSStateVector = sorted(
                self.m_VSStateVector, key=lambda x: x.LabelCost_for_lr
            )

    def GetBestValue(self, vehicle_id):
        if len(self.m_VSStateVector) >= 1:
            return [
                self.m_VSStateVector[0].LabelCost_for_lr,
                self.m_VSStateVector[0].PrimalLabelCost,
                self.m_VSStateVector[0].LabelCost,
            ]


def g_optimal_time_dependenet_dynamic_programming(
    vehicle_id,
    origin_node,
    departure_time_beginning,
    destination_node,
    arrival_time_ending,
    BestKSize,
    ULFlag,
):
    global g_time_dependent_state_vector
    global g_ending_state_vector
    global g_vehicle_passenger_visit_flag
    global g_vehicle_passenger_visit_allowed_flag
    global link_no
    global to_node_id
    global from_node_id
    global waiting_cost_flag
    global charging_cost_flag
    global next_time

    g_time_dependent_state_vector = [
        [None] * (arrival_time_ending - departure_time_beginning + 2)
    ] * g_number_of_vehicles
    if (
        arrival_time_ending > cfg.g_number_of_time_intervals
        or g_node_list[origin_node].outbound_node_size == 0
    ):
        return MAX_LABEL_COST

    # step 2: Initialization  for origin node at the preferred departure time

    for t in range(departure_time_beginning, arrival_time_ending + 1):
        g_time_dependent_state_vector[vehicle_id][t] = C_time_indexed_state_vector()
        g_time_dependent_state_vector[vehicle_id][t].Reset()
        g_time_dependent_state_vector[vehicle_id][t].current_time = t

    g_ending_state_vector[vehicle_id] = C_time_indexed_state_vector()
    g_ending_state_vector[vehicle_id].Reset()
    # origin_node
    element = CVSState()
    element.current_node_id = origin_node
    g_time_dependent_state_vector[vehicle_id][departure_time_beginning].update_state(
        element, ULFlag
    )

    # step 3:dynamic programming
    # 1 sort m_VSStateVector by labelCost for scan best k elements in step2
    for t in range(departure_time_beginning, arrival_time_ending):
        g_time_dependent_state_vector[vehicle_id][t].Sort(ULFlag)
        # 2 scan the best k elements
        for w_index in range(
            min(
                BestKSize,
                len(g_time_dependent_state_vector[vehicle_id][t].m_VSStateVector),
            )
        ):
            pElement = g_time_dependent_state_vector[vehicle_id][t].m_VSStateVector[
                w_index
            ]  # pElement is an example of  CVSState
            from_node_id = pElement.current_node_id
            # step 2.1 link from_node to to_node
            from_node = g_node_list[from_node_id]

            for i in range(from_node.outbound_node_size):
                to_node_id = from_node.outbound_node_list[i]
                to_node = g_node_list[to_node_id]
                link_no = from_node.outbound_link_list[i]
                next_time = t + link_no.spend_tm

                # step 2.2 check feasibility of node type with the current element

                # to node is destination
                if to_node_id == destination_node:
                    waiting_cost_flag = 0
                    charging_cost_flag = 0
                    new_element = CVSState()
                    new_element.mycopy(pElement)
                    # wait
                    new_element.m_visit_time_sequence.append(next_time)
                    new_element.m_visit_sequence.append(to_node_id)

                    # g_time_dependent_state_vector[vehicle_id][next_time].update_state(new_element)
                    new_element.m_visit_time_sequence.append(arrival_time_ending)
                    new_element.m_visit_sequence.append(to_node_id)
                    # g_time_dependent_state_vector[vehicle_id][next_time].update_state(new_element)
                    new_element.CalculateLabelCost(vehicle_id)
                    g_ending_state_vector[vehicle_id].update_state(new_element, ULFlag)
                    continue

                if to_node_id == origin_node:  # loading
                    continue

                # to node is activity_node
                if pElement.passenger_vehicle_visit_allowed_flag[to_node_id] == 0:
                    continue
                if pElement.passenger_vehicle_visit_allowed_flag[to_node_id] == 1:
                    if next_time > to_node.g_activity_node_ending_time:
                        continue
                    if next_time + cfg.service_length > arrival_time_ending:
                        continue
                    # feasible state transitions
                    # check capacity
                    if (
                        pElement.m_vehicle_capacity
                        > g_agent_list[vehicle_id].capacity - to_node.demand
                    ):
                        continue

                    # waiting
                    if next_time < to_node.g_activity_node_beginning_time:
                        waiting_cost_flag = 1
                        charging_cost_flag = 0
                        new_element = CVSState()
                        new_element.mycopy(pElement)
                        new_element.current_node_id = to_node_id
                        new_element.passenger_service_state[to_node_id] = 1
                        new_element.passenger_vehicle_visit_allowed_flag[to_node_id] = 0

                        # for arriving at activity node and begin wait
                        new_element.m_visit_time_sequence.append(next_time)
                        new_element.m_visit_sequence.append(to_node_id)
                        # for wait until activity node's depature time
                        new_element.m_vehicle_capacity += to_node.demand

                        new_element.m_visit_time_sequence.append(
                            to_node.g_activity_node_beginning_time
                        )
                        new_element.m_visit_sequence.append(to_node_id)
                        new_element.CalculateLabelCost(vehicle_id)
                        new_element.m_visit_time_sequence.append(
                            to_node.g_activity_node_beginning_time + cfg.service_length
                        )
                        new_element.m_visit_sequence.append(to_node_id)
                        g_time_dependent_state_vector[vehicle_id][
                            to_node.g_activity_node_beginning_time + cfg.service_length
                        ].update_state(new_element, ULFlag)
                        continue

                    else:
                        # donot need waiting
                        waiting_cost_flag = 0
                        charging_cost_flag = 0
                        new_element = CVSState()
                        new_element.mycopy(pElement)
                        new_element.current_node_id = to_node_id
                        new_element.passenger_service_state[to_node_id] = 1
                        new_element.m_visit_time_sequence.append(next_time)
                        new_element.m_visit_sequence.append(to_node_id)
                        new_element.m_vehicle_capacity += to_node.demand

                        new_element.passenger_vehicle_visit_allowed_flag[to_node_id] = 0

                        new_element.CalculateLabelCost(vehicle_id)
                        new_element.m_visit_time_sequence.append(
                            next_time + cfg.service_length
                        )
                        new_element.m_visit_sequence.append(to_node_id)
                        g_time_dependent_state_vector[vehicle_id][
                            next_time + cfg.service_length
                        ].update_state(new_element, ULFlag)
                        continue

    # logging.info("ok")
    g_ending_state_vector[vehicle_id].Sort(ULFlag)
    # logging.info(g_ending_state_vector[vehicle_id].m_VSStateVector[0].m_visit_sequence)
    # logging.info(g_ending_state_vector[vehicle_id].m_VSStateVector[0].m_visit_time_sequence)
    return g_ending_state_vector[vehicle_id].GetBestValue(vehicle_id)
