"""
alternating algorithms for VRP
@update: WR CZ SP
"""
import time

from config import *
import input as gls
from mis import mis_helper

_logger = logging.getLogger("alt")
_logger.setLevel(logging.INFO)
_progress = logging.getLogger("progress")
_progress.setLevel(logging.INFO)
_format = "{:^9s}"
_format_arr = lambda arr: (_format * len(arr)).format(*[str(x) for x in arr])

_headers = _format_arr(
    [
        "k",
        "lb",
        "lb*",
        "ub",
        "ub+",
        "ub*",
        "gap",
        "|N|",  # no. of nodes in mis
        "acc",  # accept mis
        "t",
    ]
)


def log_header():
    _logger.info("-" * (_headers.__len__()))
    _logger.info(_headers)
    _logger.info("-" * (_headers.__len__()))


def admm():
    # loop for each ADMM iterations
    _logger.name = "admm"
    start_time = time.time()
    for i in range(g_number_of_ADMM_iterations):
        if i % 20 == 0:
            log_header()
        used_v = 0
        gls.path_no_seq.append([])
        gls.path_time_seq.append([])
        gls.service_times.append([0] * gls.g_number_of_nodes)
        gls.service_times_lr.append([])
        gls.record_profit.append([])
        gls.repeat_served.append([])
        gls.un_served.append([])
        if i != 0:
            gls.service_times[i] = gls.service_times[i - 1]

        stepsize = 1.0 / (i + 1)

        null_seq = []
        null_time_seq = []
        gls.ADMM_local_upperbound[i] = 0.0
        gls.ADMM_local_lowerbound[i] = 0.0
        for v in range(gls.g_number_of_vehicles - 1):
            service_times_unchange = gls.service_times
            if gls.g_ending_state_vector[v] is not None:
                for n in range(1, gls.g_number_of_nodes - 1):
                    gls.service_times[i][n] -= (
                        gls.g_ending_state_vector[v]
                        .m_VSStateVector[0]
                        .passenger_service_state[n]
                    )
            for n in range(1, gls.g_number_of_nodes - 1):
                # classical
                if gls.alg.alg_dual == "classical":
                    gls.g_node_list[n].base_profit_for_searching = (
                        gls.g_node_list[n].base_profit_for_lr
                        + (1 - 2 * gls.service_times[i][n]) * gls.rho / 2.0
                    )

                # proximal linear
                elif gls.alg.alg_dual == "prox_linear":
                    gls.g_node_list[n].base_profit_for_searching = (
                        gls.g_node_list[n].base_profit_for_lr
                        + (1 - service_times_unchange[i][n]) * gls.rho
                        - 0.1
                        * (
                            0.5
                            - gls.g_ending_state_vector[v]
                            .m_VSStateVector[0]
                            .passenger_service_state[n]
                        )
                        if gls.g_ending_state_vector[v] != None
                        else gls.g_node_list[n].base_profit_for_lr
                        + (1 - service_times_unchange[i][n]) * gls.rho
                        - 0.05
                    )
                else:
                    raise ValueError("unknown dual algorithm")

            gls.g_optimal_time_dependenet_dynamic_programming(
                v,
                origin_node,
                departure_time_beginning,
                destination_node,
                arrival_time_ending,
                gls.BestKSize,
                0,
            )
            gls.ADMM_local_upperbound[i] += (
                gls.g_ending_state_vector[v].m_VSStateVector[0].PrimalLabelCost
            )
            gls.path_no_seq[i].append(
                gls.g_ending_state_vector[v].m_VSStateVector[0].m_visit_sequence
            )
            gls.path_time_seq[i].append(
                gls.g_ending_state_vector[v].m_VSStateVector[0].m_visit_time_sequence
            )

            for n in range(1, gls.g_number_of_nodes - 1):
                gls.service_times[i][n] += (
                    gls.g_ending_state_vector[v]
                    .m_VSStateVector[0]
                    .passenger_service_state[n]
                )
            if len(gls.path_no_seq[i][v]) != 2:
                used_v += 1

        for n in range(1, gls.g_number_of_nodes - 1):
            if gls.service_times[i][n] > 1:
                gls.repeat_served[i].append(n)
            if gls.service_times[i][n] == 0:
                gls.un_served[i].append(n)

                gls.ADMM_local_upperbound[i] = gls.ADMM_local_upperbound[i] + 500
            gls.record_profit[i].append(gls.g_node_list[n].base_profit_for_lr)

        # Calculate_lower_bound(i)
        gls.g_optimal_time_dependenet_dynamic_programming(
            gls.g_number_of_vehicles - 1,
            origin_node,
            departure_time_beginning,
            destination_node,
            arrival_time_ending,
            gls.BestKSize,
            1,
        )

        for vv in range(gls.g_number_of_vehicles - 1):
            gls.ADMM_local_lowerbound[i] = gls.ADMM_local_lowerbound[i] + min(
                gls.g_ending_state_vector[gls.g_number_of_vehicles - 1]
                .m_VSStateVector[vv]
                .LabelCost_for_lr,
                0,
            )  # v shortest paths
        for n in range(1, gls.g_number_of_nodes - 1):
            #     if g_ending_state_vector[g_number_of_vehicles-1].m_VSStateVector[0]==1:
            #         ADMM_local_lowerbound[i] = ADMM_local_lowerbound[i] +rho*(g_number_of_vehicles-2)**2
            #     else:
            #         ADMM_local_lowerbound[i] = ADMM_local_lowerbound[i] + rho
            gls.ADMM_local_lowerbound[i] = (
                gls.ADMM_local_lowerbound[i] + gls.g_node_list[n].base_profit_for_lr
            )

        #         rho+=1
        for n in range(1, gls.g_number_of_nodes - 1):
            gls.g_node_list[n].base_profit_for_lr = (
                gls.g_node_list[n].base_profit_for_lr
                + (1 - gls.service_times[i][n]) * gls.rho
            )

        ########################################################
        # @update by C.Z.
        # primal fixing and better solutions
        # if primal = 1,
        # use the maximum weight independent set heuristic for better solutions.
        # update all associate keys
        ADMM_local_upperbound_seq = gls.ADMM_local_upperbound[i]
        bool_use_mis = False
        if gls.alg.alg_primal != "classical":
            #
            mis_helper.update(gls.g_ending_state_vector)
            ff, cost, visits = mis_helper.compute_best_collection(
                range(1, gls.g_number_of_nodes - 1)
            )
            bool_use_mis = ff is not None and cost < ADMM_local_upperbound_seq
            if bool_use_mis:
                # use new solutions
                for vn in range(len(ff)):
                    gls.path_no_seq[i][vn] = list(ff[vn][0].seq)
                    gls.path_time_seq[i][vn] = list(ff[vn][0].tm)

                for vn in range(len(ff), gls.g_number_of_vehicles - 1):
                    gls.path_no_seq[i][vn] = null_seq
                    gls.path_time_seq[i][vn] = null_time_seq
                gls.ADMM_local_upperbound[i] = cost

                gls.repeat_served[i] = []
                gls.un_served[i] = []
                gls.record_profit[i] = []

                # count service time
                for n in range(1, gls.g_number_of_nodes - 1):
                    if visits[n] > 1:
                        gls.repeat_served[i].append(n)
                    if visits[n] == 0:
                        gls.un_served[i].append(n)
                    gls.record_profit[i].append(gls.g_node_list[n].base_profit_for_lr)

        ########################################################
        b_lb = max(gls.ADMM_local_lowerbound).__round__(2)
        b_ub = min(gls.ADMM_local_upperbound).__round__(2)
        _logger.info(
            _format_arr(
                [
                    i,
                    gls.ADMM_local_lowerbound[i].__round__(2),
                    b_lb,
                    ADMM_local_upperbound_seq.__round__(2),
                    gls.ADMM_local_upperbound[i].__round__(2),
                    b_ub,
                    f"{min(10, ((b_ub - b_lb) / (1e-4 + abs(b_lb)))):.2%}",
                    mis_helper.size,
                    int(bool_use_mis),
                    (time.time() - start_time).__round__(2),
                ]
            )
        )
