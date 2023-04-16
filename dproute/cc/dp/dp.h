//
// Created by chuwen on 2021/2/6.
//

#ifndef DUALNEWSVENDOR_DP_H
#define DUALNEWSVENDOR_DP_H

#include <iostream>
#include <fstream>
#include <array>
#include <random>
#include <thread>
#include <future>
#include "Eigen/Dense"
#include "Eigen/Core"
#include "nlohmann/json.hpp"
#include "problem_queue.h"
#include "sol.h"

using json = nlohmann::json;
using array = Eigen::ArrayXd;
using int_array = Eigen::ArrayXi;

std::vector<double>
run_dp_single(
        int n,
        double C,
        double *f,
        double *D,
        double *V,
        double *E,
        double *a,
        double *b
);

json parse_json(char *fp);

json parse_json(const std::string &fp);

int run_test(char *fp, bool bool_batch_test, bool bool_speed_test);

#endif //DUALNEWSVENDOR_DP_H
