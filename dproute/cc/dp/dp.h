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
using arrayd = Eigen::ArrayXd;
using arrayf = Eigen::ArrayXf;
using int_array = Eigen::ArrayXi;

bool bool_valid_route(state, double);

bool bool_valid_route(state, double, double, double);

struct problem_data{
    int n;     // node size
    int m;     // edge size
    double *f; // cost array of E
    double *D; // distance array of E
    int *I;    // i~ of (i,j) := e in E
    int *J;    // j~ of (i,j) := e in E
    int *V;    // nodes
    double *c; // capacity usage
    double *T; // time needed to serve
    double *a; // lb of time-window
    double *b; // ub of time-window
    double C;  // capacity
};


//Solution run_dp_single_sol(
void run_dp_single_sol(
        int n,     // node size
        int m,     // edge size
        double *f, // cost array of E
        double *D, // distance array of E
        int *I,    // i~ of (i,j) := e in E
        int *J,    // j~ of (i,j) := e in E
        int *V,    // nodes
        double *c, // capacity usage
        double *T, // time needed to serve
        double *a, // lb of time-window
        double *b, // ub of time-window
        double C,  // capacity
        bool verbose
);

std::vector<double>
run_dp_single(
        int n,     // node size
        int m,     // edge size
        double *f, // cost array of E
        double *D, // distance array of E
        int *I,    // i~ of (i,j) := e in E
        int *J,    // j~ of (i,j) := e in E
        int *V,    // nodes
        double *c, // capacity usage
        double *T, // time needed to serve
        double *a, // lb of time-window
        double *b, // ub of time-window
        double C  // capacity
);

problem_data parse_data(const std::string &fp);
problem_data parse_data(char *fp);

json parse_json(char *fp);

json parse_json(const std::string &fp);


int run_test(char *fp, bool bool_batch_test, bool bool_speed_test);

#endif //DUALNEWSVENDOR_DP_H
