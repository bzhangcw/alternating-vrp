//
// Created by C. Zhang on 2021/2/4.
//

#include <iostream>
#include <queue>
#include <unordered_map>

#include "Eigen/Dense"
#include "nlohmann/json.hpp"

#include "problem_queue.h"
#include "dp.h"

using Eigen::MatrixXf;
using namespace std;
using namespace Eigen;

int main(int argc, char *argv[]) {

    /*
     * TEST DATA WITH BENCHMARK RESULTS
     * @date: 2021/02/05
     *
     * */
    nlohmann::json test = parse_json(argv[1]); // benchmark stored at "src/test/test.json"

    vector<int> V = test["V"];
    vector<int> I = test["I"];
    vector<int> J = test["J"];
    int n = V.size();
    MatrixXf D(n, n);
    int idx = 0;

    // query c and C; knapsack
    vector<double> c = test["c"];
    vector<double> a = test["a"];
    vector<double> b = test["b"];
    double C = test["C"];
    return 0;
}