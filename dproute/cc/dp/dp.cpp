#include "dp.h"

/*
 * Universal functions
 *
 * */
//double evaluate(state &s, action &ac, double multiplier) {
//    return int(ac.is_work == 1) * multiplier;
//}


#define DPVERBOSE 1

bool bool_valid_route(state s, double C, double lb, double ub) {
    if (s.c > C) {
        return false;
    } else if (s.t > ub) {
        return false;
    } else if (s.t < lb) {
        return false;
    } else {
        return true;
    }
}

bool bool_valid_route(state s, double C) {
    if (s.c > C) {
        return false;
    } else {
        return true;
    }
}


void void_run_dp_single_sol(
        int n,
        int m,
        double *f,
        double *D,
        int *I,
        int *J,
        int *V,
        double *c,
        double *T,
        double *S,
        double *a,
        double *b,
        double C = 200.0,
        bool verbose = false
) {
    using namespace std;
    auto ac = run_dp_single(n, m, f, D, I, J, V, c, T, S, a, b, C, verbose, false);
    if (verbose) {
        cout << "@best policy:" <<
             endl;
        for (auto cc: ac)
            cout << cc.to_string() << "=" <<
                 endl;
    }
}

std::vector<int>
run_dp(int n, int m, double *f, double *D, int *I, int *J, int *V, double *c, double *T, double *S, double *a,
       double *b, double C, bool verbose, bool inexact=false) {
    auto ac = run_dp_single(n, m, f, D, I, J, V, c, T, S, a, b, C, verbose, inexact);
    std::vector<int> ans{};
    for (auto cc: ac) {
        ans.push_back(cc.i);
    }
    return ans;
}


std::vector<action> run_dp_single(
        int n,
        int m,
        double *f,
        double *D,
        int *I,
        int *J,
        int *V,
        double *c,
        double *T,
        double *S,
        double *a,
        double *b,
        double C = 200.0,
        bool verbose = false,
        bool inexact = false
) {
    /*
     * define containers
     * */
    using namespace std;
    state_map state_dict = state_map();
    action_map action_dict = action_map();
    value_map value_dict = value_map();
    tail_map tail_dict = tail_map();
    best_tail_map tail_star_dict = best_tail_map();

    /* Debug logs
        Eigen::ArrayXd c_arr(N);
        for (int i = 0; i < N; i++) {
            cout << *(c + i) << ",";
            c_arr[i] = c[i];
        }
        cout << endl;
    */
    /*
     * define problem LIFO queue
     * which is actually a stack (LIFO)
     * */
    auto queue = problem_queue();
    Eigen::MatrixXf Dm(n, n); // cost
    Eigen::MatrixXd Ex(n, n); // id matrix


    for (int e = 0; e < m; e++) {
        Dm(I[e], J[e]) = f[e];
        Ex(I[e], J[e]) = e;
    }


    state s_init = state(0, 0.0, 0.0, 0.0, V, n);
    const string k_init = s_init.to_string();
    state_dict[k_init] = s_init;
    queue.insert(s_init);
    while (!queue.is_empty()) {
        auto kv_pair = queue.get_last();
        string k = kv_pair.first;
        state s = kv_pair.second;
//        if (s.unv.empty() || s.c > 0) {
//            // which means you reach the last stage
//            value_dict[k] = 0.0;
//            queue.pop();
//            continue;
//        }
        if (verbose) { cout << s.to_string() << endl; }

        auto _tails = tail_dict[k];
        // generating tail problems
        if (!_tails.empty() || (value_dict.find(k) != value_dict.end())) {
            // tail problems already defined or there is none
        } else {
            // this gives new tail problems
            for (auto &j: s.unv) {
                if ((s.s == 0) && (j == 0)) continue;

                double cost = Dm(s.s, j);
                if (inexact) {
                    if (cost >= 0 and s.s != 0 and not(j == 0)) continue;
                }

                int eid = Ex(s.s, j);
                action ac = action(
                        s.s,
                        j,
                        cost,
                        T[eid],
                        c[j]
                );
                auto new_state = s.apply(ac, S[j]);

                // what is a new state?
//                string new_state_k = new_state.to_string();
                double lb = a[j];
                double ub = b[j];
                new_state.adjust(lb, ub, b);
                if (!bool_valid_route(new_state, C, lb, ub)) {
//                if (!bool_valid_route(new_state, C)) {
                    continue;
                }

                auto new_tail = tail(new_state, ac);
                if (verbose) {
                    fprintf(
                            stdout,
                            "---\n|--action: %s \n|--state: %s\n|--tail: %s\n",
                            ac.to_string().c_str(),
                            new_state.to_string().c_str(),
                            new_tail.to_string().c_str()
                    );

                }
                _tails.push_back(new_tail);
            }

            tail_dict[k] = _tails;
            if (_tails.empty()) {
                // no tail problem, summarize
                value_dict[k] = 0.0;
            } else {
                for (auto tl: _tails) {
                    auto new_state_k = tl.st.to_string();
                    if (tl.st.s == 0) {
                        // back to depot
                        value_dict[new_state_k] = 0.0;
                        continue;
                    }
                    auto got = value_dict.find(new_state_k);
                    if (got == value_dict.end()) {
                        // not exists
                        state_dict[new_state_k] = tl.st;
                        queue.insert(tl.st);
                    }
                }

            }
            continue;
        }

        /*
         * all tail problems has been proposed,
         * do value eval
         * summarize all tail problem
         *
         * */
        double min_val = 1e5;
        action best_ac;
        tail best_tail;
        string best_st_k;

        for (auto &tl: _tails) {
            auto tl_s_k = tl.st.to_string();
            double value = value_dict[tl_s_k] + tl.ac.f;
            if (value < min_val) {
                min_val = value;
                best_ac = tl.ac;
                best_tail = tl;
                best_st_k = tl_s_k;
            }
        }
        value_dict[k] = min_val;
        tail_star_dict[k] = best_tail;
        queue.pop();
    }
    // generate the best policy
    string current_k = k_init;
    vector<action> ac;
    while (true) {
        state s = state_dict[current_k];
//        if (s.s >= N) break;
//        output.col(3)[s.s] = s.s;
        auto got = tail_star_dict.find(current_k);
        if (got == tail_star_dict.end())
            break;
        current_k = got->second.st.to_string();
//        auto is_work = got->second.ac.is_work;
//        output.col(0)[s.s] = (is_work == 1) * lambda[s.s];
//        output.col(1)[s.s] = float(is_work == -1);
//        output.col(2)[s.s] = float(is_work == 1);
        ac.push_back(got->second.ac);
    }
    if (verbose) {
        cout << "@best value:" << value_dict[k_init] << endl;
        cout << "@best policy:" <<
             endl;
        for (auto cc: ac)
            cout << cc.to_string() << "=" <<
                 endl;
    }
    return ac;
}


json parse_json(char *fp) {
    using namespace std;
    ifstream ifs(fp);
    json _json = json::parse(ifs);
    return _json;
}

json parse_json(const std::string &fp) {
    using namespace std;
    ifstream ifs(fp);
    json _json = json::parse(ifs);
    return _json;
}

problem_data parse_data(const std::string &fp) {
    using namespace std;
    auto p = problem_data();
    nlohmann::json t = parse_json(fp);
    p.C = t["C"];
    p.n = t["n"];
    p.m = t["m"];
    p.c = t["c"].get<std::vector<double>>();
    p.a = t["a"].get<std::vector<double>>();
    p.b = t["b"].get<std::vector<double>>();
    p.T = t["T"].get<std::vector<double>>();
    p.S = t["S"].get<std::vector<double>>();
    p.I = t["I"].get<std::vector<int>>();
    p.J = t["J"].get<std::vector<int>>();
    p.V = t["V"].get<std::vector<int>>();
    p.f = t["f"].get<std::vector<double>>();
    p.D = t["D"].get<std::vector<double>>();

    // verbose logging
    fprintf(stdout, "number of nodes: %d, edges: %d, total capacity: %f\n",
            p.n, p.m, p.C);

    return p;
}


problem_data parse_data(char *fp) {
    using namespace std;
    auto p = problem_data();
    nlohmann::json t = parse_json(fp);
    p.C = t["C"];
    p.n = t["n"];
    p.m = t["m"];
    p.c = t["c"].get<std::vector<double>>();
    p.a = t["a"].get<std::vector<double>>();
    p.b = t["b"].get<std::vector<double>>();
    p.T = t["T"].get<std::vector<double>>();
    p.S = t["S"].get<std::vector<double>>();
    p.I = t["I"].get<std::vector<int>>();
    p.J = t["J"].get<std::vector<int>>();
    p.V = t["V"].get<std::vector<int>>();
    p.f = t["f"].get<std::vector<double>>();
    p.D = t["D"].get<std::vector<double>>();

    // verbose logging
    fprintf(stdout, "number of nodes: %d, edges: %d, total capacity: %f\n",
            p.n, p.m, p.C);

    return p;
}


