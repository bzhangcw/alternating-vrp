#include "dp.h"

/*
 * Universal functions
 *
 * */
//double evaluate(state &s, action &ac, double multiplier) {
//    return int(ac.is_work == 1) * multiplier;
//}


Solution run_dp_single_sol(
        int n,
        int m, // edge size
        double *f,
        double *D,
        int *V,
        int *I,
        int *J,
        int *T,
        double *a,
        double *b
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
     * define problem queue
     * which is actually a stack (LIFO)
     * */
    auto queue = problem_queue();
    Eigen::MatrixXf Dm(n,n);

    for (int e=0; e<m; e++){
        Dm(I[e], J[e]) = f[e];
    }
    state s_init = state(0, 0.0, 0.0, V, n);
    const string k_init = s_init.to_string();
    state_dict[k_init] = s_init;
    queue.insert(s_init);
    while (!queue.is_empty()) {
        auto kv_pair = queue.get_last();
        string k = kv_pair.first;
        state s = kv_pair.second;
        int n = s.s;
        // cout << s.to_string() << endl;
        if (s.s >= N) {
            // which means you reach the last stage
            value_dict[k] = state::evaluate();
            queue.pop();
            continue;
        }
        auto _tails = tail_dict[k];
        if (!_tails.empty()) {
            /*9
             * tail problems already defined
             * */
        } else {
            /*
             * this gives new tail problems
             * */
            for (auto &_ac_val: {0, 1, -1}) {
                action ac = action(_ac_val);
                auto new_stage_and_tail = s.apply(ac);
                auto new_state = new_stage_and_tail.second;
                string new_state_k = new_state.to_string();
                if (new_state.s < L) {
                    continue;
                    // which violates the lower bound;
                }
                auto new_tail = tail(new_state, ac);
                auto got = value_dict.find(new_state_k);
                if (got == value_dict.end()) {
                    queue.insert(new_state);
                }
                state_dict[new_state_k] = new_state;
                _tails.push_back(new_tail);
            }
            tail_dict[k] = _tails;
            continue;
        }
        /*
         * all tail problems has been solved,
         * do value eval
         * summarize all tail problem
         *
         * */
        double min_val = 1e5;
        action best_ac;
        tail best_tail;
        string best_st_k;

        for (auto &tl : _tails) {
            auto tl_s_k = tl.st.to_string();
            double value = value_dict[tl_s_k] + evaluate(tl.st, tl.ac);
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

    Eigen::ArrayXXd output = Eigen::ArrayXXd::Zero(N, 4);
    /*
     * generate the best policy
     *
     */
    string current_k = k_init;
    while (true) {
        state s = state_dict[current_k];
        if (s.s >= N) break;
        output.col(3)[s.s] = s.s;
        auto got = tail_star_dict.find(current_k);
        if (got == tail_star_dict.end())
            break;
        current_k = got->second.st.to_string();
        auto is_work = got->second.ac.is_work;
        output.col(0)[s.s] = (is_work == 1) * lambda[s.s];
        output.col(1)[s.s] = float(is_work == -1);
        output.col(2)[s.s] = float(is_work == 1);

    }
    if (print) {
        cout << "@best value:" << value_dict[k_init] << endl;
        cout << "@best policy: \n"
                "reward repair work lifespan\n"
             << output
             << endl;
    }

    auto solStruct = Solution(output, value_dict[k_init]);
    return solStruct;

}


std::vector<double> run_dp_single(
        int n,
        double *f,
        double *D,
        double *V,
        double *E,
        double *a,
        double *b,
) {
    auto sol = run_dp_single_sol(n, f, D, V, E, a, b);
    auto array = get_solutions(sol, N, print);
    return array;
}

//
//std::vector<double> run_dp_batch(
//        int size,
//        double *lambda,
//        double *c,
//        int N,
//        double *a,
//        double *b,
//        double L,
//        int *tau,
//        double *s0,
//        bool print = true,
//        bool truncate = true // whether we truncate strictly @stage N
//) {
//    /*
//     * auto sol = run_dp_single_sol(c, N, a, b, L, tau, s0, print, truncate);
//     * auto array = get_solutions(sol, N, print);
//     * return array;
//     */
//    unsigned int nthreads = size;
//
//    std::vector<std::future<Solution>> futures(nthreads);
//    std::vector<Solution> outputs(nthreads);
//    for (decltype(futures)::size_type i = 0; i < nthreads; ++i) {
//        futures[i] = std::async(
//                run_dp_single_sol,
//                lambda, c[i], N, a[i], b[i], L, tau[i], s0[i], print, truncate
//        );
//    }
//    for (decltype(futures)::size_type i = 0; i < nthreads; ++i) {
//        outputs[i] = futures[i].get();
//    }
//
//    auto array = get_solutions(outputs, N);
//    return array;
//}


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
