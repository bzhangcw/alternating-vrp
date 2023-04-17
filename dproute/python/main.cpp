#include "dp.h"

int main(int argc, char *argv[]) {
    using namespace std;
    nlohmann::json test = parse_json(argv[1]); // benchmark stored at "src/test/test.json"
    problem_data p = parse_data(argv[1]);
    void_run_dp_single_sol(
            p.n,
            p.m,
            p.f.data(),
            p.D.data(),
            p.I.data(),
            p.J.data(),
            p.V.data(),
            p.c.data(),
            p.T.data(),
            p.S.data(),
            p.a.data(),
            p.b.data(),
            p.C,
            true
    );
    return 1;
}
