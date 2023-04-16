//
// Created by C. Zhang on 2021/2/4.
//
#include <string>
#include "state.h"

state::state(int s, double v, double t, const int *vs, int n) {
    this->v = v; // v(s)
    this->s = s; // s
    this->t = t; // vs
    this->unv = std::unordered_set<int>(vs, vs + n);
}
state::state(int s, double v, double t, std::unordered_set<int> unc) {
    this->v = v; // v(s)
    this->s = s; // s
    this->t = t; // vs
    this->unv = unc;
}

state::state(const state &s) {
    *this = s;
}

std::string state::to_string() const {

    return std::to_string(this->s)
           + ":" + std::to_string(this->t)
           + "@" + std::to_string(this->v);

}

bool operator==(const state &lhs, const state &rhs) {
    return lhs.s == rhs.s && lhs.v == rhs.v;
}


state state::apply(const action &ac) {
    auto cc = std::unordered_set<int>(this->unv);
    cc.erase(ac.i);
    state s = state(
            ac.j,
            this->v + ac.f,
            this->t + ac.t,
            cc
    );
}

state state::apply() {
    return state();
}



