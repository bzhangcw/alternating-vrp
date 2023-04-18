//
// Created by C. Zhang on 2021/2/4.
//

#ifndef REPAIRCPP_STATE_H
#define REPAIRCPP_STATE_H

#include "action.h"
#include <vector>
#include <iterator>

class state {
public:
    int s{};    // current position
    double v{}; // value function
    double t{}; // accumulated time
    double c{}; // accumulated used capacity
    std::vector<int> unv{};

    state() {
        this->v = 0.0;
        this->s = 0;
        this->t = 0.0;
        this->c = 0.0;
        this->unv = std::vector<int>();
    };

    state(
            int, double, double, double,
            const int *, int // initialize unvisited via a data buffer.
    );

    state(int s, double v, double t, double c, std::vector<int> unc);

    state(state const &s);

    std::string to_string() const;

    state apply(const action &, double);

    void adjust(double, double, double *);
    double apply();
};


#endif //REPAIRCPP_STATE_H
