//
// Created by C. Zhang on 2021/2/4.
//

#ifndef REPAIRCPP_STATE_H
#define REPAIRCPP_STATE_H

# include "action.h"
# include<unordered_set>

class state {
public:
    int s{};
    double v{};
    double t{};
    std::unordered_set<int> unv{};

    state() {
        this->v = 0.0;
        this->s = 0;
        this->t = 0.0;
        this->unv = std::unordered_set<int>();
    };

    state(int, double, double, const int *, int);
    state(int s, double v, double t, std::unordered_set<int> unc);

    state(state const &s);

    std::string to_string() const;

    state apply(const action &ac);
    state apply();
};


#endif //REPAIRCPP_STATE_H
