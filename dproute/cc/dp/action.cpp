//
// Created by C. Zhang on 2021/2/4.
//

#include "action.h"
#include <iostream>

action::~action() = default;


action::action(const action &action) {
    *this = action;
}

action::action(int i, int j, double f, double t) {
    this->i = i;
    this->j = j;
    this->f = f;
    this->t = t;
}

//std::string action::to_string() const {
//    return std::to_string(
//            fprintf(stdout, "(%d,%d):%.1e", this->i, this->j, this->f)
//    );
//}