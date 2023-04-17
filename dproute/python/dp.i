/* File: dp.i */
%module dp
%include "carrays.i"
%include "std_vector.i"
%include "cpointer.i"

// arrays and vectors
%array_class(double, double_array_py);
%array_class(int, int_array_py);

%pointer_functions(double, doubleP);

namespace std
        {
                %template(DoubleVector) vector<double>;
        }


%{
#define SWIG_FILE_WITH_INIT

#include "dp.h"

%}

void void_run_dp_single_sol(
        int n,     // node size
        int m,     // edge size
        double *f, // cost array of E
        double *D, // distance array of E
        int *I,    // i~ of (i,j) := e in E
        int *J,    // j~ of (i,j) := e in E
        int *V,    // nodes
        double *c, // capacity usage
        double *T, // time needed to travel
        double *S, // time needed to serve
        double *a, // lb of time-window
        double *b, // ub of time-window
        double C,  // capacity
        bool verbose
);

//std::vector<int>
//run_dp_single(
//        int n,     // node size
//        int m,     // edge size
//        double *f, // cost array of E
//        double *D, // distance array of E
//        int *I,    // i~ of (i,j) := e in E
//        int *J,    // j~ of (i,j) := e in E
//        int *V,    // nodes
//        double *c, // capacity usage
//        double *T, // time needed to travel
//        double *S, // time needed to serve
//        double *a, // lb of time-window
//        double *b, // ub of time-window
//        double C  // capacity
//);
