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
#include "sol.h"

%}


std::vector<double> run_dp_single(
        double *lambda,
        double c,
        int N,
        double a,
        double b,
        double L,
        int tau,
        double s0,
        bool print,
        bool truncate // whether we truncate strictly @stage N
);


std::vector<double> run_dp_batch(
        int size,
        double *lambda,
        double *c,
        int N,
        double *a,
        double *b,
        double L,
        int *tau,
        double *s0,
        bool print,
        bool truncate
);

