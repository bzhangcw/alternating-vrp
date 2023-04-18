import numpy as np

from .pydp import *


def convert_to_c_arr(size, lambda_k):
    c_arr = double_array_py(size)
    for i in range(size):
        c_arr[i] = lambda_k[i]

    return c_arr


def convert_to_c_arr_int(size, lambda_k):
    c_arr = int_array_py(size)
    for i in range(size):
        c_arr[i] = lambda_k[i]

    return c_arr


def solve_by_dp_cc(data, verbose=True, inexact=False):
    sol = run_dp(
        data['n'],
        data['m'],
        convert_to_c_arr(data['m'], data['f']),
        convert_to_c_arr(data['m'], data['D']),
        convert_to_c_arr_int(data['m'], data['I']),
        convert_to_c_arr_int(data['m'], data['J']),
        convert_to_c_arr_int(data['n'], data['V']),
        convert_to_c_arr(data['n'], data['c']),
        convert_to_c_arr(data['m'], data['T']),
        convert_to_c_arr(data['n'], data['S']),
        convert_to_c_arr(data['n'], data['a']),
        convert_to_c_arr(data['n'], data['b']),
        data['C'],
        verbose,
        inexact,
        20.0
    )

    return [*sol, 0]
