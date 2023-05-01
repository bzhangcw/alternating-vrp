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


def solve_by_dp_cc(data, select=None, verbose=True, inexact=False):
    if select is None:
        sol = run_dp(
            data['n'],
            data['m'],
            convert_to_c_arr(data['m'], data['f']),
            convert_to_c_arr(data['m'], data['D']), # not really used
            convert_to_c_arr_int(data['m'], data['I'].tolist()),
            convert_to_c_arr_int(data['m'], data['J'].tolist()),
            convert_to_c_arr_int(data['n'], data['V'].tolist()),
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

    _idx_v, _idx_e = select
    _n, _m = len(_idx_v), len(_idx_e)
    sol = run_dp(
        _n,
        _m,
        convert_to_c_arr(_m, data['f'][_idx_e]),
        convert_to_c_arr(_m, data['D'][_idx_e]),  # not really used
        convert_to_c_arr_int(_m, data['I'][_idx_e].tolist()),
        convert_to_c_arr_int(_m, data['J'][_idx_e].tolist()),
        convert_to_c_arr_int(_n, data['V'][_idx_v].tolist()),
        convert_to_c_arr(_n, data['c'][_idx_v]),
        convert_to_c_arr(_m, data['T'][_idx_e]),
        convert_to_c_arr(_n, data['S'][_idx_v]),
        convert_to_c_arr(_n, data['a'][_idx_v]),
        convert_to_c_arr(_n, data['b'][_idx_v]),
        data['C'],
        verbose,
        inexact,
        20.0
    )
    return [*sol, 0]


