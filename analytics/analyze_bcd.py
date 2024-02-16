import os
import sys
import pandas as pd
import json
import numpy as np

pd.set_option("display.max_columns", None)
# usage: python analyze_bcd.py <directory> <best_sol>
REMAME_MAP = {
    "f_best": r"$f^*$",
    # "k": "$k$",
    # "t": "$t$",
    "f1": "$f1$",
    "gap_rel1": r"$\epsilon_f1$",
    "f2": "$f2$",
    "gap_rel2": r"$\epsilon_f2$",
    "f_ortools": r"$f^O$",
    "nv": "$|J|$",
    "nc": "$|C|$",
    "f": "$f$",
    # "f_h": "$f_h$",
    # "lb": r"$\underline f$",
    # "eps_axb": r"$\epsilon_A$",
    # "eps_cap": r"$\epsilon_c$",
    # "eps_fixpoint": r"$\epsilon_x$",
    # "calls_route_tsp": r"$\mathcal K_c$",
    # "avgtm_route_tsp": r"$t_c$",
    # "gap": r"$\Delta f$",
    "gap_rel": r"$\epsilon_f$",
}


# load all json files in a directory
def load_json_files(directory):
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]
    json_data = []
    for file in json_files:
        name_data = file.split(".")[0]
        nv_nc = file.split(".")[1]
        nv, nc = nv_nc.split("-")
        # oracle data
        with open(f"{directory}/{file}") as f:
            results = json.load(f)
            _calls = results["oracle_calls"]
            _times = results["oracle_avgtm"]
            del results["oracle_calls"]
            del results["oracle_avgtm"]
            for k, v in _calls.items():
                if k.endswith("route_tsp"):
                    results[f"calls_{k}"] = v
                    results[f"avgtm_{k}"] = _times[k]

            json_data.append(
                {"instance": name_data, "nc": int(nc), "nv": int(nv), **results}
            )
    return json_data


# each as a record and merge to a pandas dataframe
json_data = load_json_files(sys.argv[1])
# read best know
df1 = (
    pd.read_csv(sys.argv[2])
    .assign(
        instance=lambda df: df["Problem"].apply(lambda x: x.split(".")[0].lower()),
        nc=lambda df: df["Problem"].apply(lambda x: int(x.split(".")[1])),
        nv=lambda df: df["NV"].astype(int),
    )
    .set_index(["instance", "nc", "nv"])
    .rename(columns={"Distance": "f_best"})
)

# ortools
df2 = (
    pd.read_csv(sys.argv[3])
    .assign(
        instance=lambda df: df["filename"].apply(lambda x: x.split(".")[0].lower()),
        nc=lambda df: df["n_customers"],
        nv=lambda df: df["n_vehicles"].astype(int),
    )
    .set_index(["instance", "nc", "nv"])
    .rename(columns={"travel_time": "f1"})
)

df3 = (
    pd.read_csv(sys.argv[3])
    .assign(
        instance=lambda df: df["filename"].apply(lambda x: x.split(".")[0].lower()),
        nc=lambda df: df["n_customers"],
        nv=lambda df: df["n_vehicles"].astype(int),
    )
    .set_index(["instance", "nc", "nv"])
    .rename(columns={"travel_time": "f2"})
)
df2["f2"] = df3["f2"]


df2 = (
    df1.join(df2)
    .assign(
        gap1=lambda df: df["f1"] - df["f_best"],
        gap_rel1=lambda df: (100 * df["gap1"] / df["f_best"]).apply(
            lambda x: f"{x:.1f}\%" if np.isnan(x) == False else "-"
        ),
        gap2=lambda df: df["f2"] - df["f_best"],
        gap_rel2=lambda df: (100 * df["gap2"] / df["f_best"]).apply(
            lambda x: f"{x:.1f}\%" if np.isnan(x) == False else "-"
        ),
    )
    .fillna("-")
)

df = (
    df2.join(
        pd.DataFrame.from_records(json_data)
        .set_index(["instance", "nc", "nv"])
        .sort_index()
    )
    .assign(
        bool_succ=lambda df: (df["eps_axb"] < 1e-1) * (df["eps_cap"] < 1e-1),
        f=lambda df: df.apply(
            lambda row: row["f"] if row["bool_succ"] else np.nan, axis=1
        ),
        gap=lambda df: df["f"] - df["f_best"],
        gap_rel=lambda df: (100 * df["gap"] / df["f_best"]).apply(
            lambda x: f"{x:.1f}\%" if np.isnan(x) == False else "-"
        ),
    )
    .fillna("-")
)
print(
    df.rename(columns=REMAME_MAP)
    .rename_axis(index=REMAME_MAP)
    .to_latex(longtable=True, escape=False, float_format="%.1f"),
    file=open("table1.tex", "w"),
)
columns = df.columns
print(
    df[[c for c in REMAME_MAP.keys() if c in columns]]
    .rename(columns=REMAME_MAP)
    .rename_axis(index=REMAME_MAP)
    .to_latex(longtable=True, escape=False, float_format="%.1f"),
    file=open("table2.tex", "w"),
)
