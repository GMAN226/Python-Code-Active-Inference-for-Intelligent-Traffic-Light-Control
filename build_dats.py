"""Regenerate the pgfplots data files from the per-timestep traces.

Reads results/traces_all.csv and writes three data files used by the
Scenario~4 time-series figures in the paper. For each controller in
{rule-based, DQN, active inference}, computes the mean and standard
deviation of the relevant per-step trace across the 100 SUMO seeds at
each evaluation tick, and emits both as separate columns so the
pgfplots templates can render a mean line with a fill-between band.
"""
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
RESULTS = os.path.join(SCRIPT_DIR, "results")
# .dat files live next to the paper .tex one level up so pgfplots can find them
OUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

traces = pd.read_csv(os.path.join(RESULTS, "traces_all.csv"))
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_idle"] = df["idle_time_NS"] + df["idle_time_EW"]
    df = df.sort_values(["controller", "seed", "ttime"]).reset_index(drop=True)
    df["cum_idle"] = df.groupby(["controller", "seed"])["total_idle"].cumsum()
    df["switch"] = (df.groupby(["controller", "seed"])["phase"]
                    .diff().fillna(0) != 0).astype(int)
    df["cum_switch"] = df.groupby(["controller", "seed"])["switch"].cumsum()
    return df


s4 = prepare(traces[traces["scenario"] == 4])
s2 = prepare(traces[traces["scenario"] == 2])

CTRL_KEY = {"Rule-Based": "rb", "DQN": "dqn", "Active Inference": "ai"}


def agg_at_t(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    g = df.groupby(["controller", "ttime"])[value_col].agg(["mean", "std"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    pivot_mean = g.pivot(index="ttime", columns="controller", values="mean")
    pivot_std = g.pivot(index="ttime", columns="controller", values="std")
    out = pd.DataFrame({"t": pivot_mean.index})
    for ctrl, key in CTRL_KEY.items():
        mean = pivot_mean[ctrl].values
        std = pivot_std[ctrl].values
        out[key] = mean
        out[f"{key}_lo"] = np.maximum(mean - std, 0.0)
        out[f"{key}_hi"] = mean + std
    return out.reset_index(drop=True)


def write_dat(path: str, df: pd.DataFrame):
    cols = ["t", "rb", "rb_lo", "rb_hi",
            "dqn", "dqn_lo", "dqn_hi",
            "ai", "ai_lo", "ai_hi"]
    df = df[cols]
    # DQN runs past 3640 s while RB/AI do not, so trailing rows have NaNs for
    # the missing controllers. pgfplots breaks line plots on NaN; truncate to
    # the largest t where every column has a value.
    df = df[df.notna().all(axis=1)]
    with open(path, "w") as f:
        f.write(" ".join(cols) + "\n")
        for _, r in df.iterrows():
            vals = [int(r["t"])] + [f"{r[c]:.3f}" for c in cols[1:]]
            f.write(" ".join(str(v) for v in vals) + "\n")


write_dat(os.path.join(OUT_DIR, "cumidle_s4.dat"),
          agg_at_t(s4, "cum_idle"))
write_dat(os.path.join(OUT_DIR, "idle_s4.dat"),
          agg_at_t(s4, "total_idle"))
write_dat(os.path.join(OUT_DIR, "switches_s4.dat"),
          agg_at_t(s4, "cum_switch"))
write_dat(os.path.join(OUT_DIR, "switches_s2.dat"),
          agg_at_t(s2, "cum_switch"))

print("Wrote cumidle_s4.dat, idle_s4.dat, switches_s4.dat, switches_s2.dat")
