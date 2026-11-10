#!/usr/bin/env python3
"""
train_eta_v2.py - the ETA model, version 2.

What is new compared with train_eta.py
  1. LIVE "RECENT BUSES" INPUTS: how much slower or faster than normal the buses that passed a stop pair in the last
     10 minutes (all routes) and the last 20 minutes (this route) were. This tells the model about today's traffic.
  2. HISTORY BY TIME OF DAY: a factor for every hour and for weekday against weekend, learned on the training days.
  3. STOP INPUTS: distance since the last stop, and how far the bus is through the current stop-to-stop section.
  4. THE MODEL LEARNS THE CORRECTION to the historical ETA (target = real time minus the historical ETA), not the whole time.
  5. PREDICTION RANGES: two more models give a lower and an upper guess (default 10% and 90%): "8-12 min".
  6. LEAVE-ONE-DAY-OUT TEST (--mode cv): train on all other days, test on each day, so the result does not depend on one day.
  7. A per-route check: where is the model worse than the historical ETA?

   python train_eta_v2.py                    final model: train 60% / validation 20% / test 20% in time, ranges, ablations
   python train_eta_v2.py --mode cv          leave-one-day-out test (slower)
   python train_eta_v2.py --step 10          use every 10th ping (faster)
Keep train_eta.py in the same folder (this script uses its loading code). Needs lightgbm (uv add lightgbm), else scikit-learn is used.
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from train_eta import load, ping_features, make_cases, metrics
except ImportError:
    raise SystemExit("Keep train_eta.py in the same folder as train_eta_v2.py.")

FEATURES = ["dist", "h", "v1", "v2", "v3", "v_mean3", "standing_s", "hour", "dow", "is_weekend", "v_route", "eta_route", "hist_eta", "tod_factor",
            "hist_tod", "rc10", "rc10_n", "rr20", "rr20_n", "dist_prev", "seg_frac", "prog", "rep"]
RECENT = ["rc10", "rc10_n", "rr20", "rr20_n"]
MIN_SUPPORT = 8


def day_of(ts):
    return ((np.asarray(ts) + 19800) // 86400).astype(int)


# ---------------------------------------------------------------- what is learned from the training days
def seg_pairs(arr):
    a = arr.sort_values(["run_key", "stop_idx"])
    g = a.groupby("run_key")
    a = a.assign(next_t=g["arrive_ts"].shift(-1), next_i=g["stop_idx"].shift(-1))
    s = a[a["next_i"] == a["stop_idx"] + 1].copy()
    s["t"] = s["next_t"] - s["arrive_ts"]
    return s[s["t"].between(3, 600)]


def seg_medians(arr_train):
    s = seg_pairs(arr_train)
    g = s.groupby(["route_id", "stop_idx"])["t"].agg(["median", "size"])
    g = g[g["size"] >= MIN_SUPPORT]
    return g["median"].to_dict()


def route_tables(arr, med, v_global):
    sp = arr.groupby(["route_id", "stop_idx"])["stop_prog_m"].first().reset_index().sort_values(["route_id", "stop_idx"])
    tables = {}
    for r, g2 in sp.groupby("route_id"):
        idx, prog = g2["stop_idx"].values, g2["stop_prog_m"].values
        if len(idx) < 3 or np.any(np.diff(prog) < 0):
            continue
        T = np.empty(len(idx) - 1)
        for m in range(len(idx) - 1):
            T[m] = med.get((r, idx[m]), np.nan) if idx[m + 1] == idx[m] + 1 else np.nan
            if np.isnan(T[m]):
                T[m] = (prog[m + 1] - prog[m]) / v_global
        tables[r] = (idx, prog, np.concatenate([[0.0], np.cumsum(T)]))
    return tables


def make_events(arr, med):
    """Every observed stop-to-stop time compared with normal: ratio > 1 = slower than normal. Time = when the bus reached the second stop."""
    s = seg_pairs(arr)
    m = pd.DataFrame([(r, i, v) for (r, i), v in med.items()], columns=["route_id", "stop_idx", "med"])
    s = s.merge(m, on=["route_id", "stop_idx"], how="inner")
    s["ratio"] = (s["t"] / s["med"]).clip(0.3, 3.0)
    return s[["route_id", "next_t", "ratio"]].rename(columns={"next_t": "t_end"}).sort_values("t_end").reset_index(drop=True)


def hist_feats(cases, tables):
    n = len(cases)
    eta, dp, sf = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    pos = {}
    for k, r in enumerate(cases["route_id"].values):
        pos.setdefault(r, []).append(k)
    p, si = cases["prog"].values, cases["sidx"].values
    for r, rows in pos.items():
        t = tables.get(r)
        if t is None:
            continue
        idx, prog, C = t
        rows = np.array(rows)
        j = np.minimum(np.searchsorted(idx, si[rows]), len(idx) - 1)
        ok = idx[j] == si[rows]
        eta[rows] = np.where(ok, C[j] - np.interp(p[rows], prog, C), np.nan)
        k2 = np.clip(np.searchsorted(prog, p[rows], side="right") - 1, 0, len(prog) - 2)
        dp[rows] = p[rows] - prog[k2]
        sf[rows] = dp[rows] / np.maximum(prog[k2 + 1] - prog[k2], 1)
    return eta, dp, sf


def window_mean(ev_t, ev_r, ts, window):
    cs = np.concatenate([[0.0], np.cumsum(ev_r)])
    hi = np.searchsorted(ev_t, ts, side="left")                 # events that ended before the ping
    lo = np.searchsorted(ev_t, ts - window, side="left")
    n = hi - lo
    return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan), n


def recent_feats(cases, ev):
    ts = cases["ts"].values
    rc, rcn = window_mean(ev["t_end"].values, ev["ratio"].values, ts, 600)
    rr, rrn = np.full(len(cases), np.nan), np.zeros(len(cases))
    eidx = ev.groupby("route_id").indices
    cpos = {}
    for k, r in enumerate(cases["route_id"].values):
        cpos.setdefault(r, []).append(k)
    for r, rows in cpos.items():
        e = eidx.get(r)
        if e is None:
            continue
        rows = np.array(rows)
        rr[rows], rrn[rows] = window_mean(ev["t_end"].values[e], ev["ratio"].values[e], ts[rows], 1200)
    return rc, rcn, rr, rrn


class Learned:
    pass


def learn(tr, arr_train, arr_all):
    L = Learned()
    vt = (tr["dist"] / tr["actual"]).clip(1, 20)
    L.v_global = float(vt.median()); L.v_route = vt.groupby(tr["route_id"]).median().to_dict()
    L.med = seg_medians(arr_train)
    L.tables = route_tables(arr_all, L.med, L.v_global)
    L.events = make_events(arr_all, L.med)
    tmp = tr[["route_id", "sidx", "prog", "dist", "actual", "hour", "is_weekend"]].copy()
    tmp["hist_eta"] = hist_feats(tmp, L.tables)[0]
    ok = tmp["hist_eta"] > 10
    r = (tmp.loc[ok, "actual"] / tmp.loc[ok, "hist_eta"]).clip(0.3, 3)
    g = r.groupby([tmp.loc[ok, "is_weekend"], np.floor(tmp.loc[ok, "hour"])]).agg(["median", "size"])
    L.tod = {k: v["median"] for k, v in g.iterrows() if v["size"] >= 200}
    return L


def add_feats(d, L):
    d = d.copy()
    d["v_route"] = d["route_id"].map(L.v_route).fillna(L.v_global).clip(1, 20)
    d["eta_route"] = d["dist"] / d["v_route"]
    d["hist_eta"], d["dist_prev"], d["seg_frac"] = hist_feats(d, L.tables)
    d["tod_factor"] = [L.tod.get((w, h), 1.0) for w, h in zip(d["is_weekend"].values, np.floor(d["hour"].values))]
    d["hist_tod"] = d["hist_eta"] * d["tod_factor"]
    d["rc10"], d["rc10_n"], d["rr20"], d["rr20_n"] = recent_feats(d, L.events)
    d["base"] = d["hist_tod"].fillna(d["eta_route"])
    return d


# ---------------------------------------------------------------- models
def fit_model(Xtr, ytr, Xva, yva, backend, kind="l1", alpha=0.5, quiet=False):
    if backend == "lightgbm":
        import lightgbm as lgb
        params = dict(learning_rate=0.08, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=0)
        params.update(objective="regression_l1") if kind == "l1" else params.update(objective="quantile", alpha=alpha)
        dtr = lgb.Dataset(Xtr, ytr); dva = lgb.Dataset(Xva, yva, reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        if not quiet:
            print(f"      trained {m.best_iteration} trees")
        return m, m.predict
    from sklearn.ensemble import HistGradientBoostingRegressor
    kw = dict(loss="absolute_error") if kind == "l1" else dict(loss="quantile", quantile=alpha)
    m = HistGradientBoostingRegressor(learning_rate=0.1, max_iter=300, max_leaf_nodes=63, min_samples_leaf=100, early_stopping=True,
                                      validation_fraction=0.1, n_iter_no_change=20, random_state=0, **kw)
    m.fit(Xtr, ytr)
    if not quiet:
        print(f"      trained {m.n_iter_} trees (scikit-learn)")
    return m, m.predict


def to_target(d, target):
    return {"residual": d["actual"] - d["base"], "ratio": d["actual"] / d["base"].clip(5, None), "raw": d["actual"]}[target]


def from_pred(d, p, target):
    return {"residual": d["base"].values + p, "ratio": d["base"].values * np.clip(p, 0.2, 5), "raw": p}[target]


def pick_backend(b):
    if b != "auto":
        return b
    try:
        import lightgbm  # noqa
        return "lightgbm"
    except ImportError:
        return "sklearn"


def prepare(a):
    pings, arr, files = load(a.arrivals, a.date)
    print(f"1. LOAD      {len(files)} day(s): {len(pings):,} pings, {len(arr):,} arrivals")
    pings = ping_features(pings)
    cases = make_cases(pings, arr, a.step, a.hmax)
    cases = cases[cases["dist"] > 0].reset_index(drop=True)
    if len(cases) < 3000:
        raise SystemExit(f"Only {len(cases):,} cases. Collect and process more data (or use a smaller --step).")
    return cases, arr