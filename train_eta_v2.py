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

# ---------------------------------------------------------------- final model: time split, ranges, ablations
def run_final(a, cases, arr, backend):
    c1, c2 = cases["ts"].quantile(0.6), cases["ts"].quantile(0.8)
    tr, va, te = cases[cases["ts"] < c1], cases[(cases["ts"] >= c1) & (cases["ts"] < c2)], cases[cases["ts"] >= c2]
    print(f"2. CASES     {len(cases):,}: train {len(tr):,}, validation {len(va):,}, test {len(te):,} (test = after {pd.to_datetime(c2 + 19800, unit='s'):%d %b %H:%M} IST)")
    L = learn(tr, arr[arr["arrive_ts"] < c1], arr)
    print(f"3. LEARN     typical speed {L.v_global * 3.6:.0f} km/h; {len(L.v_route):,} routes; stop-to-stop times of {len(L.tables):,} routes; {len(L.events):,} observed stop-to-stop times; "
          f"time-of-day factors for {len(L.tod)} hour groups")
    sub = tr.sample(min(len(tr), a.max_train), random_state=0) if len(tr) > a.max_train else tr
    sub, va, te = add_feats(sub, L), add_feats(va, L), add_feats(te, L)
    ev = te.copy()
    ev["r_hist_tod"] = ev["base"]; ev["r_hist"] = ev["hist_eta"].fillna(ev["eta_route"]); ev["r_route"] = ev["eta_route"]
    ev["r_calc"] = ev["dist"] / (ev["v1"] / 3.6).clip(2, 14).fillna(L.v_global); ev["r_feed"] = ev["dist"] / ev["rep"].clip(2, 14).fillna(L.v_global)
    rules, models = [], {}

    def train(name, col, feats, kind="l1", alpha=0.5, store=None):
        print(f"4. TRAIN     {name} ...")
        m, pr = fit_model(sub[feats].astype("float32"), to_target(sub, a.target), va[feats].astype("float32"), to_target(va, a.target), backend, kind, alpha)
        ev[col] = from_pred(ev, pr(ev[feats].astype("float32")), a.target)
        if store:
            models[store] = (m, feats)
        return m
    main = train("main model (all inputs)", "r_ai", FEATURES, store="point"); rules.append(("AI model v2 (all inputs)", "r_ai"))
    if not a.no_ablation:
        train("without the feed number", "r_ai_nofeed", [f for f in FEATURES if f != "rep"]); rules.append(("AI v2 without feed number", "r_ai_nofeed"))
        train("without the recent-buses inputs", "r_ai_norecent", [f for f in FEATURES if f not in RECENT]); rules.append(("AI v2 without recent-buses inputs", "r_ai_norecent"))
    rules += [("historical x time of day", "r_hist_tod"), ("historical stop-to-stop times", "r_hist"), ("route typical speed", "r_route"), ("calculated speed", "r_calc"), ("feed number as speed", "r_feed")]

    print(f"\n=== RESULT ON THE TEST PART ({len(ev):,} cases the model never saw; error = guess minus real time) ===")
    print(f"   {'rule':<36}{'mean error':>11}{'typical error':>14}{'bias':>9}{'within 1 min':>14}{'within 2 min':>14}")
    rows = sorted(((n, metrics(ev[c].values, ev["actual"].values)) for n, c in rules), key=lambda x: x[1]["mae"])
    for n, m in rows:
        print(f"   {n:<36}{m['mae']:>9.0f} s{m['med']:>12.0f} s{m['bias']:>+7.0f} s{m['w60']:>12.0f}%{m['w120']:>13.0f}%")
    base = min(m["mae"] for n, m in rows if not n.startswith("AI")); mm = metrics(ev["r_ai"].values, ev["actual"].values)["mae"]
    gain = 100 * (base - mm) / base
    print(f"   -> the AI model's mean error is {abs(gain):.0f}% {'lower (better)' if gain >= 0 else 'higher (worse)'} than the best simple rule ({base:.0f} s)")
    show = [r for r in rules if r[1] in ("r_ai", "r_hist_tod", "r_hist", "r_route")]
    print("\nBY HOW FAR AHEAD THE STOP IS (mean error in seconds)")
    groups = [("1 stop", ev["h"] == 1), ("2 stops", ev["h"] == 2), ("3-4 stops", ev["h"].between(3, 4)), (f"5-{a.hmax}", ev["h"] >= 5)]
    print(f"   {'rule':<36}" + "".join(f"{n:>11}" for n, _ in groups))
    for n, c in show:
        print(f"   {n:<36}" + "".join(f"{np.nanmean(np.abs(ev.loc[m, c] - ev.loc[m, 'actual'])):>9.0f} s" for _, m in groups))
    print("\nBY WHAT THE BUS DOES AT THE MOMENT")
    for name, m in (("standing (under 2 km/h)", ev["v1"] < 2), ("moving (5 km/h or more)", ev["v1"] >= 5)):
        if m.sum() >= 500:
            print(f"   {name}: {int(m.sum()):,} cases   " + "   ".join(f"{n}: {np.nanmean(np.abs(ev.loc[m, c] - ev.loc[m, 'actual'])):.0f} s" for n, c in show))

    print("\nPER-ROUTE CHECK (routes with enough test cases): where is the model worse than the historical ETA?")
    ev["e_ai"] = (ev["r_ai"] - ev["actual"]).abs(); ev["e_hist"] = (ev["r_hist"] - ev["actual"]).abs()
    pr = ev.groupby("route_id").agg(n=("actual", "size"), ai=("e_ai", "mean"), hist=("e_hist", "mean"))
    pr = pr[pr["n"] >= max(100, len(ev) // 3000)]
    if len(pr):
        print(f"   {len(pr):,} routes: the model is better on {100 * (pr['ai'] < pr['hist']).mean():.0f}% of them")
        print("   worst routes for the model (cases, model error, historical error):")
        for r, v in pr.assign(d=pr["ai"] - pr["hist"]).sort_values("d", ascending=False).head(6).iterrows():
            print(f"     route_id {r:<8}{int(v['n']):>8,} cases   {v['ai']:>5.0f} s   {v['hist']:>5.0f} s")

    if not a.no_ranges:
        lo_a, hi_a = a.low, a.high
        print(f"\nPREDICTION RANGES (a lower and an upper guess; the range should hold the real time about {100 * (hi_a - lo_a):.0f}% of the time)")
        residual_feats = FEATURES
        mlo, plo = fit_model(sub[residual_feats].astype("float32"), to_target(sub, "residual"), va[residual_feats].astype("float32"), to_target(va, "residual"), backend, "q", lo_a, quiet=True)
        mhi, phi = fit_model(sub[residual_feats].astype("float32"), to_target(sub, "residual"), va[residual_feats].astype("float32"), to_target(va, "residual"), backend, "q", hi_a, quiet=True)
        ev["lo"] = ev["base"].values + plo(ev[residual_feats].astype("float32")); ev["hi"] = ev["base"].values + phi(ev[residual_feats].astype("float32"))
        ev["hi"] = np.maximum(ev["hi"], ev["lo"] + 1)
        before = ((ev["actual"] >= ev["lo"]) & (ev["actual"] <= ev["hi"])).mean()
        # calibration on the validation part: widen both ends by the amount that makes the range hold the real time as often as wanted
        vlo = va["base"].values + plo(va[residual_feats].astype("float32")); vhi = np.maximum(va["base"].values + phi(va[residual_feats].astype("float32")), vlo + 1)
        score = np.maximum(vlo - va["actual"].values, va["actual"].values - vhi)
        want = hi_a - lo_a
        hg = lambda h: np.where(h <= 2, h, np.where(h <= 4, 3, 4))           # groups of stops ahead: 1, 2, 3-4, 5 or more
        qfun = lambda sc: float(np.quantile(sc, min(1.0, np.ceil((len(sc) + 1) * want) / len(sc))))
        qall, qg = qfun(score), {}
        for g in (1, 2, 3, 4):
            m_ = hg(va["h"].values) == g
            qg[g] = qfun(score[m_]) if m_.sum() >= 200 else qall
        qhat = qall
        widen = np.array([qg[g] for g in hg(ev["h"].values)])
        ev["lo"], ev["hi"] = ev["lo"] - widen, ev["hi"] + widen
        inside = (ev["actual"] >= ev["lo"]) & (ev["actual"] <= ev["hi"])
        print(f"   before calibration the range held the real time in {100 * before:.0f}% of the test cases; the ends were widened (calibrated on the validation part, separately for 1, 2, 3-4 and 5+ stops ahead: by " + ", ".join(f"{qg[g]:.0f} s" for g in (1, 2, 3, 4)) + ")")
        print(f"   range holds the real time in {100 * inside.mean():.0f}% of the test cases (wanted about {100 * want:.0f}%); typical width {np.median(ev['hi'] - ev['lo']):.0f} s")
        print(f"   {'':<14}" + "".join(f"{n:>11}" for n, _ in groups))
        print(f"   {'holds':<14}" + "".join(f"{100 * inside[m].mean():>10.0f}%" for _, m in groups))
        print(f"   {'width (s)':<14}" + "".join(f"{np.median((ev['hi'] - ev['lo'])[m]):>11.0f}" for _, m in groups))
        models["low"], models["high"] = (mlo, residual_feats), (mhi, residual_feats)

    print("\nWHAT THE MODEL USES MOST")
    if backend == "lightgbm":
        imp = dict(zip(FEATURES, main.feature_importance("gain"))); tot = sum(imp.values()) or 1
        for k, v in sorted(imp.items(), key=lambda x: -x[1])[:10]:
            print(f"   {k:<14}{100 * v / tot:>6.1f}%")
    else:
        from sklearn.inspection import permutation_importance
        s = ev.sample(min(len(ev), 20000), random_state=0)
        r = permutation_importance(main, s[FEATURES].astype("float32"), to_target(s, a.target), n_repeats=1, random_state=0, scoring="neg_mean_absolute_error")
        for k, v in sorted(zip(FEATURES, r.importances_mean), key=lambda x: -x[1])[:10]:
            print(f"   {k:<14} error grows by {v:>6.1f} s if this input is shuffled")
    out = Path(a.out); out.mkdir(exist_ok=True)
    (out / "eta_v2_models.pkl").write_bytes(pickle.dumps({"backend": backend, "target": a.target, "models": {k: v[0] for k, v in models.items()}, "features": {k: v[1] for k, v in models.items()}}))
    (out / "eta_v2_tables.pkl").write_bytes(pickle.dumps({"v_global": L.v_global, "v_route": L.v_route, "med": L.med, "tables": L.tables, "tod": L.tod}))
    (out / "eta_v2_info.json").write_text(json.dumps({"backend": backend, "target": a.target, "features": FEATURES, "test_mean_error_s": round(float(mm), 1), "cases": len(cases)}, indent=1))
    if not a.no_ranges:
        (out / "eta_v2_calibration.json").write_text(json.dumps({"low_q": a.low, "high_q": a.high, "widen_s_by_stops_ahead_1_2_3to4_5plus": [round(qg[g], 1) for g in (1, 2, 3, 4)]}))
    print(f"\nSaved the models and the learned tables in {out}/")


# ---------------------------------------------------------------- leave-one-day-out
def run_cv(a, cases, arr, backend):
    cases = cases.assign(day=day_of(cases["ts"]))
    cnt = cases["day"].value_counts()
    days = sorted(d for d, n in cnt.items() if n >= max(2000, 0.02 * len(cases)))
    if len(days) < 3:
        raise SystemExit("Leave-one-day-out needs at least 3 days with enough cases.")
    print(f"2. CASES     {len(cases):,} cases on {len(days)} days. Each day is tested with a model trained on all the other days.\n")
    print(f"   {'day':<12}{'cases':>9}{'AI v2':>9}{'historical x tod':>18}{'historical':>12}{'route typical':>15}{'AI gain vs best simple':>25}")
    rows = []
    for d in days:
        te, trn = cases[cases["day"] == d], cases[cases["day"] != d]
        L = learn(trn, arr[day_of(arr["arrive_ts"]) != d], arr)
        trn = trn.sample(min(len(trn), a.max_train), random_state=0)
        va = trn.sample(frac=0.1, random_state=1); tr2 = trn.drop(va.index)
        tr2, va, te = add_feats(tr2, L), add_feats(va, L), add_feats(te, L)
        m, pr = fit_model(tr2[FEATURES].astype("float32"), to_target(tr2, a.target), va[FEATURES].astype("float32"), to_target(va, a.target), backend, quiet=True)
        ai = from_pred(te, pr(te[FEATURES].astype("float32")), a.target)
        res = dict(ai=metrics(ai, te["actual"].values)["mae"], ht=metrics(te["base"].values, te["actual"].values)["mae"],
                   h=metrics(te["hist_eta"].fillna(te["eta_route"]).values, te["actual"].values)["mae"], rt=metrics(te["eta_route"].values, te["actual"].values)["mae"])
        best = min(res["ht"], res["h"], res["rt"])
        rows.append((d, len(te), res, best))
        print(f"   {pd.to_datetime(d * 86400, unit='s'):%a %d %b}  {len(te):>9,}{res['ai']:>7.0f} s{res['ht']:>16.0f} s{res['h']:>10.0f} s{res['rt']:>13.0f} s{100 * (best - res['ai']) / best:>22.0f}%")
    gains = [100 * (b - r["ai"]) / b for _, _, r, b in rows]
    print(f"\n   average gain of the AI model over the best simple rule: {np.mean(gains):.0f}% (best day {max(gains):.0f}%, worst day {min(gains):.0f}%)")
    print(f"   average mean error: AI v2 {np.mean([r['ai'] for _, _, r, _ in rows]):.0f} s, historical x time of day {np.mean([r['ht'] for _, _, r, _ in rows]):.0f} s, "
          f"historical {np.mean([r['h'] for _, _, r, _ in rows]):.0f} s, route typical {np.mean([r['rt'] for _, _, r, _ in rows]):.0f} s")
    print("   (each day is tested with history learned from the OTHER days, which may include later days: it shows how well the model works on a new day)")


def main():
    ap = argparse.ArgumentParser(description="ETA model, version 2")
    ap.add_argument("--arrivals", default="data/arrivals"); ap.add_argument("--date")
    ap.add_argument("--step", type=int, default=6); ap.add_argument("--hmax", type=int, default=6)
    ap.add_argument("--max-train", type=int, default=1_200_000); ap.add_argument("--backend", choices=["auto", "lightgbm", "sklearn"], default="auto")
    ap.add_argument("--mode", choices=["final", "cv"], default="final"); ap.add_argument("--target", choices=["residual", "ratio", "raw"], default="residual")
    ap.add_argument("--no-ablation", action="store_true"); ap.add_argument("--no-ranges", action="store_true")
    ap.add_argument("--low", type=float, default=0.1); ap.add_argument("--high", type=float, default=0.9); ap.add_argument("--out", default="models")
    a = ap.parse_args()
    backend = pick_backend(a.backend)
    print(f"model: {'LightGBM' if backend == 'lightgbm' else 'scikit-learn HistGradientBoosting'}   target: {a.target}   mode: {a.mode}")
    cases, arr = prepare(a)
    (run_final if a.mode == "final" else run_cv)(a, cases, arr, backend)
    print("\nSend this whole output back.")


if __name__ == "__main__":
    main()
