#!/usr/bin/env python3
"""
train_eta.py - the first AI model for the ETA: gradient boosting that learns from the observed arrival times.

Case      = one ping of a bus + one of the next stops ahead (like eta_eval.py).
Answer    = the real time the bus needed to reach that stop (from the arrival table).
Inputs    = distance, stops ahead, speed of the last 3 intervals, how long the bus has been standing, hour, weekday,
            the typical speed of the route, the historical travel time through the stops of the route, position on the
            route, and the raw number of the feed's speed field.
Split     = by TIME: first 60% train, next 20% validation (stops the training), last 20% test.
            The typical speeds and the historical stop-to-stop times are learned on the training part only.
Output    = test errors of the model next to the simple rules, a test without the feed number, what the model uses most,
            and the saved model in models/.

   python train_eta.py                      all days in data/arrivals
   python train_eta.py --step 10            use every 10th ping (faster)
Needs lightgbm (uv add lightgbm). If it is missing, scikit-learn's HistGradientBoostingRegressor is used instead.
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd

MIN_REMAIN, MAX_REMAIN = 5, 1500
FEATURES = ["dist", "h", "v1", "v2", "v3", "v_mean3", "standing_s", "hour", "dow", "is_weekend", "v_route", "eta_route", "hist_eta", "prog", "rep"]


def load(folder, date):
    folder = Path(folder)
    files = sorted(folder.glob("*_pings.csv.gz"))
    if date:
        files = [f for f in files if f.name.startswith(date)]
    if not files:
        raise SystemExit(f"No *_pings.csv.gz files in {folder}. Run build_arrivals.py first.")
    P, A = [], []
    for k, f in enumerate(files):
        pg = pd.read_csv(f, dtype={"vehicle_id": str, "route_id": str})
        ar = pd.read_csv(str(f).replace("_pings", "_arrivals"), dtype={"vehicle_id": str, "route_id": str})
        if "stop_prog_m" not in ar:
            raise SystemExit("The arrivals files have no stop_prog_m column. Run:  python build_arrivals.py run --force")
        pg["run_key"] = k * 10_000_000 + pg["run_id"]; ar["run_key"] = k * 10_000_000 + ar["run_id"]
        P.append(pg); A.append(ar)
    return pd.concat(P, ignore_index=True), pd.concat(A, ignore_index=True), files


def ping_features(p):
    """Features that only need the pings of one run: speeds of the last intervals, standing time, hour, weekday."""
    p = p.sort_values(["run_key", "ts"]).reset_index(drop=True)
    g = p.groupby("run_key")
    p["v1"] = p["calc_route_kmh"]; p["v2"] = g["calc_route_kmh"].shift(1); p["v3"] = g["calc_route_kmh"].shift(2)
    p["v_mean3"] = p[["v1", "v2", "v3"]].mean(axis=1)
    still = g["prog"].diff().abs() < 15
    last_move = p["ts"].where(~still).groupby(p["run_key"]).ffill()
    p["standing_s"] = (p["ts"] - last_move).fillna(0)
    t = p["ts"] + 19800
    p["hour"] = (t % 86400) / 3600.0
    p["dow"] = ((t // 86400) + 3) % 7                      # 0 = Monday
    p["is_weekend"] = (p["dow"] >= 5).astype(int)
    return p


def make_cases(pings, arr, step, hmax):
    arr = arr.sort_values(["run_key", "stop_prog_m"]).reset_index(drop=True)
    pidx, aidx = pings.groupby("run_key").indices, arr.groupby("run_key").indices
    P_ts, P_prog = pings["ts"].values, pings["prog"].values
    A_ts, A_sp, A_si = arr["arrive_ts"].values, arr["stop_prog_m"].values, arr["stop_idx"].values
    out = {"pi": [], "h": [], "dist": [], "actual": [], "sidx": []}
    for key, ia in aidx.items():
        ip = pidx.get(key)
        if ip is None or len(ip) < 3:
            continue
        ip = ip[::step]
        spr, atr, sir, prog, ts = A_sp[ia], A_ts[ia], A_si[ia], P_prog[ip], P_ts[ip]
        j0 = np.searchsorted(spr, prog + 20, side="right")
        for h in range(hmax):
            j = j0 + h
            ok = j < len(spr)
            if not ok.any():
                break
            ii, jj = np.where(ok)[0], j[ok]
            remain = atr[jj] - ts[ii]
            g = (remain >= MIN_REMAIN) & (remain <= MAX_REMAIN)
            out["pi"].append(ip[ii][g]); out["h"].append(np.full(g.sum(), h + 1)); out["dist"].append(spr[jj][g] - prog[ii][g])
            out["actual"].append(remain[g]); out["sidx"].append(sir[jj][g])
    c = pd.DataFrame({k: np.concatenate(v) if v else np.array([]) for k, v in out.items()})
    pf = pings.iloc[c["pi"].values][["ts", "route_id", "vehicle_id", "prog", "v1", "v2", "v3", "v_mean3", "standing_s", "hour", "dow", "is_weekend", "speed_rep"]].reset_index(drop=True)
    pf = pf.rename(columns={"speed_rep": "rep"})
    return pd.concat([c.drop(columns="pi").reset_index(drop=True), pf], axis=1)


def history_tables(arr, cut, v_global):
    """Median travel time between two neighbouring stops of a route, from arrivals BEFORE the cut. Returns per route: stop idx, progress, cumulative time."""
    a = arr[arr["arrive_ts"] < cut].sort_values(["run_key", "stop_idx"])
    g = a.groupby("run_key")
    a = a.assign(next_t=g["arrive_ts"].shift(-1), next_i=g["stop_idx"].shift(-1))
    seg = a[a["next_i"] == a["stop_idx"] + 1].copy()
    seg["t"] = seg["next_t"] - seg["arrive_ts"]
    med = seg[seg["t"].between(3, 600)].groupby(["route_id", "stop_idx"])["t"].median()
    sp = arr.groupby(["route_id", "stop_idx"])["stop_prog_m"].first().reset_index().sort_values(["route_id", "stop_idx"])
    tables = {}
    medd = med.to_dict()
    for r, g2 in sp.groupby("route_id"):
        idx, prog = g2["stop_idx"].values, g2["stop_prog_m"].values
        if len(idx) < 3 or np.any(np.diff(prog) < 0):
            continue
        T = np.empty(len(idx) - 1)
        for m in range(len(idx) - 1):
            T[m] = medd.get((r, idx[m]), np.nan) if idx[m + 1] == idx[m] + 1 else np.nan
            if np.isnan(T[m]):
                T[m] = (prog[m + 1] - prog[m]) / v_global                      # no data: distance / typical speed
        tables[r] = (idx, prog, np.concatenate([[0.0], np.cumsum(T)]))
    return tables


def hist_eta(cases, tables):
    out = np.full(len(cases), np.nan)
    pos = {}
    for k, r in enumerate(cases["route_id"].values):
        pos.setdefault(r, []).append(k)
    p = cases["prog"].values; si = cases["sidx"].values
    for r, rows in pos.items():
        t = tables.get(r)
        if t is None:
            continue
        idx, prog, C = t
        rows = np.array(rows)
        j = np.searchsorted(idx, si[rows])
        ok = (j < len(idx)) & (idx[np.minimum(j, len(idx) - 1)] == si[rows])
        tpos = np.interp(p[rows], prog, C)
        res = np.where(ok, C[np.minimum(j, len(idx) - 1)] - tpos, np.nan)
        out[rows] = res
    return out


def metrics(pred, actual):
    e = pred - actual
    return dict(mae=np.nanmean(np.abs(e)), med=np.nanmedian(np.abs(e)), bias=np.nanmean(e), w60=100 * np.nanmean(np.abs(e) <= 60), w120=100 * np.nanmean(np.abs(e) <= 120))


def fit_model(Xtr, ytr, Xva, yva, backend):
    if backend == "lightgbm":
        import lightgbm as lgb
        params = dict(objective="regression_l1", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.9,
                      bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=0)
        dtr = lgb.Dataset(Xtr, ytr); dva = lgb.Dataset(Xva, yva, reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        print(f"      trained {m.best_iteration} trees")
        return m, m.predict
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.1, max_iter=300, max_leaf_nodes=63, min_samples_leaf=100,
                                      early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=0)
    m.fit(Xtr, ytr)
    print(f"      trained {m.n_iter_} trees (scikit-learn)")
    return m, m.predict


def main():
    ap = argparse.ArgumentParser(description="Train an ETA model on observed arrival times")
    ap.add_argument("--arrivals", default="data/arrivals"); ap.add_argument("--date")
    ap.add_argument("--step", type=int, default=6, help="use every N-th ping"); ap.add_argument("--hmax", type=int, default=6, help="stops ahead (1..N)")
    ap.add_argument("--max-train", type=int, default=1_500_000); ap.add_argument("--backend", choices=["auto", "lightgbm", "sklearn"], default="auto")
    ap.add_argument("--no-ablation", action="store_true"); ap.add_argument("--out", default="models")
    a = ap.parse_args()
    backend = a.backend
    if backend == "auto":
        try:
            import lightgbm  # noqa
            backend = "lightgbm"
        except ImportError:
            backend = "sklearn"
    print(f"model: {'LightGBM' if backend == 'lightgbm' else 'scikit-learn HistGradientBoosting'}")
    pings, arr, files = load(a.arrivals, a.date)
    print(f"1. LOAD      {len(files)} day(s): {len(pings):,} pings, {len(arr):,} arrivals")
    pings = ping_features(pings)
    cases = make_cases(pings, arr, a.step, a.hmax)
    cases = cases[cases["dist"] > 0].reset_index(drop=True)
    if len(cases) < 3000:
        raise SystemExit(f"Only {len(cases):,} cases. Collect and process more data (or use a smaller --step).")
    c1, c2 = cases["ts"].quantile(0.6), cases["ts"].quantile(0.8)
    tr, va, te = cases[cases["ts"] < c1], cases[(cases["ts"] >= c1) & (cases["ts"] < c2)], cases[cases["ts"] >= c2]
    print(f"2. CASES     {len(cases):,} cases (a ping + one of the next 1-{a.hmax} stops): train {len(tr):,}, validation {len(va):,}, test {len(te):,}"
          f"  (test = after {pd.to_datetime(c2 + 19800, unit='s'):%d %b %H:%M} IST)")
    vt = (tr["dist"] / tr["actual"]).clip(1, 20)
    v_global = float(vt.median()); v_route = vt.groupby(tr["route_id"]).median()
    tables = history_tables(arr, c1, v_global)
    print(f"3. LEARN     typical speed {v_global:.1f} m/s ({v_global * 3.6:.0f} km/h); typical speed of {len(v_route):,} routes; stop-to-stop times of {len(tables):,} routes (training part only)")
    for d in (tr, va, te):
        d["v_route"] = d["route_id"].map(v_route).fillna(v_global).clip(1, 20)
        d["eta_route"] = d["dist"] / d["v_route"]
        d["hist_eta"] = hist_eta(d, tables)
    sub = tr.sample(min(len(tr), a.max_train), random_state=0) if len(tr) > a.max_train else tr
    ev = te.copy()
    ev["r_route"] = ev["eta_route"]; ev["r_hist"] = ev["hist_eta"].fillna(ev["eta_route"])
    ev["r_calc"] = ev["dist"] / (ev["v1"] / 3.6).clip(2, 14).fillna(v_global); ev["r_feed"] = ev["dist"] / ev["rep"].clip(2, 14).fillna(v_global)
    print("4. TRAIN     main model with all inputs ...")
    model, pred = fit_model(sub[FEATURES].astype("float32"), sub["actual"], va[FEATURES].astype("float32"), va["actual"], backend)
    ev["r_model"] = pred(ev[FEATURES].astype("float32"))
    rules = [("AI model (all inputs)", "r_model")]
    if not a.no_ablation:
        f2 = [f for f in FEATURES if f != "rep"]
        print("   TRAIN     second model WITHOUT the feed number (to see if it helps) ...")
        m2, p2 = fit_model(sub[f2].astype("float32"), sub["actual"], va[f2].astype("float32"), va["actual"], backend)
        ev["r_model2"] = p2(ev[f2].astype("float32")); rules.append(("AI model without feed number", "r_model2"))
    rules += [("historical stop-to-stop times", "r_hist"), ("route typical speed", "r_route"), ("calculated speed", "r_calc"), ("feed number as speed", "r_feed")]

    print(f"\n=== RESULT ON THE TEST PART ({len(ev):,} cases the model never saw; error = guess minus real time) ===")
    print(f"   {'rule':<32}{'mean error':>11}{'typical error':>14}{'bias':>9}{'within 1 min':>14}{'within 2 min':>14}")
    rows = sorted(((n, metrics(ev[c].values, ev["actual"].values)) for n, c in rules), key=lambda x: x[1]["mae"])
    for n, m in rows:
        print(f"   {n:<32}{m['mae']:>9.0f} s{m['med']:>12.0f} s{m['bias']:>+7.0f} s{m['w60']:>12.0f}%{m['w120']:>13.0f}%")
    base = min(m["mae"] for n, m in rows if not n.startswith("AI"))
    mm = metrics(ev["r_model"].values, ev["actual"].values)["mae"]
    gain = 100 * (base - mm) / base
    print(f"   -> the AI model's mean error is {abs(gain):.0f}% {'lower (better)' if gain >= 0 else 'higher (worse)'} than the best simple rule ({base:.0f} s)")
    print("\nBY HOW FAR AHEAD THE STOP IS (mean error in seconds)")
    groups = [("1 stop", ev["h"] == 1), ("2 stops", ev["h"] == 2), ("3-4 stops", ev["h"].between(3, 4)), (f"5-{a.hmax} stops", ev["h"] >= 5)]
    print(f"   {'rule':<32}" + "".join(f"{n:>11}" for n, _ in groups))
    for n, c in rules:
        print(f"   {n:<32}" + "".join(f"{np.nanmean(np.abs(ev.loc[m, c] - ev.loc[m, 'actual'])):>9.0f} s" for _, m in groups))
    print("\nBY WHAT THE BUS DOES AT THE MOMENT")
    for name, m in (("standing (under 2 km/h)", ev["v1"] < 2), ("moving (5 km/h or more)", ev["v1"] >= 5)):
        if m.sum() >= 500:
            print(f"   {name}: {int(m.sum()):,} cases   " + "   ".join(f"{n}: {np.nanmean(np.abs(ev.loc[m, c] - ev.loc[m, 'actual'])):.0f} s" for n, c in rules[:3]))
    print("\nWHAT THE MODEL USES MOST")
    if backend == "lightgbm":
        imp = dict(zip(FEATURES, model.feature_importance("gain"))); tot = sum(imp.values()) or 1
        top = sorted(imp.items(), key=lambda x: -x[1])[:10]
        for k, v in top: print(f"   {k:<14}{100 * v / tot:>6.1f}%")
    else:
        from sklearn.inspection import permutation_importance
        s = ev.sample(min(len(ev), 30000), random_state=0)
        r = permutation_importance(model, s[FEATURES].astype("float32"), s["actual"], n_repeats=1, random_state=0, scoring="neg_mean_absolute_error")
        for k, v in sorted(zip(FEATURES, r.importances_mean), key=lambda x: -x[1])[:10]: print(f"   {k:<14} error grows by {v:>6.1f} s if this input is shuffled")
    out = Path(a.out); out.mkdir(exist_ok=True)
    (out / "eta_model.pkl").write_bytes(pickle.dumps({"backend": backend, "model": model, "features": FEATURES}))
    (out / "eta_tables.pkl").write_bytes(pickle.dumps({"v_global": v_global, "v_route": v_route.to_dict(), "hist_tables": tables}))
    (out / "eta_info.json").write_text(json.dumps({"backend": backend, "features": FEATURES, "test_mean_error_s": round(float(mm), 1), "cases": len(cases)}, indent=1))
    print(f"\nSaved the model and the learned tables in {out}/")
    print("\nSend this whole output back.")


if __name__ == "__main__":
    main()
