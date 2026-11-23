#!/usr/bin/env python3
"""
replay_eta.py - replays ONE saved day through live_eta.py, poll by poll, and compares it with the training pipeline.

Why: the live engine must give the same numbers as train_eta_v2.py, otherwise the model sees different inputs than it was trained on.
The saved clean file (data/clean/<day>.csv.gz) is fed to the engine in the order the feed delivered it (poll by poll, like live).

It checks, in this order:
  1. FLAGS      the gps_jump flags of the live rules against the flags the collector saved
  2. ARRIVALS   the arrivals found live against data/arrivals/<day>_arrivals.csv.gz (made by build_arrivals.py)
  3. INPUTS     every model input of every case against the inputs of train_eta_v2.py (same bus, same ping, same stop)
  4. RECENT     the "recent buses" numbers (rc10, rr20): live-strict (only what was known at that poll), live with hindsight, and the training ones
  5. MODEL      (needs lightgbm and models/eta_v2_models.pkl) ETA error of the model with training inputs against the model with live inputs
  6. SPEED      how many pings per second the engine handles

   python replay_eta.py --clean data/clean/2026-11-10.csv.gz --capture-every 6
   python replay_eta.py --clean data/clean/2026-10-20.csv.gz --capture-every 1 --no-model       (features and arrivals only)
Keep live_eta.py, build_arrivals.py, train_eta.py and train_eta_v2.py in the same folder.
"""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

import live_eta as le
import train_eta as t1
import train_eta_v2 as t2

FEATS = t2.FEATURES


def say(title):
    print(f"\n==== {title} ====", flush=True)


class FakeBooster:
    """Only for testing the code path without lightgbm: returns a fixed correction."""
    def __init__(self, v):
        self.v = v

    def predict(self, X):
        return np.full(len(X), self.v, dtype=float)


def read_clean(path):
    want = {"fetched_at", "ts", "vehicle_id", "route_id", "trip_id", "lat", "lon", "speed_rep", "speed_ms", "flag", "source"}
    df = pd.read_csv(path, usecols=lambda c: c in want, dtype={"vehicle_id": str, "route_id": str, "trip_id": str, "flag": str, "source": str}, low_memory=False)
    if "speed_ms" in df and "speed_rep" not in df:
        df = df.rename(columns={"speed_ms": "speed_rep"})
    df["trip_id"] = df["trip_id"].fillna("")
    return df.sort_values(["fetched_at", "ts"], kind="stable").reset_index(drop=True)


def same(a, b, atol=1e-6):
    return np.isclose(a, b, atol=atol, rtol=0, equal_nan=True)


def diff_table(a, b, cols, title, mask=None):
    print(f"   {title}")
    print(f"   {'input':<12}{'same':>9}{'median |diff|':>16}{'90% |diff|':>13}{'NaN differs':>13}")
    for c in cols:
        x, y = a[c].values.astype(float), b[c].values.astype(float)
        if mask is not None:
            x, y = x[mask], y[mask]
        ok = same(x, y)
        d = np.abs(x - y)[~ok & ~np.isnan(x) & ~np.isnan(y)]
        print(f"   {c:<12}{100 * ok.mean():>8.1f}%{(np.median(d) if len(d) else 0):>16.4g}{(np.quantile(d, .9) if len(d) else 0):>13.4g}{100 * (np.isnan(x) != np.isnan(y)).mean():>12.1f}%")


def main():
    ap = argparse.ArgumentParser(description="Replay a saved day through live_eta.py")
    ap.add_argument("--clean", required=True, help="data/clean/<day>.csv.gz")
    ap.add_argument("--arrivals", default="data/arrivals"); ap.add_argument("--models", default="models"); ap.add_argument("--gtfs", default="gtfs")
    ap.add_argument("--patterns-cache", default=None, help="read the route lines from this .arrival_patterns_shapes.pkl instead of the GTFS folder")
    ap.add_argument("--capture-every", type=int, default=6, help="check the inputs of every N-th ping of a run (1 = all pings; the engine always sees all pings)")
    ap.add_argument("--no-model", action="store_true", help="skip the model test")
    ap.add_argument("--fake-model", action="store_true", help="test the prediction code with a dummy model (numbers are meaningless)")
    ap.add_argument("--warmup-min", type=float, default=20, help="minutes after the first poll that are left out of the 'recent buses' comparison")
    a = ap.parse_args()
    day = Path(a.clean).name.split(".")[0]

    # ------------------------------------------------------------ engine
    say(f"LOAD   {day}")
    have_models = False
    eng = None
    if not a.no_model and not a.fake_model:
        try:
            import lightgbm  # noqa
            eng = le.LiveETA.load(a.models, a.gtfs, a.patterns_cache, with_models=True, record=True); have_models = True
        except ImportError:
            print("   lightgbm is not installed here: the model test (part 5) is skipped.")
    if eng is None:
        eng = le.LiveETA.load(a.models, a.gtfs, a.patterns_cache, with_models=False, record=True)
        if a.fake_model:
            eng.models = {"target": "residual", "models": {"point": FakeBooster(0.0), "low": FakeBooster(-60.0), "high": FakeBooster(90.0)},
                          "features": {k: FEATS for k in ("point", "low", "high")}}
            have_models = True
    print(f"   {len(eng.pats):,} route lines, {len(eng.L.tables):,} routes with history tables, {len(eng.med):,} stop-pair medians, calibration widen {eng.widen}")
    df = read_clean(a.clean)
    t0, t1_ = df["ts"].min(), df["ts"].max()
    print(f"   {len(df):,} pings of {df['vehicle_id'].nunique():,} buses, {df['fetched_at'].nunique()} polls, {pd.to_datetime(t0 + 19800, unit='s'):%H:%M} to {pd.to_datetime(t1_ + 19800, unit='s'):%H:%M} IST")

    # ------------------------------------------------------------ replay, poll by poll
    say("REPLAY   poll by poll (like live)")
    cap, spent = [], 0.0
    for fa, g in df.groupby("fetched_at", sort=True):
        src = g["source"].iloc[0]
        rows = [{"position.latitude": r.lat, "position.longitude": r.lon, "timestamp": r.ts, "vehicle.id": r.vehicle_id, "trip.route_id": r.route_id,
                 "trip.trip_id": r.trip_id, "position.speed": r.speed_rep}
                for r in g.itertuples()]
        t_ = time.time()
        touched = eng.update(rows, now=float(fa), source=src)
        keys = [k for k in touched if eng.groups[k].run.cur["idx"] % a.capture_every == 0]
        c = eng.cases(keys=keys, now=float(fa))                     # what the explorer would compute right after this poll
        spent += time.time() - t_
        if len(c):
            c["poll"] = fa
            cap.append(c)
    live = pd.concat(cap, ignore_index=True) if cap else pd.DataFrame()
    print(f"   done: {len(df):,} pings in {spent:.1f} s = {len(df) / max(spent, 1e-9):,.0f} pings/s (a live snapshot has about 2,000 new pings per poll); {len(live):,} cases captured")
    print("   engine counters: " + ", ".join(f"{k} {v:,}" for k, v in sorted(eng.stats.items())))

    # ------------------------------------------------------------ 1. flags
    say("1. FLAGS   live gps_jump rule against the flags the collector saved")
    fl_file = {(r.vehicle_id, int(r.ts)): (r.flag == "gps_jump") for r in df.itertuples()}
    fl_live = {(v, t): f for v, t, f in eng.flag_log}
    both = [k for k in fl_file if k in fl_live]
    bad = [k for k in both if fl_file[k] != fl_live[k]]
    print(f"   {len(both):,} pings compared, {len(bad)} with a different flag (file flags {sum(fl_file.values())}, live flags {sum(fl_live[k] for k in both)})"
          + (f"; first: {bad[:3]}" if bad else ""))
    print(f"   pings the engine did not accept at all: {len(fl_file) - len(both)} (should be 0)")

    # ------------------------------------------------------------ 2. arrivals
    say("2. ARRIVALS   found live against build_arrivals.py")
    arr = pd.read_csv(Path(a.arrivals) / f"{day}_arrivals.csv.gz", dtype={"vehicle_id": str, "route_id": str, "trip_id": str})
    arr["trip_id"] = arr["trip_id"].fillna("")
    lv = pd.DataFrame(eng.arrival_log, columns=["vehicle_id", "route_id", "trip_id", "stop_idx", "t_live"])
    keycols = ["vehicle_id", "route_id", "trip_id", "stop_idx"]
    arr = arr.sort_values("arrive_ts"); lv = lv.sort_values("t_live")
    # match each offline arrival with the live arrival of the same bus, trip and stop that is closest in time (at most 5 s apart)
    arr, lv = arr.reset_index(drop=True), lv.reset_index(drop=True)
    arr["_a"], lv["_l"] = arr.index, lv.index
    a_s = arr.assign(_k=arr[keycols].astype(str).agg("|".join, axis=1)).sort_values("arrive_ts")
    l_s = lv.assign(_k=lv[keycols].astype(str).agg("|".join, axis=1)).sort_values("t_live")
    mm = pd.merge_asof(a_s, l_s[["_k", "t_live", "_l"]], left_on="arrive_ts", right_on="t_live", by="_k", direction="nearest", tolerance=5.0)
    mm = mm.sort_values("_a").drop_duplicates("_l", keep="first") if mm["_l"].notna().any() else mm
    left = arr.merge(mm[["_a", "t_live", "_l"]], on="_a", how="left")
    left["_merge"] = np.where(left["_l"].notna(), "both", "left_only")
    right = lv[~lv["_l"].isin(left["_l"].dropna())].copy(); right["_merge"] = "right_only"
    m = pd.concat([left, right], ignore_index=True)
    both_ = m[m["_merge"] == "both"]
    dt = (both_["t_live"] - both_["arrive_ts"]).abs()
    print(f"   offline {len(arr):,} | live {len(lv):,} | found by both {len(both_):,} ({100 * len(both_) / max(len(arr), 1):.2f}% of the offline ones)")
    print(f"   times of the common ones: same within 0.15 s {100 * (dt <= 0.15).mean():.2f}%, largest difference {dt.max():.1f} s")
    off_only, live_only = m[m["_merge"] == "left_only"], m[m["_merge"] == "right_only"]
    print(f"   only offline: {len(off_only):,}   only live: {len(live_only):,}")
    if len(off_only):
        print("   only offline, by quality: " + ", ".join(f"{k} {v}" for k, v in off_only["quality"].value_counts().items()))
        first = arr.groupby("run_id")["arrive_ts"].transform("min")                    # arrivals at the start of a run
        near = off_only.merge(arr[["run_id", "stop_idx", "arrive_ts"]].assign(first=first), on=["run_id", "stop_idx", "arrive_ts"], how="left")
        print(f"   only offline and within 60 s of the first arrival of its run: {100 * ((near['arrive_ts'] - near['first']) <= 60).mean():.0f}%")
        gb = [(eng.groups.get((r.vehicle_id, r.route_id, r.trip_id))) for r in off_only.itertuples()]
        print(f"   only offline: group was marked 'against its direction' at some time but not at the end: {sum(1 for g_ in gb if g_ is not None and g_.ever_bad and not g_.bad)} of {len(gb)}"
              f"; group not known to the engine at all: {sum(1 for g_ in gb if g_ is None)}")
        print("   examples (vehicle, route, stop_idx, time):", [(r.vehicle_id, r.route_id, int(r.stop_idx), r.arrive_ist) for r in off_only.head(4).itertuples()])
    if len(live_only):
        gb = [(eng.groups.get((r.vehicle_id, r.route_id, r.trip_id))) for r in live_only.itertuples()]
        print(f"   only live: the bus group is marked 'against its route direction' at the end of the day for {sum(1 for g_ in gb if g_ is not None and g_.bad)} of {len(gb)} "
              f"(offline throws those groups away with hindsight; live cannot know it at the time)")
        print("   examples only live (vehicle, route, stop_idx, t):", [(r.vehicle_id, r.route_id, int(r.stop_idx), round(r.t_live, 1)) for r in live_only.head(4).itertuples()])

    if live.empty:
        print("\nNo cases captured, stopping here."); return

    # ------------------------------------------------------------ 3. inputs against the training pipeline
    say("3. INPUTS   every model input against train_eta_v2.py")
    pings, arr2, files = t1.load(a.arrivals, day)
    pings = t1.ping_features(pings)
    cases = t1.make_cases(pings, arr2, a.capture_every, 6)
    cases = cases[cases["dist"] > 0].reset_index(drop=True)
    L = eng.L
    L_off = t2.Learned(); L_off.v_global, L_off.v_route, L_off.tables, L_off.tod = L.v_global, L.v_route, L.tables, L.tod
    L_off.events = t2.make_events(arr2, eng.med)                                   # events of THIS day only, found with hindsight
    off = t2.add_feats(cases, L_off)
    for d_ in (off, live):
        d_["ts"] = d_["ts"].astype("int64")
    key = ["vehicle_id", "route_id", "ts", "sidx"]
    j = live.merge(off, on=key, suffixes=("_live", "_off"))
    print(f"   training cases {len(off):,} | live cases {len(live):,} | same bus, ping and stop in both {len(j):,} ({100 * len(j) / max(len(off), 1):.1f}% of the training cases)")
    print(f"   stops ahead (h) different: {100 * (j['h_live'] != j['h_off']).mean():.2f}% of the common cases  (training counts only stops that have a detected arrival, live counts the real next stops)")
    A = j[[c + "_live" for c in FEATS]].rename(columns=lambda c: c[:-5]); B = j[[c + "_off" for c in FEATS]].rename(columns=lambda c: c[:-4])
    diff_table(A, B, [c for c in FEATS if c not in t2.RECENT], "inputs that do not depend on other buses (should be about 100% the same):")

    # ------------------------------------------------------------ 4. recent buses
    say("4. RECENT   the 'recent buses' numbers")
    ev_live = pd.DataFrame(eng.events.log, columns=["route_id", "t_end", "ratio"]).sort_values("t_end").reset_index(drop=True)
    ev_off = L_off.events
    print(f"   stop-to-stop events: live {len(ev_live):,}, training (with hindsight) {len(ev_off):,}; mean ratio live {ev_live['ratio'].mean():.3f}, training {ev_off['ratio'].mean():.3f}")
    # same cases, hindsight window but with the events found live: separates "events found differently" from "events known too late"
    cs = pd.DataFrame({"ts": j["ts"].astype(float), "route_id": j["route_id"]})
    h_rc, h_rcn, h_rr, h_rrn = t2.recent_feats(cs, ev_live)
    j["rc10_hind"], j["rc10_n_hind"], j["rr20_hind"], j["rr20_n_hind"] = h_rc, h_rcn, h_rr, h_rrn
    start = t0
    for w_name, msk in (("all cases", np.ones(len(j), bool)), (f"after the first {a.warmup_min:.0f} min", (j["ts"] - start >= a.warmup_min * 60).values)):
        if msk.sum() == 0:
            continue
        print(f"\n   {w_name}: {msk.sum():,} cases")
        for tit, x, y in (("live-strict vs live-hindsight (= events known too late)", "_live", "_hind"), ("live-hindsight vs training (= events found differently)", "_hind", "_off"), ("live-strict vs training (= total difference the model sees)", "_live", "_off")):
            for c, c2 in (("rc10", "rc10"), ("rr20", "rr20")):
                xs = j[f"{c}_live"] if x == "_live" else j[f"{c}_hind"]
                ys = j[f"{c}_off"] if y == "_off" else j[f"{c}_hind"]
                dd = np.abs(xs.values - ys.values)[msk]
                dd = dd[~np.isnan(dd)]
                print(f"     {tit:<62}{c}: median |diff| {np.median(dd) if len(dd) else float('nan'):.3f}, 90% {np.quantile(dd, .9) if len(dd) else float('nan'):.3f}, NaN differs {100 * (np.isnan(xs.values[msk]) != np.isnan(ys.values[msk])).mean():.1f}%")
        print(f"     events counted in the window (median): rc10_n live {j.loc[msk, 'rc10_n_live'].median():.0f}, hindsight {j.loc[msk, 'rc10_n_hind'].median():.0f}, training {j.loc[msk, 'rc10_n_off'].median():.0f};"
              f"  rr20_n live {j.loc[msk, 'rr20_n_live'].median():.0f}, hindsight {j.loc[msk, 'rr20_n_hind'].median():.0f}, training {j.loc[msk, 'rr20_n_off'].median():.0f}")
    mins = ((j["ts"] - start) // 300 * 5).astype(int)
    wu = j.groupby(mins).agg(n=("ts", "size"), rc10_n=("rc10_n_live", "median"), rr20_n=("rr20_n_live", "median"), no_rc10=("rc10_live", lambda s: 100 * s.isna().mean()))
    print("\n   WARM-UP (minutes since the first poll -> cases, median events in 10 min (all buses), median events in 20 min (same route), % cases without rc10):")
    for k, v in wu.iterrows():
        print(f"     {k:>3}-{k + 5:<3} min {int(v['n']):>7,} cases  rc10_n {v['rc10_n']:>6.0f}  rr20_n {v['rr20_n']:>4.0f}  no rc10 {v['no_rc10']:>5.1f}%")

    # ------------------------------------------------------------ 4b. the store against the training formula
    rng = np.random.default_rng(0)
    pick = rng.choice(len(ev_off), min(300, len(ev_off)), replace=False)
    st = le.Events(keep_s=1e12)                       # the live engine forgets events older than 1 h; this test asks about the whole day
    for r in ev_off.itertuples():
        st.add(r.route_id, r.t_end, r.ratio)
    q = pd.DataFrame({"ts": ev_off["t_end"].values[pick] + rng.uniform(-30, 700, len(pick)), "route_id": ev_off["route_id"].values[pick]})
    ref = t2.recent_feats(q, ev_off)
    got = np.array([st.recent(r, t) for r, t in zip(q["route_id"], q["ts"])], dtype=float)
    ok = all(np.allclose(got[:, i], ref[i], equal_nan=True, atol=1e-9) for i in range(4))
    print(f"\n   event store against train_eta_v2.recent_feats on {len(q)} random queries: {'IDENTICAL' if ok else 'DIFFERENT (!)'}")

    # ------------------------------------------------------------ 5. model
    say("5. MODEL   ETA error with the training inputs against the live inputs")
    if not have_models:
        print("   skipped (no lightgbm here, or --no-model). On your computer run this script without --no-model.")
    else:
        if a.fake_model:
            print("   FAKE MODEL: only the code path is tested, the numbers below mean nothing.")
        base_cols = ["route_id", "trip_id_live", "ts", "h_live", "sidx", "dist_live", "prog_live"]
        def frame(sfx, rc=None):
            d = pd.DataFrame({c: j[c + sfx] for c in FEATS if c + sfx in j})
            d["route_id"], d["ts"], d["sidx"], d["vehicle_id"] = j["route_id"], j["ts"], j["sidx"], j["vehicle_id"]
            d["base"] = j["base" + sfx]
            if rc:
                for c in t2.RECENT:
                    d[c] = j[c + "_hind"] if c + "_hind" in j else d[c]
            return d
        variants = {"AI with training inputs": frame("_off"), "AI with live inputs (strict)": frame("_live")}
        hind = frame("_live")
        for c in t2.RECENT:
            hind[c] = j[c + "_hind"]
        variants["AI with live inputs + hindsight window"] = hind
        hh = frame("_live"); hh["h"] = j["h_off"].values
        variants["AI with live inputs, h as in training"] = hh
        act = j["actual"].values if "actual" in j else j["actual_off"].values
        print(f"   {len(j):,} cases   {'':<4}{'mean error':>11}{'typical':>9}{'bias':>8}{'within 1 min':>14}{'range holds':>13}{'range width':>13}")
        rows = {}
        for name, d in variants.items():
            p = eng.predict_cases(d)
            mt = t1.metrics(p["eta_raw"].values, act)
            cov = ((act >= p["lo_raw"].values) & (act <= p["hi_raw"].values)).mean() * 100 if "lo_raw" in p else float("nan")
            wid = np.median(p["hi_raw"].values - p["lo_raw"].values) if "lo_raw" in p else float("nan")
            print(f"   {name:<42}{mt['mae']:>9.0f} s{mt['med']:>7.0f} s{mt['bias']:>+6.0f} s{mt['w60']:>12.0f}%{cov:>12.0f}%{wid:>11.0f} s")
        mh = t1.metrics(j["base_off"].values, act)
        print(f"   {'history x time of day (no AI)':<42}{mh['mae']:>9.0f} s{mh['med']:>7.0f} s{mh['bias']:>+6.0f} s{mh['w60']:>12.0f}%")
        print("   (a day that was part of the training data gives optimistic errors; compare the three AI lines with each other, not with the report numbers)")
        now = float(live["poll"].max())
        pr = eng.predict(now=now)
        print(f"\n   eng.predict() smoke test at the last poll: {len(pr):,} rows for {pr['vehicle_id'].nunique():,} buses, columns {list(pr.columns)[:8]} ...")
        if len(pr) and "lo_s" in pr:
            print(f"   eta_s inside [lo_s, hi_s] for {100 * ((pr['eta_s'] >= pr['lo_s']) & (pr['eta_s'] <= pr['hi_s'])).mean():.0f}% (should be 100), warm rows {100 * pr['warm'].mean():.0f}%")
    print("\nSend this whole output back.")


if __name__ == "__main__":
    main()
