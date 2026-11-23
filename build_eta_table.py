#!/usr/bin/env python3
"""
build_eta_table.py - makes the "typical travel time between two stops" table used by delhi_bus_tui.py (history ETA).

It reads data/arrivals/*_arrivals.csv.gz (made by build_arrivals.py) and, for every route_id and every pair of
neighbouring stops, takes the MEDIAN time buses needed to get from the first stop to the next one. The tracker then
adds these times up for the stops that are still ahead of a live bus (see README: "ETA from history").

Run (next to your data/ folder, after build_arrivals.py):
  python build_eta_table.py build                    all days in data/arrivals/
  python build_eta_table.py build --days 2026-11-06,2026-11-07     only these days
  python build_eta_table.py check                    honest test: learn on all days except the LAST one, then predict the last day
  python build_eta_table.py selftest

Writes:  data/eta_table.csv.gz        route_id, from_stop, to_stop, n, median_s, p25_s, p75_s
         data/eta_table_info.json     typical speeds (fallback for stop pairs without data), days used, build time
Run it again whenever you have collected new days: the table then learns from them.
"""
import argparse, json, sys, tempfile, time
from pathlib import Path
import numpy as np
import pandas as pd

MIN_T, MAX_T = 3, 600       # seconds: stop-to-stop times outside this range are not used (same rule as train_eta.py)
MIN_N = 3                   # a stop pair needs at least this many observations
WANT = {"run_id", "route_id", "vehicle_id", "stop_idx", "stop_id", "arrive_ts", "quality", "stop_prog_m"}


def load(folder, days):
    files = sorted(Path(folder).glob("*_arrivals.csv.gz"))
    if days:
        files = [f for f in files if f.name[:10] in days]
    if not files:
        raise SystemExit(f"No *_arrivals.csv.gz files found in {folder} (run build_arrivals.py first).")
    parts = []
    for k, f in enumerate(files):
        d = pd.read_csv(f, usecols=lambda c: c in WANT, dtype={"route_id": str, "stop_id": str, "vehicle_id": str})
        d["run_key"] = k * 10_000_000 + d["run_id"]
        parts.append(d)
    return pd.concat(parts, ignore_index=True), [f.name[:10] for f in files]


def build(a):
    if "quality" in a:
        a = a[a["quality"] != "coarse"]
    a = a.sort_values(["run_key", "stop_idx"]).reset_index(drop=True)
    ok = (a["run_key"] == a["run_key"].shift()) & ((a["stop_idx"] - a["stop_idx"].shift()) == 1)
    seg = pd.DataFrame({"route_id": a["route_id"], "from_stop": a["stop_id"].shift(), "to_stop": a["stop_id"],
                        "t": a["arrive_ts"] - a["arrive_ts"].shift(), "m": a["stop_prog_m"] - a["stop_prog_m"].shift()})[ok]
    seg = seg[seg["t"].between(MIN_T, MAX_T)]
    g = seg.groupby(["route_id", "from_stop", "to_stop"])["t"]
    tab = g.agg(n="size", median_s="median", p25_s=lambda x: x.quantile(0.25), p75_s=lambda x: x.quantile(0.75)).reset_index() if len(seg) < 200_000 else None
    if tab is None:                                                      # faster path for big data
        q = g.quantile([0.25, 0.75]).unstack(); q.columns = ["p25_s", "p75_s"]
        tab = pd.concat([g.size().rename("n"), g.median().rename("median_s"), q], axis=1).reset_index()
    tab = tab[tab["n"] >= MIN_N].copy()
    for c in ("median_s", "p25_s", "p75_s"):
        tab[c] = tab[c].round(1)
    v = (seg["m"] / seg["t"]).where(seg["m"] > 0).dropna().clip(1, 20)
    seg_v = seg.assign(v=(seg["m"] / seg["t"]).where(seg["m"] > 0)).dropna(subset=["v"])
    seg_v["v"] = seg_v["v"].clip(1, 20)
    v_route = seg_v.groupby("route_id")["v"].median().round(2)
    info = {"v_global": round(float(v.median()), 2), "v_route": v_route.to_dict(), "segments_seen": int(len(seg)),
            "segment_pairs": int(len(tab)), "routes": int(tab["route_id"].nunique())}
    return tab, info


def main():
    ap = argparse.ArgumentParser(description="typical stop-to-stop time table for the tracker")
    ap.add_argument("cmd", choices=["build", "check", "selftest"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--days", help="comma separated dates, for example 2026-11-06,2026-11-07 (default: all)")
    a_ = ap.parse_args()
    if a_.cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    t0 = time.time()
    if a_.cmd == "check":
        arr, days = load(Path(a_.data) / "arrivals", set(a_.days.split(",")) if a_.days else None)
        check(arr, days)
        return
    arr, days = load(Path(a_.data) / "arrivals", set(a_.days.split(",")) if a_.days else None)
    tab, info = build(arr)
    info.update({"days": days, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "arrivals_used": int(len(arr))})
    out = Path(a_.data) / "eta_table.csv.gz"
    tab.to_csv(out, index=False)
    (Path(a_.data) / "eta_table_info.json").write_text(json.dumps(info))
    cnt = tab["n"]
    print(f"days used: {len(days)} ({days[0]} .. {days[-1]}) | arrivals {len(arr):,}")
    print(f"stop pairs with a typical time: {len(tab):,} on {info['routes']:,} route_ids | observations per pair: median {cnt.median():.0f}, 10+ in {100 * (cnt >= 10).mean():.0f}% of pairs")
    print(f"typical speed (fallback when a pair has no data): {info['v_global']:.1f} m/s = {info['v_global'] * 3.6:.0f} km/h overall, {len(info['v_route']):,} route_ids with their own value")
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB) and data/eta_table_info.json  ({time.time() - t0:.0f} s)")
    print("Start the tracker with:  python delhi_bus_tui.py   (it finds data/eta_table.csv.gz by itself; --eta speed turns history off)")


def check(arr, days):
    """Learn the table on all days but the last, then for every bus of the last day: from its arrival at stop i, predict its arrival
    at the stops i+1 .. i+6 by adding the table times (exactly what the tracker does) and compare with the real arrival."""
    if len(days) < 2:
        raise SystemExit("check needs at least 2 days of arrivals (it learns on the earlier days and tests on the last one).")
    arr = arr[arr["quality"] != "coarse"] if "quality" in arr else arr
    day_of = arr["run_key"] // 10_000_000
    last = day_of.max()
    tr, te = arr[day_of < last], arr[day_of == last].copy()
    tab, info = build(tr)
    seg = {(r, a, b): t for r, a, b, t in zip(tab["route_id"], tab["from_stop"], tab["to_stop"], tab["median_s"])}
    vr, vg = info["v_route"], info["v_global"]
    seq = arr.sort_values(["route_id", "stop_idx"]).groupby(["route_id", "stop_idx"]).agg(stop_id=("stop_id", "first"), prog=("stop_prog_m", "first")).reset_index()
    parts = []
    for r, g in seq.groupby("route_id"):
        idx, sid, prog = g["stop_idx"].to_numpy(), g["stop_id"].to_numpy(), g["prog"].to_numpy()
        T = np.zeros(len(idx)); H = np.zeros(len(idx))                         # time and data flag of the section that STARTS at this row
        v = vr.get(r, vg)
        for m in range(len(idx) - 1):
            t = seg.get((r, sid[m], sid[m + 1])) if idx[m + 1] == idx[m] + 1 else None
            if t is None:
                T[m] = max(prog[m + 1] - prog[m], 0) / v
            else:
                T[m], H[m] = t, 1
        parts.append(pd.DataFrame({"route_id": r, "stop_idx": idx, "C": np.cumsum(T) - T, "CH": np.cumsum(H) - H, "prog_c": prog}))
    cum = pd.concat(parts)
    te = te.merge(cum, on=["route_id", "stop_idx"], how="left")
    te = te[te["route_id"].isin(set(tab["route_id"]))].sort_values(["run_key", "stop_idx"]).reset_index(drop=True)
    out = []
    for k in range(1, 7):
        sh = lambda c: te[c].shift(-k)
        ok = (te["run_key"] == te["run_key"].shift(-k)) & ((te["stop_idx"].shift(-k) - te["stop_idx"]) == k)
        d = pd.DataFrame({"k": k, "actual": sh("arrive_ts") - te["arrive_ts"], "hist": sh("C") - te["C"], "dist": sh("prog_c") - te["prog_c"],
                          "cov": (sh("CH") - te["CH"]) / k})[ok]
        out.append(d)
    d = pd.concat(out).dropna()
    d = d[d["actual"] > 0]
    d["typ"] = d["dist"].clip(lower=0) / vg
    d["bucket"] = pd.cut(d["k"], [0, 1, 2, 4, 6], labels=["1 stop", "2 stops", "3-4 stops", "5-6 stops"])
    def row(x):
        e, e2 = (x["hist"] - x["actual"]).abs(), (x["typ"] - x["actual"]).abs()
        return pd.Series({"cases": len(x), "history_mean_s": e.mean(), "history_median_s": e.median(), "within_1min": 100 * (e <= 60).mean(),
                          "within_2min": 100 * (e <= 120).mean(), "bias_s": (x["hist"] - x["actual"]).mean(), "one_speed_mean_s": e2.mean(), "data_share": 100 * x["cov"].mean()})
    res = d.groupby("bucket", observed=True).apply(row, include_groups=False)
    res.loc["all"] = row(d)
    print(f"CHECK: table learned on {len(days) - 1} day(s), tested on the last day ({days[-1]}): {len(d):,} (bus, stop, stop-ahead) cases")
    print(res.round(1).to_string())
    print("history_* = table ETA error in seconds | within = % of cases within 1 / 2 minutes | bias = predicted minus real (positive = too slow) | "
          "one_speed = distance / one typical speed | data_share = % of the sections that came from data (the rest use the route's typical speed)")
    print("Note: this measures ETA from a bus's arrival at one stop to later stops, not from a ping in between, so it is a little easier than the live case.")


def selftest():
    rng = np.random.default_rng(3)
    rows, rid = [], 0
    for r in ("A", "B"):
        for j in range(60):
            rid += 1; t = 1000.0 + j * 600
            for i in range(6):
                if i:
                    t += {"A": 50.0, "B": 90.0}[r] * (1 + (0.1 if i == 3 else 0)) + rng.normal(0, 3)
                rows.append((rid, r, f"V{j}", i, f"{r}{i}", t, "good", i * 500.0))
    df = pd.DataFrame(rows, columns=["run_id", "route_id", "vehicle_id", "stop_idx", "stop_id", "arrive_ts", "quality", "stop_prog_m"])
    df["run_key"] = df["run_id"]
    tab, info = build(df)
    t = tab.set_index(["route_id", "from_stop", "to_stop"])["median_s"]
    ok = [abs(t[("A", "A0", "A1")] - 50) < 3, abs(t[("B", "B0", "B1")] - 90) < 3, abs(t[("A", "A2", "A3")] - 55) < 3, abs(info["v_route"]["A"] - 10) < 0.8]
    print("SELFTEST", "PASS" if all(ok) else f"FAIL {ok}", f"| A0->A1 {t[('A', 'A0', 'A1')]:.0f} s (50), B0->B1 {t[('B', 'B0', 'B1')]:.0f} s (90), A2->A3 {t[('A', 'A2', 'A3')]:.0f} s (55), speed A {info['v_route']['A']} m/s (10)")
    return all(ok)


if __name__ == "__main__":
    main()
