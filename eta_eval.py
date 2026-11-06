#!/usr/bin/env python3
"""
eta_eval.py - which speed gives the best ETA? Test simple ETA rules against the OBSERVED arrival times.

It uses the files of build_arrivals.py (data/arrivals/<date>_pings.csv.gz and <date>_arrivals.csv.gz).
A test case is: one ping of a bus + one of the next 1-5 stops ahead. The right answer is the real time the bus
needed to reach that stop (from the arrival table). Each rule makes a guess: distance along the route / speed.
Rules (all clipped to 2-14 m/s like the explorer):
   feed number           the number of the feed's speed field (treated as m/s)
   feed + 12 s per stop  the same, plus a guess of 12 s for each stop in between (like the explorer)
   calculated            speed along the route between this ping and the previous one
   calculated + 12 s     the same, plus 12 s per stop in between
   route typical         the typical speed of this route, learned on the first 60% of the data (earlier times)
   one typical speed     one speed for all routes, learned on the same earlier 60%
The rules are tested only on the LAST 40% of the data in time, so the learned speeds never saw those cases.

   python eta_eval.py                         all days in data/arrivals
   python eta_eval.py --date 2026-10-30
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

HMAX, MIN_REMAIN, MAX_REMAIN, DWELL = 5, 5, 1500, 12
RULES = [("feed number", "p_feed"), ("feed + 12 s per stop", "p_feed_dw"), ("calculated", "p_calc"), ("calculated + 12 s per stop", "p_calc_dw"),
         ("route typical", "p_route"), ("one typical speed", "p_const")]


def load(folder, date):
    folder = Path(folder)
    files = sorted(folder.glob("*_pings.csv.gz"))
    if date:
        files = [f for f in files if f.name.startswith(date)]
    if not files:
        raise SystemExit(f"No *_pings.csv.gz files in {folder}. Run build_arrivals.py (without --no-pings) first.")
    P, A = [], []
    for k, f in enumerate(files):
        pg = pd.read_csv(f, dtype={"vehicle_id": str, "route_id": str})
        ar = pd.read_csv(str(f).replace("_pings", "_arrivals"), dtype={"vehicle_id": str, "route_id": str})
        if "stop_prog_m" not in ar:
            raise SystemExit("The arrivals files have no stop_prog_m column. Run:  python build_arrivals.py run --force")
        pg["run_key"] = k * 10_000_000 + pg["run_id"]; ar["run_key"] = k * 10_000_000 + ar["run_id"]
        P.append(pg); A.append(ar)
    return pd.concat(P, ignore_index=True), pd.concat(A, ignore_index=True), files


def build_samples(pings, arr, step):
    pings = pings.sort_values(["run_key", "ts"]).reset_index(drop=True)
    arr = arr.sort_values(["run_key", "stop_prog_m"]).reset_index(drop=True)
    pidx, aidx = pings.groupby("run_key").indices, arr.groupby("run_key").indices
    P_ts, P_prog = pings["ts"].values, pings["prog"].values
    A_ts, A_sp = arr["arrive_ts"].values, arr["stop_prog_m"].values
    out = {"pi": [], "h": [], "dist": [], "actual": []}
    for key, ia in aidx.items():
        ip = pidx.get(key)
        if ip is None or len(ip) < 3:
            continue
        ip = ip[::step]
        spr, atr, prog, ts = A_sp[ia], A_ts[ia], P_prog[ip], P_ts[ip]
        j0 = np.searchsorted(spr, prog + 20, side="right")
        for h in range(HMAX):
            j = j0 + h
            ok = j < len(spr)
            if not ok.any():
                break
            ii, jj = np.where(ok)[0], j[ok]
            remain = atr[jj] - ts[ii]
            g = (remain >= MIN_REMAIN) & (remain <= MAX_REMAIN)
            out["pi"].append(ip[ii][g]); out["h"].append(np.full(g.sum(), h + 1)); out["dist"].append(spr[jj][g] - prog[ii][g]); out["actual"].append(remain[g])
    s = pd.DataFrame({k: np.concatenate(v) if v else np.array([]) for k, v in out.items()})
    pg = pings.iloc[s["pi"].values].reset_index(drop=True)
    s["ts"], s["route_id"], s["vehicle_id"] = pg["ts"].values, pg["route_id"].values, pg["vehicle_id"].values
    s["rep"], s["calc"] = pg["speed_rep"].values, pg["calc_route_kmh"].values
    return s.drop(columns="pi"), pings


def metrics(df, col):
    e = df[col] - df["actual"]
    return dict(mae=e.abs().mean(), med=e.abs().median(), bias=e.mean(), w60=100 * (e.abs() <= 60).mean(), w120=100 * (e.abs() <= 120).mean())


def table(df, title):
    print(f"\n{title}   ({len(df):,} test cases)")
    print(f"   {'rule':<28}{'mean error':>11}{'typical error':>14}{'bias':>9}{'within 1 min':>14}{'within 2 min':>14}")
    rows = sorted(((n, metrics(df, c)) for n, c in RULES), key=lambda x: x[1]["mae"])
    for n, m in rows:
        print(f"   {n:<28}{m['mae']:>9.0f} s{m['med']:>12.0f} s{m['bias']:>+7.0f} s{m['w60']:>12.0f}%{m['w120']:>13.0f}%")
    return rows[0][0]


def main():
    ap = argparse.ArgumentParser(description="Test ETA rules against observed arrival times")
    ap.add_argument("--arrivals", default="data/arrivals"); ap.add_argument("--date"); ap.add_argument("--step", type=int, default=3, help="use every N-th ping")
    a = ap.parse_args()
    pings, arr, files = load(a.arrivals, a.date)
    s, pings = build_samples(pings, arr, a.step)
    print(f"=== ETA RULES TESTED AGAINST OBSERVED ARRIVALS: {len(files)} day(s), {len(s):,} cases (a ping + one of the next 1-{HMAX} stops) ===")
    s = s[s["rep"].notna() & s["calc"].notna() & s["dist"].gt(0)].copy()
    if len(s) < 2000:
        raise SystemExit(f"Only {len(s):,} usable cases (need a reported speed and a calculated speed). Collect and process more data.")
    cut = s["ts"].quantile(0.6)
    train, test = s[s["ts"] < cut].copy(), s[s["ts"] >= cut].copy()
    vt = (train["dist"] / train["actual"]).clip(1, 20)
    v_const = vt.median(); v_route = vt.groupby(train["route_id"]).median()
    print(f"learned on the first 60% (until {pd.to_datetime(cut + 19800, unit='s'):%d %b %H:%M} IST): one typical speed {v_const:.1f} m/s ({v_const * 3.6:.0f} km/h); typical speed of {len(v_route):,} routes")
    v_feed = test["rep"].clip(2, 14); v_calc = (test["calc"] / 3.6).clip(2, 14)
    dw = DWELL * (test["h"] - 1)
    test["p_feed"] = test["dist"] / v_feed; test["p_feed_dw"] = test["p_feed"] + dw
    test["p_calc"] = test["dist"] / v_calc; test["p_calc_dw"] = test["p_calc"] + dw
    test["p_route"] = test["dist"] / test["route_id"].map(v_route).fillna(v_const).clip(1, 20); test["p_const"] = test["dist"] / v_const
    best = table(test, "1. ALL TEST CASES  (error = guess minus real time; bias > 0 means the guess is too long)")
    print(f"   -> lowest mean error: {best}.  Remember: the arrival times themselves are uncertain by about half the gap between two pings (15 s or more).")

    print("\n2. BY HOW FAR AHEAD THE STOP IS (mean error in seconds)")
    print(f"   {'rule':<28}{'1 stop':>9}{'2 stops':>9}{'3-5 stops':>11}")
    for n, c in RULES:
        print(f"   {n:<28}" + "".join(f"{(g[c] - g['actual']).abs().mean():>8.0f} s" if k < 2 else f"{(g[c] - g['actual']).abs().mean():>10.0f} s" for k, g in enumerate([test[test['h'] == 1], test[test['h'] == 2], test[test['h'] >= 3]])))

    print("\n3. BY WHAT THE BUS DOES AT THE MOMENT (calculated speed along the route)")
    for name, g in (("standing (under 2 km/h)", test[test["calc"] < 2]), ("moving (5 km/h or more)", test[test["calc"] >= 5])):
        if len(g) >= 500:
            print(f"   {name}: {len(g):,} cases   " + "   ".join(f"{n.split(' +')[0] if 'stop' not in n else n}: {(g[c] - g['actual']).abs().mean():.0f} s" for n, c in RULES))

    nz = pings[pings["speed_rep"] > 0].copy()
    f = (nz["speed_rep"] * 3.6) % 1
    nz["sig"] = (f < 0.03) | (f > 0.97)
    per = nz.groupby("vehicle_id").agg(n=("sig", "size"), sig=("sig", "mean"))
    real = set(per[(per["n"] >= 15) & (per["sig"] >= 0.8)].index)
    test["real_like"] = test["vehicle_id"].isin(real)
    print(f"\n4. BY KIND OF BUS: buses whose feed numbers look like real speeds (whole km/h / 3.6 on at least 80% of their non-zero numbers): {len(real):,}")
    for name, g in (("'real speed' buses", test[test["real_like"]]), ("all other buses", test[~test["real_like"]])):
        if len(g) >= 300:
            print(f"   {name}: {len(g):,} cases   " + "   ".join(f"{n}: {(g[c] - g['actual']).abs().mean():.0f} s" for n, c in RULES[:3]))
    print("   -> if the feed number is clearly better for 'real speed' buses and worse for the others, use it only for those buses")
    print("\nSend this whole output back.")


if __name__ == "__main__":
    main()
