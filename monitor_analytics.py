#!/usr/bin/env python3
"""
monitor_analytics.py - MONITORING and ANALYTICS from the stop arrival times made by build_arrivals.py (offline).

It reads data/arrivals/<date>_arrivals.csv.gz (all days, or one day) and answers:

  MONITORING  (is the service running well?)
    1 HEADWAYS      time between two buses of the same route_id at the same stop
    2 BUNCHING      buses that travel too close together (headway under 25% of normal) and long GAPS (over 2x normal)
    3 WAIT TIME     "excess wait" = how much longer a passenger waits because buses are not evenly spaced
    4 SLOW WINDOWS  15-minute windows where a route (or the whole city) is much slower than its normal for that hour
    5 STALLS        a bus that took more than 5 minutes (and 6x normal) to go from one stop to the next

  ANALYTICS  (what does the network look like?)
    6 ROUTE RELIABILITY   how much the stop-to-stop time varies (buffer index) and the median speed
    7 SPEED BY HOUR       median stop-to-stop speed by hour, weekday vs weekend
    8 SLOWEST SECTIONS    the stop pairs with the lowest speed
    9 STOPS & FLEET       busiest stops, and how many buses were active per hour

Commands (run next to your data/ and gtfs/ folders):
  python monitor_analytics.py selftest                   simulated service with known problems: checks that they are found
  python monitor_analytics.py run                        all days found in data/arrivals/
  python monitor_analytics.py run --date 2026-11-09      one day
  python monitor_analytics.py run --plots                also save PNG charts (needs matplotlib)
Results are printed and saved as CSV files in data/analytics/.

Honest limits (also say them in the report):
  * Trip IDs do not link to the timetable, so there is NO "late against timetable" here. Everything is measured
    between buses (headways) or against the usual time of the same route (baseline).
  * A missed arrival (pings too far apart) makes a headway look longer, so "long gaps" can be false alarms. Bunching is the safer signal.
  * Headways above --max-headway (45 min) are dropped (sessions end, low frequency routes), so rare-bus routes are under-counted.
  * The "usual" time is the median of your own data for that route, section and hour. With few days it also contains the day being tested.
"""
import argparse, sys, tempfile, time
from pathlib import Path
import numpy as np
import pandas as pd

IST = 19800
MIN_H_S = 10          # headways shorter than this are duplicates, not buses
MAX_H_S = 2700        # headways longer than this (45 min) are dropped
BUNCH_FRAC = 0.25     # bunched = headway below this share of the normal headway
GAP_FACTOR = 2.0      # long gap = headway above this multiple of the normal headway
SLOW_RATIO = 1.6      # slow window = median (time / usual time) at or above this
MIN_OBS_WINDOW = 15   # a 15-minute window needs at least this many observations
STALL_MIN_S, STALL_FACTOR, STALL_MAX_S = 300, 6.0, 7200
FULL_HOUR_SHARE = 0.5   # an hour with fewer arrivals than this share of the busiest hour of that day counts as partly covered
MIN_SECTION_M = 150   # shorter sections are not ranked by speed (timing is too coarse)
MAX_SEG_KMH = 80      # faster stop-to-stop speeds are treated as errors
WANT = {"run_id", "route_id", "vehicle_id", "stop_idx", "stop_id", "arrive_ts", "quality", "stop_prog_m"}


def say(i, n, title, plain):
    print(f"\n[{i}/{n}] {title}\n      {plain}", flush=True)


def hhmm(slot15):
    m = int(slot15) * 15
    return f"{m // 60:02d}:{m % 60:02d}"


def dayname(dayno):
    return pd.Timestamp(int(dayno) * 86400, unit="s").strftime("%Y-%m-%d %a")


# ============================ load ============================
def load_names(gtfs):
    names, stops = {}, {}
    g = Path(gtfs)
    try:
        r = pd.read_csv(g / "routes.txt", dtype=str)
        col = "route_short_name" if "route_short_name" in r else None
        for rid, sh in zip(r["route_id"], r[col] if col else r["route_id"]):
            names[rid] = rid if (not isinstance(sh, str) or sh == rid) else f"{sh} [{rid}]"
    except Exception:
        pass
    try:
        s = pd.read_csv(g / "stops.txt", dtype=str, usecols=["stop_id", "stop_name"])
        stops = dict(zip(s["stop_id"], s["stop_name"]))
    except Exception:
        pass
    return names, stops


def load_arrivals(folder, date, keep_coarse):
    files = sorted(Path(folder).glob("*_arrivals.csv.gz"))
    if date:
        files = [f for f in files if f.name.startswith(date)]
    if not files:
        raise SystemExit(f"No *_arrivals.csv.gz files in {folder}. Run build_arrivals.py first.")
    parts = []
    for k, f in enumerate(files):
        d = pd.read_csv(f, usecols=lambda c: c in WANT, dtype={"vehicle_id": str, "route_id": str, "stop_id": str})
        d["run_key"] = k * 10_000_000 + d["run_id"]
        parts.append(d)
    a = pd.concat(parts, ignore_index=True)
    n0 = len(a)
    if not keep_coarse and "quality" in a:
        a = a[a["quality"] != "coarse"]
    local = a["arrive_ts"] + IST
    a["day"] = (local // 86400).astype(int)
    a["hour"] = ((local % 86400) // 3600).astype(int)
    a["bin15"] = ((local % 86400) // 900).astype(int)
    hc = a.groupby(["day", "hour"]).size().rename("hc")             # hours at the start / end of a session are only partly covered
    a = a.join(hc, on=["day", "hour"])
    a["full_hour"] = a["hc"] >= FULL_HOUR_SHARE * a.groupby("day")["hc"].transform("max")
    a = a.drop(columns="hc")
    a = a.sort_values(["route_id", "vehicle_id", "stop_idx", "arrive_ts"])
    same = (a["route_id"] == a["route_id"].shift()) & (a["vehicle_id"] == a["vehicle_id"].shift()) & (a["stop_idx"] == a["stop_idx"].shift())
    dup = same & ((a["arrive_ts"] - a["arrive_ts"].shift()).abs() < 120)       # the same bus, same stop, twice (overlapping runs)
    a = a[~dup].reset_index(drop=True)
    return a, files, n0


# ============================ 1-3 headways, bunching, wait time ============================
def headways(a):
    h = a.sort_values(["day", "route_id", "stop_idx", "arrive_ts"])
    key = (h["day"] == h["day"].shift()) & (h["route_id"] == h["route_id"].shift()) & (h["stop_idx"] == h["stop_idx"].shift()) & (h["vehicle_id"] != h["vehicle_id"].shift())
    h = h.assign(headway_s=(h["arrive_ts"] - h["arrive_ts"].shift()).where(key), prev_veh=h["vehicle_id"].shift())
    h = h[h["headway_s"].between(MIN_H_S, MAX_H_S) & h["full_hour"]].copy()      # partly covered hours are dropped: they only show short headways
    h["pair"] = h["day"].astype(str) + "|" + h["vehicle_id"] + "|" + h["prev_veh"]
    g = ["day", "route_id", "hour"]
    # the normal headway needs 8 different bus pairs (the same pair seen at 30 stops is still one pair)
    n_ref = h.groupby(g)["pair"].transform("nunique"); ref = h.groupby(g)["headway_s"].transform("median")
    n2 = h.groupby(["day", "route_id"])["pair"].transform("nunique"); ref2 = h.groupby(["day", "route_id"])["headway_s"].transform("median")
    h["ref_s"] = np.where(n_ref >= 8, ref, np.where(n2 >= 8, ref2, np.nan))
    ok = h["ref_s"].notna()
    h["bunched"] = (h["headway_s"] < BUNCH_FRAC * h["ref_s"]) & ok
    h["gap"] = (h["headway_s"] > GAP_FACTOR * h["ref_s"]) & ok
    h["h2"] = h["headway_s"] ** 2
    h["norm"] = (h["headway_s"] / h["ref_s"]).clip(upper=5)      # capped at 5x normal so one huge gap cannot decide the CV
    return h


def route_headway_table(h, a):
    g = h.groupby("route_id").agg(headways=("headway_s", "size"), mean_s=("headway_s", "mean"), median_s=("headway_s", "median"),
                                   sd_s=("headway_s", "std"), sum_h=("headway_s", "sum"), sum_h2=("h2", "sum"),
                                   norm_mean=("norm", "mean"), norm_sd=("norm", "std"), bus_pairs=("pair", "nunique"),
                                   pct_bunched=("bunched", "mean"), pct_long_gap=("gap", "mean"))
    g["cv"] = g["norm_sd"] / g["norm_mean"]          # spread of (headway / normal headway of that hour): 0 = even, about 1 = random arrivals
    g["avg_wait_min"] = g["sum_h2"] / (2 * g["sum_h"]) / 60            # what a passenger arriving at random waits on average
    g["even_wait_min"] = g["mean_s"] / 2 / 60                          # the wait if buses were perfectly evenly spaced
    g["excess_wait_min"] = g["avg_wait_min"] - g["even_wait_min"]
    g["median_headway_min"] = g["median_s"] / 60
    g["buses"] = a.groupby("route_id")["vehicle_id"].nunique().reindex(g.index)
    g["pct_bunched"] *= 100; g["pct_long_gap"] *= 100
    return g.drop(columns=["sum_h", "sum_h2", "sd_s", "mean_s", "median_s", "norm_mean", "norm_sd"]).reset_index()


# ============================ segments (stop to stop) ============================
def segments(a):
    """returns (stop-to-stop table, time-of-day factors)"""
    s = a.sort_values(["run_key", "stop_idx"]).reset_index(drop=True)
    ok = (s["run_key"] == s["run_key"].shift()) & ((s["stop_idx"] - s["stop_idx"].shift()) == 1)
    s["seg_s"] = (s["arrive_ts"] - s["arrive_ts"].shift()).where(ok)
    has_m = "stop_prog_m" in s
    s["seg_m"] = (s["stop_prog_m"] - s["stop_prog_m"].shift()).where(ok) if has_m else np.nan
    s["prev_stop_id"] = s["stop_id"].shift().where(ok)
    s = s[ok & (s["seg_s"] > 0)].copy()
    s["kmh"] = s["seg_m"] / s["seg_s"] * 3.6
    if has_m:
        s = s[(s["seg_m"] > 0) & (s["kmh"] <= MAX_SEG_KMH)]
    # usual time: route + section + hour (needs 10 cases), else route + section
    bh = s.groupby(["route_id", "stop_idx", "hour"])["seg_s"].agg(base_h="median", n_h="size").reset_index()
    b0 = s.groupby(["route_id", "stop_idx"])["seg_s"].agg(base_0="median", n_0="size").reset_index()
    s = s.merge(bh, on=["route_id", "stop_idx", "hour"], how="left").merge(b0, on=["route_id", "stop_idx"], how="left")
    s["base_s"] = np.where(s["n_h"] >= 10, s["base_h"], np.where(s["n_0"] >= 10, s["base_0"], np.nan))
    s["ratio"] = np.where(s["base_s"] >= 5, s["seg_s"] / s["base_s"], np.nan)
    s["weekend"] = ((s["day"] + 3) % 7 >= 5)
    # Most sections have fewer than 10 cases per hour, so the usual time is mostly an all-day value and rush hours look slow.
    # The time-of-day factor (median ratio of everything in that hour) is divided out, so only the surprise remains.
    fac = s.groupby(["weekend", "hour"]).agg(hour_factor=("ratio", "median"), factor_days=("day", "nunique")).reset_index()
    s = s.merge(fac, on=["weekend", "hour"], how="left")
    s["ratio_adj"] = s["ratio"] / s["hour_factor"]
    mx = a.groupby("route_id")["stop_idx"].max()
    s["near_end"] = (s["stop_idx"] <= 2) | (s["stop_idx"] >= s["route_id"].map(mx) - 2)     # first or last sections: terminal halts are normal there
    return s.drop(columns=["base_h", "n_h", "base_0", "n_0"]), fac


def slow_windows(s):
    v = s[s["ratio_adj"].notna() & ~s["near_end"]]
    w = v.groupby(["day", "bin15", "route_id"])["ratio_adj"].agg(ratio_median="median", observations="size").reset_index()
    w = w[(w["observations"] >= MIN_OBS_WINDOW) & (w["ratio_median"] >= SLOW_RATIO)].sort_values("ratio_median", ascending=False)
    c = v.groupby(["day", "bin15"]).agg(ratio_median=("ratio_adj", "median"), observations=("ratio_adj", "size"), factor_days=("factor_days", "first")).reset_index()
    return w, c[c["observations"] >= 200]


def stalls(s):
    t = s[s["base_s"].notna() & (s["seg_s"] > np.maximum(STALL_MIN_S, STALL_FACTOR * s["base_s"])) & (s["seg_s"] < STALL_MAX_S)].copy()
    t["kind"] = np.where(t["near_end"], "terminal", "on route")
    t["time_ist"] = pd.to_datetime(t["arrive_ts"] + IST, unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    t["took_min"] = (t["seg_s"] / 60).round(1); t["usual_min"] = (t["base_s"] / 60).round(1)
    return t.sort_values("seg_s", ascending=False)[["time_ist", "kind", "route_id", "vehicle_id", "prev_stop_id", "stop_id", "took_min", "usual_min"]]


def reliability(s):
    q = s.groupby(["route_id", "stop_idx"])["seg_s"].quantile([0.5, 0.9]).unstack()
    q.columns = ["p50", "p90"]; q["n"] = s.groupby(["route_id", "stop_idx"]).size()
    q = q[(q["n"] >= 15) & (q["p50"] >= 5)]
    q["buffer"] = (q["p90"] - q["p50"]) / q["p50"]
    r = q.groupby("route_id").agg(sections=("buffer", "size"), buffer_index=("buffer", "median"))
    r["median_speed_kmh"] = s.groupby("route_id")["kmh"].median().reindex(r.index) if s["kmh"].notna().any() else np.nan
    r["observations"] = s.groupby("route_id").size().reindex(r.index)
    return r.reset_index(), q


# ============================ report ============================
def run_all(a, files, n0, out, names, stops, plots, min_h, top):
    lab = lambda r: names.get(r, r)
    stn = lambda x: (stops.get(x, x) if isinstance(x, str) else "-")
    out.mkdir(parents=True, exist_ok=True)
    days = sorted(a["day"].unique())
    print("\n================ MONITORING & ANALYTICS REPORT ================")
    print(f"files: {len(files)} | arrivals used: {len(a):,} of {n0:,} (coarse-quality and duplicates removed) | buses {a['vehicle_id'].nunique():,} | route_ids {a['route_id'].nunique():,}")
    print("days: " + ", ".join(dayname(d) for d in days))

    # ---- 1-3
    say(1, 6, "HEADWAYS, BUNCHING AND WAIT TIME", f"Time between two buses of one route_id at one stop (different buses, {MIN_H_S} s to {MAX_H_S // 60:.0f} min). Normal headway = median of that route_id, day and hour. Partly covered hours (session start / end) are left out.")
    h = headways(a)
    rt = route_headway_table(h, a); rt.to_csv(out / "route_headway.csv", index=False)
    print(f"      {len(h):,} headways on {h['route_id'].nunique():,} route_ids. All routes together: median headway {h['headway_s'].median() / 60:.1f} min, "
          f"bunched {100 * h['bunched'].mean():.1f}%, long gaps {100 * h['gap'].mean():.1f}%")
    print("      BENCHMARK: buses that arrive completely at random (no control at all) would give bunched about 16%, long gaps about 25%, CV 1.0.")
    print("      A perfectly regular service gives 0%, 0% and CV 0. So compare each route with these two ends, not with zero.")
    big = rt[rt["bus_pairs"] >= min_h].copy()
    print(f"      {len(big):,} route_ids have at least {min_h} different bus pairs (the same two buses seen at many stops count once) and are ranked below.")
    if len(big):
        bands = pd.cut(big["cv"], [-1, 0.5, 0.8, 1.2, 99], labels=["regular (CV under 0.5)", "somewhat regular (0.5-0.8)", "random-like (0.8-1.2)", "clustered (over 1.2)"]).value_counts().reindex(
            ["regular (CV under 0.5)", "somewhat regular (0.5-0.8)", "random-like (0.8-1.2)", "clustered (over 1.2)"])
        print("      Ranked routes by spacing: " + " | ".join(f"{k} {v} ({100 * v / len(big):.0f}%)" for k, v in bands.items()))
    cols = lambda d: [(lab(r.route_id), int(r.bus_pairs), f"{r.median_headway_min:.1f}", f"{r.cv:.2f}", f"{r.pct_bunched:.1f}", f"{r.pct_long_gap:.1f}", f"{r.excess_wait_min:.1f}") for r in d.itertuples()]
    head = f"      {'route':<30}{'bus pairs':>10}{'median min':>11}{'CV':>6}{'bunched%':>9}{'gaps%':>7}{'extra wait min':>15}"
    print("\n      Most clustered / uneven buses (highest CV, a scale-free measure):")
    print(head)
    for row in cols(big.sort_values("cv", ascending=False).head(top)):
        print(f"      {row[0]:<30}{row[1]:>10}{row[2]:>11}{row[3]:>6}{row[4]:>9}{row[5]:>7}{row[6]:>15}")
    print("\n      Most bunching:")
    print(head)
    for row in cols(big.sort_values("pct_bunched", ascending=False).head(top)):
        print(f"      {row[0]:<30}{row[1]:>10}{row[2]:>11}{row[3]:>6}{row[4]:>9}{row[5]:>7}{row[6]:>15}")
    print("\n      Most regular routes (lowest CV):")
    print(head)
    for row in cols(big.sort_values("cv").head(5)):
        print(f"      {row[0]:<30}{row[1]:>10}{row[2]:>11}{row[3]:>6}{row[4]:>9}{row[5]:>7}{row[6]:>15}")
    print("      (Extra wait = minutes a passenger waits on average more than with perfectly even buses. It grows with the headway, so rank by CV.)")
    hb = h.groupby("hour").agg(headways=("headway_s", "size"), bunched=("bunched", "mean"), gap=("gap", "mean"))
    hb = hb[hb["headways"] >= 500]
    if len(hb):
        print("\n      Bunching by hour of day (all routes): " + " | ".join(f"{i:02d}h {100 * r.bunched:.1f}%" for i, r in hb.iterrows()))
        hb.reset_index().to_csv(out / "bunching_by_hour.csv", index=False)

    # ---- segments
    say(2, 6, "STOP-TO-STOP TIMES", "For every bus and every pair of neighbouring stops that both got an arrival, the travel time and speed are calculated.")
    s, fac = segments(a)
    print(f"      {len(s):,} stop-to-stop trips on {s['route_id'].nunique():,} route_ids | median speed {s['kmh'].median():.1f} km/h")
    rel, q = reliability(s[~s["near_end"]]); rel.to_csv(out / "route_reliability.csv", index=False)
    fac.to_csv(out / "time_of_day_factor.csv", index=False)
    for wk, nm in ((False, "weekday"), (True, "weekend")):
        t = fac[fac["weekend"] == wk]
        if len(t):
            print(f"      Time-of-day effect, time / all-day usual time ({nm}): " + " | ".join(f"{r.hour:02d}h {r.hour_factor:.2f}" for r in t.itertuples()))
    print("      (above 1 = slower than the all-day usual: this is the normal rush-hour pattern, so it is divided out before looking for slow windows)")

    # ---- 4 slow windows
    say(3, 6, "SLOW WINDOWS", f"15-minute windows where a route_id is at least {SLOW_RATIO}x slower than its usual time for that hour (min {MIN_OBS_WINDOW} observations; first and last sections left out). City-wide windows are judged only for hours that have 3+ days of data.")
    w, city = slow_windows(s)
    w["time_ist"] = [f"{dayname(d)[:10]} {hhmm(b)}" for d, b in zip(w["day"], w["bin15"])]
    w.to_csv(out / "slow_windows.csv", index=False); city.to_csv(out / "city_ratio_by_15min.csv", index=False)
    if len(w):
        print(f"      {len(w):,} slow route windows. Worst:")
        for r in w.head(top).itertuples():
            print(f"      {r.time_ist}  {lab(r.route_id):<28} {r.ratio_median:.2f}x usual time ({r.observations} observations)")
        cl = w.groupby(["day", "bin15"]).size().reset_index(name="routes").sort_values("routes", ascending=False)
        cl = cl[cl["routes"] >= 5]
        if len(cl):
            print("      Times when 5 or more routes were slow together (a shared cause such as rain, an event or a closed road is possible): " + ", ".join(f"{dayname(r.day)[:10]} {hhmm(r.bin15)} ({r.routes} routes)" for r in cl.head(8).itertuples()))
    else:
        print("      no slow route windows found.")
    cc = city[city["factor_days"] >= 3]
    cs = cc[cc["ratio_median"] >= 1.15]
    if len(cs):
        print("      City-wide slow windows (median at least 1.15x the usual for that hour): " + ", ".join(f"{dayname(r.day)[:10]} {hhmm(r.bin15)} ({r.ratio_median:.2f}x)" for r in cs.head(8).itertuples()))
    elif len(cc):
        print(f"      City-wide: no window above 1.15x (highest {cc['ratio_median'].max():.2f}x at {dayname(cc.loc[cc['ratio_median'].idxmax(), 'day'])[:10]} {hhmm(cc.loc[cc['ratio_median'].idxmax(), 'bin15'])}).")
    else:
        print("      City-wide check needs hours with 3+ days of data: not available.")

    # ---- 5 stalls
    say(4, 6, "STALLS AND LONG HALTS", f"A bus needed more than {STALL_MIN_S // 60} minutes and more than {STALL_FACTOR:.0f}x the usual time between two neighbouring stops (up to {STALL_MAX_S // 3600} h). Pairs at the first / last 3 stops of a route are marked 'terminal'.")
    st = stalls(s); st.to_csv(out / "stalls.csv", index=False)
    on = st[st["kind"] == "on route"]
    print(f"      {len(st):,} long halts ({100 * len(st) / max(len(s), 1):.3f}% of stop-to-stop trips): {len(st) - len(on):,} at terminals, {len(on):,} on the route.")
    if len(on):
        bd = pd.cut(on["took_min"], [0, 10, 20, 30, 60, 1e9], labels=["5-10 min", "10-20", "20-30", "30-60", "over 60"]).value_counts().reindex(["5-10 min", "10-20", "20-30", "30-60", "over 60"])
        print("      On the route, by duration: " + " | ".join(f"{k} {v:,}" for k, v in bd.items()))
        print("      Longest 5-30 minute halts on the route (long ones are mostly parked buses and driver breaks, not faults):")
        mid = on[(on["took_min"] <= 30)]
        for r in mid.head(top).itertuples():
            print(f"      {r.time_ist}  {lab(r.route_id):<28} bus {r.vehicle_id}  {stn(r.prev_stop_id)[:24]} -> {stn(r.stop_id)[:24]}: {r.took_min} min (usual {r.usual_min})")
        pr = on.groupby("route_id").size().sort_values(ascending=False).head(5)
        print("      Routes with most halts on the route: " + ", ".join(f"{lab(r)} ({n})" for r, n in pr.items()))
    print("      (a halt can be a breakdown, a jam, a driver break, a parked bus or a ping problem: check the bus with the tracker before calling it a fault)")

    # ---- 6 analytics
    say(5, 6, "ROUTE RELIABILITY, SPEED BY HOUR, SLOWEST SECTIONS", "Buffer index = (slowest 10% time - median time) / median time for a stop pair; the route value is the median over its stop pairs (first and last 3 stops left out). Higher = less predictable.")
    rb = rel[rel["sections"] >= 10]
    print(f"      {len(rb):,} route_ids with at least 10 stop pairs of 15+ observations. Median buffer index {rb['buffer_index'].median():.2f}, median route speed {rb['median_speed_kmh'].median():.1f} km/h")
    print("      Least predictable routes:")
    for r in rb.sort_values("buffer_index", ascending=False).head(top).itertuples():
        print(f"      {lab(r.route_id):<22} buffer index {r.buffer_index:.2f} | median speed {r.median_speed_kmh:.1f} km/h | {r.observations:,} observations")
    print("      Most predictable routes:")
    for r in rb.sort_values("buffer_index").head(5).itertuples():
        print(f"      {lab(r.route_id):<22} buffer index {r.buffer_index:.2f} | median speed {r.median_speed_kmh:.1f} km/h | {r.observations:,} observations")
    s["weekend"] = ((s["day"] + 3) % 7 >= 5)
    sp = s.groupby(["weekend", "hour"])["kmh"].agg(median_kmh="median", observations="size").reset_index()
    sp = sp[sp["observations"] >= 500]; sp.to_csv(out / "speed_by_hour.csv", index=False)
    for wk, nm in ((False, "weekday"), (True, "weekend")):
        t = sp[sp["weekend"] == wk]
        if len(t):
            print(f"      Median stop-to-stop speed by hour ({nm}): " + " | ".join(f"{r.hour:02d}h {r.median_kmh:.1f}" for r in t.itertuples()))
    sec = s[s["kmh"].notna() & ~s["near_end"] & (s["seg_m"] >= MIN_SECTION_M)].groupby(["route_id", "stop_idx"]).agg(median_kmh=("kmh", "median"), observations=("kmh", "size"), stop_id=("stop_id", "first"), prev_stop_id=("prev_stop_id", "first")).reset_index()
    sec = sec[sec["observations"] >= 30].sort_values("median_kmh")
    sec["from_stop"] = sec["prev_stop_id"].map(stn); sec["to_stop"] = sec["stop_id"].map(stn)
    sec.to_csv(out / "slowest_sections.csv", index=False)
    hs = s[s["kmh"].notna() & ~s["near_end"] & (s["seg_m"] >= MIN_SECTION_M)].groupby(["prev_stop_id", "stop_id"]).agg(
        median_kmh=("kmh", "median"), observations=("kmh", "size"), routes=("route_id", "nunique")).reset_index()
    hs = hs[(hs["routes"] >= 3) & (hs["observations"] >= 100)].sort_values("median_kmh")
    hs["from_stop"] = hs["prev_stop_id"].map(stn); hs["to_stop"] = hs["stop_id"].map(stn)
    hs.to_csv(out / "congestion_hotspots.csv", index=False)
    print(f"      CONGESTION HOTSPOTS: the same stop pair on 3+ different route_ids, 100+ observations ({len(hs):,} pairs qualify). More reliable than a single route because several routes agree:")
    for r in hs.head(top).itertuples():
        print(f"      {r.median_kmh:5.1f} km/h  {str(r.from_stop)[:30]} -> {str(r.to_stop)[:30]}  ({r.routes} routes, {r.observations} observations)")
    print(f"      Slowest sections (stop pair, 30+ observations, at least {MIN_SECTION_M} m long, not at the route ends):")
    for r in sec.head(top).itertuples():
        print(f"      {r.median_kmh:5.1f} km/h  {lab(r.route_id):<20} {str(r.from_stop)[:26]} -> {str(r.to_stop)[:26]}  ({r.observations})")

    # ---- 9 stops, fleet
    say(6, 6, "BUSIEST STOPS AND FLEET ACTIVITY", "Stop activity = bus passes per day at the stop. Fleet activity = different buses with at least one arrival in that hour.")
    sa = a.groupby("stop_id").agg(passes=("vehicle_id", "size"), buses=("vehicle_id", "nunique"), routes=("route_id", "nunique")).reset_index().sort_values("passes", ascending=False)
    sa["stop_name"] = sa["stop_id"].map(stn); sa["passes_per_day"] = (sa["passes"] / max(len(days), 1)).round(0)
    sa.head(500).to_csv(out / "busiest_stops.csv", index=False)
    print("      Busiest stops (passes per day in the hours you collected):")
    for r in sa.head(top).itertuples():
        print(f"      {str(r.stop_name)[:34]:<36} {r.passes_per_day:>8.0f} passes/day | {r.routes} route_ids | {r.buses} buses")
    fl = a.groupby(["day", "hour"])["vehicle_id"].nunique().unstack(fill_value=0)
    fl.index = [dayname(d) for d in fl.index]; fl.to_csv(out / "fleet_by_hour.csv")
    print("      Active buses per hour (a partly covered first or last hour shows a low number):")
    for d, r in fl.iterrows():
        print(f"      {d}: " + " ".join(f"{hr:02d}h:{int(v):,}" for hr, v in r.items() if v > 0))
    print(f"\nCSV files saved in {out}/")
    if plots:
        make_plots(out, rt, city, sp, min_h)
    return dict(h=h, rt=rt, s=s, w=w, city=city, st=st, rel=rel)


def make_plots(out, rt, city, sp, min_h):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not installed: no charts (uv add matplotlib)"); return
    big = rt[rt["bus_pairs"] >= min_h]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(big["median_headway_min"], big["excess_wait_min"], s=10, alpha=0.5)
    ax.set_xlabel("median headway (min)"); ax.set_ylabel("extra waiting time (min)"); ax.set_title("Extra waiting time per route_id")
    fig.tight_layout(); fig.savefig(out / "plot_excess_wait.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for wk, nm in ((False, "weekday"), (True, "weekend")):
        t = sp[sp["weekend"] == wk]
        if len(t):
            ax.plot(t["hour"], t["median_kmh"], marker="o", label=nm)
    ax.set_xlabel("hour of day (IST)"); ax.set_ylabel("median stop-to-stop speed (km/h)"); ax.legend(); ax.set_title("Speed by hour")
    fig.tight_layout(); fig.savefig(out / "plot_speed_by_hour.png", dpi=150); plt.close(fig)
    if len(city):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for d, t in city.groupby("day"):
            ax.plot(t["bin15"] * 0.25, t["ratio_median"], marker=".", label=dayname(d)[:10])
        ax.axhline(1.0, color="grey", lw=0.8); ax.set_xlabel("hour of day (IST)"); ax.set_ylabel("time / usual time"); ax.legend(fontsize=7); ax.set_title("City-wide slowness (1.0 = usual)")
        fig.tight_layout(); fig.savefig(out / "plot_city_slowness.png", dpi=150); plt.close(fig)
    print(f"charts saved in {out}/ (plot_*.png)")


# ============================ selftest ============================
def simulate(folder):
    """2 days x 6 routes. Known problems: R_BUNCH sends buses in pairs 25 s apart; R_STALL has one bus stopped 12 min; R_SLOW is 2x slower 10:00-10:30 on day 2."""
    rng = np.random.default_rng(7)
    folder.mkdir(parents=True, exist_ok=True)
    routes = {"R_EVEN": 600, "R_RANDOM": 600, "R_BUNCH": 600, "R_STALL": 600, "R_SLOW": 600, "R_OTHER": 480}
    n_stops, spacing, base_seg = 30, 500.0, 60.0
    truth = {}
    for di, day in enumerate(["2026-11-02", "2026-11-03"]):
        t0 = int(pd.Timestamp(day).timestamp()) - IST + 9 * 3600
        rows, run_id = [], 0
        for rid, H in routes.items():
            deps = []
            for k in range(int(4 * 3600 / H)):
                base = t0 + k * H
                if rid == "R_BUNCH":
                    deps += [base, base + 25]                # pairs: headway pattern 25 s / 575 s (2 buses per cycle of 2H)
                elif rid == "R_RANDOM":
                    deps.append(base + rng.integers(-250, 250))
                else:
                    deps.append(base + rng.integers(-20, 20))
            if rid == "R_BUNCH":
                deps = [d + (0 if i % 2 == 0 else 0) for i, d in enumerate(deps)]
            for j, dep in enumerate(sorted(deps)):
                run_id += 1; t = float(dep)
                veh = f"{rid}-{j % 40:02d}"
                for idx in range(n_stops):
                    seg = base_seg * rng.uniform(0.85, 1.2)
                    if rid == "R_SLOW" and di == 1 and 10 * 3600 <= (t - t0 + 9 * 3600) < 10.5 * 3600:
                        seg *= 2.0
                    if rid == "R_STALL" and di == 0 and j == 12 and idx == 15:
                        seg += 720; truth["stall_vehicle"] = veh
                    if rid == "R_OTHER" and di == 0 and j == 5 and idx == 1:
                        seg += 1200; truth["layover_vehicle"] = veh
                    if idx > 0:
                        t += seg
                    rows.append((run_id, rid, veh, idx, f"S{rid}_{idx}", round(t, 1), "good", idx * spacing))
        pd.DataFrame(rows, columns=["run_id", "route_id", "vehicle_id", "stop_idx", "stop_id", "arrive_ts", "quality", "stop_prog_m"]).to_csv(folder / f"{day}_arrivals.csv.gz", index=False)
    return truth


def selftest():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        truth = simulate(tmp / "arr")
        a, files, n0 = load_arrivals(tmp / "arr", None, False)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = run_all(a, files, n0, tmp / "out", {}, {}, False, 30, 5)
        print(buf.getvalue()[:0], end="")
        rt = res["rt"].set_index("route_id")
        checks = []
        checks.append(("R_BUNCH has the highest bunching share", rt["pct_bunched"].idxmax() == "R_BUNCH"))
        checks.append(("R_BUNCH bunching above 30%", rt.loc["R_BUNCH", "pct_bunched"] > 30))
        checks.append(("R_EVEN bunching below 2%", rt.loc["R_EVEN", "pct_bunched"] < 2))
        checks.append(("R_EVEN excess wait below R_RANDOM", rt.loc["R_EVEN", "excess_wait_min"] < rt.loc["R_RANDOM", "excess_wait_min"]))
        checks.append(("R_BUNCH has more excess wait than R_EVEN", rt.loc["R_BUNCH", "excess_wait_min"] > rt.loc["R_EVEN", "excess_wait_min"] + 1))
        st = res["st"]
        checks.append(("stall of 12 min found for the right bus", len(st) >= 1 and (st[st["kind"] == "on route"].iloc[0]["vehicle_id"] == truth["stall_vehicle"]) and st[st["kind"] == "on route"].iloc[0]["took_min"] > 11))
        checks.append(("only that stall found on the route", (st["kind"] == "on route").sum() == 1))
        checks.append(("20 min halt at stop 1 is marked terminal, not on route", ((st["kind"] == "terminal") & (st["vehicle_id"] == truth["layover_vehicle"])).any() and not ((st["kind"] == "on route") & (st["vehicle_id"] == truth["layover_vehicle"])).any()))
        w = res["w"]
        checks.append(("slow window found on R_SLOW", len(w) >= 1 and (w["route_id"] == "R_SLOW").any()))
        checks.append(("slow windows only on R_SLOW", set(w["route_id"]) <= {"R_SLOW"}))
        checks.append(("no city-wide slow window", (res["city"]["ratio_median"] < 1.25).all() if len(res["city"]) else True))
        print("\nSELFTEST (simulated service with known problems)")
        for name, ok in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        print(f"  bunched share: " + ", ".join(f"{r} {rt.loc[r, 'pct_bunched']:.1f}%" for r in rt.index))
        print(f"  extra wait min: " + ", ".join(f"{r} {rt.loc[r, 'excess_wait_min']:.1f}" for r in rt.index))
        print("ALL CHECKS PASSED" if all(ok for _, ok in checks) else "SOME CHECKS FAILED")
        return all(ok for _, ok in checks)


def main():
    global MAX_H_S
    ap = argparse.ArgumentParser(description="Monitoring and analytics from the stop arrival files")
    ap.add_argument("cmd", choices=["run", "selftest"])
    ap.add_argument("--data", default="data"); ap.add_argument("--gtfs", default="gtfs")
    ap.add_argument("--date"); ap.add_argument("--plots", action="store_true")
    ap.add_argument("--include-coarse", action="store_true", help="also use arrivals whose two pings were more than 90 s apart")
    ap.add_argument("--min-pairs", type=int, default=30, help="a route_id needs this many different bus pairs to be ranked")
    ap.add_argument("--top", type=int, default=10, help="rows shown in each list")
    ap.add_argument("--max-headway", type=float, default=MAX_H_S / 60, help="minutes; longer headways are dropped")
    a_ = ap.parse_args()
    if a_.cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    MAX_H_S = a_.max_headway * 60
    t0 = time.time()
    names, stops = load_names(a_.gtfs)
    a, files, n0 = load_arrivals(Path(a_.data) / "arrivals", a_.date, a_.include_coarse)
    run_all(a, files, n0, Path(a_.data) / "analytics", names, stops, a_.plots, a_.min_pairs, a_.top)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
