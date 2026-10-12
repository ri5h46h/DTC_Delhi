#!/usr/bin/env python3
"""
build_arrivals.py - turns saved bus pings into STOP ARRIVAL TIMES (offline, run it after a collection session).

The feed only gives dots ("bus X was at lat/lon at time T"). This script finds WHEN each bus reached each stop:
  1 LOAD      stop list of every route_id from your GTFS (cached after the first run)
  2 MATCH     for every ping: how far is it from its route line, and how many metres along the route is it?
  3 RUNS      cut each bus's pings into trips ("runs"); check the bus moves in the direction its route_id says
  4 ARRIVALS  between two pings that are before / after a stop, interpolate the time the bus passed the stop
  5 SAVE      write data/arrivals/<date>_arrivals.csv.gz and <date>_runs.csv.gz, then print a quality report

Commands (put this file next to otd_collector.py, so the default folders 'data' and 'gtfs' work):
  python build_arrivals.py selftest                     simulated buses with known truth: shows how accurate the method is
  python build_arrivals.py run                          process every day found in data/clean/
  python build_arrivals.py run --date 2026-10-11        one day;   --routes 2097,1957  only these route_ids
  python build_arrivals.py show --vehicle DL1PC1234     step-by-step view of one bus (pings -> progress -> arrivals)
"""
import argparse, math, pickle, sys, tempfile, time
from pathlib import Path
import numpy as np
import pandas as pd

# ---- settings (can also be changed with command-line options) ----
ON_ROUTE_M = 100      # a ping within this distance of the route line counts as "on the route"
BACK_M = 250          # progress falling back by more than this = a new trip started
RUN_GAP_S = 300       # no on-route ping for this long = the bus run is cut here
MAX_GAP_S = 120       # arrival is only kept if the two pings around the stop are at most this far apart in time
IST = 19800


def say(i, n, title, plain):
    print(f"\n[{i}/{n}] {title}\n      {plain}", flush=True)


# ============================ GTFS -> route lines ============================
class Pattern:
    """Ordered stops of one route_id, joined by straight lines. progress = metres along this line."""
    def __init__(self, route_id, short, stop_ids, stops):
        pts = [(s,) + tuple(stops[s]) for s in stop_ids if s in stops]
        self.route_id, self.short = route_id, short
        self.stop_ids = [p[0] for p in pts]
        self.names = [p[1] for p in pts]
        lat = np.array([p[2] for p in pts], float); lon = np.array([p[3] for p in pts], float)
        self.ok = len(pts) >= 2
        if self.ok:
            self.lat0, self.lon0 = float(lat.mean()), float(lon.mean())
            self.kx = 111320.0 * math.cos(math.radians(self.lat0))
            self.x, self.y = (lon - self.lon0) * self.kx, (lat - self.lat0) * 110540.0
            self.seglen = np.hypot(np.diff(self.x), np.diff(self.y))
            self.cum = np.concatenate([[0.0], np.cumsum(self.seglen)])      # progress of each stop
            self.total = float(self.cum[-1])

    def project_many(self, lat, lon, chunk=20000):
        """For many pings: (distance to the line in m, progress along the line in m)."""
        px, py = (lon - self.lon0) * self.kx, (lat - self.lat0) * 110540.0
        ax, ay, bx, by = self.x[:-1], self.y[:-1], self.x[1:], self.y[1:]
        dx, dy = bx - ax, by - ay
        l2 = np.where(dx * dx + dy * dy == 0, 1.0, dx * dx + dy * dy)
        dist, prog = np.empty(len(px)), np.empty(len(px))
        for s in range(0, len(px), chunk):
            X, Y = px[s:s + chunk, None], py[s:s + chunk, None]
            t = np.clip(((X - ax) * dx + (Y - ay) * dy) / l2, 0, 1)
            d = np.hypot(X - (ax + t * dx), Y - (ay + t * dy))
            i = d.argmin(1); r = np.arange(len(i))
            dist[s:s + chunk] = d[r, i]
            prog[s:s + chunk] = self.cum[i] + t[r, i] * self.seglen[i]
        return dist, prog


def load_patterns(gtfs):
    gtfs = Path(gtfs)
    need = ["routes.txt", "trips.txt", "stops.txt", "stop_times.txt"]
    for f in need:
        if not (gtfs / f).exists():
            raise SystemExit(f"Missing {gtfs / f}. Put your GTFS .txt files in '{gtfs}/' (or use --gtfs).")
    sig = [(f, (gtfs / f).stat().st_mtime, (gtfs / f).stat().st_size) for f in need]
    cache = gtfs / ".arrival_patterns.pkl"
    d = None
    if cache.exists():
        try:
            d = pickle.load(open(cache, "rb"))
            if d["sig"] != sig:
                d = None
        except Exception:
            d = None
    if d is None:
        print("      first time: reading stop_times.txt (can take a few minutes, then it is cached) ...", flush=True)
        routes = pd.read_csv(gtfs / "routes.txt", dtype=str)
        short = dict(zip(routes["route_id"], routes["route_short_name"])) if "route_short_name" in routes else {}
        sdf = pd.read_csv(gtfs / "stops.txt", dtype={"stop_id": str})
        sdf["stop_lat"] = pd.to_numeric(sdf["stop_lat"], errors="coerce"); sdf["stop_lon"] = pd.to_numeric(sdf["stop_lon"], errors="coerce")
        sdf = sdf.dropna(subset=["stop_lat", "stop_lon"])
        stops = {r.stop_id: (str(r.stop_name), float(r.stop_lat), float(r.stop_lon)) for r in sdf.itertuples()}
        trips = pd.read_csv(gtfs / "trips.txt", dtype=str, usecols=["route_id", "trip_id"])
        sizes = pd.Series(dtype="float64")
        for ch in pd.read_csv(gtfs / "stop_times.txt", usecols=["trip_id"], dtype=str, chunksize=1_000_000):
            sizes = sizes.add(ch["trip_id"].value_counts(), fill_value=0)
        trips["n"] = trips["trip_id"].map(sizes).fillna(0)
        rep = trips.sort_values("n", ascending=False).drop_duplicates("route_id")      # longest trip of each route_id
        rep = rep[rep["n"] >= 2]
        want = set(rep["trip_id"]); parts = []
        for ch in pd.read_csv(gtfs / "stop_times.txt", usecols=["trip_id", "stop_id", "stop_sequence"], dtype=str, chunksize=1_000_000):
            parts.append(ch[ch["trip_id"].isin(want)])
        st = pd.concat(parts)
        st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
        seq = st.sort_values(["trip_id", "stop_sequence"]).groupby("trip_id")["stop_id"].apply(list).to_dict()
        raw = {r.route_id: (short.get(r.route_id, r.route_id), seq.get(r.trip_id, [])) for r in rep.itertuples()}
        d = {"sig": sig, "stops": stops, "raw": raw}
        pickle.dump(d, open(cache, "wb"))
    pats = {}
    for rid, (sh, sids) in d["raw"].items():
        p = Pattern(rid, sh, sids, d["stops"])
        if p.ok:
            pats[rid] = p
    return pats


# ============================ the 4 processing steps ============================
def match_pings(df, pats, a):
    """Step 2: distance to route line + progress along the route for every ping."""
    df = df[df["route_id"].isin(pats)].reset_index(drop=True)
    lat, lon = df["lat"].values, df["lon"].values
    perp, prog = np.full(len(df), np.nan), np.full(len(df), np.nan)
    groups = df.groupby("route_id").indices
    step = max(1, len(groups) // 4)
    for k, (rid, ix) in enumerate(groups.items(), 1):
        perp[ix], prog[ix] = pats[rid].project_many(lat[ix], lon[ix])
        if k % step == 0 or k == len(groups):
            print(f"      matched {k}/{len(groups)} route_ids", flush=True)
    df["perp"], df["prog"] = perp, prog
    df["on_route"] = df["perp"] <= a.on_route
    return df


def build_runs(df, a):
    """Step 3: cut pings into runs. Returns (on-route pings with run_id, table of runs)."""
    allk = df.groupby(["vehicle_id", "route_id", "trip_id"]).agg(n_all=("ts", "size"), n_off=("on_route", lambda s: int((~s).sum())))
    p = df[df["on_route"]].sort_values(["vehicle_id", "route_id", "trip_id", "ts"])
    p = p.drop_duplicates(["vehicle_id", "route_id", "trip_id", "ts"]).reset_index(drop=True)
    ts, pr = p["ts"].values, p["prog"].values
    run_id = np.full(len(p), -1); runs = []; rid = 0
    for key, ix in p.groupby(["vehicle_id", "route_id", "trip_id"], sort=False).indices.items():
        ix = np.sort(ix); t, g = ts[ix], pr[ix]
        st = allk.loc[key]
        base = dict(vehicle_id=key[0], route_id=key[1], trip_id=key[2], off_route_share=round(st.n_off / st.n_all, 3))
        step = np.diff(g)
        fwd, bwd = int((step > 20).sum()), int((step < -20).sum())
        if bwd >= 3 and bwd > 2 * fwd:                                   # moves AGAINST the direction of its route_id
            run_id[ix] = rid
            runs.append(dict(run_id=rid, **base, t_start=t[0], t_end=t[-1], n_pings=len(ix), prog_start=g[0], prog_end=g[-1], direction_ok=False, usable=False))
            rid += 1
            continue
        s = 0
        while s < len(ix):
            e, rmax = s + 1, g[s]
            while e < len(ix) and t[e] - t[e - 1] <= a.run_gap and g[e] >= rmax - BACK_M:
                rmax = max(rmax, g[e]); e += 1
            seg = ix[s:e]; run_id[seg] = rid
            gg = pr[seg]
            runs.append(dict(run_id=rid, **base, t_start=ts[seg[0]], t_end=ts[seg[-1]], n_pings=len(seg), prog_start=gg[0], prog_end=gg.max(),
                             direction_ok=True, usable=len(seg) >= 3 and gg.max() - gg[0] >= 200))
            rid += 1; s = e
    p["run_id"] = run_id
    return p, pd.DataFrame(runs)


def find_arrivals(p, runs, pats, a):
    """Step 4: where a run goes from 'before the stop' to 'after the stop', interpolate the time it passed."""
    ts, pr = p["ts"].values, p["prog"].values
    idx_by_run = p.groupby("run_id").indices
    cols = {k: [] for k in ("run_id", "stop_idx", "t", "gap", "span")}
    expected = 0
    for r in runs[runs["usable"]].itertuples():
        ix = np.sort(idx_by_run[r.run_id]); t = ts[ix]
        pm = np.maximum.accumulate(pr[ix])                  # progress never goes backwards (removes GPS wobble)
        P = pats[r.route_id].cum
        expected += int(((P > pm[0]) & (P <= pm[-1])).sum())
        j = np.searchsorted(pm, P, side="left")             # first ping at or after each stop
        ok = (j > 0) & (j < len(pm))
        k = np.where(ok)[0]; j = j[ok]; i = j - 1           # pm[i] < stop <= pm[j]
        frac = (P[k] - pm[i]) / (pm[j] - pm[i])
        gap = t[j] - t[i]
        keep = gap <= a.max_gap
        k, i, j, frac, gap = k[keep], i[keep], j[keep], frac[keep], gap[keep]
        cols["run_id"].append(np.full(len(k), r.run_id)); cols["stop_idx"].append(k)
        cols["t"].append(t[i] + frac * gap); cols["gap"].append(gap); cols["span"].append(pm[j] - pm[i])
    arr = pd.DataFrame({c: (np.concatenate(v) if v else np.array([])) for c, v in cols.items()})
    if arr.empty:
        return arr, expected
    info = runs.set_index("run_id")[["vehicle_id", "route_id", "trip_id"]]
    arr = arr.join(info, on="run_id")
    arr["stop_id"] = [pats[r].stop_ids[k] for r, k in zip(arr["route_id"], arr["stop_idx"].astype(int))]
    arr["stop_idx"] = arr["stop_idx"].astype(int)
    arr["arrive_ts"] = arr["t"].round(1)
    arr["arrive_ist"] = pd.to_datetime(arr["t"] + IST, unit="s").dt.strftime("%H:%M:%S")
    arr["gap_s"] = arr["gap"].round(0).astype(int); arr["span_m"] = arr["span"].round(0).astype(int)
    arr["speed_kmh"] = (arr["span"] / arr["gap"].clip(lower=1) * 3.6).round(1)
    arr["quality"] = np.where(arr["gap"] <= 40, "good", np.where(arr["gap"] <= 90, "ok", "coarse"))
    arr = arr.sort_values(["run_id", "stop_idx"]).reset_index(drop=True)
    return arr[["run_id", "route_id", "vehicle_id", "trip_id", "stop_idx", "stop_id", "arrive_ts", "arrive_ist", "gap_s", "span_m", "speed_kmh", "quality"]], expected


def process(df, pats, a, quiet=False):
    n0 = len(df)
    df = df[df["flag"].fillna("") == ""]
    if not quiet:
        say(2, 5, "MATCH", f"{n0:,} pings read, {n0 - len(df):,} dropped because the collector flagged them as GPS jumps.")
    df = match_pings(df, pats, a)
    if df.empty:
        return df, df, df, 0
    if not quiet:
        print(f"      {df['on_route'].mean() * 100:.1f}% of {len(df):,} pings are within {a.on_route} m of their route line (the rest are depot trips, diversions, wrong route_id or bad GPS)")
        say(3, 5, "RUNS", "Each bus's pings are cut into trips; a trip that moves backwards along its route is marked as a direction mismatch.")
    p, runs = build_runs(df, a)
    if not quiet:
        say(4, 5, "ARRIVALS", "For every stop between a bus's first and last ping, the pass time is interpolated between the two pings around the stop.")
    arr, expected = find_arrivals(p, runs, pats, a)
    return p, runs, arr, expected


def report(df_in, p, runs, arr, expected, pats):
    print("\n================ QUALITY REPORT ================")
    if runs.empty:
        print("no runs found (are the route_ids in your data present in the GTFS?)"); return
    u = runs["usable"].sum(); bw = (~runs["direction_ok"]).sum()
    print(f"runs: {len(runs):,}  usable {u:,}  | moving against their route_id direction: {bw:,} ({100 * bw / len(runs):.1f}%)  | too short: {len(runs) - u - bw:,}")
    if arr.empty:
        print("no arrivals detected"); return
    print(f"arrivals: {len(arr):,} for {arr['vehicle_id'].nunique():,} buses on {arr['route_id'].nunique():,} route_ids")
    print("  quality (gap between the two pings around the stop):  " + ", ".join(f"{k} {v:,}" for k, v in arr["quality"].value_counts().items()))
    print(f"  coverage: {100 * len(arr) / max(expected, 1):.1f}% of the stops each bus drove past got an arrival time (rest skipped: pings too far apart)")
    print(f"  median gap between the pings used: {arr['gap_s'].median():.0f} s, median speed around the stop: {arr['speed_kmh'].median():.1f} km/h")
    a2 = arr.copy(); a2["pat_cum"] = [pats[r].cum[k] for r, k in zip(a2["route_id"], a2["stop_idx"])]
    a2["next_t"] = a2.groupby("run_id")["arrive_ts"].shift(-1); a2["next_cum"] = a2.groupby("run_id")["pat_cum"].shift(-1)
    s = a2.dropna(subset=["next_t"]); dt = s["next_t"] - s["arrive_ts"]
    ok = dt > 0; v = (s["next_cum"] - s["pat_cum"])[ok] / dt[ok] * 3.6
    print(f"  stop-to-stop travel: {len(s):,} pairs | median speed {v.median():.1f} km/h | suspicious (>60 km/h): {100 * (v > 60).mean():.1f}% | zero/negative time: {100 * (~ok).mean():.1f}%")
    top = arr["route_id"].value_counts().head(5)
    print("  most arrivals: " + ", ".join(f"{pats[r].short} ({n:,})" for r, n in top.items()))


# ============================ commands ============================
def read_clean(path):
    df = pd.read_csv(path, usecols=["ts", "vehicle_id", "route_id", "trip_id", "lat", "lon", "flag"],
                     dtype={"vehicle_id": str, "route_id": str, "trip_id": str, "flag": str}, low_memory=False)
    df["trip_id"] = df["trip_id"].fillna("")              # a missing trip_id must not drop the ping
    return df.dropna(subset=["vehicle_id", "route_id", "ts", "lat", "lon"])


def run_file(path, pats, a, outdir):
    day = Path(path).name.split(".")[0]
    print(f"\n######## {day} ########")
    say(1, 5, "LOAD", f"reading {path}")
    df = read_clean(path)
    if a.routes:
        df = df[df["route_id"].isin(a.routes.split(","))]
    p, runs, arr, expected = process(df, pats, a)
    if runs.empty:
        print("nothing to save"); return None
    outdir.mkdir(parents=True, exist_ok=True)
    say(5, 5, "SAVE", f"writing files into {outdir}/")
    arr.to_csv(outdir / f"{day}_arrivals.csv.gz", index=False)
    runs.to_csv(outdir / f"{day}_runs.csv.gz", index=False)
    print(f"      {outdir / (day + '_arrivals.csv.gz')}  ({len(arr):,} rows)\n      {outdir / (day + '_runs.csv.gz')}  ({len(runs):,} rows)")
    report(df, p, runs, arr, expected, pats)
    return arr


def cmd_run(a):
    files = sorted((Path(a.data) / "clean").glob("*.csv.gz"))
    if a.date:
        files = [f for f in files if f.name.startswith(a.date)]
    if not files:
        raise SystemExit(f"No clean files found in {a.data}/clean/ (date filter: {a.date}).")
    pats = load_patterns(a.gtfs)
    print(f"      {len(pats):,} route_ids have a usable stop list")
    for f in files:
        run_file(f, pats, a, Path(a.data) / "arrivals")


def cmd_show(a):
    files = sorted((Path(a.data) / "clean").glob("*.csv.gz"))
    if a.date:
        files = [f for f in files if f.name.startswith(a.date)]
    pats = load_patterns(a.gtfs)
    for f in files:
        df = read_clean(f); df = df[df["vehicle_id"] == a.vehicle]
        if df.empty:
            continue
        p, runs, arr, _ = process(df, pats, a, quiet=True)
        print(f"\n===== {a.vehicle} on {f.name} =====")
        d = df[df["flag"].fillna("") == ""]
        d = match_pings(d, pats, a).sort_values("ts")
        for r in d.itertuples():
            pt = pats[r.route_id]
            t = time.strftime("%H:%M:%S", time.gmtime(r.ts + IST))
            print(f"{t}  route {pt.short:<12} progress {r.prog:7.0f} m of {pt.total:6.0f}  {r.perp:6.0f} m from line  {'on-route' if r.on_route else 'OFF-ROUTE'}")
        print("\nruns:"); print(runs.to_string(index=False) if not runs.empty else "none")
        print("\narrivals:")
        for r in arr.itertuples():
            print(f"  {r.arrive_ist}  stop #{r.stop_idx:<3} {pats[r.route_id].names[r.stop_idx][:30]:<30} gap {r.gap_s:>3}s  {r.quality}")
        return
    print("vehicle not found in the selected files")


# ============================ self test with simulated buses ============================
def cmd_selftest(a):
    rng = np.random.default_rng(7)
    tmp = Path(tempfile.mkdtemp()); (tmp / "gtfs").mkdir(); (tmp / "data" / "clean").mkdir(parents=True)
    R, nst, lat0, lon0 = 2500.0, 24, 28.60, 77.20                   # a curved road (radius 2.5 km) with 24 stops
    S = np.sort(np.concatenate([[0], rng.uniform(300, 520, nst - 1).cumsum()]))
    def xy(s): th = s / R; return R * np.sin(th), R * (1 - np.cos(th))
    def ll(x, y): return lat0 + y / 110540.0, lon0 + x / (111320.0 * math.cos(math.radians(lat0)))
    sx, sy = xy(S); slat, slon = ll(sx, sy)
    (tmp / "gtfs" / "routes.txt").write_text("route_id,route_short_name\n1,TESTUP\n2,TESTDOWN\n")
    (tmp / "gtfs" / "trips.txt").write_text("route_id,trip_id\n1,T1\n2,T2\n")
    (tmp / "gtfs" / "stops.txt").write_text("stop_id,stop_name,stop_lat,stop_lon\n" + "".join(f"S{k},Stop {k},{slat[k]:.6f},{slon[k]:.6f}\n" for k in range(nst)))
    (tmp / "gtfs" / "stop_times.txt").write_text("trip_id,stop_id,stop_sequence\n" + "".join(f"T1,S{k},{k + 1}\n" for k in range(nst)) + "".join(f"T2,S{k},{nst - k}\n" for k in range(nst)))
    def simulate(start, reverse=False):
        """true bus motion at 1 s steps: random speed, dwell at some stops"""
        s, v, t, truth, nxt, dwell = 0.0, 7.0, start, {}, 0, 0
        path = [(t, 0.0)]
        while s < S[-1] and t < start + 4000:
            if dwell > 0: dwell -= 1; vv = 0.0
            else:
                v = float(np.clip(v + rng.normal(0, 0.6), 2.5, 12)); vv = v
            s += vv; t += 1
            while nxt < nst and s >= S[nxt]:
                truth[nxt] = t; nxt += 1
                if rng.random() < 0.7: dwell = int(rng.integers(8, 30))
            path.append((t, s))
        return np.array(path), truth
    rows, truths = [], {}
    base = 1_800_000_000
    for b in range(30):                                              # 30 normal buses
        path, truth = simulate(base + b * 120)
        pt = base + b * 120 + rng.uniform(0, 15)
        while pt < path[-1, 0]:
            s = np.interp(pt, path[:, 0], path[:, 1]); x, y = xy(min(s, S[-1])); la, lo = ll(x + rng.normal(0, 6), y + rng.normal(0, 6))
            rows.append((pt, f"BUS{b}", "1", f"T{b}", la, lo))
            pt += rng.uniform(10, 30)
        truths[f"BUS{b}"] = truth
    path, _ = simulate(base + 5000); pt = base + 5000                # 2 buses that drive the opposite way but are labelled route 1
    for b in range(2):
        pt = base + 5000 + b * 60
        while pt < path[-1, 0] + b * 60:
            s = S[-1] - np.interp(pt - b * 60, path[:, 0], path[:, 1]); x, y = xy(max(s, 0)); la, lo = ll(x + rng.normal(0, 6), y + rng.normal(0, 6))
            rows.append((pt, f"WRONG{b}", "1", "TW", la, lo)); pt += rng.uniform(10, 30)
    for k in range(40):                                              # a bus far away from its route line (depot trip)
        rows.append((base + 9000 + k * 20, "DEPOT", "1", "TD", lat0 + 0.05, lon0 + 0.05))
    path, truth = simulate(base + 7000); truths["TWICE"] = {}        # one vehicle doing the route twice with the SAME trip_id
    for rep in range(2):
        off = rep * 4200; path, truth = simulate(base + 7000 + off)
        if rep == 0: truths["TWICE"] = dict(truth)
        pt = base + 7000 + off
        while pt < path[-1, 0]:
            s = np.interp(pt, path[:, 0], path[:, 1]); x, y = xy(min(s, S[-1])); la, lo = ll(x + rng.normal(0, 6), y + rng.normal(0, 6))
            rows.append((pt, "TWICE", "1", "TT", la, lo)); pt += rng.uniform(10, 30)
    df = pd.DataFrame(rows, columns=["ts", "vehicle_id", "route_id", "trip_id", "lat", "lon"]); df["flag"] = ""
    df.to_csv(tmp / "data" / "clean" / "2030-01-01.csv.gz", index=False)
    print(f"Simulated {len(df):,} pings: 30 normal buses, 2 buses driving against their route label, 1 depot bus, 1 bus doing 2 trips with the same trip_id.")
    a.data, a.gtfs, a.routes = str(tmp / "data"), str(tmp / "gtfs"), None
    pats = load_patterns(a.gtfs)
    arr = run_file(tmp / "data" / "clean" / "2030-01-01.csv.gz", pats, a, tmp / "data" / "arrivals")
    # compare with the truth
    errs = []; first_runs = {}
    for r in arr.itertuples():
        tt = truths.get(r.vehicle_id, {}).get(r.stop_idx)
        if tt is not None and r.vehicle_id != "TWICE":
            errs.append(r.arrive_ts - tt)
    errs = np.array(errs)
    tw = arr[arr["vehicle_id"] == "TWICE"]
    print("\n================ SELF-TEST RESULT (simulated data, known truth) ================")
    print(f"compared {len(errs):,} detected arrivals with the true pass times of 30 normal buses")
    print(f"  error: mean {errs.mean():+.1f} s | median absolute {np.median(abs(errs)):.1f} s | 90th percentile {np.percentile(abs(errs), 90):.1f} s | worst {abs(errs).max():.0f} s")
    print(f"  buses driving against their label produced {int((arr['vehicle_id'].str.startswith('WRONG')).sum())} arrivals (should be 0)")
    print(f"  depot bus produced {int((arr['vehicle_id'] == 'DEPOT').sum())} arrivals (should be 0)")
    print(f"  same trip_id used twice was cut into {tw['run_id'].nunique()} runs (should be 2)")
    print("These numbers are for SIMULATED buses with GPS noise of about 6 m and pings every 10-30 s. Real accuracy has to be checked with real data.")


def main():
    ap = argparse.ArgumentParser(description="Detect bus stop arrivals from collected pings")
    ap.add_argument("cmd", choices=["run", "show", "selftest"])
    ap.add_argument("--data", default="data"); ap.add_argument("--gtfs", default="gtfs")
    ap.add_argument("--date"); ap.add_argument("--routes", help="comma separated route_ids"); ap.add_argument("--vehicle")
    ap.add_argument("--on-route", type=float, default=ON_ROUTE_M); ap.add_argument("--max-gap", type=float, default=MAX_GAP_S)
    ap.add_argument("--run-gap", type=float, default=RUN_GAP_S)
    a = ap.parse_args()
    if a.cmd == "show" and not a.vehicle:
        raise SystemExit("show needs --vehicle VEHICLE_ID")
    {"run": cmd_run, "show": cmd_show, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
