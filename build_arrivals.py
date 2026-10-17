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
  python build_arrivals.py run                          process only NEW or CHANGED days found in data/clean/
  python build_arrivals.py run --date 2026-10-11        one day;   --force  redo days that are up to date;   --routes 2097,1957  only these route_ids
  python build_arrivals.py run --no-shapes              straight lines between stops instead of the real road shape (shapes.txt is the default now)
  python build_arrivals.py show --vehicle DL1PC1234     step-by-step view of one bus (pings -> progress -> arrivals)
"""
import argparse, json, math, pickle, sys, tempfile, time
from pathlib import Path
import numpy as np
import pandas as pd

# ---- settings (can also be changed with command-line options) ----
ON_ROUTE_M = 100      # a ping within this distance of the route line counts as "on the route"
BACK_M = 250          # progress falling back by more than this = a new trip started
RUN_GAP_S = 300       # no on-route ping for this long = the bus run is cut here
MAX_GAP_S = 120       # arrival is only kept if the two pings around the stop are at most this far apart in time
VMAX = 20.0           # m/s: progress cannot jump faster than this (72 km/h) between two pings
SLACK_M = 100         # extra metres allowed on top of that for GPS noise
IST = 19800


def settings(a):
    return {"version": 2, "shapes": bool(a.shapes), "on_route": a.on_route, "max_gap": a.max_gap, "run_gap": a.run_gap, "vmax": a.vmax}


def say(i, n, title, plain):
    print(f"\n[{i}/{n}] {title}\n      {plain}", flush=True)


# ============================ GTFS -> route lines ============================
class Pattern:
    """One route_id: its stops, and the line the bus follows (straight lines between stops, or the real shape).
    progress = metres along that line.  stop_prog = progress of each stop."""
    def __init__(self, route_id, short, stop_ids, stops, shape=None):
        pts = [(s,) + tuple(stops[s]) for s in stop_ids if s in stops]
        self.route_id, self.short = route_id, short
        self.stop_ids = [p[0] for p in pts]
        self.names = [p[1] for p in pts]
        slat = np.array([p[2] for p in pts], float); slon = np.array([p[3] for p in pts], float)
        self.ok = len(pts) >= 2
        if not self.ok:
            return
        self.geometry = "stops"
        plat, plon = slat, slon
        if shape is not None and len(shape[0]) >= 2:
            plat, plon, self.geometry = np.asarray(shape[0], float), np.asarray(shape[1], float), "shape"
        self.lat0, self.lon0 = float(plat.mean()), float(plon.mean())
        self.kx = 111320.0 * math.cos(math.radians(self.lat0))
        self.x, self.y = (plon - self.lon0) * self.kx, (plat - self.lat0) * 110540.0
        self.seglen = np.hypot(np.diff(self.x), np.diff(self.y))
        self.cum = np.concatenate([[0.0], np.cumsum(self.seglen)])           # progress of each line vertex
        self.total = float(self.cum[-1])
        self.stop_prog = self.cum.copy() if self.geometry == "stops" else self._stop_progress(slat, slon)

    def _stop_progress(self, slat, slon):
        """Progress of each stop on the shape line. Stops are placed in order (never behind the previous stop)."""
        ax, ay, bx, by = self.x[:-1], self.y[:-1], self.x[1:], self.y[1:]
        dx, dy = bx - ax, by - ay
        l2 = np.where(dx * dx + dy * dy == 0, 1.0, dx * dx + dy * dy)
        out, prev = [], 0.0
        for lat, lon in zip(slat, slon):
            X, Y = (lon - self.lon0) * self.kx, (lat - self.lat0) * 110540.0
            t = np.clip(((X - ax) * dx + (Y - ay) * dy) / l2, 0, 1)
            d = np.hypot(X - (ax + t * dx), Y - (ay + t * dy))
            d = np.where(self.cum[1:] >= prev - 50, d, np.inf)
            i = int(d.argmin()) if np.isfinite(d).any() else len(d) - 1
            prev = max(prev, float(self.cum[i] + t[i] * self.seglen[i]))
            out.append(prev)
        return np.array(out)

    def project_many(self, lat, lon, chunk=None):
        """For many pings: (distance to the line in m, progress along the line in m)."""
        px, py = (lon - self.lon0) * self.kx, (lat - self.lat0) * 110540.0
        ax, ay, bx, by = self.x[:-1], self.y[:-1], self.x[1:], self.y[1:]
        dx, dy = bx - ax, by - ay
        l2 = np.where(dx * dx + dy * dy == 0, 1.0, dx * dx + dy * dy)
        dist, prog = np.empty(len(px)), np.empty(len(px))
        chunk = chunk or max(200, int(2_000_000 / max(len(ax), 1)))      # at most about 2 million numbers per temporary array
        for s in range(0, len(px), chunk):
            X, Y = px[s:s + chunk, None], py[s:s + chunk, None]
            t = np.clip(((X - ax) * dx + (Y - ay) * dy) / l2, 0, 1)
            d = np.hypot(X - (ax + t * dx), Y - (ay + t * dy))
            i = d.argmin(1); r = np.arange(len(i))
            dist[s:s + chunk] = d[r, i]
            prog[s:s + chunk] = self.cum[i] + t[r, i] * self.seglen[i]
        return dist, prog


def load_patterns(gtfs, use_shapes=False):
    gtfs = Path(gtfs)
    need = ["routes.txt", "trips.txt", "stops.txt", "stop_times.txt"]
    for f in need:
        if not (gtfs / f).exists():
            raise SystemExit(f"Missing {gtfs / f}. Put your GTFS .txt files in '{gtfs}/' (or use --gtfs).")
    if use_shapes:
        has_col = "shape_id" in pd.read_csv(gtfs / "trips.txt", nrows=0).columns
        if not (gtfs / "shapes.txt").exists() or not has_col:
            print("      WARNING: --shapes needs shapes.txt and a shape_id column in trips.txt. Using straight lines between stops instead.")
            use_shapes = False
        else:
            need.append("shapes.txt")
    sig = [(f, (gtfs / f).stat().st_mtime, (gtfs / f).stat().st_size) for f in need]
    cache = gtfs / (".arrival_patterns_shapes.pkl" if use_shapes else ".arrival_patterns.pkl")
    d = None
    if cache.exists():
        try:
            d = pickle.load(open(cache, "rb"))
            if d["sig"] != sig:
                d = None
        except Exception:
            d = None
    if d is None:
        print("      first time: reading stop_times.txt" + (" and shapes.txt" if use_shapes else "") + " (can take a few minutes, then it is cached) ...", flush=True)
        routes = pd.read_csv(gtfs / "routes.txt", dtype=str)
        short = dict(zip(routes["route_id"], routes["route_short_name"])) if "route_short_name" in routes else {}
        sdf = pd.read_csv(gtfs / "stops.txt", dtype={"stop_id": str})
        sdf["stop_lat"] = pd.to_numeric(sdf["stop_lat"], errors="coerce"); sdf["stop_lon"] = pd.to_numeric(sdf["stop_lon"], errors="coerce")
        sdf = sdf.dropna(subset=["stop_lat", "stop_lon"])
        stops = {r.stop_id: (str(r.stop_name), float(r.stop_lat), float(r.stop_lon)) for r in sdf.itertuples()}
        trips = pd.read_csv(gtfs / "trips.txt", dtype=str, usecols=["route_id", "trip_id"] + (["shape_id"] if use_shapes else []))
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
        shp = {}
        if use_shapes:
            ws, parts = set(rep["shape_id"].dropna()), []
            for ch in pd.read_csv(gtfs / "shapes.txt", usecols=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"], dtype=str, chunksize=1_000_000):
                parts.append(ch[ch["shape_id"].isin(ws)])
            sh = pd.concat(parts)
            for c in ("shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"):
                sh[c] = pd.to_numeric(sh[c], errors="coerce")
            sh = sh.dropna().sort_values(["shape_id", "shape_pt_sequence"])
            shp = {k: (g["shape_pt_lat"].values, g["shape_pt_lon"].values) for k, g in sh.groupby("shape_id")}
        raw = {r.route_id: (short.get(r.route_id, r.route_id), seq.get(r.trip_id, []), shp.get(getattr(r, "shape_id", None))) for r in rep.itertuples()}
        d = {"sig": sig, "stops": stops, "raw": raw}
        pickle.dump(d, open(cache, "wb"))
    pats = {}
    for rid, tup in d["raw"].items():
        sh_, sids, *rest = tup
        p = Pattern(rid, sh_, sids, d["stops"], rest[0] if rest else None)
        if p.ok:
            pats[rid] = p
    n_shape = sum(1 for p in pats.values() if p.geometry == "shape")
    print(f"      route line: {'real road shape for %d route_ids, straight lines between stops for the rest' % n_shape if use_shapes else 'straight lines between stops'}")
    return pats


# ============================ the 4 processing steps ============================
def match_pings(df, pats, a, verbose=True):
    """Step 2: distance to route line + progress along the route for every ping."""
    df = df[df["route_id"].isin(pats)].reset_index(drop=True)
    lat, lon = df["lat"].values, df["lon"].values
    perp, prog = np.full(len(df), np.nan), np.full(len(df), np.nan)
    groups = df.groupby("route_id").indices
    step = max(1, len(groups) // 4)
    for k, (rid, ix) in enumerate(groups.items(), 1):
        perp[ix], prog[ix] = pats[rid].project_many(lat[ix], lon[ix])
        if verbose and (k % step == 0 or k == len(groups)):
            print(f"      matched {k}/{len(groups)} route_ids", flush=True)
    df["perp"], df["prog"] = perp, prog
    df["on_route"] = df["perp"] <= a.on_route
    return df


def build_runs(df, a):
    """Step 3: cut pings into runs, drop single-ping 'spikes'. Returns (on-route pings with run_id, table of runs, number of spikes)."""
    allk = df.groupby(["vehicle_id", "route_id", "trip_id"]).agg(n_all=("ts", "size"), n_off=("on_route", lambda s: int((~s).sum())))
    p = df[df["on_route"]].sort_values(["vehicle_id", "route_id", "trip_id", "ts"])
    p = p.drop_duplicates(["vehicle_id", "route_id", "trip_id", "ts"]).reset_index(drop=True)
    ts, pr = p["ts"].values, p["prog"].values
    run_id = np.full(len(p), -1); runs = []; rid = 0; spikes = 0
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
        segs, cur, pend = [], [0], []
        for k in range(1, len(ix)):
            last = cur[-1]; dt = t[k] - t[last]; dg = g[k] - g[last]
            if dt > a.run_gap:                                           # long silence: new run
                spikes += len(pend); segs.append(cur); cur, pend = [k], []
                continue
            if dg > a.vmax * dt + SLACK_M or dg < -BACK_M:               # not believable for the same trip
                if pend and abs(g[k] - g[pend[-1]]) > a.vmax * (t[k] - t[pend[-1]]) + SLACK_M:
                    spikes += len(pend); pend = []
                pend.append(k)
                if len(pend) >= 3:                                       # 3 pings agree with each other: a genuinely new trip
                    segs.append(cur); cur, pend = list(pend), []
                continue
            spikes += len(pend); pend = []                               # earlier suspicious pings were one-off spikes
            cur.append(k)
        spikes += len(pend); segs.append(cur)
        for seg_local in segs:
            seg = ix[seg_local]; run_id[seg] = rid; gg = pr[seg]
            runs.append(dict(run_id=rid, **base, t_start=ts[seg[0]], t_end=ts[seg[-1]], n_pings=len(seg), prog_start=gg[0], prog_end=gg.max(),
                             direction_ok=True, usable=len(seg) >= 3 and gg.max() - gg[0] >= 200))
            rid += 1
    p["run_id"] = run_id
    return p[p["run_id"] >= 0].reset_index(drop=True), pd.DataFrame(runs), spikes


def find_arrivals(p, runs, pats, a):
    """Step 4: where a run goes from 'before the stop' to 'after the stop', interpolate the time it passed."""
    ts, pr = p["ts"].values, p["prog"].values
    idx_by_run = p.groupby("run_id").indices
    cols = {k: [] for k in ("run_id", "stop_idx", "t", "gap", "span")}
    expected = 0
    for r in runs[runs["usable"]].itertuples():
        ix = np.sort(idx_by_run[r.run_id]); t = ts[ix]
        pm = np.maximum.accumulate(pr[ix])                  # progress never goes backwards (removes GPS wobble)
        P = pats[r.route_id].stop_prog
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
    df = match_pings(df, pats, a, verbose=not quiet)
    if df.empty:
        return df, df, df, 0, df, 0
    if not quiet:
        print(f"      {df['on_route'].mean() * 100:.1f}% of {len(df):,} pings are within {a.on_route:.0f} m of their route line (the rest are depot trips, diversions, wrong route_id or bad GPS)")
        say(3, 5, "RUNS", "Each bus's pings are cut into trips; one-off position spikes are removed; a trip that moves backwards along its route is marked as a direction mismatch.")
    p, runs, spikes = build_runs(df, a)
    if not quiet:
        say(4, 5, "ARRIVALS", "For every stop between a bus's first and last ping, the pass time is interpolated between the two pings around the stop.")
    arr, expected = find_arrivals(p, runs, pats, a)
    return p, runs, arr, expected, df, spikes


def report(dfm, p, runs, arr, expected, spikes, pats):
    print("\n================ QUALITY REPORT ================")
    if runs.empty:
        print("no runs found (are the route_ids in your data present in the GTFS?)"); return
    bins = [-1, 50, 100, 200, 500, 1000, np.inf]; names = ["<=50 m", "50-100", "100-200", "200-500", "500-1000", ">1 km"]
    sh = pd.cut(dfm["perp"], bins, labels=names).value_counts(normalize=True).reindex(names).fillna(0)
    print("pings by distance from their route line: " + " | ".join(f"{n} {100 * v:.1f}%" for n, v in sh.items()))
    print("   (many pings at 100-500 m = the line is too rough, try --shapes;  many above 1 km = depot trips or wrong route_id)")
    g = dfm.groupby("route_id").agg(n=("perp", "size"), on=("on_route", "mean"))
    g = g[g["n"] >= 100].sort_values("on").head(5)
    print("routes with the lowest share of pings on the line: " + ", ".join(f"{pats[r].short} ({100 * v.on:.0f}% of {int(v.n):,})" for r, v in g.iterrows()))
    print(f"one-off position spikes removed: {spikes:,} pings")
    u = runs["usable"].sum(); bw = (~runs["direction_ok"]).sum()
    print(f"runs: {len(runs):,}  usable {u:,}  | moving against their route_id direction: {bw:,} ({100 * bw / len(runs):.1f}%)  | too short: {len(runs) - u - bw:,}")
    if arr.empty:
        print("no arrivals detected"); return
    print(f"arrivals: {len(arr):,} for {arr['vehicle_id'].nunique():,} buses on {arr['route_id'].nunique():,} route_ids")
    print("  quality (gap between the two pings around the stop):  " + ", ".join(f"{k} {v:,}" for k, v in arr["quality"].value_counts().items()))
    print(f"  coverage: {100 * len(arr) / max(expected, 1):.1f}% of the stops each bus drove past got an arrival time (rest skipped: pings too far apart)")
    print(f"  median gap between the pings used: {arr['gap_s'].median():.0f} s, median speed around the stop: {arr['speed_kmh'].median():.1f} km/h")
    a2 = arr.copy(); a2["pat_cum"] = [pats[r].stop_prog[k] for r, k in zip(a2["route_id"], a2["stop_idx"])]
    a2["next_t"] = a2.groupby("run_id")["arrive_ts"].shift(-1); a2["next_cum"] = a2.groupby("run_id")["pat_cum"].shift(-1)
    s = a2.dropna(subset=["next_t"]).copy(); dt = s["next_t"] - s["arrive_ts"]
    s["v"] = np.where(dt > 0, (s["next_cum"] - s["pat_cum"]) / dt.where(dt > 0, np.nan) * 3.6, np.nan)
    sus = s["v"] > 60
    print(f"  stop-to-stop travel: {len(s):,} pairs | median speed {s['v'].median():.1f} km/h | suspicious (>60 km/h): {100 * sus.mean():.1f}% | zero/negative time: {100 * s['v'].isna().mean():.1f}%")
    if sus.any():
        byq = s.groupby("quality")["v"].apply(lambda x: 100 * (x > 60).mean()).round(1).to_dict()
        circ = {r: math.hypot(p_.x[0] - p_.x[-1], p_.y[0] - p_.y[-1]) < 300 for r, p_ in pats.items()}
        sc = s[sus]["route_id"].map(circ).mean() * 100
        top = s[sus]["route_id"].value_counts().head(5)
        print(f"     suspicious share by quality: {byq}   | on circular routes: {sc:.0f}% of the suspicious pairs")
        print("     routes with most suspicious pairs: " + ", ".join(f"{pats[r].short} ({n:,})" for r, n in top.items()))
    top = arr["route_id"].value_counts().head(5)
    print("  most arrivals: " + ", ".join(f"{pats[r].short} ({n:,})" for r, n in top.items()))


# ============================ commands ============================
def read_clean(path):
    df = pd.read_csv(path, usecols=["ts", "vehicle_id", "route_id", "trip_id", "lat", "lon", "flag"],
                     dtype={"vehicle_id": str, "route_id": str, "trip_id": str, "flag": str}, low_memory=False)
    df["trip_id"] = df["trip_id"].fillna("")              # a missing trip_id must not drop the ping
    return df.dropna(subset=["vehicle_id", "route_id", "ts", "lat", "lon"])


def run_file(path, pats, a, outdir, quiet=False):
    day = Path(path).name.split(".")[0]
    if not quiet:
        print(f"\n######## {day} ########")
        say(1, 5, "LOAD", f"reading {path}")
    df = read_clean(path)
    if a.routes:
        df = df[df["route_id"].isin(a.routes.split(","))]
    p, runs, arr, expected, dfm, spikes = process(df, pats, a, quiet=quiet)
    if runs.empty:
        print("nothing to save"); return None
    outdir.mkdir(parents=True, exist_ok=True)
    arr.to_csv(outdir / f"{day}_arrivals.csv.gz", index=False)
    runs.to_csv(outdir / f"{day}_runs.csv.gz", index=False)
    (outdir / f"{day}_settings.json").write_text(json.dumps(settings(a)))
    if not quiet:
        say(5, 5, "SAVE", f"writing files into {outdir}/")
        print(f"      {outdir / (day + '_arrivals.csv.gz')}  ({len(arr):,} rows)\n      {outdir / (day + '_runs.csv.gz')}  ({len(runs):,} rows)")
        report(dfm, p, runs, arr, expected, spikes, pats)
    return arr, runs, dfm, spikes


def cmd_run(a):
    files = sorted((Path(a.data) / "clean").glob("*.csv.gz"))
    if a.date:
        files = [f for f in files if f.name.startswith(a.date)]
    if not files:
        raise SystemExit(f"No clean files found in {a.data}/clean/ (date filter: {a.date}).")
    outdir = Path(a.data) / "arrivals"
    todo = []
    for f in files:
        day = f.name.split('.')[0]
        out, meta = outdir / f"{day}_arrivals.csv.gz", outdir / f"{day}_settings.json"
        same = False
        if meta.exists():
            try:
                same = json.loads(meta.read_text()) == settings(a)
            except Exception:
                same = False
        if out.exists() and out.stat().st_mtime >= f.stat().st_mtime and same and not a.force and not a.routes:
            print(f"skip {f.name}: arrivals are already up to date with the same settings (use --force to redo)")
        else:
            why = "new day" if not out.exists() else ("data changed" if out.stat().st_mtime < f.stat().st_mtime else "different settings or older version" if not same else "forced")
            print(f"{day}: will be processed ({why})")
            todo.append(f)
    if not todo:
        print("nothing new to process."); return
    pats = load_patterns(a.gtfs, a.shapes)
    print(f"      {len(pats):,} route_ids have a usable stop list")
    for f in todo:
        run_file(f, pats, a, outdir)


def cmd_show(a):
    files = sorted((Path(a.data) / "clean").glob("*.csv.gz"))
    if a.date:
        files = [f for f in files if f.name.startswith(a.date)]
    pats = load_patterns(a.gtfs, a.shapes)
    for f in files:
        df = read_clean(f); df = df[df["vehicle_id"] == a.vehicle]
        if df.empty:
            continue
        p, runs, arr, _, _, _ = process(df, pats, a, quiet=True)
        print(f"\n===== {a.vehicle} on {f.name} =====")
        d = df[df["flag"].fillna("") == ""]
        d = match_pings(d, pats, a, verbose=False).sort_values("ts")
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
    ss = np.arange(0, S[-1] + 25, 25.0); hx, hy = xy(ss); hlat, hlon = ll(hx, hy)      # the real road shape: a point every 25 m
    g = tmp / "gtfs"
    (g / "routes.txt").write_text("route_id,route_short_name\n1,TESTUP\n2,TESTDOWN\n")
    (g / "trips.txt").write_text("route_id,trip_id,shape_id\n1,T1,SH1\n2,T2,SH2\n")
    (g / "stops.txt").write_text("stop_id,stop_name,stop_lat,stop_lon\n" + "".join(f"S{k},Stop {k},{slat[k]:.6f},{slon[k]:.6f}\n" for k in range(nst)))
    (g / "stop_times.txt").write_text("trip_id,stop_id,stop_sequence\n" + "".join(f"T1,S{k},{k + 1}\n" for k in range(nst)) + "".join(f"T2,S{k},{nst - k}\n" for k in range(nst)))
    (g / "shapes.txt").write_text("shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\n" + "".join(f"SH1,{la:.6f},{lo:.6f},{i}\n" for i, (la, lo) in enumerate(zip(hlat, hlon)))
                                  + "".join(f"SH2,{la:.6f},{lo:.6f},{i}\n" for i, (la, lo) in enumerate(zip(hlat[::-1], hlon[::-1]))))
    def simulate(start):
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
    rows, truths, nspike = [], {}, 0
    base = 1_800_000_000
    for b in range(30):                                              # 30 normal buses (1% of pings are position spikes)
        path, truth = simulate(base + b * 120)
        pt = base + b * 120 + rng.uniform(0, 15)
        while pt < path[-1, 0]:
            s = np.interp(pt, path[:, 0], path[:, 1])
            if rng.random() < 0.01: s = s + 900 if s < S[-1] - 900 else s - 900; nspike += 1
            x, y = xy(min(max(s, 0), S[-1])); la, lo = ll(x + rng.normal(0, 6), y + rng.normal(0, 6))
            rows.append((pt, f"BUS{b}", "1", f"T{b}", la, lo)); pt += rng.uniform(10, 30)
        truths[f"BUS{b}"] = truth
    path, _ = simulate(base + 5000)
    for b in range(2):                                               # 2 buses that drive the opposite way but are labelled route 1
        pt = base + 5000 + b * 60
        while pt < path[-1, 0] + b * 60:
            s = S[-1] - np.interp(pt - b * 60, path[:, 0], path[:, 1]); x, y = xy(max(s, 0)); la, lo = ll(x + rng.normal(0, 6), y + rng.normal(0, 6))
            rows.append((pt, f"WRONG{b}", "1", "TW", la, lo)); pt += rng.uniform(10, 30)
    for k in range(40):                                              # a bus far away from its route line (depot trip)
        rows.append((base + 9000 + k * 20, "DEPOT", "1", "TD", lat0 + 0.05, lon0 + 0.05))
    for rep in range(2):                                             # one vehicle doing the route twice with the SAME trip_id
        off = rep * 4200; path, truth = simulate(base + 7000 + off)
        if rep == 0: truths["TWICE"] = dict(truth)
        pt = base + 7000 + off
        while pt < path[-1, 0]:
            s = np.interp(pt, path[:, 0], path[:, 1]); x, y = xy(min(s, S[-1])); la, lo = ll(x + rng.normal(0, 6), y + rng.normal(0, 6))
            rows.append((pt, "TWICE", "1", "TT", la, lo)); pt += rng.uniform(10, 30)
    df = pd.DataFrame(rows, columns=["ts", "vehicle_id", "route_id", "trip_id", "lat", "lon"]); df["flag"] = ""
    path_clean = tmp / "data" / "clean" / "2030-01-01.csv.gz"; df.to_csv(path_clean, index=False)
    print(f"Simulated {len(df):,} pings: 30 normal buses ({nspike} pings moved 900 m = spikes), 2 buses against their label, 1 depot bus, 1 bus with 2 trips under one trip_id.")
    a.routes = None; res = {}
    for mode in ("stops", "shapes"):
        pats = load_patterns(str(g), mode == "shapes")
        arr, runs, dfm, spikes = run_file(path_clean, pats, a, tmp / "data" / f"arrivals_{mode}", quiet=(mode == "shapes"))
        errs = np.array([r.arrive_ts - truths[r.vehicle_id][r.stop_idx] for r in arr.itertuples() if r.vehicle_id.startswith("BUS") and r.stop_idx in truths[r.vehicle_id]])
        res[mode] = dict(on=100 * dfm["on_route"].mean(), n=len(errs), mean=errs.mean(), med=np.median(abs(errs)), p90=np.percentile(abs(errs), 90), worst=abs(errs).max(),
                         wrong=int(arr["vehicle_id"].str.startswith("WRONG").sum()), depot=int((arr["vehicle_id"] == "DEPOT").sum()),
                         twice=arr[arr["vehicle_id"] == "TWICE"]["run_id"].nunique(), spikes=spikes)
    print("\n================ SELF-TEST RESULT (simulated data, known truth) ================")
    print(f"{'route line used':<34}{'on line':>9}{'arrivals':>10}{'mean err':>10}{'median |err|':>14}{'p90 |err|':>11}{'worst':>7}")
    for m, lab in (("stops", "straight lines between stops"), ("shapes", "real road shape (--shapes)")):
        r = res[m]; print(f"{lab:<34}{r['on']:>8.1f}%{r['n']:>10}{r['mean']:>+9.1f}s{r['med']:>13.1f}s{r['p90']:>10.1f}s{r['worst']:>6.0f}s")
    r = res["shapes"]
    print(f"checks (both modes): wrong-direction buses gave {res['stops']['wrong']}/{r['wrong']} arrivals (should be 0) | depot bus {res['stops']['depot']}/{r['depot']} (0) | same trip_id twice -> {res['stops']['twice']}/{r['twice']} runs (2)")
    print(f"position spikes injected: {nspike} | removed by the spike filter: {res['stops']['spikes']} (stops mode) / {r['spikes']} (shapes mode)")
    print("These numbers are for SIMULATED buses with GPS noise of about 6 m and pings every 10-30 s. Real accuracy has to be checked with real data.")


def main():
    ap = argparse.ArgumentParser(description="Detect bus stop arrivals from collected pings")
    ap.add_argument("cmd", choices=["run", "show", "selftest"])
    ap.add_argument("--data", default="data"); ap.add_argument("--gtfs", default="gtfs")
    ap.add_argument("--date"); ap.add_argument("--routes", help="comma separated route_ids"); ap.add_argument("--vehicle")
    ap.add_argument("--on-route", type=float, default=ON_ROUTE_M); ap.add_argument("--max-gap", type=float, default=MAX_GAP_S)
    ap.add_argument("--run-gap", type=float, default=RUN_GAP_S)
    ap.add_argument("--vmax", type=float, default=VMAX, help="fastest believable bus speed in m/s for the spike filter")
    ap.add_argument("--shapes", action="store_true", help="(default now) use shapes.txt, the real road line")
    ap.add_argument("--no-shapes", action="store_true", help="use straight lines between stops instead of shapes.txt")
    ap.add_argument("--force", action="store_true", help="redo days whose arrivals are already up to date")
    a = ap.parse_args()
    a.shapes = not a.no_shapes
    if a.cmd == "show" and not a.vehicle:
        raise SystemExit("show needs --vehicle VEHICLE_ID")
    {"run": cmd_run, "show": cmd_show, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
