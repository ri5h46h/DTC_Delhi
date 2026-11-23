#!/usr/bin/env python3
"""
live_eta.py - the AI ETA for the live bus explorer.  STAGE A: the engine only (no screens yet).

What it does, every time the explorer gets a new feed snapshot:
  1. takes the new pings of each bus (same rules as otd_collector.py: no duplicates, GPS jumps dropped),
  2. puts every ping on its route line with the SAME shape-based progress as build_arrivals.py,
  3. cuts the pings of a bus into runs and removes one-off spikes with the SAME rules and numbers as build_arrivals.py
     (but step by step, because in live mode the future pings are not known),
  4. detects stop arrivals the same way as build_arrivals.py (interpolates the time the bus passed the stop),
  5. keeps the "recent buses" numbers: observed / normal stop-to-stop time of all buses in the last 10 min (rc10)
     and of the same route in the last 20 min (rr20), the same definition as train_eta_v2.py,
  6. builds the model inputs of a bus for its next 1..6 stops with the training code (train_eta_v2.add_feats),
  7. predicts the ETA and the 10-90% range with the saved v2 models (point model, low model, high model, calibration).

Code that is REUSED, not rewritten (keep these files in the same folder):
  build_arrivals.py : Pattern (route line, stop progress, project_many) and the constants ON_ROUTE_M, BACK_M, RUN_GAP_S, MAX_GAP_S, VMAX, SLACK_M
  train_eta_v2.py   : FEATURES, add_feats (history ETA, time-of-day factor, stop inputs), hist_feats
  Small rules that had to be written again for live use are marked "same as ..." in the comments, and replay_eta.py checks them.

Usage from the explorer (Stage B):
    eng = LiveETA.load(models_dir="models", gtfs_dir="gtfs")
    eng.update(fleet_rows, now=time.time(), source="tstack")        # once per feed snapshot (rows = the flattened vehicle rows)
    table = eng.predict(now=time.time())                              # one row per bus and next stop (1..6 stops ahead)
"""
import bisect, json, math, pickle, threading
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import build_arrivals as ba
    import train_eta_v2 as t2
except ImportError:
    raise SystemExit("Keep build_arrivals.py, train_eta.py and train_eta_v2.py in the same folder as live_eta.py.")

FEATURES = t2.FEATURES
HMAX = 6                                     # the models were trained for 1..6 stops ahead
# ---- rules of otd_collector.py (same as there; the collector checks pings before they are saved) ----
BBOX = (28.20, 29.10, 76.60, 77.70)          # lat_min, lat_max, lon_min, lon_max
MAX_AGE_S = 600                              # ping older than this = bus not live
JUMP_KMH = 100                               # implied speed above this (and moved > 100 m) = gps_jump -> the ping is dropped by build_arrivals.py
IST = 19800
WARM_MIN_EVENTS = 100                        # the recent-buses numbers count as "warm" when at least this many stop-to-stop times of all buses are in the last 10 min


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def hav(lat1, lon1, lat2, lon2):             # same as otd_collector.py
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(a))


# ============================================================ route lines
def load_patterns(gtfs_dir="gtfs", cache=None):
    """Route lines with the real road shapes, exactly as build_arrivals.py makes them.
    With the GTFS folder present this simply calls build_arrivals.load_patterns(use_shapes=True).
    If stop_times.txt is not there (for example in a test), the cache file .arrival_patterns_shapes.pkl is read directly."""
    gtfs = Path(gtfs_dir)
    if cache is None and (gtfs / "stop_times.txt").exists():
        return ba.load_patterns(gtfs, True)
    cache = Path(cache) if cache else gtfs / ".arrival_patterns_shapes.pkl"
    d = pickle.load(open(cache, "rb"))
    pats = {}
    for rid, tup in d["raw"].items():            # same loop as the end of build_arrivals.load_patterns
        sh_, sids, *rest = tup
        p = ba.Pattern(rid, sh_, sids, d["stops"], rest[0] if rest else None)
        if p.ok:
            pats[rid] = p
    return pats


# ============================================================ recent-buses numbers
class EventStore:
    """Observed / normal stop-to-stop time ("ratio", above 1 = slower than normal) and the time the bus reached the second stop.
    Kept sorted by that time. window(ts, w) = mean ratio and count of the events with  ts - w <= t_end < ts  (same as train_eta_v2.window_mean)."""
    def __init__(self, keep_s=3600.0):
        self.t, self.r, self.keep, self._adds = [], [], keep_s, 0

    def add(self, t_end, ratio):
        i = bisect.bisect_right(self.t, t_end)
        self.t.insert(i, t_end); self.r.insert(i, ratio)
        self._adds += 1
        if self._adds % 500 == 0 and self.t and self.t[-1] - self.t[0] > 2 * self.keep:      # forget what is far older than the longest window
            cut = bisect.bisect_left(self.t, self.t[-1] - self.keep)
            del self.t[:cut]; del self.r[:cut]

    def window(self, ts, window):
        hi = bisect.bisect_left(self.t, ts)
        lo = bisect.bisect_left(self.t, ts - window)
        n = hi - lo
        return (sum(self.r[lo:hi]) / n if n else float("nan")), n

    def __len__(self):
        return len(self.t)


class Events:
    """All buses (rc10: last 10 min) and one store per route_id (rr20: last 20 min)."""
    def __init__(self, record=False, keep_s=3600.0):
        self.all, self.route, self.n = EventStore(keep_s), defaultdict(lambda: EventStore(keep_s)), 0
        self.log = [] if record else None          # (route_id, t_end, ratio) of every event, only for the replay test

    def add(self, route_id, t_end, ratio):
        self.all.add(t_end, ratio); self.route[route_id].add(t_end, ratio); self.n += 1
        if self.log is not None:
            self.log.append((route_id, t_end, ratio))

    def recent(self, route_id, ts):
        rc, rcn = self.all.window(ts, 600)
        st = self.route.get(route_id)
        rr, rrn = st.window(ts, 1200) if st is not None else (float("nan"), 0)
        return rc, rcn, rr, rrn


# ============================================================ state of one bus
class _Run:
    """One continuous trip of a bus (what build_runs() calls a run)."""
    __slots__ = ("n", "g_first", "g_max", "last_t", "last_g", "last_gr", "hv1", "hv2", "last_move", "last_arr", "pending", "released", "cur")

    def __init__(self):
        self.n, self.g_first, self.g_max = 0, None, -1e18
        self.last_t = self.last_g = self.last_gr = None
        self.hv1 = self.hv2 = float("nan")        # v1 of the previous ping and of the one before (= v2 and v3 of the next ping)
        self.last_move, self.last_arr = None, None
        self.pending, self.released = [], False   # arrivals wait here until the run is "usable" (3 pings, 200 m)
        self.cur = None                           # inputs of the latest ping of this run


class _Group:
    """All pings of one (vehicle, route_id, trip_id), like the groups of build_runs()."""
    __slots__ = ("vid", "route", "trip", "pat", "run", "pend", "fwd", "bwd", "last_on_g", "bad", "ever_bad", "spikes", "last_ts")

    def __init__(self, vid, route, trip, pat):
        self.vid, self.route, self.trip, self.pat = vid, route, trip, pat
        self.run, self.pend, self.fwd, self.bwd, self.last_on_g = None, [], 0, 0, None
        self.bad, self.ever_bad, self.spikes, self.last_ts = False, False, 0, 0.0


class LiveETA:
    def __init__(self, patterns, tables, models=None, calibration=None, record=False):
        self.pats = patterns
        self.med = tables["med"]
        self.L = t2.Learned()                     # what train_eta_v2.add_feats needs (the events are kept live, not here)
        self.L.v_global, self.L.v_route, self.L.tables, self.L.tod = tables["v_global"], tables["v_route"], tables["tables"], tables["tod"]
        self.L.events = pd.DataFrame({"route_id": pd.Series(dtype=str), "t_end": pd.Series(dtype=float), "ratio": pd.Series(dtype=float)})
        self.models = models                      # dict: backend, target, models{point,low,high}, features{...}  or None
        self.widen = (calibration or {}).get("widen_s_by_stops_ahead_1_2_3to4_5plus", [0.0, 0.0, 0.0, 0.0])
        self.events = Events(record)
        self.groups, self.by_vid, self.last_seen = {}, {}, {}
        self._p1 = {}                             # stop progress rounded to 0.1 m per route (the arrivals file that trained the model has 0.1 m)
        self.record, self.arrival_log, self.flag_log = record, [], []
        self.stats = defaultdict(int)
        self.lock = threading.RLock()
        self.last_now = 0.0

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, models_dir="models", gtfs_dir="gtfs", patterns_cache=None, with_models=True, record=False):
        md = Path(models_dir)
        tables = pickle.load(open(md / "eta_v2_tables.pkl", "rb"))
        cal = json.loads((md / "eta_v2_calibration.json").read_text()) if (md / "eta_v2_calibration.json").exists() else None
        models = None
        if with_models:
            try:
                models = pickle.load(open(md / "eta_v2_models.pkl", "rb"))        # needs lightgbm when the models were trained with it
            except ModuleNotFoundError as e:
                raise SystemExit(f"Cannot open the saved models: {e}. Install it with  uv add lightgbm  (or use with_models=False for features only).")
            if models.get("target") != "residual":
                raise SystemExit("live_eta.py supports the 'residual' target only (the default of train_eta_v2.py).")
        return cls(load_patterns(gtfs_dir, patterns_cache), tables, models, cal, record)

    # ------------------------------------------------------------ step 1: one row of the feed
    def _accept(self, r, now, source):
        """Same checks and the same gps_jump rule as otd_collector.validate(). Returns None (row not used) or a tuple."""
        lat, lon = to_float(r.get("position.latitude")), to_float(r.get("position.longitude"))
        if lat is None or lon is None or not (BBOX[0] <= lat <= BBOX[1] and BBOX[2] <= lon <= BBOX[3]):
            self.stats["bad_position"] += 1; return None
        ts = to_float(r.get("timestamp"))
        if not ts:
            self.stats["no_time"] += 1; return None
        if now is not None and (ts > now + 120 or now - ts > MAX_AGE_S):
            self.stats["stale_or_future"] += 1; return None
        vid = str(r.get("vehicle.id") or r.get("vehicle.label") or r.get("entity_id"))
        prev, flag = self.last_seen.get(vid), False
        if prev:
            if ts <= prev[0]:
                self.stats["duplicate"] += 1; return None
            dt = ts - prev[0]
            if dt <= 300:
                moved = hav(prev[1], prev[2], lat, lon)
                if moved / dt * 3.6 > JUMP_KMH and moved > 100:
                    flag = True
        self.last_seen[vid] = (ts, lat, lon)       # also for flagged pings, like the collector
        route = str(r.get("trip.route_id") or "").strip()
        trip = "" if r.get("trip.trip_id") in (None, "") else str(r.get("trip.trip_id"))
        rep = to_float(r.get("position.speed")) if source == "tstack" else None      # OTD always sends 0: not a speed
        return (vid, ts, lat, lon, route, trip, float("nan") if rep is None else rep, flag)

    # ------------------------------------------------------------ the update of one snapshot
    def update(self, rows, now=None, source="tstack"):
        """Feed the rows of one snapshot. Returns the list of (vehicle_id, route_id, trip_id) whose newest ping was added to a run."""
        with self.lock:
            self.stats["snapshots"] += 1
            self.stats["rows"] += len(rows)
            acc = []
            for r in rows:
                a = self._accept(r, now, source)
                if a is None:
                    continue
                if self.record:
                    self.flag_log.append((a[0], int(a[1]), a[7]))
                if a[7]:
                    self.stats["gps_jump"] += 1; continue                       # build_arrivals.py drops flagged pings
                acc.append(a)
            if now is not None:
                self.last_now = max(self.last_now, now)
            # progress along the route line: one call per route_id (same as build_arrivals.match_pings)
            byr = defaultdict(list)
            for i, a in enumerate(acc):
                if a[4] in self.pats:
                    byr[a[4]].append(i)
                else:
                    self.stats["unresolved_route"] += 1
            perp, prog = np.full(len(acc), np.nan), np.full(len(acc), np.nan)
            for route, ix in byr.items():
                ix = np.array(ix)
                perp[ix], prog[ix] = self.pats[route].project_many(np.array([acc[i][2] for i in ix]), np.array([acc[i][3] for i in ix]))
            touched = []
            for i in sorted(range(len(acc)), key=lambda i: acc[i][1]):         # oldest ping first
                vid, ts, lat, lon, route, trip, rep, _ = acc[i]
                if route not in self.pats:
                    continue
                if perp[i] > ba.ON_ROUTE_M:
                    self.stats["off_route"] += 1; continue
                key = (vid, route, trip)
                g = self.groups.get(key)
                if g is None:
                    g = self.groups[key] = _Group(vid, route, trip, self.pats[route])
                if self._ping(g, ts, float(prog[i]), rep):
                    g.last_ts = ts
                    self.by_vid[vid] = key
                    touched.append(key)
            if now is not None and not self.record and self.stats["snapshots"] % 40 == 0:      # (the replay test keeps everything, to be able to explain differences)
                self._prune(now)
            return touched

    def _prune(self, now):
        old = [k for k, g in self.groups.items() if now - g.last_ts > 3600]
        for k in old:
            del self.groups[k]
            if self.by_vid.get(k[0]) == k:
                del self.by_vid[k[0]]
        for v in [v for v, (t, _, _) in self.last_seen.items() if now - t > 7200]:
            del self.last_seen[v]

    # ------------------------------------------------------------ steps 2-4: runs, spikes, arrivals (same rules as build_runs / find_arrivals)
    def _ping(self, g, ts, prog, rep):
        """One on-route ping of a group. Returns True if it became part of a run."""
        if g.last_on_g is not None:                                            # direction check: counts of forward / backward steps of ALL on-route pings
            step = prog - g.last_on_g
            if step > 20: g.fwd += 1
            elif step < -20: g.bwd += 1
            g.bad = g.bwd >= 3 and g.bwd > 2 * g.fwd                            # moves AGAINST the direction of its route_id
            g.ever_bad = g.ever_bad or g.bad
        g.last_on_g = prog
        run = g.run
        if run is None:
            self._new_run(g, [(ts, prog, rep)]); return True
        dt, dg = ts - run.last_t, prog - run.last_g
        if dt > ba.RUN_GAP_S:                                                  # long silence: new run
            g.spikes += len(g.pend); g.pend = []
            self._new_run(g, [(ts, prog, rep)]); return True
        if dg > ba.VMAX * dt + ba.SLACK_M or dg < -ba.BACK_M:                  # not believable for the same trip: wait for more pings
            if g.pend and abs(prog - g.pend[-1][1]) > ba.VMAX * (ts - g.pend[-1][0]) + ba.SLACK_M:
                g.spikes += len(g.pend); g.pend = []
            g.pend.append((ts, prog, rep))
            if len(g.pend) >= 3:                                               # 3 pings agree with each other: a genuinely new trip
                seed, g.pend = g.pend, []
                self._new_run(g, seed); return True
            self.stats["pending_ping"] += 1
            return False
        g.spikes += len(g.pend); g.pend = []                                   # earlier suspicious pings were one-off spikes
        self._append(g, run, ts, prog, rep)
        return True

    def _new_run(self, g, seeds):
        run = g.run = _Run()
        for ts, prog, rep in seeds:
            self._append(g, run, ts, prog, rep)

    def _append(self, g, run, ts, prog, rep):
        gr = float(np.round(prog, 1))                                          # the saved pings file has progress rounded to 0.1 m
        v1, still = float("nan"), False
        if run.n > 0:
            dt = ts - run.last_t
            if 15 <= dt <= 120:                                                # same as calc_route_kmh in build_arrivals.py
                v1 = float(np.round((prog - run.last_g) / dt * 3.6, 1))
            still = abs(gr - run.last_gr) < 15                                 # same as ping_features(): moved less than 15 m
            # --- arrivals: stops between the furthest point so far and this ping (same as find_arrivals)
            pm_new = max(run.g_max, prog)
            gap = dt
            if pm_new > run.g_max and gap <= ba.MAX_GAP_S:
                P = g.pat.stop_prog
                a, b = int(np.searchsorted(P, run.g_max, side="right")), int(np.searchsorted(P, pm_new, side="right"))
                for k in range(a, b):
                    frac = (P[k] - run.g_max) / (pm_new - run.g_max)
                    self._arrival(g, run, k, round(float(run.last_t + frac * gap), 1))
        else:
            run.g_first = prog
        if not still:
            run.last_move = ts
        v2, v3 = run.hv1, run.hv2
        vs = [v for v in (v1, v2, v3) if v == v]
        run.cur = dict(ts=ts, prog=gr, v1=v1, v2=v2, v3=v3, v_mean3=(sum(vs) / len(vs) if vs else float("nan")), standing_s=ts - run.last_move, rep=rep, idx=run.n)
        run.hv2, run.hv1 = run.hv1, v1
        run.n, run.g_max, run.last_t, run.last_g, run.last_gr = run.n + 1, max(run.g_max, prog), ts, prog, gr
        if not run.released and run.n >= 3 and run.g_max - run.g_first >= 200:     # "usable" run, same as build_runs()
            run.released = True
            pend, run.pending = run.pending, []
            for k, t in pend:
                self._emit(g, run, k, t)

    def _arrival(self, g, run, k, t):
        if run.released:
            self._emit(g, run, k, t)
        else:
            run.pending.append((k, t))

    def _emit(self, g, run, k, t):
        if g.bad:
            return
        self.stats["arrivals"] += 1
        if self.record:
            self.arrival_log.append((g.vid, g.route, g.trip, k, t))
        if run.last_arr is not None and run.last_arr[0] == k - 1:              # stop-to-stop time of two neighbouring stops (same as make_events)
            d = t - run.last_arr[1]
            m = self.med.get((g.route, k - 1))
            if 3 <= d <= 600 and m:
                self.events.add(g.route, t, min(max(d / m, 0.3), 3.0))
        run.last_arr = (k, t)

    # ------------------------------------------------------------ step 5-6: the model inputs
    def cases(self, keys=None, now=None, hmax=HMAX, max_age_s=300):
        """One row per bus and next stop (1..hmax stops ahead), with all the inputs of the model.
        keys = list of (vehicle, route, trip) from update(); default = every bus whose newest ping is at most max_age_s old (needs now)."""
        with self.lock:
            if keys is None:
                keys = [k for k in self.by_vid.values() if now is None or now - self.groups[k].last_ts <= max_age_s]
            cols = defaultdict(list)
            for key in keys:
                g = self.groups.get(key)
                if g is None or g.bad or g.run is None or g.run.n < 1 or g.run.cur is None:
                    continue
                c = g.run.cur
                P = self._p1.get(g.route)
                if P is None:
                    P = self._p1[g.route] = np.round(g.pat.stop_prog, 1)
                i0 = int(np.searchsorted(P, c["prog"] + 20, side="right"))      # first stop more than 20 m ahead (same as make_cases)
                for h in range(1, hmax + 1):
                    k = i0 + h - 1
                    if k >= len(P):
                        break
                    for name in ("ts", "prog", "v1", "v2", "v3", "v_mean3", "standing_s", "rep"):
                        cols[name].append(c[name])
                    cols["vehicle_id"].append(g.vid); cols["route_id"].append(g.route); cols["trip_id"].append(g.trip)
                    cols["h"].append(h); cols["sidx"].append(k); cols["dist"].append(float(P[k]) - c["prog"]); cols["run_ping"].append(c["idx"])
                    cols["stop_id"].append(g.pat.stop_ids[k])
            if not cols:
                return pd.DataFrame(columns=["vehicle_id", "route_id", "trip_id", "stop_id", "ts", "h", "sidx", "dist", "prog"] + FEATURES[2:])
            d = pd.DataFrame(cols)
            t = d["ts"] + IST                                                  # same as ping_features()
            d["hour"] = (t % 86400) / 3600.0
            d["dow"] = ((t // 86400) + 3) % 7                                  # 0 = Monday
            d["is_weekend"] = (d["dow"] >= 5).astype(int)
            d = t2.add_feats(d, self.L)                                        # history ETA, time-of-day factor, stop inputs (the training code)
            rec = {}
            for r_, ts_ in set(zip(d["route_id"].values, d["ts"].values)):     # the recent-buses numbers come from the live store
                rec[(r_, ts_)] = self.events.recent(r_, ts_)
            vals = [rec[(r_, ts_)] for r_, ts_ in zip(d["route_id"].values, d["ts"].values)]
            d["rc10"], d["rc10_n"], d["rr20"], d["rr20_n"] = [np.array([v[i] for v in vals], dtype=float) for i in range(4)]
            return d

    # ------------------------------------------------------------ step 7: the prediction
    def predict_cases(self, d):
        """Adds eta_raw / lo_raw / hi_raw (seconds from the time of the bus's last ping) to a cases table. Same as run_final() in train_eta_v2.py."""
        if self.models is None or d.empty:
            return d
        d = d.copy()
        mm, ff = self.models["models"], self.models["features"]
        p = {k: mm[k].predict(d[ff[k]].astype("float32")) for k in ("point", "low", "high") if k in mm}
        base = d["base"].values
        d["eta_raw"] = base + p["point"]                                       # target = real time minus the historical ETA
        if "low" in p and "high" in p:
            lo = base + p["low"]
            hi = np.maximum(base + p["high"], lo + 1)
            hg = np.where(d["h"].values <= 2, d["h"].values, np.where(d["h"].values <= 4, 3, 4))
            w = np.array(self.widen)[hg - 1]                                   # calibration: widen both ends, separately for 1, 2, 3-4, 5+ stops ahead
            d["lo_raw"], d["hi_raw"] = lo - w, hi + w
        return d

    def predict(self, keys=None, now=None, hmax=HMAX, max_age_s=300):
        """The table the explorer shows: eta_s / lo_s / hi_s are seconds from NOW (the age of the bus's last ping is subtracted, like the history ETA).
        eta_s is never below 0 and lies inside [lo_s, hi_s] (display fix: the three models are separate, so their ends can cross a little).
        The raw model values stay in eta_raw / lo_raw / hi_raw."""
        d = self.predict_cases(self.cases(keys, now, hmax, max_age_s))
        if d.empty or "eta_raw" not in d:
            return d
        age = np.maximum(0.0, (now - d["ts"].values)) if now is not None else np.zeros(len(d))
        d["age_s"] = age
        eta = np.maximum(d["eta_raw"].values - age, 0.0)
        d["eta_s"] = eta
        if "lo_raw" in d:
            d["lo_s"] = np.minimum(np.maximum(d["lo_raw"].values - age, 0.0), eta)
            d["hi_s"] = np.maximum(d["hi_raw"].values - age, eta)
        d["hist_s"] = np.maximum(d["base"].values - age, 0.0)                  # history x time of day, for the fallback and for comparison
        d["warm"] = d["rc10_n"] >= WARM_MIN_EVENTS                              # after a restart rc10 needs about 10 minutes to fill; until then show the history ETA
        return d

    def status(self, now=None):
        now = now if now is not None else self.last_now
        rc, rcn = self.events.all.window(now, 600) if now else (float("nan"), 0)
        out = dict(self.stats)
        out.update(buses=len(self.by_vid), groups=len(self.groups), events_10min=rcn, city_ratio_10min=rc)
        return out
