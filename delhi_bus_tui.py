#!/usr/bin/env python3
"""
Delhi Live Bus Explorer - a terminal app in the spirit of the One Delhi app.

  * MAP view  : live bus "bubbles" (route number) around any area, zoom/pan, route filter
  * STOP board: pick a stop -> upcoming buses of every route with ETA
  * BUS detail: every field the OTD feed gives for a bus, plus derived values
  * FIELDS    : which fields the feed really fills (use these numbers in your report)

Run:   python delhi_bus_tui.py            (live, needs API key + GTFS folder)
       python delhi_bus_tui.py --demo     (fake buses, no key or files needed)
Press ? inside the app for keys.
"""
import argparse, curses, math, os, pickle, random, threading, time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# ========================= CONFIG (edit these) =========================
API_KEY = os.getenv("API_KEY")     # <-- paste your OTD key here (or set env var OTD_API_KEY)
GTFS_DIR = "gtfs"                        # folder with routes.txt, trips.txt, stops.txt, stop_times.txt
BASE_URL = "https://otd.delhi.gov.in"
RT_PATH = "/api/realtime/VehiclePositions.pb"
REFRESH_SECONDS = 15                     # feed updates about every 10 s; do not go lower than 10
START_LAT, START_LON = 28.6315, 77.2167  # Connaught Place
# =======================================================================

ZOOMS = [10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400]   # metres per terminal column
DEFAULT_ZOOM = 3
OFF_ROUTE_M = 300          # a bus farther than this from its route line is flagged "off-route"
DWELL_S = 12               # seconds added per intermediate stop in the ETA estimate
ASSUMED_SPEED = 4.5        # m/s (about 16 km/h) when no speed can be found
STALE_S = 300


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fmt0(v):
    f = to_float(v)
    return '-' if f is None else f'{f:.0f}'


def hav(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def angdiff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


# ============================ STATIC GTFS ============================
class Pattern:
    """One ordered list of stops for a route and direction (the longest trip we found)."""
    def __init__(self, route_id, direction, headsign, stop_ids, stops):
        pts = [(s,) + tuple(stops[s]) for s in stop_ids if s in stops]
        self.route_id, self.direction, self.headsign = route_id, str(direction), headsign
        self.stop_ids = [p[0] for p in pts]
        self.names = [p[1] for p in pts]
        self.lat = np.array([p[2] for p in pts], float)
        self.lon = np.array([p[3] for p in pts], float)
        self.ok = len(pts) >= 2
        if self.ok:
            self.lat0, self.lon0 = float(self.lat.mean()), float(self.lon.mean())
            self.kx = 111320.0 * math.cos(math.radians(self.lat0))
            self.x = (self.lon - self.lon0) * self.kx
            self.y = (self.lat - self.lat0) * 110540.0
            self.seglen = np.hypot(np.diff(self.x), np.diff(self.y))
            self.cum = np.concatenate([[0.0], np.cumsum(self.seglen)])
            self.total = float(self.cum[-1])

    def project(self, lat, lon):
        """Return (distance to line in m, progress along line in m, local heading in degrees)."""
        px, py = (lon - self.lon0) * self.kx, (lat - self.lat0) * 110540.0
        ax, ay, bx, by = self.x[:-1], self.y[:-1], self.x[1:], self.y[1:]
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / np.where(l2 == 0, 1, l2), 0, 1)
        d = np.hypot(px - (ax + t * dx), py - (ay + t * dy))
        i = int(np.argmin(d))
        heading = (math.degrees(math.atan2(dx[i], dy[i])) + 360) % 360
        return float(d[i]), float(self.cum[i] + t[i] * self.seglen[i]), heading


class Static:
    def __init__(self, stops, routes, trip_info, raw_patterns):
        self.stops = stops                                   # stop_id -> (name, lat, lon)
        self.routes = routes                                 # route_id -> short name
        self.short_to_id = {}
        for rid, sn in routes.items():
            self.short_to_id.setdefault(str(sn), rid)
        self.trip_info = trip_info                           # trip_id -> (route_id, direction)
        self.patterns = defaultdict(list)
        self.stop_routes = defaultdict(list)                 # stop_id -> [(pattern, index)]
        for rid, dr, hs, sids in raw_patterns:
            p = Pattern(rid, dr, hs, sids, stops)
            if p.ok:
                self.patterns[rid].append(p)
                for k, sid in enumerate(p.stop_ids):
                    self.stop_routes[sid].append((p, k))
        self.stop_ids = list(stops)
        self.stop_lat = np.array([stops[s][1] for s in self.stop_ids], float)
        self.stop_lon = np.array([stops[s][2] for s in self.stop_ids], float)
        self.stop_names_l = [str(stops[s][0]).lower() for s in self.stop_ids]

    def label(self, rid):
        return str(self.routes.get(rid, rid))

    def resolve_route(self, route_val, trip_val):
        rv = None if route_val in (None, "") else str(route_val).strip()
        if rv in self.routes:
            return rv, True
        if rv in self.short_to_id:
            return self.short_to_id[rv], True
        if trip_val not in (None, "") and str(trip_val) in self.trip_info:
            return self.trip_info[str(trip_val)][0], True
        return None, False

    def search_stops(self, q, n=15):
        toks = q.lower().split()
        hits = [i for i, nm in enumerate(self.stop_names_l)
                if all(t in nm or t == str(self.stop_ids[i]).lower() for t in toks)]
        hits.sort(key=lambda i: len(self.stop_names_l[i]))
        return [self.stop_ids[i] for i in hits[:n]]

    def search_routes(self, q, n=15):
        q = q.lower()
        hits = [rid for rid, sn in self.routes.items() if q in str(sn).lower() or q in str(rid).lower()]
        hits.sort(key=lambda r: (str(self.routes[r]).lower() != q, len(str(self.routes[r])), str(self.routes[r])))
        return hits[:n]


def load_static(folder):
    folder = Path(folder)
    need = ["routes.txt", "trips.txt", "stops.txt", "stop_times.txt"]
    for f in need:
        if not (folder / f).exists():
            raise SystemExit(f"Missing {folder / f}. Put your GTFS .txt files in the '{folder}' folder (GTFS_DIR in the script).")
    sig = [(f, (folder / f).stat().st_mtime, (folder / f).stat().st_size) for f in need]
    cache = folder / ".bus_cache.pkl"
    if cache.exists():
        try:
            d = pickle.load(open(cache, "rb"))
            if d["sig"] == sig:
                return Static(d["stops"], d["routes"], d["trip_info"], d["raw_patterns"])
        except Exception:
            pass
    print("First run: indexing GTFS (can take a few minutes for stop_times.txt)...")
    routes_df = pd.read_csv(folder / "routes.txt", dtype=str)
    rn = routes_df["route_short_name"] if "route_short_name" in routes_df else routes_df["route_id"]
    routes = dict(zip(routes_df["route_id"], rn.fillna(routes_df["route_id"])))
    sdf = pd.read_csv(folder / "stops.txt", dtype={"stop_id": str})
    sdf["stop_lat"] = pd.to_numeric(sdf["stop_lat"], errors="coerce")
    sdf["stop_lon"] = pd.to_numeric(sdf["stop_lon"], errors="coerce")
    sdf = sdf.dropna(subset=["stop_lat", "stop_lon"])
    stops = {r.stop_id: (str(r.stop_name), float(r.stop_lat), float(r.stop_lon)) for r in sdf.itertuples()}
    trips = pd.read_csv(folder / "trips.txt", dtype=str)
    if "direction_id" in trips:
        trips["dirkey"] = trips["direction_id"].fillna("0")
    elif "trip_headsign" in trips:
        trips["dirkey"] = trips["trip_headsign"].fillna("0")
    else:
        trips["dirkey"] = "0"
    sizes = pd.Series(dtype="float64")
    for ch in pd.read_csv(folder / "stop_times.txt", usecols=["trip_id"], dtype=str, chunksize=1_000_000):
        sizes = sizes.add(ch["trip_id"].value_counts(), fill_value=0)
    trips["ncount"] = trips["trip_id"].map(sizes).fillna(0)
    rep = trips.sort_values("ncount", ascending=False).drop_duplicates(["route_id", "dirkey"])
    rep = rep[rep["ncount"] >= 2].groupby("route_id").head(4)
    want = set(rep["trip_id"])
    parts = []
    for ch in pd.read_csv(folder / "stop_times.txt", usecols=["trip_id", "stop_id", "stop_sequence"], dtype=str, chunksize=1_000_000):
        parts.append(ch[ch["trip_id"].isin(want)])
    st = pd.concat(parts)
    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    st = st.sort_values(["trip_id", "stop_sequence"])
    seq = st.groupby("trip_id")["stop_id"].apply(list).to_dict()
    raw = []
    for r in rep.itertuples():
        sids = seq.get(r.trip_id, [])
        hs = getattr(r, "trip_headsign", None)
        if not isinstance(hs, str) or not hs:
            hs = stops.get(sids[-1], ("?",))[0] if sids else "?"
        raw.append((r.route_id, r.dirkey, hs, sids))
    trip_info = {t: (r, d) for t, r, d in zip(trips["trip_id"], trips["route_id"], trips["dirkey"])}
    pickle.dump({"sig": sig, "stops": stops, "routes": routes, "trip_info": trip_info, "raw_patterns": raw}, open(cache, "wb"))
    return Static(stops, routes, trip_info, raw)


# ============================ LIVE FLEET ============================
class Bus:
    def __init__(self, key, row, lat, lon, ts):
        self.key, self.row, self.lat, self.lon, self.ts = key, row, lat, lon, ts
        self.hist = deque(maxlen=14)
        self.route_id = None
        self.route_ok = False
        self.assign = None
        self.assigned = False


class Assign:
    def __init__(self, pat, prog, perp, conf, heading, alts=None):
        self.pat, self.prog, self.perp, self.conf, self.heading = pat, prog, perp, conf, heading
        self.alts = alts or [(pat, prog, perp)]          # other directions that fit equally well


class Fleet:
    def age(self, b):
        return max(0.0, time.time() - b.ts)

    def __init__(self, static):
        self.static = static
        self.buses, self.by_route, self.rows = {}, {}, []
        self.snap, self.last_ok, self.last_err, self.header_ts = 0, None, None, None
        self.bad_snaps = 0                 # empty / suspiciously short feed answers that were ignored

    def update(self, rows, header_ts=None):
        now = time.time()
        new, by_route = {}, defaultdict(list)
        for r in rows:
            lat, lon = to_float(r.get("position.latitude")), to_float(r.get("position.longitude"))
            if lat is None or lon is None or (lat == 0 and lon == 0):
                continue
            key = str(r.get("vehicle.id") or r.get("vehicle.label") or r.get("entity_id"))
            ts = to_float(r.get("timestamp")) or now
            old = self.buses.get(key)
            b = Bus(key, r, lat, lon, ts)
            if old:
                b.hist = old.hist
            if not b.hist or ts > b.hist[-1][0]:
                b.hist.append((ts, lat, lon))
            b.route_id, b.route_ok = self.static.resolve_route(r.get("trip.route_id"), r.get("trip.trip_id"))
            new[key] = b
            if b.route_id:
                by_route[b.route_id].append(b)
        self.buses, self.by_route, self.rows = new, by_route, rows
        self.snap += 1
        self.last_ok, self.last_err, self.header_ts = now, None, header_ts

    # ---- which pattern/direction is the bus on? ----
    def assignment(self, b):
        if b.assigned:
            return b.assign
        b.assigned = True
        pats = self.static.patterns.get(b.route_id, []) if b.route_id else []
        if not pats:
            return None
        cands = [(*p.project(b.lat, b.lon), p) for p in pats]          # (perp, prog, heading, pat)
        tid = b.row.get("trip.trip_id")
        if tid not in (None, "") and str(tid) in self.static.trip_info:
            dr = str(self.static.trip_info[str(tid)][1])
            for c in cands:
                if c[3].direction == dr:
                    b.assign = Assign(c[3], c[1], c[0], "trip", c[2])
                    return b.assign
        cands.sort(key=lambda c: c[0])
        best = cands[0]
        conf = "only"
        if len(cands) > 1:
            conf = "guess"
            second = cands[1]
            if second[0] - best[0] < 60:                                   # both directions look equally close
                brg = to_float(b.row.get("position.bearing"))
                if brg:
                    best = min(cands[:2], key=lambda c: angdiff(brg, c[2]))
                    conf = "bearing"
                elif len(b.hist) >= 2:
                    old = b.hist[0]
                    scored = []
                    for c in cands[:2]:
                        _, pr0, _ = c[3].project(old[1], old[2])
                        scored.append((c[1] - pr0, c))
                    scored.sort(key=lambda s: -s[0])
                    if scored[0][0] > 30 and scored[1][0] <= 30:
                        best, conf = scored[0][1], "history"
            else:
                conf = "nearest"
        alts = [(c[3], c[1], c[0]) for c in cands if c[0] - cands[0][0] < 60] if conf == "guess" else None
        b.assign = Assign(best[3], best[1], best[0], conf, best[2], alts)
        return b.assign

    def speed(self, b):
        """(metres/second, source). Derived from positions first, because the feed speed may be 0."""
        t1, la1, lo1 = b.hist[-1]
        for t0, la0, lo0 in b.hist:
            dt = t1 - t0
            if 20 <= dt <= 180:
                return min(max(hav(la0, lo0, la1, lo1) / dt, 2.0), 14.0), "derived"
        s = to_float(b.row.get("position.speed"))
        if s and s > 0.5:
            return min(s, 14.0), "feed"
        return ASSUMED_SPEED, "assumed"

    def stop_board(self, stop_id):
        rows = []
        for pat, k in self.static.stop_routes.get(stop_id, []):
            target = pat.cum[k]
            for b in self.by_route.get(pat.route_id, []):
                a = self.assignment(b)
                if not a:
                    continue
                mine = [(pp, pr, pe) for pp, pr, pe in a.alts if pp is pat]
                if not mine or mine[0][2] > OFF_ROUTE_M or mine[0][1] > target + 40:
                    continue
                prog = mine[0][1]
                rem = max(0.0, target - prog)
                v, src = self.speed(b)
                between = int(np.searchsorted(pat.cum, target) - np.searchsorted(pat.cum, prog))
                eta = max(0.0, rem / v + DWELL_S * max(0, between - 1) - self.age(b))   # bus moved since its last ping
                rows.append({"bus": b, "route": self.static.label(pat.route_id), "to": pat.headsign,
                             "eta_s": eta, "dist": rem, "speed": v * 3.6, "src": src, "conf": "dir?" if a.conf == "guess" else a.conf})
        rows.sort(key=lambda r: r["eta_s"])
        return rows


# ============================ DATA SOURCES ============================
class LiveSource:
    def __init__(self, key):
        self.key = key

    def fetch(self):
        from google.transit import gtfs_realtime_pb2
        from google.protobuf.json_format import MessageToDict
        r = requests.get(BASE_URL + RT_PATH, params={"key": self.key}, timeout=20)
        if r.status_code in (401, 403):
            raise RuntimeError(f"HTTP {r.status_code}: API key rejected. Check API_KEY in the script.")
        r.raise_for_status()
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(r.content)
        rows = []
        for e in feed.entity:
            if e.HasField("vehicle"):
                row = flatten(MessageToDict(e.vehicle, preserving_proto_field_name=True))
                row["entity_id"] = e.id
                rows.append(row)
        return rows, (feed.header.timestamp if feed.header.HasField("timestamp") else None)


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def demo_static():
    stops, routes, trip_info, raw = {}, {}, {}, []
    defs = {"534": ((28.60, 77.15), (28.66, 77.30)), "425": ((28.72, 77.22), (28.54, 77.22)), "DL7": ((28.58, 77.16), (28.68, 77.27))}
    for n, (a, b) in defs.items():
        routes["R" + n] = n
        for d, (p, q) in enumerate([(a, b), (b, a)]):
            sids = []
            for i in range(26):
                sid = f"{n}_{d}_{i}" if d == 0 else f"{n}_0_{25 - i}"
                t = i / 25
                if sid not in stops:
                    stops[sid] = (f"{n} Stop {sid.split('_')[-1]}", p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
                sids.append(sid)
            raw.append(("R" + n, str(d), f"{n} Terminal {'B' if d == 0 else 'A'}", sids))
            for j in range(4):
                trip_info[f"T{n}_{d}_{j}"] = ("R" + n, str(d))
    return Static(stops, routes, trip_info, raw)


class DemoSource:
    def __init__(self, static):
        self.static, self.state, self.t = static, [], time.time()
        random.seed(3)
        i = 0
        for rid, pats in static.patterns.items():
            for p in pats:
                for j in range(9):
                    self.state.append({"id": f"DL1PC{1000 + i}", "pat": p, "pos": random.random() * p.total,
                                       "v": random.uniform(3, 9), "trip": j % 2 == 0, "j": j})
                    i += 1

    def fetch(self):
        now = time.time(); dt = min(now - self.t, 60); self.t = now
        rows = []
        for s in self.state:
            p = s["pat"]
            s["pos"] = (s["pos"] + s["v"] * max(dt, 15)) % p.total
            k = int(np.searchsorted(p.cum, s["pos"]) - 1); k = max(0, min(k, len(p.seglen) - 1))
            t = (s["pos"] - p.cum[k]) / max(p.seglen[k], 1)
            lat = p.lat[k] + (p.lat[k + 1] - p.lat[k]) * t + random.uniform(-1, 1) * 1e-4
            lon = p.lon[k] + (p.lon[k + 1] - p.lon[k]) * t + random.uniform(-1, 1) * 1e-4
            row = {"entity_id": s["id"], "vehicle.id": s["id"], "position.latitude": lat, "position.longitude": lon,
                   "position.speed": 0.0, "timestamp": str(int(now - random.randint(0, 20))),
                   "trip.route_id": p.route_id}
            if s["trip"]:
                row["trip.trip_id"] = f"T{p.route_id[1:]}_{p.direction}_{s['j'] % 4}"
            else:
                row["position.bearing"] = (math.degrees(math.atan2(p.x[k + 1] - p.x[k], p.y[k + 1] - p.y[k])) + 360) % 360
            rows.append(row)
        return rows, int(now)


# ============================ SCREENS ============================
class GridScreen:
    """Text-only screen used for tests."""
    def __init__(self, h=40, w=120):
        self.h, self.w = h, w
        self.g = [[" "] * w for _ in range(h)]

    def size(self):
        return self.h, self.w

    def clear(self):
        self.g = [[" "] * self.w for _ in range(self.h)]

    def put(self, r, c, text, style="n"):
        if r < 0 or r >= self.h:
            return
        for i, ch in enumerate(str(text)):
            if 0 <= c + i < self.w:
                self.g[r][c + i] = ch

    def refresh(self):
        pass

    def dump(self):
        return "\n".join("".join(row).rstrip() for row in self.g)


class CursesScreen:
    def __init__(self, stdscr):
        self.s = stdscr
        curses.curs_set(0)
        stdscr.keypad(True)
        stdscr.timeout(250)
        self.styles = {"n": 0, "dim": curses.A_DIM, "bold": curses.A_BOLD, "rev": curses.A_REVERSE}
        if curses.has_colors():
            curses.start_color()
            try:
                curses.use_default_colors(); bg = -1
            except curses.error:
                bg = curses.COLOR_BLACK
            for i, col in enumerate([curses.COLOR_CYAN, curses.COLOR_GREEN, curses.COLOR_YELLOW,
                                     curses.COLOR_MAGENTA, curses.COLOR_RED, curses.COLOR_BLUE], 1):
                curses.init_pair(i, col, bg)
                self.styles[f"c{i - 1}"] = curses.color_pair(i) | curses.A_BOLD
            curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLUE)
            self.styles["hdr"] = curses.color_pair(7) | curses.A_BOLD
            self.styles["warn"] = self.styles["c4"]
            self.styles["ok"] = self.styles["c1"]
        else:
            for i in range(6):
                self.styles[f"c{i}"] = curses.A_BOLD
            self.styles.update(hdr=curses.A_REVERSE, warn=curses.A_BOLD, ok=0)

    def size(self):
        return self.s.getmaxyx()

    def clear(self):
        self.s.erase()

    def put(self, r, c, text, style="n"):
        h, w = self.s.getmaxyx()
        if r < 0 or r >= h or c >= w:
            return
        text = str(text)[: max(0, w - c - (1 if r == h - 1 else 0))]
        try:
            self.s.addstr(r, max(c, 0), text, self.styles.get(style, 0))
        except curses.error:
            pass

    def refresh(self):
        self.s.refresh()

    def getkey(self):
        return self.s.getch()


# ============================ APP / UI ============================
class App:
    def __init__(self, static, fleet, lat, lon):
        self.static, self.fleet = static, fleet
        self.clat, self.clon, self.z = lat, lon, DEFAULT_ZOOM
        self.mode = "map"
        self.sel = None                # selected bus key
        self.stop = None               # selected stop id
        self.route = None              # route filter (route_id)
        self.board_i = 0
        self.scroll = 0
        self.search = {"kind": "stop", "q": "", "res": [], "i": 0}
        self.quit = False
        self.force = threading.Event()
        self.msg = ""

    # ---------- helpers ----------
    def style_for(self, rid):
        return f"c{sum(map(ord, str(rid))) % 6}"

    def view(self, scr):
        H, W = scr.size()
        mh = max(5, H - 1 - 1 - 8)
        return H, W, mh

    def tocell(self, lat, lon, W, mh):
        mpc = ZOOMS[self.z]
        kx = 111320.0 * math.cos(math.radians(self.clat))
        return (int(round(mh / 2 - (lat - self.clat) * 110540.0 / (mpc * 2))) + 1,
                int(round(W / 2 + (lon - self.clon) * kx / mpc)))

    def visible_buses(self, W, mh):
        out = []
        for b in self.fleet.buses.values():
            if self.route and b.route_id != self.route:
                continue
            r, c = self.tocell(b.lat, b.lon, W, mh)
            if 1 <= r <= mh and 0 <= c < W:
                out.append((b, r, c))
        return out

    def age(self, b):
        return max(0, time.time() - b.ts)

    # ---------- keys ----------
    def on_key(self, k, scr):
        H, W, mh = self.view(scr)
        if k in (-1, curses.KEY_RESIZE):
            return
        if self.mode == "search":
            return self.key_search(k)
        if self.mode == "help":
            self.mode = "map"
            return
        if self.mode in ("raw", "fields"):
            if k in (27, ord("q"), ord("\n"), curses.KEY_ENTER, ord("?")):
                self.mode = "map"; self.scroll = 0
            elif k in (curses.KEY_DOWN, ord("j")):
                self.scroll += 1
            elif k in (curses.KEY_UP, ord("k")):
                self.scroll = max(0, self.scroll - 1)
            elif k == curses.KEY_NPAGE:
                self.scroll += 10
            elif k == curses.KEY_PPAGE:
                self.scroll = max(0, self.scroll - 10)
            return
        if self.mode == "board":
            rows = self.fleet.stop_board(self.stop)
            if k in (27, ord("q")):
                self.mode = "map"
            elif k == curses.KEY_DOWN:
                self.board_i = min(self.board_i + 1, max(0, len(rows) - 1))
            elif k == curses.KEY_UP:
                self.board_i = max(0, self.board_i - 1)
            elif k in (ord("\n"), curses.KEY_ENTER) and rows:
                b = rows[min(self.board_i, len(rows) - 1)]["bus"]
                self.sel = b.key; self.clat, self.clon = b.lat, b.lon; self.mode = "map"
            elif k == ord("m"):
                _, la, lo = self.static.stops[self.stop]; self.clat, self.clon = la, lo; self.mode = "map"
            elif k == ord("f"):
                self.force.set()
            return
        # ---- map mode ----
        step_c = max(3, W // 8)
        mpc = ZOOMS[self.z]
        dlon = step_c * mpc / (111320.0 * math.cos(math.radians(self.clat)))
        dlat = max(2, mh // 6) * mpc * 2 / 110540.0
        if k == ord("q"): self.quit = True
        elif k == curses.KEY_LEFT or k == ord("h"): self.clon -= dlon
        elif k == curses.KEY_RIGHT or k == ord("l"): self.clon += dlon
        elif k == curses.KEY_UP or k == ord("k"): self.clat += dlat
        elif k == curses.KEY_DOWN or k == ord("j"): self.clat -= dlat
        elif k in (ord("+"), ord("=")): self.z = max(0, self.z - 1)
        elif k in (ord("-"), ord("_")): self.z = min(len(ZOOMS) - 1, self.z + 1)
        elif k == ord("0"): self.clat, self.clon, self.z = START_LAT, START_LON, DEFAULT_ZOOM
        elif k == ord("f"): self.force.set(); self.msg = "refreshing..."
        elif k == ord("b"):
            vis = sorted(self.visible_buses(W, mh), key=lambda t: (t[1] - mh / 2) ** 2 + ((t[2] - W / 2) / 2) ** 2)
            if vis:
                keys = [t[0].key for t in vis]
                self.sel = keys[(keys.index(self.sel) + 1) % len(keys)] if self.sel in keys else keys[0]
        elif k == ord("B"):
            self.sel = None
        elif k in (ord("\n"), curses.KEY_ENTER) and self.sel in self.fleet.buses:
            self.mode = "raw"; self.scroll = 0
        elif k == ord("s"):
            self.open_board(self.nearest_stop(self.clat, self.clon))
        elif k == ord("S") and self.sel in self.fleet.buses:
            self.open_board(self.next_stop_of(self.fleet.buses[self.sel]))
        elif k == ord("/"):
            self.search = {"kind": "stop", "q": "", "res": [], "i": 0}; self.mode = "search"
        elif k == ord("r"):
            self.search = {"kind": "route", "q": "", "res": [], "i": 0}; self.mode = "search"
        elif k == ord("x"):
            self.route = None; self.sel = None
        elif k == ord("i"): self.mode = "fields"; self.scroll = 0
        elif k == ord("?"): self.mode = "help"; self.scroll = 0

    def nearest_stop(self, lat, lon):
        st = self.static
        d = (st.stop_lat - lat) ** 2 + ((st.stop_lon - lon) * math.cos(math.radians(lat))) ** 2
        return st.stop_ids[int(np.argmin(d))] if len(d) else None

    def next_stop_of(self, b):
        a = self.fleet.assignment(b)
        if not a:
            self.msg = "this bus is not matched to a route pattern"; return None
        k = int(np.searchsorted(a.pat.cum, a.prog - 1))
        return a.pat.stop_ids[min(k, len(a.pat.stop_ids) - 1)]

    def open_board(self, sid):
        if sid:
            self.stop, self.board_i, self.mode = sid, 0, "board"
            _, la, lo = self.static.stops[sid]
            self.clat, self.clon = la, lo

    def key_search(self, k):
        s = self.search
        if k == 27:
            self.mode = "map"
        elif k in (curses.KEY_BACKSPACE, 127, 8):
            s["q"] = s["q"][:-1]
        elif k == curses.KEY_DOWN:
            s["i"] = min(s["i"] + 1, max(0, len(s["res"]) - 1))
        elif k == curses.KEY_UP:
            s["i"] = max(0, s["i"] - 1)
        elif k in (ord("\n"), curses.KEY_ENTER):
            if s["kind"] == "route" and not s["q"]:
                self.route = None; self.mode = "map"; return
            if s["res"]:
                pick = s["res"][min(s["i"], len(s["res"]) - 1)]
                if s["kind"] == "stop":
                    self.open_board(pick)
                else:
                    self.route = pick; self.mode = "map"
                    pats = self.static.patterns.get(pick, [])
                    if pats:
                        self.clat, self.clon = pats[0].lat0, pats[0].lon0
                        self.z = max(self.z, 5)
        elif 32 <= k < 127:
            s["q"] += chr(k); s["i"] = 0
        if s["kind"] == "stop":
            s["res"] = self.static.search_stops(s["q"]) if s["q"] else []
        else:
            s["res"] = self.static.search_routes(s["q"]) if s["q"] else []

    # ---------- drawing ----------
    def draw(self, scr):
        scr.clear()
        H, W, mh = self.view(scr)
        if self.mode == "search": self.draw_search(scr, H, W)
        elif self.mode == "board": self.draw_board(scr, H, W)
        elif self.mode == "raw": self.draw_raw(scr, H, W)
        elif self.mode == "fields": self.draw_fields(scr, H, W)
        elif self.mode == "help": self.draw_help(scr, H, W)
        else: self.draw_map(scr, H, W, mh)
        scr.refresh()

    def header(self, scr, W, extra=""):
        f = self.fleet
        age = "-" if not f.last_ok else f"{time.time() - f.last_ok:.0f}s"
        txt = f" Delhi Live Bus Explorer | buses {len(f.buses)} | updated {age} ago | zoom {ZOOMS[self.z]} m/col"
        if f.bad_snaps: txt += f" | empty feeds ignored: {f.bad_snaps}"
        if self.route: txt += f" | route {self.static.label(self.route)}"
        txt += extra
        scr.put(0, 0, txt.ljust(W), "hdr")

    def footer(self, scr, H, text):
        err = self.fleet.last_err
        scr.put(H - 1, 0, (f" ERROR: {err}" if err else " " + text)[: 400], "warn" if err else "dim")

    def draw_map(self, scr, H, W, mh):
        self.header(scr, W)
        mpc = ZOOMS[self.z]
        # route overlay
        if self.route:
            for p in self.static.patterns.get(self.route, []):
                for i in range(len(p.seglen)):
                    n = max(1, min(300, int(p.seglen[i] / max(mpc * 0.7, 1))))
                    for j in range(n + 1):
                        t = j / n
                        r, c = self.tocell(p.lat[i] + (p.lat[i + 1] - p.lat[i]) * t, p.lon[i] + (p.lon[i + 1] - p.lon[i]) * t, W, mh)
                        if 1 <= r <= mh and 0 <= c < W: scr.put(r, c, ".", "dim")
                for la, lo in zip(p.lat, p.lon):
                    r, c = self.tocell(la, lo, W, mh)
                    if 1 <= r <= mh and 0 <= c < W: scr.put(r, c, "+", "dim")
        # stops
        if mpc <= 200:
            st = self.static
            span_lat = mh * mpc * 2 / 110540.0 / 2 + 0.002
            span_lon = W * mpc / (111320.0 * math.cos(math.radians(self.clat))) / 2 + 0.002
            idx = np.where((abs(st.stop_lat - self.clat) < span_lat) & (abs(st.stop_lon - self.clon) < span_lon))[0]
            for i in idx[:1500]:
                r, c = self.tocell(st.stop_lat[i], st.stop_lon[i], W, mh)
                if 1 <= r <= mh and 0 <= c < W:
                    scr.put(r, c, ".", "dim")
                    if mpc <= 50 and len(idx) < 60:
                        scr.put(r, c + 2, str(st.stops[st.stop_ids[i]][0])[:16], "dim")
        # buses
        vis = self.visible_buses(W, mh)
        if mpc >= 400:
            cells = defaultdict(list)
            for b, r, c in vis:
                cells[(r, c)].append(b)
            for (r, c), bl in cells.items():
                n = len(bl)
                scr.put(r, c, "*" if n == 1 else (str(n) if n < 10 else "#"), self.style_for(bl[0].route_id) if n == 1 else "bold")
        else:
            used = set()
            order = sorted(vis, key=lambda t: (t[1] - mh / 2) ** 2 + ((t[2] - W / 2) / 2) ** 2)
            for b, r, c in order:
                if b.key == self.sel: continue
                lab = self.static.label(b.route_id) if b.route_id else (str(b.row.get("trip.route_id") or "?"))
                lab = lab[:6]
                text = f"({lab})"
                c0 = c - len(text) // 2
                cells_needed = {(r, x) for x in range(c0 - 1, c0 + len(text) + 1)}
                style = "dim" if self.age(b) > STALE_S else self.style_for(b.route_id)
                if not (cells_needed & used) and c0 >= 0 and c0 + len(text) < W:
                    scr.put(r, c0, text, style); used |= cells_needed
                elif (r, c) not in used:
                    scr.put(r, c, "o", style); used.add((r, c))
        # selected bus
        if self.sel in self.fleet.buses:
            b = self.fleet.buses[self.sel]
            r, c = self.tocell(b.lat, b.lon, W, mh)
            lab = self.static.label(b.route_id) if b.route_id else "?"
            if 1 <= r <= mh and 0 <= c < W:
                scr.put(r, max(0, c - (len(lab) + 2) // 2), f"[{lab}]", "rev")
        # crosshair
        scr.put(1 + mh // 2, W // 2, "+", "bold")
        self.draw_panel(scr, H, W, mh, vis)
        self.footer(scr, H, self.msg or "arrows pan | +/- zoom | b next bus | Enter bus fields | s stop board | / find stop | r route | x clear | i feed fields | ? help | q quit")

    def draw_panel(self, scr, H, W, mh, vis):
        top = mh + 1
        scr.put(top, 0, "-" * W, "dim")
        f = self.fleet
        if self.sel in f.buses:
            b = f.buses[self.sel]
            a = f.assignment(b)
            v, src = f.speed(b)
            rid = self.static.label(b.route_id) if b.route_id else "unknown"
            scr.put(top + 1, 1, f"BUS {b.key}   route {rid}" + (f"  ->  {a.pat.headsign}" if a else ""), "bold")
            fs = to_float(b.row.get("position.speed"))
            scr.put(top + 2, 1, f"pos {b.lat:.5f}, {b.lon:.5f}   speed {v * 3.6:.0f} km/h ({src})   feed speed {fs if fs is not None else '-'}   bearing {fmt0(b.row.get('position.bearing'))}   ping age {self.age(b):.0f}s")
            if a:
                k = int(np.searchsorted(a.pat.cum, a.prog - 1)); k = min(k, len(a.pat.names) - 1)
                warn = "   OFF-ROUTE" if a.perp > OFF_ROUTE_M else ""
                scr.put(top + 3, 1, f"progress {100 * a.prog / max(a.pat.total, 1):.0f}% of route   {a.perp:.0f} m from route line   direction by: {a.conf}{warn}", "warn" if warn else "n")
                scr.put(top + 4, 1, "next stops:", "dim")
                x = 13
                for kk in range(k, min(k + 3, len(a.pat.names))):
                    d = max(0, a.pat.cum[kk] - a.prog)
                    eta = d / v + DWELL_S * max(0, kk - k)
                    t = f"{a.pat.names[kk][:18]} {eta / 60:.0f}min  "
                    scr.put(top + 4, x, t); x += len(t)
            else:
                scr.put(top + 3, 1, "not matched to a static route/trip (route_id or trip_id missing or not in GTFS)", "warn")
            scr.put(top + 5, 1, f"trip_id {b.row.get('trip.trip_id', '-')}   route_id {b.row.get('trip.route_id', '-')}   (Enter = all fields, S = stop board for next stop)", "dim")
        else:
            cnt = Counter(self.static.label(b.route_id) if b.route_id else "?" for b, _, _ in vis)
            scr.put(top + 1, 1, f"{len(vis)} buses in view. Routes: " + "  ".join(f"{k}({v})" for k, v in cnt.most_common(14)))
            unk = sum(1 for b, _, _ in vis if not b.route_id)
            if unk: scr.put(top + 2, 1, f"{unk} of them cannot be matched to a route (see 'i' for ID match rates)", "warn")
            scr.put(top + 3, 1, "Press b to select the bus nearest the centre (+), s for the nearest stop's board.", "dim")

    def draw_board(self, scr, H, W):
        sid = self.stop
        name, la, lo = self.static.stops[sid]
        self.header(scr, W, f" | STOP BOARD")
        routes = sorted({self.static.label(p.route_id) for p, _ in self.static.stop_routes.get(sid, [])})
        scr.put(2, 1, f"{name}   (stop {sid})   {la:.5f}, {lo:.5f}", "bold")
        scr.put(3, 1, f"{len(routes)} routes serve this stop: " + " ".join(routes[:40]), "dim")
        scr.put(5, 1, f"{'ROUTE':<8}{'TOWARDS':<30}{'ETA':>8}{'DIST':>9}{'SPEED':>8}  {'SPEED SRC':<10}{'VEHICLE':<14}{'DIR':<8}", "rev")
        rows = self.fleet.stop_board(sid)
        if not rows:
            scr.put(7, 1, "No tracked bus is approaching this stop right now (or buses cannot be matched to routes).", "warn")
        for i, r in enumerate(rows[: H - 9]):
            eta = r["eta_s"]
            e = "due" if eta < 45 else f"{eta / 60:.0f} min"
            line = f"{r['route']:<8}{str(r['to'])[:28]:<30}{e:>8}{r['dist']:>8.0f}m{r['speed']:>6.0f}km  {r['src']:<10}{r['bus'].key:<14}{r['conf']:<8}"
            scr.put(6 + i, 1, line, "rev" if i == self.board_i else self.style_for(r["route"]))
        scr.put(H - 3, 1, "ETA = remaining distance along the route / current speed + 12 s per stop - age of the last ping. This is the simple baseline your AI models must beat.", "dim")
        self.footer(scr, H, "Up/Down choose bus | Enter show on map | m centre map on stop | f refresh | Esc back")

    def draw_search(self, scr, H, W):
        s = self.search
        self.header(scr, W, " | SEARCH")
        scr.put(2, 1, f"{'Find stop (name or id)' if s['kind'] == 'stop' else 'Route filter (number; empty + Enter clears)'}: {s['q']}_", "bold")
        for i, pick in enumerate(s["res"][: H - 6]):
            if s["kind"] == "stop":
                nm, la, lo = self.static.stops[pick]
                txt = f"{nm}  (id {pick})  {la:.4f}, {lo:.4f}"
            else:
                txt = f"{self.static.label(pick)}   (route_id {pick})   {len(self.static.patterns.get(pick, []))} patterns"
            scr.put(4 + i, 2, ("> " if i == s["i"] else "  ") + txt, "rev" if i == s["i"] else "n")
        self.footer(scr, H, "type to search | Up/Down choose | Enter select | Esc cancel")

    def draw_raw(self, scr, H, W):
        self.header(scr, W, " | BUS FIELDS")
        b = self.fleet.buses.get(self.sel)
        if not b:
            scr.put(2, 1, "bus no longer in feed"); return
        a = self.fleet.assignment(b); v, src = self.fleet.speed(b)
        lines = [("-- raw fields from the OTD feed --", "bold")] + [(f"{k:<32} {val}", "n") for k, val in sorted(b.row.items())]
        lines += [("", "n"), ("-- derived by this app --", "bold"),
                  (f"{'route (resolved)':<32} {self.static.label(b.route_id) if b.route_id else 'NOT RESOLVED'}", "n"),
                  (f"{'speed used for ETA':<32} {v * 3.6:.1f} km/h ({src})", "n"),
                  (f"{'ping age':<32} {self.age(b):.0f} s", "n"),
                  (f"{'positions remembered':<32} {len(b.hist)}", "n")]
        if a:
            lines += [(f"{'pattern direction / headsign':<32} {a.pat.direction} / {a.pat.headsign}", "n"),
                      (f"{'distance from route line':<32} {a.perp:.0f} m", "n"),
                      (f"{'progress along route':<32} {a.prog:.0f} m of {a.pat.total:.0f} m", "n"),
                      (f"{'direction decided by':<32} {a.conf}", "n")]
        self.scroll = min(self.scroll, max(0, len(lines) - (H - 3)))
        for i, (t, st) in enumerate(lines[self.scroll: self.scroll + H - 3]):
            scr.put(2 + i, 1, t, st)
        self.footer(scr, H, "Up/Down scroll | Esc back")

    def feed_census(self):
        rows = self.fleet.rows
        n = len(rows) or 1
        present, zero, num, ex = Counter(), Counter(), Counter(), {}
        for r in rows:
            for k, v in r.items():
                present[k] += 1; ex.setdefault(k, v)
                x = to_float(v)
                if x is not None: num[k] += 1; zero[k] += (x == 0)
        return n, present, zero, num, ex

    def draw_fields(self, scr, H, W):
        self.header(scr, W, " | FEED FIELDS")
        n, present, zero, num, ex = self.feed_census()
        buses = list(self.fleet.buses.values())
        with_trip = [b for b in buses if b.row.get("trip.trip_id") not in (None, "")]
        trip_ok = sum(1 for b in with_trip if str(b.row["trip.trip_id"]) in self.static.trip_info)
        with_route = [b for b in buses if b.row.get("trip.route_id") not in (None, "")]
        route_ok = sum(1 for b in with_route if b.route_ok)
        stale = sum(1 for b in buses if self.age(b) > STALE_S)
        lines = [(f"vehicles in latest snapshot: {len(self.fleet.rows)}   usable (valid position): {len(buses)}   stale (> {STALE_S}s): {stale}", "bold"),
                 (f"trip_id present: {len(with_trip)}  found in static trips.txt: {trip_ok} ({100 * trip_ok / max(1, len(with_trip)):.0f}%)", "warn" if with_trip and trip_ok / len(with_trip) < .5 else "n"),
                 (f"route_id present: {len(with_route)}  matched to static routes: {route_ok} ({100 * route_ok / max(1, len(with_route)):.0f}%)", "n"),
                 (f"buses resolved to a route (by either ID): {sum(1 for b in buses if b.route_id)} of {len(buses)}", "n"), ("", "n"),
                 (f"{'FIELD':<34}{'PRESENT':>9}{'ZERO':>8}   EXAMPLE", "rev")]
        for k in sorted(present):
            z = f"{100 * zero[k] / num[k]:.0f}%" if num[k] else "-"
            lines.append((f"{k:<34}{100 * present[k] / n:>8.0f}%{z:>8}   {str(ex[k])[:40]}", "n"))
        self.scroll = min(self.scroll, max(0, len(lines) - (H - 3)))
        for i, (t, st) in enumerate(lines[self.scroll: self.scroll + H - 3]):
            scr.put(2 + i, 1, t, st)
        self.footer(scr, H, "Up/Down scroll | Esc back   (these numbers answer: do trip IDs match? is speed all zero?)")

    def draw_help(self, scr, H, W):
        self.header(scr, W, " | HELP")
        txt = ["MAP", "  arrows / hjkl   pan            + / -   zoom in / out        0   back to Connaught Place",
               "  b   select next bus near centre (+)     B  deselect       Enter  all fields of selected bus",
               "  s   stop board of the stop nearest the centre             S  board for the selected bus's next stop",
               "  /   search a stop by name/id      r   show only one route (also draws its line and stops)     x  clear",
               "  i   what fields does the feed fill, do IDs match      f  refresh now      q  quit", "",
               "BUBBLES  (534) = bus on route 534.  o = bus whose label did not fit.  At far zoom: * one bus, 2..9 / # = many buses.",
               "         [534] highlighted = selected bus.   + in the middle = map centre.   . = bus stop (zoom in to see)", "",
               "STOP BOARD  upcoming buses sorted by ETA. ETA = distance along the route / speed. Direction 'by' tells how the app",
               "  decided the bus direction: trip (trip_id matched), bearing, history (it moved that way), nearest, guess.", "",
               "Dim bubbles = no ping for more than 5 minutes."]
        for i, t in enumerate(txt):
            scr.put(2 + i, 1, t)
        self.footer(scr, H, "any key: back")


def fetch_loop(source, fleet, app, interval):
    while not app.quit:
        try:
            rows, hts = source.fetch()
            prev = len(fleet.rows)
            if not rows or (prev >= 200 and len(rows) < 0.2 * prev):
                # The server sometimes answers with an empty or tiny file. Keep the last good data
                # instead of wiping the map (this was the "0 buses" blank screen).
                fleet.bad_snaps += 1
                fleet.last_ok = fleet.last_ok or time.time()
                app.msg = f"Feed sent only {len(rows)} vehicles (before: {prev}). Kept the old data."
            else:
                fleet.update(rows, hts)
                app.msg = ""
        except Exception as e:
            fleet.last_err = f"{type(e).__name__}: {e}"[:200]
        app.force.wait(timeout=max(10, interval))
        app.force.clear()


def run(stdscr, app, source, interval):
    scr = CursesScreen(stdscr)
    threading.Thread(target=fetch_loop, args=(source, app.fleet, app, interval), daemon=True).start()
    while not app.quit:
        app.draw(scr)
        app.on_key(scr.getkey(), scr)


def main():
    ap = argparse.ArgumentParser(description="Delhi live bus explorer")
    ap.add_argument("--demo", action="store_true", help="use fake buses (no key, no GTFS files needed)")
    ap.add_argument("--gtfs", default=GTFS_DIR)
    ap.add_argument("--refresh", type=int, default=REFRESH_SECONDS)
    a = ap.parse_args()
    if a.demo:
        static = demo_static(); source = DemoSource(static)
    else:
        key = os.getenv("OTD_API_KEY") or API_KEY
        if key == "PASTE_YOUR_API_KEY_HERE":
            raise SystemExit("Open delhi_bus_tui.py and paste your key into API_KEY (or set OTD_API_KEY).")
        static = load_static(a.gtfs); source = LiveSource(key)
    fleet = Fleet(static)
    app = App(static, fleet, START_LAT, START_LON)
    curses.wrapper(run, app, source, a.refresh)


if __name__ == "__main__":
    main()
