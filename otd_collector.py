#!/usr/bin/env python3
"""
OTD Live Collector - collects Delhi bus GPS data and cleans it, while SHOWING what it does.

Every poll goes through 5 visible steps:
   1 FETCH     download the live file from the OTD server
   2 DECODE    turn the binary (protobuf) file into one row per bus
   3 VALIDATE  check every row: drop unusable ones (with a reason), flag suspicious ones
   4 SAVE      append the new, valid pings to today's file in data/clean/
   5 HEALTH    write one line about this poll to data/health/ (feed reliability record)

Commands:
   python otd_collector.py run              live dashboard (keys: p pause, f fetch now, +/- speed, e explain, q quit)
   python otd_collector.py run --plain      same steps printed as readable text (good for a server / log file)
   python otd_collector.py run --demo       fake feed, no API key needed (to learn how it works)
   python otd_collector.py inspect          summary of the data collected so far
"""
import argparse, csv, curses, gzip, math, os, random, re, sys, threading, time
from collections import Counter, deque
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ============================ CONFIG (edit these) ============================
API_KEY = os.getenv("API_KEY")     # <-- paste your OTD key (or set env var OTD_API_KEY)
GTFS_DIR = "gtfs"                       # optional: routes.txt + trips.txt, used only to flag ID matches
DATA_DIR = "data"
BASE_URL = "https://otd.delhi.gov.in"
RT_PATH = "/api/realtime/VehiclePositions.pb"
INTERVAL = 20                           # seconds between polls (the feed itself refreshes about every 10 s)
MIN_INTERVAL = 10
BBOX = (28.20, 29.10, 76.60, 77.70)     # lat_min, lat_max, lon_min, lon_max  (Delhi-NCR, generous)
MAX_AGE_S = 600                         # ping older than this is "not live" -> dropped
JUMP_KMH = 100                          # implied speed above this -> row is FLAGGED (kept)
IST = 19800                             # seconds; file dates and hours use Indian time
# ==============================================================================

COLUMNS = ["fetched_at", "ts", "vehicle_id", "route_id", "trip_id", "start_date", "start_time", "lat", "lon",
           "age_s", "dt_s", "moved_m", "speed_kmh", "flag", "trip_in_static", "route_in_static",
           "bearing", "stop_id", "stop_seq", "occupancy"]
HEALTH_COLUMNS = ["fetched_at", "http_status", "latency_s", "bytes", "vehicles_in_feed", "feed_age_s", "rows_saved",
                  "duplicate", "stale", "outside_area", "bad_row", "flag_gps_jump", "error"]
REASONS = {
    "duplicate": "bus has not sent a new position since the last poll (normal, most buses)",
    "stale": f"last ping is older than {MAX_AGE_S} s (bus not live: parked / GPS off)",
    "outside_area": "position outside the Delhi-NCR box (bad GPS)",
    "no_position": "row has no latitude/longitude",
    "no_time": "row has no timestamp",
    "future_time": "timestamp is in the future (clock error)",
}
EXPLAIN = {
    "FETCH": "Downloads the live file from the OTD server (the same data the apps use).",
    "DECODE": "The file is protobuf (binary). We turn it into one table row per bus.",
    "VALIDATE": "Each row is checked. Unusable rows are dropped with a reason; suspicious ones are only flagged.",
    "SAVE": "New valid pings are appended to today's compressed CSV in data/clean/.",
    "HEALTH": "One line per poll goes to data/health/ so we can measure how reliable the feed is.",
}


class FeedError(Exception):
    pass


def redact(text):
    """Remove the API key from any error text before it is shown or saved."""
    return re.sub(r"(key=)[^&\s'\")]+", r"\1***", str(text))


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(a))


def ist_date(t):
    return time.strftime("%Y-%m-%d", time.gmtime(t + IST))


def ist_clock(t):
    return time.strftime("%H:%M:%S", time.gmtime(t + IST))


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


# ============================ DATA SOURCES ============================
class LiveSource:
    def __init__(self, key):
        self.key = key

    def fetch(self):
        import requests
        from google.transit import gtfs_realtime_pb2
        from google.protobuf.json_format import MessageToDict
        t0 = time.time()
        r = requests.get(BASE_URL + RT_PATH, params={"key": self.key}, timeout=25)
        latency = time.time() - t0
        if r.status_code in (401, 403):
            raise FeedError(f"HTTP {r.status_code}: the server rejected the API key. Check API_KEY.")
        if r.status_code == 429:
            raise FeedError("HTTP 429: too many requests. Slow down (press - on the dashboard).")
        r.raise_for_status()
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(r.content)
        rows = []
        for e in feed.entity:
            if e.HasField("vehicle"):
                row = flatten(MessageToDict(e.vehicle, preserving_proto_field_name=True))
                row["entity_id"] = e.id
                rows.append(row)
        hts = feed.header.timestamp if feed.header.HasField("timestamp") else None
        return {"rows": rows, "status": r.status_code, "bytes": len(r.content), "latency": latency, "header_ts": hts, "raw": r.content}


class FakeSource:
    """Pretend feed that contains every kind of problem, so you can see the cleaning work."""
    def __init__(self, n=400):
        random.seed(1)
        self.buses = [{"id": f"DL1PC{1000 + i}", "route": random.choice(["534", "425", "1707", "DL7"]),
                       "lat": random.uniform(28.50, 28.75), "lon": random.uniform(77.05, 77.30),
                       "ts": 0, "alive": random.random() > 0.05} for i in range(n)]
        self.n = 0

    def fetch(self):
        self.n += 1
        now = time.time()
        if self.n % 7 == 0:
            raise FeedError("simulated network problem (demo)")
        rows = []
        if self.n % 9 != 0:                                   # every 9th poll the demo feed is empty
            for b in self.buses:
                if b["alive"] and random.random() < 0.45:     # only some buses send a new ping each poll
                    b["lat"] += random.uniform(-0.001, 0.001); b["lon"] += random.uniform(-0.001, 0.001)
                    b["ts"] = int(now - random.randint(0, 15))
                elif not b["ts"]:
                    b["ts"] = int(now - 5000)                 # dead bus: very old ping
                row = {"entity_id": b["id"], "vehicle.id": b["id"], "vehicle.label": b["id"],
                       "trip.route_id": b["route"], "trip.trip_id": f"{b['route']}_16_20",
                       "trip.start_date": ist_date(now).replace("-", ""), "trip.start_time": "16:13:46",
                       "trip.schedule_relationship": "SCHEDULED", "position.speed": 0.0,
                       "position.latitude": b["lat"], "position.longitude": b["lon"], "timestamp": str(b["ts"])}
                rows.append(row)
            if len(rows) > 10:
                rows[3]["position.latitude"] = 0.0; rows[3]["position.longitude"] = 0.0     # bad GPS
                rows[5]["position.latitude"] = 19.07; rows[5]["position.longitude"] = 72.87   # Mumbai!
                del rows[7]["position.latitude"]
                rows[9]["position.latitude"] += 0.5                                           # GPS jump (flag)
        time.sleep(0.3)
        return {"rows": rows, "status": 200, "bytes": 250 * len(rows), "latency": 0.3, "header_ts": int(now) - 4,
                "raw": ("\n".join(str(r) for r in rows)).encode()}


def load_static(folder):
    import pandas as pd
    p = Path(folder)
    if not (p / "trips.txt").exists() or not (p / "routes.txt").exists():
        return None
    trips = pd.read_csv(p / "trips.txt", dtype=str, usecols=["trip_id"])
    routes = pd.read_csv(p / "routes.txt", dtype=str, usecols=["route_id"])
    return {"trips": set(trips["trip_id"]), "routes": set(routes["route_id"])}


# ============================ THE PIPELINE ============================
class Collector:
    def __init__(self, source, static, data_dir, interval, save_raw=False):
        self.source, self.static, self.dir = source, static, Path(data_dir)
        self.save_raw = save_raw
        self.interval = interval
        self.last_seen = {}                       # vehicle_id -> (ts, lat, lon) of last SAVED ping
        self.events = deque(maxlen=300)
        self.report = None
        self.counts = deque(maxlen=10)
        self.t_start = time.time()
        self.stats = {"polls": 0, "ok": 0, "failed": 0, "empty": 0, "saved": 0, "bytes": 0, "cycles_bytes": 0}
        self.rejects, self.flags, self.vehicles = Counter(), Counter(), set()
        self.paused, self.stop, self.force = False, False, False
        self.wake = threading.Event()
        self.next_at = time.time()
        self.on_report = None
        for sub in ("clean", "health"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)

    def log(self, msg, level="info"):
        self.events.append((time.time(), level, msg))

    # ---- step 3: validation + enrichment of one row ----
    def validate(self, r, now):
        lat, lon = to_float(r.get("position.latitude")), to_float(r.get("position.longitude"))
        if lat is None or lon is None:
            return "no_position", None
        if not (BBOX[0] <= lat <= BBOX[1] and BBOX[2] <= lon <= BBOX[3]):
            return "outside_area", None
        ts = to_float(r.get("timestamp"))
        if not ts:
            return "no_time", None
        if ts > now + 120:
            return "future_time", None
        age = now - ts
        if age > MAX_AGE_S:
            return "stale", None
        vid = str(r.get("vehicle.id") or r.get("vehicle.label") or r.get("entity_id"))
        prev = self.last_seen.get(vid)
        dt = moved = speed = None
        flag = ""
        if prev:
            if ts <= prev[0]:
                return "duplicate", None
            dt = ts - prev[0]
            moved = hav(prev[1], prev[2], lat, lon)
            if dt <= 300:
                speed = moved / dt * 3.6
                if speed > JUMP_KMH and moved > 100:
                    flag = "gps_jump"
        trip, route = r.get("trip.trip_id"), r.get("trip.route_id")
        row = [int(now), int(ts), vid, route or "", trip or "", r.get("trip.start_date", ""), r.get("trip.start_time", ""),
               round(lat, 6), round(lon, 6), int(age), "" if dt is None else int(dt),
               "" if moved is None else round(moved, 1), "" if speed is None else round(speed, 1), flag,
               "" if not self.static else int(str(trip) in self.static["trips"]),
               "" if not self.static else int(str(route) in self.static["routes"]),
               r.get("position.bearing", ""), r.get("stop_id", ""), r.get("current_stop_sequence", ""),
               r.get("occupancy_status", "")]
        self.last_seen[vid] = (ts, lat, lon)
        return flag or "ok", row

    # ---- one full poll ----
    def cycle(self):
        now = time.time()
        stages = []
        self.stats["polls"] += 1
        health = dict.fromkeys(HEALTH_COLUMNS, "")
        health["fetched_at"] = int(now)
        rep = {"time": now, "stages": stages, "ok": False, "kept": 0, "skipped": Counter(), "flagged": 0, "total": 0}
        try:                                                                             # 1 FETCH + 2 DECODE
            f = self.source.fetch()
        except Exception as e:
            msg = redact(f"{type(e).__name__}: {e}")[:160]
            stages.append(("FETCH", "FAIL", msg))
            self.stats["failed"] += 1
            health["error"] = msg
            self.log(f"Download failed: {msg}. Will retry.", "err")
            self.write_health(health)
            return self.finish(rep, False)
        raw_note = ""
        if self.save_raw and f.get("raw"):
            try:                                   # keep the untouched file so cleaning rules can be re-run later
                rd = self.dir / "raw" / ist_date(now)
                rd.mkdir(parents=True, exist_ok=True)
                with gzip.open(rd / f"{ist_clock(now).replace(':', '')}_{int(now * 100) % 100:02d}.pb.gz", "wb") as fh:
                    fh.write(f["raw"])
                raw_note = "  (raw copy saved)"
            except Exception as e:
                raw_note = f"  (raw save failed: {e})"
        stages.append(("FETCH", "ok", f"HTTP {f['status']}  {f['bytes'] / 1e6:.2f} MB  {f['latency']:.2f} s{raw_note}"))
        rows = f["rows"]
        n = len(rows)
        health.update(http_status=f["status"], latency_s=round(f["latency"], 2), bytes=f["bytes"], vehicles_in_feed=n)
        if f.get("header_ts"):
            health["feed_age_s"] = int(now - f["header_ts"])
        typical = sorted(self.counts)[len(self.counts) // 2] if self.counts else 0
        if n == 0 or (typical >= 200 and n < 0.2 * typical):
            stages.append(("DECODE", "WARN", f"only {n} vehicles in the file (usually about {typical}). Skipping this poll."))
            self.stats["empty"] += 1
            health["error"] = "empty_or_short_snapshot"
            self.log(f"Feed returned only {n} vehicles (usually ~{typical}). Ignored it, nothing was deleted.", "warn")
            self.write_health(health)
            self.finish(rep, False)
            return "empty"                     # the server often fixes itself in seconds: retry soon, no back-off
        self.counts.append(n)
        age_txt = f", feed time {health['feed_age_s']} s old" if health["feed_age_s"] != "" else ""
        stages.append(("DECODE", "ok", f"{n} vehicles in the file{age_txt}"))
        kept, skipped, flagged = [], Counter(), 0                                       # 3 VALIDATE
        for r in rows:
            res, row = self.validate(r, now)
            if row is None:
                skipped[res] += 1
            else:
                kept.append(row)
                if res != "ok":
                    flagged += 1; self.flags[res] += 1
        self.rejects.update(skipped)
        parts = [f"kept {len(kept)} new"] + [f"skipped {v} {k}" for k, v in skipped.most_common()]
        if flagged:
            parts.append(f"flagged {flagged} gps_jump (kept)")
        stages.append(("VALIDATE", "ok", ", ".join(parts)))
        path = self.dir / "clean" / f"{ist_date(now)}.csv.gz"                           # 4 SAVE
        try:
            new_file = not path.exists()
            with gzip.open(path, "at", newline="") as fh:
                w = csv.writer(fh)
                if new_file:
                    w.writerow(COLUMNS)
                w.writerows(kept)
            size = path.stat().st_size
            stages.append(("SAVE", "ok", f"+{len(kept)} rows -> {path}  ({size / 1e6:.1f} MB today)"))
            self.stats["saved"] += len(kept)
            self.stats["cycles_bytes"] += 1
            self.stats["bytes"] = sum(p.stat().st_size for p in (self.dir / "clean").glob("*.csv.gz"))
        except Exception as e:
            stages.append(("SAVE", "FAIL", redact(f"{type(e).__name__}: {e}")[:150]))
            health["error"] = "save_failed"
            self.log(f"Could not write file: {e}", "err")
        for r in kept:
            self.vehicles.add(r[2])
        health.update(rows_saved=len(kept), duplicate=skipped["duplicate"], stale=skipped["stale"],
                      outside_area=skipped["outside_area"],
                      bad_row=skipped["no_position"] + skipped["no_time"] + skipped["future_time"], flag_gps_jump=flagged)
        self.write_health(health)                                                       # 5 HEALTH
        stages.append(("HEALTH", "ok", "poll recorded in data/health/"))
        rep.update(kept=len(kept), skipped=skipped, flagged=flagged, total=n)
        self.stats["ok"] += 1
        self.log(f"Poll saved {len(kept)} new pings out of {n} vehicles.", "info")
        return self.finish(rep, True)

    def write_health(self, h):
        path = self.dir / "health" / f"{ist_date(h['fetched_at'])}.csv"
        new = not path.exists()
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=HEALTH_COLUMNS)
            if new:
                w.writeheader()
            w.writerow(h)

    def finish(self, rep, ok):
        rep["ok"] = ok
        self.report = rep
        if self.on_report:
            self.on_report(rep)
        return ok

    def per_day_estimate(self):
        c = self.stats["cycles_bytes"]
        if not c:
            return None
        return self.stats["bytes"] / c * (86400 / self.interval)

    def loop(self):
        fails = 0
        while not self.stop:
            if not self.paused or self.force:
                self.force = False
                res = self.cycle()
                if res is True:
                    fails, delay = 0, self.interval
                elif res == "empty":
                    delay = min(5, self.interval)
                else:
                    fails += 1
                    delay = min(self.interval * 2 ** min(fails, 3), 120)
            else:
                delay = 1
            self.next_at = time.time() + delay
            self.wake.wait(timeout=delay)
            self.wake.clear()


# ============================ PLAIN (TEXT) FRONT-END ============================
def plain_report(c):
    def cb(rep):
        print(f"\n[{ist_clock(rep['time'])}] Poll #{c.stats['polls']}")
        for name, st, detail in rep["stages"]:
            print(f"  {name:<9}{st:<5} {detail}")
        if rep["stages"] and rep["stages"][-1][0] == "HEALTH":
            for k, v in rep["skipped"].most_common(3):
                print(f"     - skipped '{k}': {REASONS[k]}")
        s = c.stats
        print(f"  TOTAL    polls {s['polls']} (ok {s['ok']}, failed {s['failed']}, empty {s['empty']}) | rows saved {s['saved']:,} "
              f"| buses seen {len(c.vehicles):,} | next poll in {c.interval} s")
    return cb


def run_plain(c, once):
    print(__doc__.split("Commands:")[0].strip())
    print(f"\nData folder: {c.dir.resolve()}   poll every {c.interval} s   (Ctrl+C to stop)\n")
    c.on_report = plain_report(c)
    try:
        if once:
            c.cycle(); return
        c.loop()
    except KeyboardInterrupt:
        c.stop = True
        s = c.stats
        print(f"\nStopped. {s['ok']} polls saved {s['saved']:,} rows for {len(c.vehicles):,} buses. Files are in {c.dir}/clean/")


# ============================ DASHBOARD (curses) ============================
class GridScreen:
    def __init__(self, h=40, w=120):
        self.h, self.w = h, w
        self.g = [[" "] * w for _ in range(h)]

    def size(self): return self.h, self.w
    def clear(self): self.g = [[" "] * self.w for _ in range(self.h)]
    def refresh(self): pass

    def put(self, r, c, text, style="n"):
        if 0 <= r < self.h:
            for i, ch in enumerate(str(text)):
                if 0 <= c + i < self.w:
                    self.g[r][c + i] = ch

    def dump(self): return "\n".join("".join(row).rstrip() for row in self.g)


class CursesScreen:
    def __init__(self, s):
        self.s = s
        curses.curs_set(0); s.keypad(True); s.timeout(250)
        self.st = {"n": 0, "bold": curses.A_BOLD, "dim": curses.A_DIM, "rev": curses.A_REVERSE}
        if curses.has_colors():
            curses.start_color()
            try: curses.use_default_colors(); bg = -1
            except curses.error: bg = curses.COLOR_BLACK
            for i, col in enumerate([curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_RED], 1):
                curses.init_pair(i, col, bg)
            curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLUE)
            self.st.update(ok=curses.color_pair(1) | curses.A_BOLD, warn=curses.color_pair(2) | curses.A_BOLD,
                           err=curses.color_pair(3) | curses.A_BOLD, hdr=curses.color_pair(4) | curses.A_BOLD)
        else:
            self.st.update(ok=0, warn=curses.A_BOLD, err=curses.A_BOLD, hdr=curses.A_REVERSE)

    def size(self): return self.s.getmaxyx()
    def clear(self): self.s.erase()
    def refresh(self): self.s.refresh()
    def getkey(self): return self.s.getch()

    def put(self, r, c, text, style="n"):
        h, w = self.s.getmaxyx()
        if 0 <= r < h and c < w:
            try: self.s.addstr(r, c, str(text)[: max(0, w - c - (1 if r == h - 1 else 0))], self.st.get(style, 0))
            except curses.error: pass


class Dashboard:
    def __init__(self, c):
        self.c, self.explain, self.quit = c, True, False

    def draw(self, scr):
        c = self.c
        scr.clear()
        H, W = scr.size()
        now = time.time()
        state = "PAUSED" if c.paused else f"next poll in {max(0, c.next_at - now):.0f}s"
        scr.put(0, 0, f" OTD Live Collector | poll every {c.interval}s | {state} | running {int((now - c.t_start) // 60)} min | folder {c.dir}".ljust(W), "hdr")
        r = 2
        scr.put(r, 1, "WHAT HAPPENED IN THE LAST POLL", "bold"); r += 1
        rep = c.report
        if not rep:
            scr.put(r, 3, "waiting for the first poll...", "dim"); r += 1
        else:
            scr.put(r - 1, 34, f"(at {ist_clock(rep['time'])} IST)", "dim")
            for i, (name, st, detail) in enumerate(rep["stages"]):
                style = {"ok": "ok", "WARN": "warn", "FAIL": "err"}.get(st, "n")
                scr.put(r, 3, f"{i + 1} {name:<9}", "bold"); scr.put(r, 16, f"[{st:<4}]", style); scr.put(r, 23, detail)
                r += 1
                if self.explain:
                    scr.put(r, 16, EXPLAIN[name], "dim"); r += 1
            if rep["ok"] and rep["total"]:
                k, tot = rep["kept"], rep["total"]
                bar = int(30 * k / tot)
                scr.put(r, 3, f"saved vs skipped: [{'#' * bar}{'.' * (30 - bar)}] {k} of {tot} rows were new and valid", "n"); r += 1
        r += 1
        s = c.stats
        scr.put(r, 1, "TOTALS SINCE START", "bold"); r += 1
        scr.put(r, 3, f"polls {s['polls']}  (ok {s['ok']}, failed {s['failed']}, empty feed {s['empty']})   rows saved {s['saved']:,}   different buses seen {len(c.vehicles):,}"); r += 1
        est = c.per_day_estimate()
        scr.put(r, 3, f"data on disk {s['bytes'] / 1e6:.1f} MB" + (f"   estimate at this rate: about {est / 1e9:.2f} GB per day" if est else "")); r += 2
        scr.put(r, 1, "WHY ROWS WERE SKIPPED (all polls so far)", "bold"); r += 1
        if not c.rejects:
            scr.put(r, 3, "nothing skipped yet", "dim"); r += 1
        for k, v in c.rejects.most_common():
            scr.put(r, 3, f"{v:>9,}  {k:<13} {REASONS[k]}"); r += 1
        if c.flags:
            scr.put(r, 3, f"{sum(c.flags.values()):>9,}  gps_jump     kept but flagged: moved faster than {JUMP_KMH} km/h", "warn"); r += 1
        r += 1
        scr.put(r, 1, "EVENT LOG", "bold"); r += 1
        for t, lvl, msg in list(c.events)[-max(3, H - r - 2):][::-1]:
            scr.put(r, 3, f"{ist_clock(t)}  {msg}", {"err": "err", "warn": "warn"}.get(lvl, "n")); r += 1
        scr.put(H - 1, 0, " p pause/resume | f fetch now | + / - poll speed | e explanations on/off | q quit", "dim")
        scr.refresh()

    def key(self, k):
        c = self.c
        if k == ord("q"): self.quit = True
        elif k == ord("p"): c.paused = not c.paused; c.log("Paused." if c.paused else "Resumed."); c.wake.set()
        elif k == ord("f"): c.force = True; c.wake.set()
        elif k in (ord("+"), ord("=")): c.interval = max(MIN_INTERVAL, c.interval - 5); c.log(f"Polling every {c.interval}s.")
        elif k in (ord("-"), ord("_")): c.interval += 5; c.log(f"Polling every {c.interval}s.")
        elif k == ord("e"): self.explain = not self.explain


def run_dashboard(c):
    def ui(stdscr):
        scr = CursesScreen(stdscr)
        d = Dashboard(c)
        threading.Thread(target=c.loop, daemon=True).start()
        while not d.quit:
            d.draw(scr)
            d.key(scr.getkey())
    curses.wrapper(ui)
    c.stop = True
    s = c.stats
    print(f"Stopped. {s['ok']} polls saved {s['saved']:,} rows for {len(c.vehicles):,} buses. Files are in {c.dir}/clean/")


# ============================ INSPECT ============================
def inspect(data_dir):
    import pandas as pd
    d = Path(data_dir)
    files = sorted((d / "clean").glob("*.csv.gz"))
    if not files:
        print(f"No data in {d}/clean/ yet. Run:  python otd_collector.py run"); return
    print("=== COLLECTED DATA (times in IST) ===")
    for f in files:
        df = pd.read_csv(f, dtype={"vehicle_id": str, "route_id": str, "trip_id": str, "flag": str, "bearing": str, "stop_id": str, "occupancy": str}, low_memory=False)
        hrs = ((df["ts"] + IST) // 3600 % 24).astype(int).value_counts().sort_index()
        print(f"\n{f.name}: {len(df):,} rows | {df['vehicle_id'].nunique():,} buses | {df['route_id'].nunique():,} routes | {f.stat().st_size / 1e6:.1f} MB")
        dt = df["dt_s"].dropna()
        if len(dt):
            print(f"  time between saved pings of one bus: median {dt.median():.0f} s, 90th percentile {dt.quantile(.9):.0f} s  <- tells how fine your ETA data can be")
        sp = df["speed_kmh"].dropna()
        if len(sp):
            print(f"  speed derived from positions: median {sp.median():.1f} km/h, share of moving pings (>2 km/h) {100 * (sp > 2).mean():.0f}%")
        def pct(col):
            v = pd.to_numeric(df[col], errors="coerce").mean()
            return "n/a (no GTFS was loaded)" if pd.isna(v) else f"{100 * v:.0f}%"
        print(f"  flagged gps_jump: {(df['flag'] == 'gps_jump').sum():,}   trip_id found in static: {pct('trip_in_static')}   route_id found: {pct('route_in_static')}")
        print("  rows per hour:  " + " ".join(f"{h:02d}h:{n:,}" for h, n in hrs.items()))
        print("  busiest routes: " + ", ".join(f"{r} ({n:,})" for r, n in df["route_id"].value_counts().head(8).items()))
    hf = sorted((d / "health").glob("*.csv"))
    if hf:
        h = pd.concat([pd.read_csv(f) for f in hf])
        ok = h["error"].isna()
        print(f"\n=== FEED HEALTH ===\npolls {len(h):,} | clean {100 * ok.mean():.1f}% | problems: {h['error'].value_counts().to_dict() or 'none'}")
        print(f"median download time {h['latency_s'].median():.2f} s | median vehicles in feed {h['vehicles_in_feed'].median():,.0f}")


def main():
    ap = argparse.ArgumentParser(description="OTD live collector")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--plain", action="store_true", help="print readable text instead of the dashboard")
    r.add_argument("--demo", action="store_true", help="fake feed (no key needed), saves into data_demo/")
    r.add_argument("--save-raw", action="store_true", help="also keep every downloaded file in data/raw/ (about 100-200 MB per 5 h)")
    r.add_argument("--once", action="store_true", help="do one poll and stop (implies --plain)")
    r.add_argument("--interval", type=int, default=INTERVAL)
    r.add_argument("--data", default=None)
    r.add_argument("--gtfs", default=GTFS_DIR)
    i = sub.add_parser("inspect"); i.add_argument("--data", default=DATA_DIR)
    a = ap.parse_args()
    if a.cmd == "inspect":
        return inspect(a.data)
    if a.demo:
        source, data = FakeSource(), a.data or "data_demo"
    else:
        key = os.getenv("OTD_API_KEY") or API_KEY
        if key == "PASTE_YOUR_API_KEY_HERE":
            raise SystemExit("Paste your key into API_KEY at the top of otd_collector.py (or set OTD_API_KEY).")
        source, data = LiveSource(key), a.data or DATA_DIR
    static = None
    try:
        static = load_static(a.gtfs)
    except Exception as e:
        print(f"(could not read GTFS from {a.gtfs}: {e})")
    if static is None:
        print(f"Note: no GTFS in '{a.gtfs}/' -> trip/route match columns will stay empty.")
    c = Collector(source, static, data, max(MIN_INTERVAL, a.interval), save_raw=a.save_raw)
    if a.plain or a.once or not sys.stdout.isatty():
        run_plain(c, a.once)
    else:
        run_dashboard(c)


if __name__ == "__main__":
    main()
