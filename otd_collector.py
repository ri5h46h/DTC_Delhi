#!/usr/bin/env python3
"""
OTD Live Collector (version 2) - collects Delhi bus GPS data and cleans it, while SHOWING what it does.

MAIN FEED   Delhi Transport Stack (has a real speed value).
FALLBACK    OTD Delhi (no speed). Only ONE feed is called per poll. After 2 problems in a row the other feed is used,
            and every 4th poll the main feed is tried once to see if it works again.

Every poll goes through 5 visible steps:
   1 FETCH     download the live file from ONE feed
   2 DECODE    turn the binary (protobuf) file into one row per bus
   3 VALIDATE  check every row: drop unusable ones (with a reason), flag suspicious ones
   4 SAVE      append the new, valid pings to today's file in data/clean/
   5 HEALTH    write one line about this poll to data/health/ (feed reliability record)

Commands:
   python otd_collector.py run                  live dashboard (keys: p pause, f fetch now, +/- speed, e explain, q quit)
   python otd_collector.py run --plain          same steps printed as readable text (good for a server / log file)
   python otd_collector.py run --save-raw       also keep every untouched download in data/raw/
   python otd_collector.py run --using-otd      OTD as the main feed (Transport Stack becomes the fallback)
   python otd_collector.py run --no-fallback    never switch feeds
   python otd_collector.py run --demo           fake feeds, no API keys needed (shows the fallback too)
   python otd_collector.py inspect              summary of the data collected so far
"""
import argparse, csv, curses, gzip, json, math, os, random, re, sys, threading, time
from collections import Counter, deque
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ============================ CONFIG (edit these) ============================
TSTACK_API_KEY = os.getenv("TSTACK_API_KEY")   # <-- paste your Delhi Transport Stack key (main feed)
OTD_API_KEY = os.getenv("OTD_API_KEY")                    # <-- paste your OTD Delhi key (fallback feed, or main with --using-otd)
GTFS_DIR = "gtfs"                       # optional: routes.txt + trips.txt, used only to flag ID matches
DATA_DIR = "data"
TSTACK_URL = "https://dts-backend.transportstack.in/api/dataset/otd/get-file"       # key goes in the x-api-key header
TSTACK_PARAMS = {"agency": "delhi-buses", "category": "realtime_gtfs", "filename": "VehiclePositions.pb"}
OTD_URL = "https://otd.delhi.gov.in/api/realtime/VehiclePositions.pb"              # key goes in the URL (?key=...)
INTERVAL = 15                           # seconds between polls (the feeds update in bursts, see 'inspect')
MIN_INTERVAL = 10
BBOX = (28.20, 29.10, 76.60, 77.70)     # lat_min, lat_max, lon_min, lon_max  (Delhi-NCR, generous)
MAX_AGE_S = 600                         # ping older than this is "not live" -> dropped
JUMP_KMH = 100                          # implied speed above this -> row is FLAGGED (kept)
IST = 19800                             # seconds; file dates and hours use Indian time
# ==============================================================================

LABEL = {"tstack": "Transport Stack", "otd": "OTD"}
COLUMNS = ["fetched_at", "source", "ts", "vehicle_id", "route_id", "trip_id", "start_date", "start_time", "schedule",
           "lat", "lon", "speed_rep", "age_s", "dt_s", "moved_m", "speed_calc_kmh", "flag", "trip_in_static", "route_in_static", "extra"]
LEGACY_COLUMNS = [("speed_ms" if c == "speed_rep" else c) for c in COLUMNS]     # layout of the first test version of the collector
HEALTH_COLUMNS = ["fetched_at", "source", "http_status", "latency_s", "bytes", "vehicles_in_feed", "feed_age_s", "rows_saved",
                  "duplicate", "stale", "outside_area", "bad_row", "flag_gps_jump", "speed_present_pct", "speed_zero_pct", "event", "error"]
# fields the collector knows. Anything else the feed sends is kept in the 'extra' column (so nothing is lost silently).
KNOWN = {"vehicle.id", "trip.route_id", "trip.trip_id", "trip.start_date", "trip.start_time", "trip.schedule_relationship",
         "position.latitude", "position.longitude", "position.speed", "timestamp", "entity_id"}
REASONS = {
    "duplicate": "bus has not sent a new position since the last poll (normal, most rows)",
    "stale": f"last ping is older than {MAX_AGE_S} s (bus not live: parked / GPS off)",
    "outside_area": "position outside the Delhi-NCR box (bad GPS)",
    "no_position": "row has no latitude/longitude",
    "no_time": "row has no timestamp",
    "future_time": "timestamp is in the future (clock error)",
}
EXPLAIN = {
    "FETCH": "Downloads the live file from ONE feed (Transport Stack, or OTD while it is the fallback).",
    "DECODE": "The file is protobuf (binary). We turn it into one table row per bus.",
    "VALIDATE": "Each row is checked. Unusable rows are dropped with a reason; suspicious ones are only flagged.",
    "SAVE": "New valid pings are appended to today's compressed CSV in data/clean/.",
    "HEALTH": "One line per poll goes to data/health/ so we can measure how reliable the feeds are.",
}
SECRETS = []


class FeedError(Exception):
    pass


def redact(text):
    for k in SECRETS:
        if k and len(k) > 4:
            text = str(text).replace(k, "***")
    return re.sub(r"(key=)[^&\s'\")]+", r"\1***", str(text))


def key_ok(k):
    return bool(k) and not str(k).startswith("PASTE")


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
    """One real feed. source = "tstack" (key in the x-api-key header) or "otd" (key in the URL)."""
    def __init__(self, source, key):
        self.source, self.key = source, key
        SECRETS.append(key)

    def http_get(self):
        import requests
        t0 = time.time()
        if self.source == "otd":
            r = requests.get(OTD_URL, params={"key": self.key}, timeout=25)
            who = "OTD (OTD_API_KEY)"
        else:
            r = requests.get(TSTACK_URL, params=TSTACK_PARAMS, headers={"x-api-key": self.key}, timeout=25)
            who = "Transport Stack (TSTACK_API_KEY)"
        latency = time.time() - t0
        if r.status_code in (401, 403):
            raise FeedError(f"HTTP {r.status_code}: the {who} key was rejected. Check the key at the top of otd_collector.py.")
        if r.status_code == 429:
            raise FeedError("HTTP 429: too many requests. Slow down (press - on the dashboard).")
        r.raise_for_status()
        if r.content[:1] in (b"{", b"<", b"["):
            raise FeedError("the server sent text instead of a feed: " + redact(r.content[:100].decode("utf-8", "replace")))
        return r.content, r.status_code, latency

    def fetch(self):
        from google.transit import gtfs_realtime_pb2
        from google.protobuf.json_format import MessageToDict
        content, status, latency = self.http_get()
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(content)
        rows = []
        for e in feed.entity:
            if e.HasField("vehicle"):
                row = flatten(MessageToDict(e.vehicle, preserving_proto_field_name=True))
                row["entity_id"] = e.id
                rows.append(row)
        hts = feed.header.timestamp if feed.header.HasField("timestamp") else None
        return {"rows": rows, "status": status, "bytes": len(content), "latency": latency, "header_ts": hts, "raw": content}


class FeedManager:
    """Calls ONE feed per poll; the other feed is only a fallback.
    - 2 problems in a row on the active feed (error, rejected key, empty or tiny file) -> switch to the other feed
    - while on the fallback, every 4th poll the main feed is tried once; if it works the manager switches back."""
    def __init__(self, primary, sources):
        self.primary = primary
        self.other = "tstack" if primary == "otd" else "otd"
        self.sources = sources                       # {"tstack": source, "otd": source}  (the other one may be missing)
        self.active, self.trying, self.fails, self.fb_cycles = primary, primary, 0, 0
        self.just_switched, self.note = False, ""

    def fetch(self):
        name = self.active
        if self.active != self.primary:
            self.fb_cycles += 1
            if self.fb_cycles % 4 == 0:
                name = self.primary                  # a quick check whether the main feed works again
        self.trying = name
        f = self.sources[name].fetch()               # problems are raised as errors
        f["source"] = name
        return f

    def mark(self, ok):
        name = self.trying
        if ok:
            if name == self.primary and self.active != self.primary:
                self.active, self.just_switched, self.note = self.primary, True, f"{LABEL[self.primary]} works again, switched back to it."
            self.fails = 0
            return
        if name != self.active:                      # a failed check of the main feed: stay on the fallback
            return
        self.fails += 1
        if self.fails >= 2 and len(self.sources) > 1:
            old, self.active = self.active, (self.other if self.active == self.primary else self.primary)
            self.fails, self.fb_cycles, self.just_switched = 0, 0, True
            self.note = f"{LABEL[old]} had problems, now using {LABEL[self.active]}."

    def state_text(self):
        if self.active == self.primary:
            return f"{LABEL[self.active]} (main)" + ("" if len(self.sources) > 1 else ", no fallback")
        return f"{LABEL[self.active]} (FALLBACK, {LABEL[self.primary]} had problems)"


class FakeSource:
    """Pretend feed with every kind of problem, so you can see the cleaning work. 'tstack' has speed, 'otd' sends 0."""
    def __init__(self, name, buses):
        self.name, self.buses, self.n = name, buses, 0

    def fetch(self):
        self.n += 1
        now = time.time()
        if self.name == "tstack" and self.n in (4, 5):
            raise FeedError("simulated Transport Stack outage (demo)")
        rows = []
        if self.n % 9 != 0:                                   # every 9th call this feed answers with an empty file
            lag = 11 if self.name == "tstack" else 0
            for b in self.buses:
                if b["alive"] and random.random() < 0.45:     # only some buses send a new ping each poll
                    b["lat"] += random.uniform(-0.001, 0.001); b["lon"] += random.uniform(-0.001, 0.001)
                    b["ts"] = int(now - random.randint(0, 15) - lag); b["v"] = random.choice([0.0, 3.2, 6.1, 8.9, 12.4])
                elif not b["ts"]:
                    b["ts"] = int(now - 5000)                 # dead bus: very old ping
                row = {"entity_id": b["id"], "vehicle.id": b["id"], "vehicle.label": b["id"],
                       "trip.route_id": b["route"], "trip.trip_id": f"{b['route']}_16_20",
                       "trip.start_date": ist_date(now).replace("-", ""), "trip.start_time": "16:13:46",
                       "trip.schedule_relationship": b["sched"], "position.speed": (b.get("v", 0.0) if self.name == "tstack" else 0.0),
                       "position.latitude": b["lat"], "position.longitude": b["lon"], "timestamp": str(b["ts"])}
                rows.append(row)
            if len(rows) > 10:
                rows[3]["position.latitude"] = 0.0; rows[3]["position.longitude"] = 0.0     # bad GPS
                rows[5]["position.latitude"] = 19.07; rows[5]["position.longitude"] = 72.87   # Mumbai!
                del rows[7]["position.latitude"]
                rows[9]["position.latitude"] += 0.5                                           # GPS jump (flag)
                rows[11]["position.bearing"] = 90.0                                           # an unexpected field -> goes to 'extra'
        time.sleep(0.3)
        return {"rows": rows, "status": 200, "bytes": 250 * len(rows), "latency": 0.3, "header_ts": int(now) - 4 - (19 if self.name == "tstack" else 0),
                "raw": ("\n".join(str(r) for r in rows)).encode()}


def demo_buses(n=400):
    random.seed(1)
    return [{"id": f"DL1PC{1000 + i}", "route": random.choice(["534UP", "425DOWN", "1707", "DL7"]), "sched": random.choice(["ADDED", "ADDED", "SCHEDULED"]),
             "lat": random.uniform(28.50, 28.75), "lon": random.uniform(77.05, 77.30), "ts": 0, "alive": random.random() > 0.05} for i in range(n)]


def load_static(folder):
    import pandas as pd
    p = Path(folder)
    if not (p / "trips.txt").exists() or not (p / "routes.txt").exists():
        return None
    trips = pd.read_csv(p / "trips.txt", dtype=str, usecols=["trip_id"])
    routes = pd.read_csv(p / "routes.txt", dtype=str, usecols=["route_id"])
    return {"trips": set(trips["trip_id"]), "routes": set(routes["route_id"])}


# ============================ THE PIPELINE ============================
def header_ok(path, columns):
    """True if the first line of the (gz) csv file is exactly our column list."""
    try:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as fh:
            return fh.readline().strip() == ",".join(columns)
    except Exception:
        return False


def check_formats(data_dir):
    """Refuse to add rows to a file of today that was written by an older collector (the columns would not line up)."""
    d, now = Path(data_dir), time.time()
    for path, cols in ((d / "clean" / f"{ist_date(now)}.csv.gz", COLUMNS), (d / "health" / f"{ist_date(now)}.csv", HEALTH_COLUMNS)):
        if path.exists() and not header_ok(path, cols):
            raise SystemExit(f"{path} has the column layout of an older collector. Rename the whole '{data_dir}' folder (for example to '{data_dir}_old') and start again.")


class Collector:
    def __init__(self, feeds, static, data_dir, interval, save_raw=False):
        self.feeds, self.static, self.dir, self.interval, self.save_raw = feeds, static, Path(data_dir), interval, save_raw
        self.last_seen = {}                       # vehicle_id -> (ts, lat, lon) of last SAVED ping
        self.events = deque(maxlen=300)
        self.report = None
        self.counts = {"tstack": deque(maxlen=10), "otd": deque(maxlen=10)}      # typical vehicle count of each feed
        self.t_start = time.time()
        self.stats = {"polls": 0, "ok": 0, "failed": 0, "empty": 0, "saved": 0, "bytes": 0, "cycles_bytes": 0}
        self.rows_by_source, self.polls_by_source = Counter(), Counter()
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
    def validate(self, r, now, source):
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
        sp = to_float(r.get("position.speed")) if source == "tstack" else None      # raw value of the feed (unit checked by inspect). OTD always sends 0: not a real speed
        extra = {k: v for k, v in r.items() if k not in KNOWN and not (k == "vehicle.label" and str(v) == vid)}
        row = [int(now), source, int(ts), vid, route or "", trip or "", r.get("trip.start_date", ""), r.get("trip.start_time", ""),
               r.get("trip.schedule_relationship", ""), round(lat, 6), round(lon, 6), "" if sp is None else round(sp, 2), int(age),
               "" if dt is None else int(dt), "" if moved is None else round(moved, 1), "" if speed is None else round(speed, 1), flag,
               "" if not self.static else int(str(trip) in self.static["trips"]),
               "" if not self.static else int(str(route) in self.static["routes"]),
               json.dumps(extra, separators=(",", ":")) if extra else ""]
        self.last_seen[vid] = (ts, lat, lon)
        return flag or "ok", row

    def note_switch(self, health):
        if self.feeds.just_switched:
            health["event"] = self.feeds.note
            self.log(self.feeds.note, "warn")

    # ---- one full poll ----
    def cycle(self):
        now = time.time()
        stages = []
        self.stats["polls"] += 1
        health = dict.fromkeys(HEALTH_COLUMNS, "")
        health["fetched_at"] = int(now)
        rep = {"time": now, "stages": stages, "ok": False, "kept": 0, "skipped": Counter(), "flagged": 0, "total": 0, "source": ""}
        try:                                                                             # 1 FETCH + 2 DECODE
            f = self.feeds.fetch()
        except Exception as e:
            src = self.feeds.trying
            rep["source"] = health["source"] = src
            msg = redact(f"{type(e).__name__}: {e}")[:160]
            stages.append(("FETCH", "FAIL", f"{LABEL[src]}: {msg}"))
            self.stats["failed"] += 1
            health["error"] = msg
            self.log(f"{LABEL[src]} download failed: {msg}", "err")
            self.feeds.mark(False)
            self.note_switch(health)
            self.write_health(health)
            return self.finish(rep, False)
        src = f["source"]
        rep["source"] = health["source"] = src
        self.polls_by_source[src] += 1
        raw_note = ""
        if self.save_raw and f.get("raw"):
            try:                                   # keep the untouched file so cleaning rules can be re-run later
                rd = self.dir / "raw" / ist_date(now)
                rd.mkdir(parents=True, exist_ok=True)
                with gzip.open(rd / f"{ist_clock(now).replace(':', '')}_{int(now * 100) % 100:02d}_{src}.pb.gz", "wb") as fh:
                    fh.write(f["raw"])
                raw_note = "  (raw copy saved)"
            except Exception as e:
                raw_note = f"  (raw save failed: {e})"
        tag = LABEL[src] + (" (fallback)" if src != self.feeds.primary else "")
        stages.append(("FETCH", "ok", f"{tag}: HTTP {f['status']}  {f['bytes'] / 1e6:.2f} MB  {f['latency']:.2f} s{raw_note}"))
        rows = f["rows"]
        n = len(rows)
        health.update(http_status=f["status"], latency_s=round(f["latency"], 2), bytes=f["bytes"], vehicles_in_feed=n)
        if f.get("header_ts"):
            health["feed_age_s"] = int(now - f["header_ts"])
        cnt = self.counts[src]
        typical = sorted(cnt)[len(cnt) // 2] if cnt else 0
        if n == 0 or (typical >= 200 and n < 0.2 * typical):
            stages.append(("DECODE", "WARN", f"only {n} vehicles in the file (usually about {typical}). Skipping this poll."))
            self.stats["empty"] += 1
            health["error"] = "empty_or_short_snapshot"
            self.log(f"{LABEL[src]} returned only {n} vehicles (usually ~{typical}). Ignored it, nothing was deleted.", "warn")
            self.feeds.mark(False)
            self.note_switch(health)
            self.write_health(health)
            self.finish(rep, False)
            return "empty"                     # the server often fixes itself in seconds: retry soon, no back-off
        cnt.append(n)
        age_txt = f", file made {health['feed_age_s']} s ago" if health["feed_age_s"] != "" else ""
        if src == "tstack":
            sp = [to_float(r.get("position.speed")) for r in rows]
            have = [x for x in sp if x is not None]
            health["speed_present_pct"] = round(100 * len(have) / n)
            health["speed_zero_pct"] = round(100 * sum(1 for x in have if x == 0) / len(have)) if have else ""
            sp_txt = f", reported speed on {health['speed_present_pct']}% (zero on {health['speed_zero_pct']}% of those)"
        else:
            sp_txt = ", no real speed in this feed"
        stages.append(("DECODE", "ok", f"{n} vehicles{age_txt}{sp_txt}"))
        kept, skipped, flagged = [], Counter(), 0                                       # 3 VALIDATE
        for r in rows:
            res, row = self.validate(r, now, src)
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
            self.rows_by_source[src] += len(kept)
            self.stats["cycles_bytes"] += 1
            self.stats["bytes"] = sum(p.stat().st_size for p in (self.dir / "clean").glob("*.csv.gz"))
        except Exception as e:
            stages.append(("SAVE", "FAIL", redact(f"{type(e).__name__}: {e}")[:150]))
            health["error"] = "save_failed"
            self.log(f"Could not write file: {e}", "err")
        for r in kept:
            self.vehicles.add(r[3])
        health.update(rows_saved=len(kept), duplicate=skipped["duplicate"], stale=skipped["stale"], outside_area=skipped["outside_area"],
                      bad_row=skipped["no_position"] + skipped["no_time"] + skipped["future_time"], flag_gps_jump=flagged)
        self.feeds.mark(True)
        self.note_switch(health)
        self.write_health(health)                                                       # 5 HEALTH
        stages.append(("HEALTH", "ok", "poll recorded in data/health/"))
        rep.update(kept=len(kept), skipped=skipped, flagged=flagged, total=n)
        self.stats["ok"] += 1
        self.log(f"{LABEL[src]}: saved {len(kept)} new pings out of {n} vehicles.", "info")
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
                try:
                    res = self.cycle()
                except Exception as e:            # one bad poll (for example a full disk) must never stop the collector
                    self.stats["failed"] += 1
                    self.log("Unexpected error in a poll: " + redact(f"{type(e).__name__}: {e}")[:150] + "  (the collector keeps running)", "err")
                    res = False
                if res is True:
                    fails, delay = 0, self.interval
                elif res == "empty":
                    delay = min(5, self.interval)
                else:
                    fails += 1
                    delay = 5 if len(self.feeds.sources) > 1 else min(self.interval * 2 ** min(fails, 3), 120)
                if self.feeds.just_switched:
                    self.feeds.just_switched = False
                    delay = 1                  # try the other feed right away
            else:
                delay = 1
            self.next_at = time.time() + delay
            self.wake.wait(timeout=delay)
            self.wake.clear()


# ============================ PLAIN (TEXT) FRONT-END ============================
def plain_report(c):
    def cb(rep):
        print(f"\n[{ist_clock(rep['time'])}] Poll #{c.stats['polls']}   feed: {c.feeds.state_text()}")
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
        scr.put(0, 0, f" OTD Live Collector v2 | poll every {c.interval}s | {state} | running {int((now - c.t_start) // 60)} min | folder {c.dir}".ljust(W), "hdr")
        fb = c.feeds.active != c.feeds.primary
        scr.put(1, 1, f"FEED NOW: {c.feeds.state_text()}", "warn" if fb else "ok")
        r = 3
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
        scr.put(r, 3, "polls / rows by feed:  " + "   ".join(f"{LABEL[k]} {c.polls_by_source[k]} polls, {c.rows_by_source[k]:,} rows" for k in ("tstack", "otd"))); r += 1
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
    import numpy as np
    import pandas as pd
    d = Path(data_dir)
    files = sorted((d / "clean").glob("*.csv.gz"))
    if not files:
        print(f"No data in {d}/clean/ yet. Run:  python otd_collector.py run"); return
    print("=== COLLECTED DATA (times in IST) ===")
    str_cols = {c: str for c in ("vehicle_id", "route_id", "trip_id", "flag", "schedule", "source", "start_date", "start_time", "extra")}
    for f in files:
        legacy = False
        if not header_ok(f, COLUMNS):
            legacy = header_ok(f, LEGACY_COLUMNS)
            if not legacy:
                print(f"\n{f.name}: older file layout, skipped (rename the old data folder)"); continue
        df = pd.read_csv(f, dtype=str_cols, low_memory=False)
        if legacy:
            df = df.rename(columns={"speed_ms": "speed_rep"})
        hrs = ((df["ts"] + IST) // 3600 % 24).astype(int).value_counts().sort_index()
        print(f"\n{f.name}: {len(df):,} rows | {df['vehicle_id'].nunique():,} buses | {df['route_id'].nunique():,} routes | {f.stat().st_size / 1e6:.1f} MB")
        print("  rows by feed: " + ", ".join(f"{LABEL.get(k, k)} {v:,}" for k, v in df["source"].value_counts().items()))
        dt = pd.to_numeric(df["dt_s"], errors="coerce").dropna()
        if len(dt):
            print(f"  time between saved pings of one bus: median {dt.median():.0f} s, 90th percentile {dt.quantile(.9):.0f} s"
                  "  <- if this is about twice your poll interval or more, polling faster brings no new pings")
        ts_rows = df[df["source"] == "tstack"]
        sp = pd.to_numeric(ts_rows["speed_rep"], errors="coerce").dropna()
        if len(ts_rows):
            nz = sp[sp > 0]
            print(f"  REPORTED speed (raw number in the feed's speed field, Transport Stack rows): present on {100 * len(sp) / len(ts_rows):.0f}% of rows, zero on {100 * (sp == 0).mean() if len(sp) else 0:.0f}% of those"
                  + (f"; non-zero: median {nz.median():.1f}, 90th percentile {nz.quantile(.9):.1f}, max {sp.max():.1f}  (if the unit were m/s: median {nz.median() * 3.6:.0f} km/h, 90th percentile {nz.quantile(.9) * 3.6:.0f} km/h)" if len(nz) else ""))
        calc = pd.to_numeric(df["speed_calc_kmh"], errors="coerce").dropna()
        if len(calc):
            print(f"  CALCULATED speed (from positions): median {calc.median():.1f} km/h, share moving (>2 km/h) {100 * (calc > 2).mean():.0f}%, among moving: median {calc[calc > 2].median():.1f}, 90th percentile {calc[calc > 2].quantile(.9):.1f} km/h")
        if len(sp):
            g = df.sort_values(["vehicle_id", "ts"]).copy()
            g["rep"] = pd.to_numeric(g["speed_rep"], errors="coerce")
            g["calc_prev"] = pd.to_numeric(g["speed_calc_kmh"], errors="coerce")             # interval that ENDS at this ping
            same = g["vehicle_id"] == g["vehicle_id"].shift(-1)
            g["calc_next"] = g["calc_prev"].shift(-1).where(same)                          # interval that STARTS at this ping
            ok = g[(g["source"] == "tstack") & g["rep"].notna() & pd.to_numeric(g["dt_s"], errors="coerce").between(10, 60)]
            pp, pn = ok.dropna(subset=["calc_prev"]), ok.dropna(subset=["calc_next"])
            if len(pp) >= 200 and len(pn) >= 200:
                r_prev, r_next = pp["rep"].corr(pp["calc_prev"], method="spearman"), pn["rep"].corr(pn["calc_next"], method="spearman")
                print(f"  SPEED CHECK on {len(pp):,} pings (10-60 s since the previous ping): rank correlation of reported speed with the calculated speed of the interval BEFORE this ping {r_prev:+.2f}, of the interval AFTER it {r_next:+.2f}")
                bins = [-0.01, 0.5, 2, 5, 8, 12, 16, 20, 1e9]
                names = ["0", "0.5-2", "2-5", "5-8", "8-12", "12-16", "16-20", "20+"]
                tb = pp.groupby(pd.cut(pp["rep"], bins, labels=names), observed=True)["calc_prev"].agg(["median", "size"])
                print("     reported (raw) -> median calculated km/h (pings): " + " | ".join(f"{k}: {v['median']:.0f} ({int(v['size']):,})" for k, v in tb.iterrows()))
                mv = pp[(pp["rep"] > 1) & (pp["calc_prev"] > 5)]
                if len(mv) >= 100:
                    k = (mv["calc_prev"] / mv["rep"]).median()
                    unit = min({"m/s": 3.6, "km/h": 1.0, "knots": 1.852, "mph": 1.609}.items(), key=lambda kv: abs(math.log(k / kv[1])))
                    guess = f"closest unit: {unit[0]}" if abs(math.log(k / unit[1])) < 0.2 else "does not match a common unit"
                else:
                    k, guess = float("nan"), "too few moving pings"
                if max(r_prev, r_next) < 0.3:
                    print(f"     VERDICT: the reported speed does NOT follow the movement of the bus (correlation below 0.3). Do not use it for ETA until this is understood. (calculated / reported = {k:.2f} km/h per unit, {guess})")
                else:
                    print(f"     VERDICT: the reported speed follows the movement of the bus. calculated / reported = {k:.2f} km/h per unit ({guess}; 3.6 = m/s, 1.0 = km/h). "
                          + ("The interval AFTER the ping fits better." if r_next > r_prev + 0.05 else "The interval BEFORE the ping fits better." if r_prev > r_next + 0.05 else ""))
        sched = df.groupby(df["schedule"].fillna("(none)")).agg(rows=("ts", "size"), trip_in_static=("trip_in_static", lambda s: pd.to_numeric(s, errors="coerce").mean()))
        print("  schedule_relationship: " + ", ".join(f"{k} {int(v.rows):,} rows (trip_id found in static: {'n/a' if pd.isna(v.trip_in_static) else f'{100 * v.trip_in_static:.0f}%'})" for k, v in sched.iterrows()))
        ex = df["extra"].dropna()
        if len(ex):
            print(f"  EXTRA fields the feed sent that we did not expect: {len(ex):,} rows, e.g. {ex.iloc[0][:100]}")
        rin = pd.to_numeric(df["route_in_static"], errors="coerce").mean()
        print(f"  flagged gps_jump: {(df['flag'] == 'gps_jump').sum():,}   route_id found in static: {'n/a (no GTFS was loaded)' if pd.isna(rin) else f'{100 * rin:.0f}%'}")
        print("  rows per hour:  " + " ".join(f"{h:02d}h:{n:,}" for h, n in hrs.items()))
        print("  busiest routes: " + ", ".join(f"{r} ({n:,})" for r, n in df["route_id"].value_counts().head(8).items()))
    hf = [f for f in sorted((d / "health").glob("*.csv")) if header_ok(f, HEALTH_COLUMNS)]
    if hf:
        h = pd.concat([pd.read_csv(f, dtype={"event": str, "error": str}) for f in hf])
        print("\n=== FEED HEALTH ===")
        for src, g in h.groupby("source"):
            ok = g["error"].isna()
            probs = g["error"].dropna().map(lambda x: x[:60]).value_counts().to_dict()
            print(f"{LABEL.get(src, src)}: {len(g):,} polls, clean {100 * ok.mean():.1f}%, median download {g['latency_s'].median():.2f} s, median vehicles {g['vehicles_in_feed'].median():,.0f}, "
                  f"median file age {pd.to_numeric(g['feed_age_s'], errors='coerce').median():.0f} s")
            print(f"    problems: {probs or 'none'}")
            gg = g[g["error"].isna()].sort_values("fetched_at").copy()
            if len(gg) >= 20:
                gap = gg["fetched_at"].diff().dropna(); gap = gap[gap < 60]
                new = pd.to_numeric(gg["rows_saved"], errors="coerce") > 0
                age = pd.to_numeric(gg["feed_age_s"], errors="coerce")
                print(f"    polls were {gap.median():.0f} s apart; {100 * (~new).mean():.0f}% of polls brought NO new rows (the feed file had not changed). "
                      f"File age on polls with new rows: median {age[new].median():.0f} s; without: median {age[~new].median():.0f} s; oldest file seen {age.quantile(.95):.0f} s (95th pct)"
                      f"  -> the feed file seems to be renewed about every {age.quantile(.95):.0f} s")
        ev = h["event"].dropna()
        print(f"feed switches: {len(ev)}" + (": " + "; ".join(ev.head(5)) if len(ev) else ""))


def main():
    ap = argparse.ArgumentParser(description="OTD live collector v2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--plain", action="store_true", help="print readable text instead of the dashboard")
    r.add_argument("--demo", action="store_true", help="fake feeds (no keys needed), saves into data_demo/")
    r.add_argument("--once", action="store_true", help="do one poll and stop (implies --plain)")
    r.add_argument("--save-raw", action="store_true", help="also keep every downloaded file in data/raw/ (about 100-200 MB per 5 h)")
    r.add_argument("--using-otd", action="store_true", help="OTD Delhi as the main feed (Transport Stack becomes the fallback)")
    r.add_argument("--no-fallback", action="store_true", help="never switch to the other feed")
    r.add_argument("--interval", type=int, default=INTERVAL)
    r.add_argument("--data", default=None)
    r.add_argument("--gtfs", default=GTFS_DIR)
    i = sub.add_parser("inspect"); i.add_argument("--data", default=DATA_DIR)
    a = ap.parse_args()
    if a.cmd == "inspect":
        return inspect(a.data)
    primary = "otd" if a.using_otd else "tstack"
    other = "tstack" if a.using_otd else "otd"
    if a.demo:
        buses = demo_buses()
        sources = {"tstack": FakeSource("tstack", buses), "otd": FakeSource("otd", buses)}
        if a.no_fallback:
            sources = {primary: sources[primary]}
        data = a.data or "data_demo"
    else:
        keys = {"otd": OTD_API_KEY, "tstack": TSTACK_API_KEY}
        if not key_ok(keys[primary]):
            raise SystemExit(f"Paste your {LABEL[primary]} API key into {'OTD_API_KEY' if primary == 'otd' else 'TSTACK_API_KEY'} at the top of otd_collector.py."
                             + ("" if a.using_otd else "  (To use OTD as the main feed instead, run with --using-otd.)"))
        sources = {primary: LiveSource(primary, keys[primary])}
        if not a.no_fallback:
            if key_ok(keys[other]):
                sources[other] = LiveSource(other, keys[other])
            else:
                print(f"Note: no {LABEL[other]} key pasted, so there is no fallback feed.")
        data = a.data or DATA_DIR
    check_formats(data)
    static = None
    try:
        static = load_static(a.gtfs)
    except Exception as e:
        print(f"(could not read GTFS from {a.gtfs}: {e})")
    if static is None:
        print(f"Note: no GTFS in '{a.gtfs}/' -> trip/route match columns will stay empty.")
    c = Collector(FeedManager(primary, sources), static, data, max(MIN_INTERVAL, a.interval), save_raw=a.save_raw)
    if a.plain or a.once or not sys.stdout.isatty():
        run_plain(c, a.once)
    else:
        run_dashboard(c)


if __name__ == "__main__":
    main()
