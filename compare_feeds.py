#!/usr/bin/env python3
"""
compare_feeds.py - compare the two live bus feeds of Delhi:

  A) OTD Delhi              https://otd.delhi.gov.in/documentation/            (key goes in the URL:  ?key=...)
  B) Delhi Transport Stack  https://delhi.transportstack.in/data-services/datadetails/7   (key goes in the header: x-api-key)

Both give a GTFS-Realtime "VehiclePositions.pb" file. This script downloads both, reads every field in them
(with its own small protobuf reader, so it also finds fields that are NOT in the standard GTFS-Realtime format)
and shows, in plain English:
   1 what was downloaded                       6 the fields that matter for this project
   2 feed headers and entity types             7 do the two feeds describe the same buses? (and do their values agree?)
   3 how fresh the data is                     8 do the ids match your static GTFS files?
   4 which fields each feed has                9 sample data
   5 extra / unknown fields                   10 summary and what it means for the project

HOW TO RUN
   1. paste your two keys below (or set env vars OTD_API_KEY and TSTACK_API_KEY)
   2. python compare_feeds.py                          one snapshot of each feed
      python compare_feeds.py --polls 4 --interval 15  also compare how often each feed changes
      python compare_feeds.py --offline feed_compare_out   analyse the files saved by an earlier run (no keys needed)
Everything printed is also saved in  feed_compare_out/report.txt  (keys are never written there).
"""
import argparse, csv, json, math, os, struct, sys, time
from collections import Counter
from pathlib import Path
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# ============================ CONFIG: paste your keys here ============================
OTD_API_KEY = os.getenv("OTD_API_KEY")
TSTACK_API_KEY = os.getenv("TSTACK_API_KEY")
# ======================================================================================
OTD_URL = "https://otd.delhi.gov.in/api/realtime/VehiclePositions.pb"
TS_URL = "https://dts-backend.transportstack.in/api/dataset/otd/get-file"
TS_PARAMS = {"agency": "delhi-buses", "category": "realtime_gtfs", "filename": "VehiclePositions.pb"}
GTFS_DIR = "gtfs"
OUT_DIR = "feed_compare_out"
NAMES = {"A": "OTD", "B": "TStack"}
LONG = {"A": "OTD Delhi", "B": "Delhi Transport Stack"}
BOX = (28.20, 29.10, 76.60, 77.70)            # Delhi-NCR box used for a sanity check of positions
IST = 19800

# ============================ tiny GTFS-Realtime reader ============================
ENUMS = {
    "Incrementality": {0: "FULL_DATASET", 1: "DIFFERENTIAL"},
    "ScheduleRelationship": {0: "SCHEDULED", 1: "ADDED", 2: "UNSCHEDULED", 3: "CANCELED", 5: "REPLACEMENT", 6: "DUPLICATED", 7: "DELETED", 8: "NEW"},
    "VehicleStopStatus": {0: "INCOMING_AT", 1: "STOPPED_AT", 2: "IN_TRANSIT_TO"},
    "CongestionLevel": {0: "UNKNOWN_CONGESTION_LEVEL", 1: "RUNNING_SMOOTHLY", 2: "STOP_AND_GO", 3: "CONGESTION", 4: "SEVERE_CONGESTION"},
    "OccupancyStatus": {0: "EMPTY", 1: "MANY_SEATS_AVAILABLE", 2: "FEW_SEATS_AVAILABLE", 3: "STANDING_ROOM_ONLY", 4: "CRUSHED_STANDING_ROOM_ONLY",
                        5: "FULL", 6: "NOT_ACCEPTING_PASSENGERS", 7: "NO_DATA_AVAILABLE", 8: "NOT_BOARDABLE"},
    "WheelchairAccessible": {0: "NO_VALUE", 1: "UNKNOWN", 2: "WHEELCHAIR_ACCESSIBLE", 3: "WHEELCHAIR_INACCESSIBLE"},
}
SCHEMA = {   # message -> field number -> (name, type)
    "FeedMessage": {1: ("header", "FeedHeader"), 2: ("entity", "FeedEntity")},
    "FeedHeader": {1: ("gtfs_realtime_version", "string"), 2: ("incrementality", "enum:Incrementality"), 3: ("timestamp", "uint"), 4: ("feed_version", "string")},
    "FeedEntity": {1: ("id", "string"), 2: ("is_deleted", "bool"), 3: ("trip_update", "raw"), 4: ("vehicle", "VehiclePosition"), 5: ("alert", "raw")},
    "VehiclePosition": {1: ("trip", "TripDescriptor"), 2: ("position", "Position"), 3: ("current_stop_sequence", "uint"),
                        4: ("current_status", "enum:VehicleStopStatus"), 5: ("timestamp", "uint"), 6: ("congestion_level", "enum:CongestionLevel"),
                        7: ("stop_id", "string"), 8: ("vehicle", "VehicleDescriptor"), 9: ("occupancy_status", "enum:OccupancyStatus"),
                        10: ("occupancy_percentage", "uint"), 11: ("multi_carriage_details", "raw")},
    "Position": {1: ("latitude", "float"), 2: ("longitude", "float"), 3: ("bearing", "float"), 4: ("odometer", "double"), 5: ("speed", "float")},
    "TripDescriptor": {1: ("trip_id", "string"), 2: ("start_time", "string"), 3: ("start_date", "string"), 4: ("schedule_relationship", "enum:ScheduleRelationship"),
                       5: ("route_id", "string"), 6: ("direction_id", "uint"), 7: ("modified_trip", "raw")},
    "VehicleDescriptor": {1: ("id", "string"), 2: ("label", "string"), 3: ("license_plate", "string"), 4: ("wheelchair_accessible", "enum:WheelchairAccessible")},
}
REPEATED = {("FeedMessage", "entity"), ("VehiclePosition", "multi_carriage_details")}
WIRE = {"string": 2, "raw": 2, "float": 5, "double": 1, "uint": 0, "bool": 0}
HINTS = {("FeedEntity", 6): "shape (experimental GTFS-RT)", ("FeedEntity", 7): "stop (experimental GTFS-RT)", ("FeedEntity", 8): "trip_modifications (experimental GTFS-RT)"}


def read_varint(buf, i):
    shift = result = 0
    while True:
        b = buf[i]; i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7
        if shift > 70:
            raise ValueError("bad varint")


def show_raw(wt, raw):
    if wt == 0: return str(raw)
    if wt == 1: return f"8 bytes, as double {struct.unpack('<d', raw)[0]:.6g}"
    if wt == 5: return f"4 bytes, as float {struct.unpack('<f', raw)[0]:.6g}"
    try:
        t = raw.decode("utf-8")
        if t.isprintable():
            return f'text "{t[:40]}"'
    except UnicodeDecodeError:
        pass
    return f"{len(raw)} bytes"


def parse_msg(buf, name, unknown):
    out, i, spec = {}, 0, SCHEMA[name]
    while i < len(buf):
        key, i = read_varint(buf, i)
        fn, wt = key >> 3, key & 7
        if wt == 0: raw, i = read_varint(buf, i)
        elif wt == 1: raw, i = buf[i:i + 8], i + 8
        elif wt == 2:
            ln, i = read_varint(buf, i); raw, i = buf[i:i + ln], i + ln
        elif wt == 5: raw, i = buf[i:i + 4], i + 4
        else: raise ValueError(f"unsupported wire type {wt} in {name}")
        f = spec.get(fn)
        if f is None:
            u = unknown.setdefault((name, fn, wt, ""), {"count": 0, "examples": []})
            u["count"] += 1
            if len(u["examples"]) < 3: u["examples"].append(show_raw(wt, raw))
            continue
        fname, ftype = f
        base = ftype.split(":")[0] if ftype.startswith("enum") else ftype
        want = 0 if ftype.startswith("enum") else WIRE.get(ftype, 2)
        if wt != want:                                            # same number but a different kind of value than expected
            u = unknown.setdefault((name, fn, wt, f" (expected {fname} as {ftype})"), {"count": 0, "examples": []})
            u["count"] += 1
            if len(u["examples"]) < 3: u["examples"].append(show_raw(wt, raw))
            continue
        if ftype in SCHEMA: val = parse_msg(raw, ftype, unknown)
        elif ftype == "raw": val = f"<{len(raw)} bytes>"
        elif ftype == "string": val = raw.decode("utf-8", "replace")
        elif ftype == "uint": val = raw
        elif ftype == "bool": val = bool(raw)
        elif ftype == "float": val = struct.unpack("<f", raw)[0]
        elif ftype == "double": val = struct.unpack("<d", raw)[0]
        else: val = ENUMS[ftype.split(":")[1]].get(raw, f"UNKNOWN_VALUE_{raw}")
        if (name, fname) in REPEATED: out.setdefault(fname, []).append(val)
        else: out[fname] = val
    return out


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flatten(v, prefix + k + "."))
        elif isinstance(v, list): out[prefix + k] = f"<{len(v)} items>"
        else: out[prefix + k] = v
    return out


def parse_feed(data):
    unknown = {}
    msg = parse_msg(data, "FeedMessage", unknown)
    rows, types = [], Counter()
    for e in msg.get("entity", []):
        for t in ("trip_update", "vehicle", "alert"):
            if t in e: types[t] += 1
        if e.get("is_deleted"): types["is_deleted"] += 1
        if "vehicle" in e:
            r = flatten(e["vehicle"]); r["entity_id"] = e.get("id", ""); rows.append(r)
    return {"header": msg.get("header", {}), "n_entities": len(msg.get("entity", [])), "types": types, "rows": rows, "unknown": unknown}


# ============================ downloading ============================
def redact(text, keys):
    for k in keys:
        if k and len(k) > 4:
            text = str(text).replace(k, "***")
    return str(text)


def download(which, key, url, timeout=25):
    t0 = time.time()
    res = {"ok": False, "status": None, "bytes": b"", "latency": None, "headers": {}, "error": None}
    try:
        if which == "A":
            r = requests.get(url, params={"key": key}, timeout=timeout)
        else:
            r = requests.get(url, params=TS_PARAMS, headers={"x-api-key": key}, timeout=timeout)
    except Exception as e:
        res["error"] = redact(f"{type(e).__name__}: {e}", [key])[:300]
        return res
    res.update(status=r.status_code, bytes=r.content, latency=time.time() - t0, headers=dict(r.headers))
    body = r.content[:300]
    if r.status_code in (401, 403):
        res["error"] = f"HTTP {r.status_code}: the server rejected the key. Check the key and how it is sent ({'?key=... in the URL' if which == 'A' else 'x-api-key header'}). Server said: {redact(body.decode('utf-8', 'replace'), [key])!r}"
    elif r.status_code == 429:
        res["error"] = "HTTP 429: too many requests. Wait a minute and try again."
    elif r.status_code != 200:
        res["error"] = f"HTTP {r.status_code}. Server said: {redact(body.decode('utf-8', 'replace'), [key])!r}"
    elif body[:1] in (b"{", b"<", b"["):
        res["error"] = f"HTTP 200 but the answer is text, not a .pb file. It says: {redact(body.decode('utf-8', 'replace'), [key])!r}"
    else:
        res["ok"] = True
    return res


# ============================ small helpers ============================
LOG = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)
def h(title):
    p(); p("=" * 100); p(title); p("=" * 100)
def say(text, indent="  "):
    import textwrap
    for line in textwrap.wrap(text, 98 - len(indent)):
        p(indent + line)
def ist(t):
    return time.strftime("%H:%M:%S", time.gmtime(t + IST))
def pct(x, n):
    return f"{100 * x / n:.0f}%" if n else "-"
def table(headers, rows, maxw=34):
    rows = [[str(c)[:maxw] for c in r] for r in rows]
    w = [max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i]) for i in range(len(headers))]
    p("  " + "  ".join(headers[i].ljust(w[i]) for i in range(len(headers))))
    p("  " + "  ".join("-" * w[i] for i in range(len(headers))))
    for r in rows:
        p("  " + "  ".join(r[i].ljust(w[i]) for i in range(len(headers))))
def hav(la1, lo1, la2, lo2):
    p1, p2 = math.radians(la1), math.radians(la2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(a))
def quant(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None
def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def census(rows):
    n, keys, out = len(rows), set(), {}
    for r in rows: keys |= r.keys()
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        isnum = all(num(v) for v in vals)
        out[k] = {"present": len(vals), "present_pct": 100 * len(vals) / n if n else 0, "numeric": isnum,
                  "zero_pct": 100 * sum(1 for v in vals if v == 0) / len(vals) if isnum and vals else None,
                  "empty_pct": 100 * sum(1 for v in vals if v == "") / len(vals) if not isnum and vals else None,
                  "distinct": len(set(vals)), "examples": list(dict.fromkeys((f"{v:.6f}" if isinstance(v, float) else str(v)) for v in vals))[:3],
                  "min": min(vals) if isnum and vals else None, "max": max(vals) if isnum and vals else None}
    return out


def informative(st):
    """Does this field really tell us something? (present, not always 0 / empty, not one constant value)"""
    if not st or st["present"] == 0: return False
    if st["numeric"] and st["zero_pct"] is not None and st["zero_pct"] >= 99: return False
    if st["empty_pct"] is not None and st["empty_pct"] >= 99: return False
    return st["distinct"] > 1


def all_schema_fields():
    out = []
    def walk(msg, prefix):
        for fn, (name, typ) in sorted(SCHEMA[msg].items()):
            if typ in SCHEMA: walk(typ, prefix + name + ".")
            else: out.append(prefix + name)
    walk("VehiclePosition", "")
    return out


IMPORTANT = [
    ("position.latitude", "Where the bus is. Needed for everything."),
    ("position.longitude", "Where the bus is. Needed for everything."),
    ("timestamp", "When the bus device recorded the position. Needed to know how fresh a ping is and to work out speed."),
    ("position.speed", "Speed reported by the bus (metres per second). If it is always 0 we must work speed out from positions."),
    ("position.bearing", "Direction the bus faces (degrees). Helps to decide UP/DOWN and to find wrong direction labels."),
    ("position.odometer", "Distance travelled by the bus. Would give exact distance between pings."),
    ("trip.route_id", "Which route (in Delhi one route_id = one direction of a route)."),
    ("trip.trip_id", "Which scheduled trip. Useful only if it matches the static timetable."),
    ("trip.direction_id", "Direction 0/1 of the trip."),
    ("trip.start_time", "Start time of the trip."),
    ("trip.start_date", "Service day of the trip."),
    ("vehicle.id", "Stable id to follow one bus over time."),
    ("vehicle.license_plate", "Number plate (another way to identify the bus)."),
    ("stop_id", "Stop the bus is at / heading to. Would give stop progress without map matching."),
    ("current_stop_sequence", "Position of that stop in the trip."),
    ("current_status", "IN_TRANSIT_TO / STOPPED_AT / INCOMING_AT. STOPPED_AT would show real halts (dwell time)."),
    ("occupancy_status", "How full the bus is."),
    ("occupancy_percentage", "How full the bus is, in percent."),
    ("congestion_level", "Traffic around the bus."),
]


# ============================ the report ============================
def main():
    ap = argparse.ArgumentParser(description="Compare the OTD Delhi and Delhi Transport Stack bus feeds")
    ap.add_argument("--polls", type=int, default=1, help="how many times to download both feeds (to compare update behaviour)")
    ap.add_argument("--interval", type=int, default=15, help="seconds between polls")
    ap.add_argument("--samples", type=int, default=3, help="how many sample vehicles to show")
    ap.add_argument("--offline", help="folder with otd.pb and transportstack.pb saved by an earlier run (no keys, no internet)")
    ap.add_argument("--gtfs", default=GTFS_DIR); ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--otd-url", default=OTD_URL); ap.add_argument("--ts-url", default=TS_URL)
    a = ap.parse_args()
    keyA = os.getenv("OTD_API_KEY") or OTD_API_KEY
    keyB = os.getenv("TSTACK_API_KEY") or TSTACK_API_KEY
    keys = [keyA, keyB]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = {"A": "otd.pb", "B": "transportstack.pb"}

    h("COMPARING THE TWO LIVE BUS FEEDS OF DELHI")
    say("Both feeds give a GTFS-Realtime VehiclePositions file: one entry per bus with its position and some details. "
        "This script downloads both, reads every field, and shows what each feed has and what it does not have. "
        "Names used below: OTD = otd.delhi.gov.in, TStack = Delhi Transport Stack.")

    # ---------- 1 download ----------
    h("1. DOWNLOAD")
    data, dl, snaps = {}, {"A": [], "B": []}, {"A": [], "B": []}
    if a.offline:
        for w in "AB":
            f = Path(a.offline) / files[w]
            if f.exists():
                data[w] = f.read_bytes(); dl[w].append({"ok": True, "status": "file", "bytes": data[w], "latency": None, "headers": {}, "error": None, "t": f.stat().st_mtime})
                p(f"  {LONG[w]}: read {len(data[w]):,} bytes from {f}")
            else:
                p(f"  {LONG[w]}: {f} not found, skipped")
    else:
        todo = {"A": keyA, "B": keyB}
        for w, k in todo.items():
            if k.startswith("PASTE_"):
                p(f"  {LONG[w]}: key not set (paste it at the top of this file or set the env variable). This feed is skipped.")
        for n in range(a.polls):
            if n:
                time.sleep(a.interval)
            for w, k in todo.items():
                if k.startswith("PASTE_"): continue
                r = download(w, k, a.otd_url if w == "A" else a.ts_url); r["t"] = time.time()
                dl[w].append(r)
                if r["ok"]:
                    data[w] = r["bytes"]
                if a.polls > 1 or not r["ok"]:
                    p(f"  poll {n + 1}/{a.polls} {LONG[w]}: " + (f"ok {len(r['bytes']):,} bytes in {r['latency']:.2f} s" if r["ok"] else "FAILED " + (r["error"] or "")))
                if r["ok"]:
                    (out / files[w]).write_bytes(r["bytes"])
    for w in "AB":
        last = dl[w][-1] if dl[w] else None
        if not last: continue
        if last["ok"]:
            if last["status"] != "file":
                p(f"  {LONG[w]}: HTTP {last['status']}, {len(last['bytes']):,} bytes, {last['latency']:.2f} s")
                hd = {k.lower(): v for k, v in last["headers"].items()}
                keep = [k for k in ("content-type", "content-length", "last-modified", "etag", "cache-control", "age", "date", "server") if k in hd]
                keep += [k for k in hd if ("rate" in k or "limit" in k) and k not in keep]
                for k in keep: p(f"      {k}: {redact(hd[k], keys)}")
        else:
            p(f"  {LONG[w]}: could not download. {last['error']}")
    parsed = {}
    for w in "AB":
        if w in data:
            try:
                parsed[w] = parse_feed(data[w])
            except Exception as e:
                p(f"  {LONG[w]}: the file could not be read as GTFS-Realtime ({type(e).__name__}: {e}). First bytes: {data[w][:60]!r}")
    try:
        from google.transit import gtfs_realtime_pb2          # optional self-check of this script's reader
        for w in parsed:
            fm = gtfs_realtime_pb2.FeedMessage(); fm.ParseFromString(data[w])
            n_off = sum(1 for e in fm.entity if e.HasField("vehicle"))
            p(f"  self-check {NAMES[w]}: this script found {len(parsed[w]['rows']):,} vehicles, the official library found {n_off:,}" + ("  (same)" if n_off == len(parsed[w]["rows"]) else "  (DIFFERENT, please report)"))
    except ImportError:
        pass
    if not parsed:
        p(); say("Nothing to compare. Paste at least one key at the top of the file (or use --offline).")
        return finish(out, keys)
    now = time.time()

    # ---------- 2 headers ----------
    h("2. FEED HEADER AND ENTITY TYPES")
    say("The header says which version of the format is used and when the whole file was made. Entity types: a feed can hold vehicle positions, trip updates (predicted arrival times per stop) and alerts.")
    rows = []
    for w in parsed:
        hd, ts = parsed[w]["header"], parsed[w]["header"].get("timestamp")
        rows.append([NAMES[w], hd.get("gtfs_realtime_version", "-"), hd.get("incrementality", "-"), (f"{ist(ts)} IST ({now - ts:.0f} s old)" if ts else "-"),
                     parsed[w]["n_entities"], parsed[w]["types"].get("vehicle", 0), parsed[w]["types"].get("trip_update", 0), parsed[w]["types"].get("alert", 0)])
    table(["feed", "version", "incrementality", "file made at", "entities", "vehicles", "trip_updates", "alerts"], rows, 40)
    for w in parsed:
        if parsed[w]["types"].get("trip_update"):
            p(f"  NOTE: {NAMES[w]} also contains {parsed[w]['types']['trip_update']} TRIP UPDATES (predicted arrival times). That would be very valuable for ETA. Look inside this feed more closely.")

    # ---------- 3 freshness ----------
    h("3. HOW FRESH IS THE DATA")
    say("Each bus has its own timestamp. 'Age' = download time minus that timestamp. Old pings mean the bus has stopped reporting.")
    rows = []
    for w in parsed:
        ts = [r["timestamp"] for r in parsed[w]["rows"] if "timestamp" in r]
        ages = [dl[w][-1].get("t", now) - t for t in ts]
        if ages:
            rows.append([NAMES[w], len(ts), f"{quant(ages, .5):.0f} s", f"{quant(ages, .9):.0f} s", f"{max(ages):.0f} s", pct(sum(1 for x in ages if x > 300), len(ages)), pct(sum(1 for x in ages if x < -30), len(ages))])
    table(["feed", "buses with time", "median age", "90th pct age", "oldest", "older than 5 min", "time in the future"], rows)
    if a.polls > 1:
        p(); say("Update behaviour over the polls (how many buses got a NEW timestamp between two polls):")
        for w in "AB":
            res = [d for d in dl[w] if d["ok"]]
            if len(res) < 2: continue
            ps = [parse_feed(d["bytes"]) for d in res]
            ch, hdr_changes = [], 0
            for x, y in zip(ps, ps[1:]):
                mx = {r.get("vehicle.id") or r["entity_id"]: r.get("timestamp") for r in x["rows"]}
                my = {r.get("vehicle.id") or r["entity_id"]: r.get("timestamp") for r in y["rows"]}
                common = [k for k in my if k in mx and my[k] is not None and mx[k] is not None]
                ch.append(sum(1 for k in common if my[k] > mx[k]) / max(1, len(common)))
                hdr_changes += y["header"].get("timestamp") != x["header"].get("timestamp")
            p(f"  {NAMES[w]}: {len(res)} good polls of {len(dl[w])}; vehicles count {min(len(x['rows']) for x in ps):,}-{max(len(x['rows']) for x in ps):,}; "
              f"header time changed in {hdr_changes}/{len(ps) - 1} steps; share of buses with a newer ping per step: " + ", ".join(f"{100 * c:.0f}%" for c in ch))

    # ---------- 4 field matrix ----------
    h("4. WHICH FIELDS DOES EACH FEED HAVE?")
    say("Every field found in the vehicle entries. 'present' = share of buses that have the field. 'zero' = share of those values that are 0 (a field that is always 0 carries no information). "
        "'distinct' = how many different values appear.")
    cen = {w: census(parsed[w]["rows"]) for w in parsed}
    fields = sorted(set().union(*[set(c) for c in cen.values()]) - {"entity_id"})
    rows = []
    def cell(w, f):
        s = cen.get(w, {}).get(f)
        if not s: return "-", "", "", ""
        z = f"{s['zero_pct']:.0f}%" if s["zero_pct"] is not None else (f"{s['empty_pct']:.0f}% empty" if s["empty_pct"] else "")
        return f"{s['present_pct']:.0f}%", z, str(s["distinct"]), ", ".join(s["examples"][:2])
    for f in fields:
        ca, cb = cell("A", f), cell("B", f)
        rows.append([f, ca[0], ca[1], ca[2], cb[0], cb[1], cb[2], (cb[3] if cb[3] else ca[3])[:30]])
    table(["field", "OTD present", "zero", "distinct", "TStack present", "zero", "distinct", "example"], rows, 30)
    missing = [f for f in all_schema_fields() if f not in fields]
    p(); say("Standard GTFS-Realtime vehicle fields that NEITHER feed has: " + (", ".join(missing) if missing else "none") + ".")

    # ---------- 5 unknown fields ----------
    h("5. EXTRA FIELDS THAT ARE NOT IN THE STANDARD FORMAT")
    say("If a feed carries fields that the standard GTFS-Realtime format does not define, they show up here with their field number and some example values. This is where 'extra information' of a feed would hide.")
    any_unknown = False
    for w in parsed:
        for (msg, fn, wt, note), u in sorted(parsed[w]["unknown"].items()):
            any_unknown = True
            p(f"  {NAMES[w]}: inside {msg}, field number {fn} ({['varint', '8-byte', 'text/bytes', '', '', '4-byte'][wt] if wt in (0, 1, 2, 5) else wt}){note or ''}"
              f"{' = ' + HINTS[(msg, fn)] if (msg, fn) in HINTS else ''}: {u['count']:,} times, examples: {', '.join(u['examples'])}")
    if not any_unknown:
        p("  none: both feeds only use standard GTFS-Realtime fields.")

    # ---------- 6 important ----------
    h("6. THE FIELDS THAT MATTER FOR THIS PROJECT")
    rows = []
    for f, why in IMPORTANT:
        def st(w):
            s = cen.get(w, {}).get(f)
            if w not in parsed: return "(no data)"
            if not s: return "missing"
            if not informative(s):
                return "always 0" if s["numeric"] and s["zero_pct"] is not None and s["zero_pct"] >= 99 else ("empty" if s["empty_pct"] and s["empty_pct"] >= 99 else f"constant ({s['examples'][0][:14]})")
            return f"yes {s['present_pct']:.0f}%"
        rows.append([f, st("A"), st("B"), why[:62]])
    table(["field", "OTD", "TStack", "why it matters"], rows, 62)
    p(); say("Meaning of the entries: 'yes' = the field is filled and varies. 'missing' = not in the feed. 'always 0' / 'empty' / 'constant' = the field exists but carries no information.")

    # ---------- 7 same buses ----------
    pair = None
    if "A" in parsed and "B" in parsed:
        h("7. DO THE FEEDS DESCRIBE THE SAME BUSES?")
        say("Buses are matched by id. The script tries every pair of id fields (vehicle.id, vehicle.label, vehicle.license_plate, entity id) and uses the pair with most buses in common.")
        norm = lambda v: str(v).strip().upper().replace(" ", "").replace("-", "")
        idf = ["vehicle.id", "vehicle.label", "vehicle.license_plate", "entity_id"]
        def idmap(w, f):
            m = {}
            for r in parsed[w]["rows"]:
                if r.get(f) not in (None, ""):
                    k = norm(r[f])
                    if k not in m or (r.get("timestamp") or 0) > (m[k].get("timestamp") or 0): m[k] = r
            return m
        best = (0, None, None)
        for fa in idf:
            ma = idmap("A", fa)
            for fb in idf:
                mb = idmap("B", fb)
                n = len(ma.keys() & mb.keys())
                if n > best[0]: best = (n, fa, fb)
        nA, nB = len(parsed["A"]["rows"]), len(parsed["B"]["rows"])
        p(f"  OTD buses: {nA:,}   TStack buses: {nB:,}")
        for w in "AB":
            ex = [r.get("vehicle.id") for r in parsed[w]["rows"][:4]]
            p(f"  example ids in {NAMES[w]}: {ex}")
        if best[0] == 0:
            p("  No bus ids are in common: the two feeds use different id formats (or different buses). Value comparison is skipped.")
        else:
            n, fa, fb = best
            ma, mb = idmap("A", fa), idmap("B", fb)
            p(f"  best match: OTD '{fa}' = TStack '{fb}'. In both: {n:,}  | only in OTD: {len(ma) - n:,}  | only in TStack: {len(mb) - n:,}")
            say(f"So TStack has {pct(n, len(ma))} of the OTD buses, and OTD has {pct(n, len(mb))} of the TStack buses.")
            common = sorted(ma.keys() & mb.keys())
            pair = (common, ma, mb)
            d, dts, same_route, same_trip = [], [], [0, 0], [0, 0]
            fieldcmp = {}
            for k in common:
                ra, rb = ma[k], mb[k]
                if all(x in ra and x in rb for x in ("position.latitude", "position.longitude")):
                    d.append(hav(ra["position.latitude"], ra["position.longitude"], rb["position.latitude"], rb["position.longitude"]))
                if "timestamp" in ra and "timestamp" in rb: dts.append(rb["timestamp"] - ra["timestamp"])
                for f in set(ra) & set(rb) - {"entity_id", "position.latitude", "position.longitude", "timestamp"}:
                    c = fieldcmp.setdefault(f, [0, 0])
                    c[0] += 1; c[1] += (ra[f] == rb[f]) or (num(ra[f]) and num(rb[f]) and abs(ra[f] - rb[f]) < 1e-6)
            if d: p(f"  distance between the two positions of the same bus: median {quant(d, .5):.0f} m, 90th pct {quant(d, .9):.0f} m, largest {max(d):,.0f} m")
            if dts:
                newer = sum(1 for x in dts if x > 0); older = sum(1 for x in dts if x < 0)
                p(f"  timestamp difference (TStack minus OTD): median {quant(dts, .5):+.0f} s | TStack newer for {pct(newer, len(dts))}, same for {pct(len(dts) - newer - older, len(dts))}, older for {pct(older, len(dts))}")
                say("A positive difference means TStack has a newer ping of that bus. (The two downloads are about a second apart, so small differences are normal.)")
            if fieldcmp:
                p(); p("  fields that exist in both feeds: do the values agree for the same bus?")
                table(["field", "buses compared", "same value"], [[f, c[0], pct(c[1], c[0])] for f, c in sorted(fieldcmp.items())])
                say("A low 'same value' means the two feeds use different formats for that field (for example different route ids). Look at the samples in section 9.")

    # ---------- 8 static ids ----------
    gt = Path(a.gtfs)
    h("8. DO THE IDS MATCH YOUR STATIC GTFS FILES?")
    ids = {}
    def readcol(fn, col):
        try:
            with open(gt / fn, newline="", encoding="utf-8-sig") as f:
                return {row[col] for row in csv.DictReader(f) if row.get(col)}
        except Exception:
            return None
    for fn, col, key in (("routes.txt", "route_id", "trip.route_id"), ("trips.txt", "trip_id", "trip.trip_id"), ("stops.txt", "stop_id", "stop_id")):
        ids[key] = readcol(fn, col)
    if all(v is None for v in ids.values()):
        p(f"  no static files found in '{gt}/' (use --gtfs). Skipped.")
    else:
        say("Share of the feed's ids that exist in your static GTFS (OTD static data). If the Transport Stack feed uses ids from a different static dataset, a low number here means you also need its static files.")
        rows = []
        for key, s in ids.items():
            if s is None: continue
            row = [key]
            for w in parsed:
                vals = [str(r[key]) for r in parsed[w]["rows"] if r.get(key) not in (None, "")]
                row.append(f"{pct(sum(1 for v in vals if v in s), len(vals))} of {len(vals):,}" if vals else "field missing")
            rows.append(row + ["-"] * (3 - len(row)))
        table(["id field", NAMES.get("A", "OTD") + " matches", NAMES.get("B", "TStack") + " matches"], [[r[0]] + r[1:1 + len(parsed)] + ["-"] * (2 - len(parsed)) for r in rows])

    # ---------- 9 samples ----------
    h("9. SAMPLE DATA")
    say("The first buses of each feed with all their fields, then the same bus from both feeds side by side (a * marks a different value).")
    for w in parsed:
        p(); p(f"  --- {LONG[w]}: first {a.samples} buses ---")
        for i, r in enumerate(parsed[w]["rows"][:a.samples], 1):
            p(f"  bus {i}:")
            for k in sorted(r):
                v = r[k]
                p(f"      {k:<30} {v:.6f}" if isinstance(v, float) else f"      {k:<26} {v}")
    if pair:
        common, ma, mb = pair
        p(); p(f"  --- same bus in both feeds (first {a.samples}) ---")
        for k in common[:a.samples]:
            ra, rb = ma[k], mb[k]
            p(f"  bus {k}")
            p(f"      {'field':<26} {'OTD':<24} {'TStack':<24}")
            for f in sorted(set(ra) | set(rb)):
                va, vb = ra.get(f, "(missing)"), rb.get(f, "(missing)")
                fa_ = f"{va:.6f}" if isinstance(va, float) else str(va)
                fb_ = f"{vb:.6f}" if isinstance(vb, float) else str(vb)
                mark = "*" if fa_ != fb_ else " "
                p(f"    {mark} {f:<30} {fa_[:23]:<24} {fb_[:23]:<24}")

    # ---------- 10 summary ----------
    h("10. SUMMARY: WHAT THE TWO FEEDS GIVE")
    if "A" in parsed and "B" in parsed:
        fa = {f for f, s in cen["A"].items() if f != "entity_id"}; fb = {f for f, s in cen["B"].items() if f != "entity_id"}
        say("Fields only in OTD: " + (", ".join(sorted(fa - fb)) or "none"))
        say("Fields only in TStack: " + (", ".join(sorted(fb - fa)) or "none"))
        say("Fields in both: " + (", ".join(sorted(fa & fb)) or "none"))
        useful_b = sorted(f for f in fb if informative(cen["B"][f]) and not informative(cen["A"].get(f)))
        if useful_b: say("Fields that carry real information in TStack but not in OTD: " + ", ".join(useful_b))
        useful_a = sorted(f for f in fa if informative(cen["A"][f]) and not informative(cen["B"].get(f)))
        if useful_a: say("Fields that carry real information in OTD but not in TStack: " + ", ".join(useful_a))
    tips = []
    for w in parsed:
        c, n = cen[w], NAMES[w]
        if informative(c.get("position.speed")): tips.append(f"{n} gives a real speed value: use it directly (and compare with the speed worked out from positions).")
        if informative(c.get("position.bearing")): tips.append(f"{n} gives bearing: use it to decide direction and to find buses moving against their route label.")
        if informative(c.get("current_status")) or informative(c.get("stop_id")) or informative(c.get("current_stop_sequence")):
            tips.append(f"{n} gives stop information (stop_id / stop sequence / status): this can replace part of the map matching, and STOPPED_AT could show real halts (dwell time).")
        if informative(c.get("occupancy_status")) or informative(c.get("occupancy_percentage")): tips.append(f"{n} gives occupancy: a new feature for ETA and for crowding analysis.")
        if informative(c.get("congestion_level")): tips.append(f"{n} gives a congestion level: a possible ETA feature.")
        if informative(c.get("position.odometer")): tips.append(f"{n} gives an odometer: exact distance travelled between pings.")
        if parsed[w]["types"].get("trip_update"): tips.append(f"{n} contains trip updates (predicted arrivals): compare them with your own ETA.")
    if tips:
        p(); say("What this means for the project:")
        for t in tips: say("- " + t, "    ")
    else:
        p(); say("Neither feed has extra fields that change the project plan (check the 'always 0' and 'missing' entries in section 6).")
    p(); say("Next steps: (1) run this once during peak hours and once at night, the answer can differ. (2) Run with --polls 4 --interval 15 to compare how often each feed changes. "
             "(3) If TStack has fields or buses that OTD lacks, we can add it as a second source to the collector.")
    finish(out, keys, cen)


def finish(out, keys, cen=None):
    text = redact("\n".join(LOG), keys)
    (out / "report.txt").write_text(text, encoding="utf-8")
    if cen:
        (out / "fields.json").write_text(redact(json.dumps(cen, indent=1, default=str), keys), encoding="utf-8")
    print(f"\nSaved: {out}/report.txt" + (f", {out}/fields.json" if cen else "") + f" and the downloaded files ({out}/otd.pb, {out}/transportstack.pb) for later --offline runs.")


if __name__ == "__main__":
    main()
