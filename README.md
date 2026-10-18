# DTC Delhi: Live Bus Explorer, Data Collector and Arrival Detection

Terminal tools for **Delhi buses (DTC and DIMTS cluster)**, built on the open **GTFS static** and **GTFS-Realtime** data from the Delhi **Open Transit Data (OTD)** portal.
The goal of the project is AI/ML on top of this data: ETA prediction, anomaly detection (bunching, irregular headways) and route analytics, with an interactive dashboard at the end.

> Status: work in progress. The three tools below are working and used on the live feed. The ML models and the dashboard are the next steps (see [Roadmap](#roadmap)).
> This is an academic project (M.Tech, AI and Data Science, public transport monitoring and ETA prediction). It is **not** an official Delhi Government, DTC or DIMTS product.

## Contents

- [What is in this repo](#what-is-in-this-repo)
- [Quick start](#quick-start)
- [Tool 1: Live bus explorer (`delhi_bus_tui.py`)](#tool-1-live-bus-explorer-delhi_bus_tuipy)
- [Tool 2: Data collector (`otd_collector.py`)](#tool-2-data-collector-otd_collectorpy)
- [Tool 3: Arrival detection (`build_arrivals.py`)](#tool-3-arrival-detection-build_arrivalspy)
- [How the tools fit together](#how-the-tools-fit-together)
- [What we learned about the Delhi feed](#what-we-learned-about-the-delhi-feed)
- [Troubleshooting](#troubleshooting)
- [Security and data notes](#security-and-data-notes)
- [Roadmap](#roadmap)

## What is in this repo

| File | What it does | Use it when |
|---|---|---|
| `delhi_bus_tui.py` | Live terminal app, like a small One Delhi: bus bubbles on a map, stop boards with ETAs, every field of the feed | You want to see and understand the live data |
| `otd_collector.py` | Downloads the live feed again and again, cleans it, and saves it to disk. Shows every step while it runs | You want to build a dataset (run it for hours) |
| `build_arrivals.py` | Turns the saved GPS dots into **stop passing times** (when each bus passed each stop) | You have collected data and want ETA, headway or bunching analysis |

## Quick start

### 1. Requirements

- Python 3.10 or newer, and [uv](https://docs.astral.sh/uv/)
- A terminal of at least **120 x 35** characters (Windows Terminal, iTerm2, Terminal.app and most Linux terminals work)
- An OTD API key (free, see step 3) and the static GTFS files (step 4)

### 2. Install with uv

```bash
# install uv first: https://docs.astral.sh/uv/
git clone https://github.com/ri5h46h/DTC_Delhi.git
cd DTC_Delhi

uv python install 3.10
uv sync                      # if the repo has a pyproject.toml
# or, if you start from an empty project:
# uv init && uv add -r requirements.txt
```

Without uv you can use the normal Python tools instead:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Packages used: `gtfs-realtime-bindings`, `requests`, `pandas`, `numpy`, `python-dotenv`, and `windows-curses` (Windows only).

### 3. Get your OTD API key

The live bus positions need a private key.

1. Open the OTD real-time page: <https://otd.delhi.gov.in/data/realtime>
2. Fill the **Request for API Access and API Private Key** form: name, email, contact number, company or institution, usage type (for study or research choose *Non-Commercial*), purpose (for example *Research* or *Student Projects*), and a short description of what you will build.
3. Accept the Terms and Conditions and submit.
4. The site then shows your secret API key. Copy it and keep it safe.
5. Read the [OTD terms](https://otd.delhi.gov.in/terms) once. Do not share your key.

The feed is requested like this (the tools do it for you): `/api/realtime/VehiclePositions.pb?key=YOUR_PRIVATE_KEY`.

### 4. Get the static GTFS files

1. Download the Delhi bus static data from <https://otd.delhi.gov.in/data/static>.
2. Unzip it into a folder called `gtfs/` next to the scripts. You need `routes.txt`, `trips.txt`, `stops.txt` and `stop_times.txt`. `shapes.txt` (the real road line of each route) is used by the arrival script and is recommended.

Note from the OTD documentation: the times inside `stop_times.txt` are rough estimates made by assuming a constant speed. We use the file for stop lists and route lines, not for real times.

### 5. Put the key in `.env`

All three tools read the key from a `.env` file, so it never sits inside the code.

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder:

```
API_KEY="paste-your-otd-api-key-here"
```

Never commit `.env` (see [Security and data notes](#security-and-data-notes)).

### 6. Folder layout

```
DTC_Delhi/
├── delhi_bus_tui.py        tool 1: live explorer
├── otd_collector.py        tool 2: collector
├── build_arrivals.py       tool 3: arrival detection
├── .env.example            copy to .env and add your key
├── requirements.txt
├── gtfs/                   you add this (static GTFS files), not in git
└── data/                   created by the tools, not in git
    ├── clean/              one file per day: cleaned GPS pings
    ├── health/             one file per day: one line per poll
    ├── raw/                optional: untouched downloads (--save-raw)
    └── arrivals/           output of build_arrivals.py
```

### 7. Run something

```bash
uv run delhi_bus_tui.py --demo              # try the explorer with fake buses (no key, no files)
uv run delhi_bus_tui.py                     # live explorer
uv run otd_collector.py run --demo          # try the collector with a fake feed
uv run otd_collector.py run                 # real collection
uv run build_arrivals.py selftest           # check the arrival method on simulated buses
```

The first run of the explorer and of the arrival script reads `stop_times.txt` (it is large) and saves a cache, so it can take a few minutes. Later runs start fast.

---

## Tool 1: Live bus explorer (`delhi_bus_tui.py`)

A terminal app in the spirit of the One Delhi app. It uses the live feed and your static GTFS files.

```bash
uv run delhi_bus_tui.py              # live
uv run delhi_bus_tui.py --demo       # fake buses, no key or GTFS files needed
uv run delhi_bus_tui.py --refresh 15 # seconds between feed updates (the feed itself updates about every 10 s)
```

### Screens

- **Map:** live bus bubbles around any area, zoom and pan, filter by route. No map image is used: stops and buses are placed on the screen from their latitude and longitude (one character is about 100 m wide and 200 m tall at the default zoom).
- **Stop board:** choose a stop and see the upcoming buses of every route, sorted by ETA, with distance, speed and the vehicle number.
- **Bus fields:** every field the feed gives for a bus, plus values the app works out itself.
- **Feed fields (`i`):** which fields the feed really fills, how often they are zero, and how many trip and route IDs match your static data.

### Keys

| Key | Action |
|---|---|
| arrows or `h j k l` | pan the map |
| `+` / `-` | zoom in / out |
| `0` | back to Connaught Place |
| `b` / `B` | select the next bus near the centre `+` / deselect |
| `Enter` | all fields of the selected bus |
| `s` | stop board of the stop nearest the map centre |
| `S` | stop board for the selected bus's next stop |
| `/` | find a stop by name or id |
| `r` | show only one route (also draws its line and stops) |
| `x` | clear route filter and selection |
| `i` | feed fields and ID match rates |
| `g` | open the selected bus (or the stop on the stop board, or the map centre) in **Google Maps** in your default browser |
| `f` | refresh now |
| `?` | help |
| `q` | quit |

**Mouse:** click the underlined `[open in Google Maps]` label next to any coordinates (bus panel, stop board, bus fields). While the app runs, the terminal passes mouse clicks to it, so to select text with the mouse hold **Option** (Terminal.app, iTerm2).

### Reading the map

| Symbol | Meaning |
|---|---|
| `(534)` | a bus on a route whose short name is 534 (long names are cut to 6 characters) |
| `o` | a bus whose label did not fit |
| `*`, `2`-`9`, `#` | far zoom: one bus, a few buses, many buses in one cell |
| `[534]` highlighted | the selected bus |
| `+` in the middle | the map centre |
| `.` | a bus stop (zoom in to see them) |
| dim bubble | no new ping for more than 5 minutes |

### How the numbers are worked out

- **Route and direction.** The feed's `route_id` already carries the direction (the static data has route IDs like `102AUP` and `102ADOWN`). The `DIR` column of the stop board tells how the app decided: `trip` (trip_id found in static data), `bearing`, `history` (the bus moved that way over its last pings), `nearest` (one direction is clearly closer), `only` (the route_id has just one stop list, which is reliable here), `dir?` (could not decide, treat with care).
- **Speed.** The feed's speed field is always 0, so speed is worked out from positions: straight-line distance between an older ping and the newest, divided by the time, using the oldest ping that is 20 to 180 seconds old. It is clipped to 2-14 m/s for ETA so a stopped bus does not give an infinite time.
- **ETA (a simple baseline).** Remaining distance **along the route** (not straight line) divided by speed, plus a time for each stop in between, minus the age of the bus's last ping.
- **Dwell.** `DWELL_S = 12` at the top of the file is a guess of seconds lost at each stop between the bus and your stop. DTC and DIMTS buses stop only when someone boards or gets off, so this can overestimate (and the speed already includes some stopping). Set `DWELL_S = 0` to see the effect. The real value is meant to be learned from data later.
- **Empty feed answers.** The server sometimes sends an empty file. The app then keeps the last good data and shows a warning, instead of clearing the map.

The ETA is a baseline for comparison, not the final model. It will differ from the One Delhi app, which uses its own ETA method.

---

## Tool 2: Data collector (`otd_collector.py`)

Collects live GPS data and cleans it, while **showing every step**. It is made to run for hours and build a dataset for the ML work.

### Commands

```bash
uv run otd_collector.py run                 # live dashboard
uv run otd_collector.py run --plain         # same steps as readable text (good for a log file: ... --plain > log.txt)
uv run otd_collector.py run --once          # one poll, then stop (a quick test of your key)
uv run otd_collector.py run --demo          # fake feed, no key needed, saves to data_demo/
uv run otd_collector.py run --interval 10   # seconds between polls (minimum 10)
uv run otd_collector.py run --save-raw      # also keep every untouched download in data/raw/
uv run otd_collector.py inspect             # summary of the data collected so far
```

Dashboard keys: `p` pause or resume, `f` fetch now, `+` / `-` poll faster / slower, `e` explanations on/off, `q` quit.

### The 5 steps of every poll (shown on screen)

| Step | What happens |
|---|---|
| 1 FETCH | Download the live file. Records the HTTP status, size and time taken. Errors are shown and it retries (it waits longer after repeated failures). |
| 2 DECODE | Binary protobuf becomes one row per bus. If the file has 0 vehicles (or under 20% of the usual number) the poll is **ignored** and retried in 5 seconds. Nothing is deleted. |
| 3 VALIDATE | Every row is checked (table below). |
| 4 SAVE | New valid pings are appended to `data/clean/YYYY-MM-DD.csv.gz` (one file per day, Indian time). |
| 5 HEALTH | One line about the poll goes to `data/health/YYYY-MM-DD.csv`: a record of how reliable the feed is. |

### Cleaning rules

Rule: **drop only what is unusable, flag what is suspicious.**

| Result | Meaning | Kept? |
|---|---|---|
| `duplicate` | the bus has not sent a new position since the last poll (most rows) | no |
| `stale` | last ping older than 600 s (parked, GPS off) | no |
| `outside_area` | position outside the Delhi-NCR box (28.2-29.1 N, 76.6-77.7 E) | no |
| `no_position`, `no_time`, `future_time` | missing or impossible values | no |
| `gps_jump` | the bus seems to move faster than 100 km/h since its last ping | **yes, flagged** |

The limits are at the top of the script (`MAX_AGE_S`, `JUMP_KMH`, `BBOX`).

<details>
<summary>Columns in <code>data/clean/*.csv.gz</code></summary>

| Column | Meaning |
|---|---|
| `fetched_at` | when we downloaded (epoch seconds) |
| `ts` | the bus's own ping time (epoch seconds) |
| `vehicle_id` | number plate style id |
| `route_id`, `trip_id`, `start_date`, `start_time` | from the feed |
| `lat`, `lon` | position |
| `age_s` | `fetched_at - ts` |
| `dt_s` | seconds since this bus's previous saved ping |
| `moved_m` | metres since the previous saved ping |
| `speed_kmh` | **derived** from positions (the feed's own speed is always 0) |
| `flag` | empty or `gps_jump` |
| `trip_in_static`, `route_in_static` | 1 or 0: does the id exist in your static GTFS |
| `bearing`, `stop_id`, `stop_seq`, `occupancy` | empty today; kept in case the feed starts to fill them |

Dropped on purpose: `position.speed` (always 0), `schedule_relationship` (always SCHEDULED), `vehicle.label` (same as the id).
</details>

### Collecting for hours

- You do **not** need 24-hour collection. Sessions of 4-5 hours work. Cover the hours that matter (morning peak about 8-11, evening peak about 17-20, some off-peak and a weekend) and write down which hours your data covers.
- On a Mac keep the laptop awake with `caffeinate`: `caffeinate -dims uv run otd_collector.py run --interval 10 --save-raw`. Plug in the charger and turn off sleep on other systems.
- A session adds to the same day file. A new session never overwrites old data. Always start from the same project folder, otherwise a new empty `data/` is created.
- Download size is about 0.5 MB per poll: about 90 MB of internet per hour at 20 s polling and about 180 MB per hour at 10 s. Saved data is small (about 15 MB per hour).
- Back up `data/` once in a while. The feed only shows the present, so lost data cannot be downloaded again.
- Use `--save-raw` at least in the first weeks. If you change a cleaning rule later, you can run the cleaning again on the raw files.
- The API key is removed from error messages before they are saved.

`inspect` shows, for every day: rows, buses, routes, size, the time between pings of one bus, derived speed, flag counts, ID match rates, rows per hour, busiest routes, and the feed health (success rate, empty answers, download time).

---

## Tool 3: Arrival detection (`build_arrivals.py`)

The feed only gives dots: "bus X was at this latitude and longitude at this time". ETA models, headways and bunching need to know **when each bus reached each stop**. This script finds that, offline, from the files the collector saved.

> "Arrival" here means the time the bus **passes** the stop position. It does not say that the bus really stopped or opened its doors (buses stop only on demand).

### Commands

```bash
uv run build_arrivals.py selftest                    # simulated buses with known truth: how accurate is the method?
uv run build_arrivals.py run                         # only NEW or CHANGED days in data/clean/
uv run build_arrivals.py run --date 2026-10-11       # one day
uv run build_arrivals.py run --force                 # redo days that are already up to date
uv run build_arrivals.py run --routes 2097,1957      # only these route_ids
uv run build_arrivals.py run --no-shapes             # straight lines between stops instead of the real road line
uv run build_arrivals.py show --vehicle DL51EV2595   # one bus step by step: pings, progress along route, arrivals
```

Stop the collector (or wait for a poll to finish) before running this, so the day file is not being written while it is read. If you get a read error, run it again.

### The 5 steps (printed while it runs)

1. **LOAD:** the stop list of every `route_id` (the longest trip in `stop_times.txt`), and the route line (real road shape from `shapes.txt` by default, otherwise straight lines between stops). Cached after the first run.
2. **MATCH:** for every ping, how far it is from its route line and how many metres along the route it is. Pings the collector flagged as GPS jumps are skipped.
3. **RUNS:** pings are cut into runs (one trip of one bus). A new run starts after 300 s without a ping. One-off position spikes are removed. A trip that mostly moves **against** the direction of its route_id is marked `direction_ok = False` and not used.
4. **ARRIVALS:** between the two pings just before and just after a stop, the pass time is interpolated: `time = t1 + (stop_progress - progress1) / (progress2 - progress1) * (t2 - t1)`.
5. **SAVE:** writes the files below, then prints a quality report.

### Output files (in `data/arrivals/`)

| File | Content |
|---|---|
| `<date>_arrivals.csv.gz` | one row per bus and stop passed |
| `<date>_runs.csv.gz` | one row per run, with checks |
| `<date>_settings.json` | the settings used; a day is rebuilt if its settings or data change |

<details>
<summary>Columns</summary>

**arrivals:** `run_id`, `route_id`, `vehicle_id`, `trip_id`, `stop_idx` (0 is the first stop of the route_id), `stop_id`, `arrive_ts` (epoch seconds), `arrive_ist` (HH:MM:SS), `gap_s` (seconds between the two pings used), `span_m` (metres between them), `speed_kmh` (speed around the stop), `quality` (`good` gap up to 40 s, `ok` up to 90 s, `coarse` above that).

**runs:** `run_id`, `vehicle_id`, `route_id`, `trip_id`, `off_route_share`, `t_start`, `t_end`, `n_pings`, `prog_start`, `prog_end`, `direction_ok`, `usable`.
</details>

### Settings

| Option | Default | Meaning |
|---|---|---|
| `--on-route` | 100 m | a ping within this distance of the route line counts as on the route |
| `--max-gap` | 120 s | an arrival is dropped if the two pings around the stop are further apart |
| `--run-gap` | 300 s | no ping for this long cuts a run |
| `--vmax` | 20 m/s | fastest believable speed for the spike filter |
| `--no-shapes` | off | use straight lines between stops |
| `--force` | off | rebuild days that are up to date |

### Days and sessions

- The collector writes one file per Indian date. Sessions on the same day are in the same file and stay apart for analysis (a 300 s silence cuts a bus's run).
- `run` only processes days that are new, whose data changed, or whose settings changed. Other days are skipped.

### Reading the quality report

The report shows the share of pings near their route line (and the distance breakdown), routes with the lowest match, how many spikes were removed, runs moving against their route label, arrival quality, coverage, and a stop-to-stop speed check. Use `show --vehicle ...` to follow one bus and compare an arrival time with what you know.

### Limits to remember

- Only the longest trip pattern of each route_id is used.
- If a bus waits at a stop, the interpolated time falls somewhere inside the ping gap, so the error is at most the gap.
- The self-test uses simulated buses (GPS noise about 6 m, pings every 10-30 s). Real accuracy must be checked on real data.

---

## How the tools fit together

```
OTD live feed ──► otd_collector.py ──► data/clean, data/health (and data/raw)
                                           │
GTFS static files (gtfs/) ─────────────────┤
                                           ▼
                                   build_arrivals.py ──► data/arrivals
                                           │
                                           ▼
                    (next) ETA dataset, baselines, models, headway / bunching, dashboard

delhi_bus_tui.py  ◄── live feed + gtfs/  (for looking at the live data)
```

## What we learned about the Delhi feed

Observations from the author's own samples (October 2026). They may change over time, so check them on your data (the `i` screen of the explorer and `otd_collector.py inspect` show these numbers).

- About 4,900 vehicles in the feed at once. The vehicle id is the number plate.
- Filled fields: position, timestamp, `route_id`, `trip_id`, `start_date`, `start_time`. **Speed is always 0.** There is no bearing, stop or occupancy field.
- `route_id` matches the static data 100%. Each static route_id is one **direction** of one route variant (names like `102AUP`, `102ADOWN`; `direction_id` is 0 everywhere), so the direction comes from the route_id.
- Only about 9% of live `trip_id`s exist in the static `trips.txt`, so trips cannot be matched to the timetable reliably.
- About 4% of polls returned an empty file (a header only, HTTP 200).
- GPS quality is good: about 0.03% of pings look like impossible jumps.
- About 1 run in 18 moves against the direction in its route_id (a wrong or late label).

## Troubleshooting

| Problem | What to do |
|---|---|
| `API key rejected` or HTTP 401 / 403 | Check `API_KEY` in `.env` (no extra spaces, correct key). |
| `Missing gtfs/...` | Put the static GTFS `.txt` files in `gtfs/` (or use `--gtfs`). |
| Map goes blank for a moment | The server sent an empty file. The app keeps the old data and shows a warning. |
| Many HTTP errors during a session | Your internet dropped (laptop sleep, hotspot). The collector retries; only the gap is lost. |
| Garbled screen | The terminal is too small. Use at least 120 x 35. |
| `curses` error on Windows | Install `windows-curses` (it is in `requirements.txt`). |
| First run is slow | It is reading `stop_times.txt`. The result is cached. |
| Read error in `build_arrivals.py` | The collector was writing the file. Stop it and run again. |

## Security and data notes

- **Never commit your API key.** `.env` must be in `.gitignore`. The key is part of the request URL, so do not paste full request URLs or log files with URLs into public places.
- Suggested `.gitignore` entries:

  ```
  .env
  data/
  data_demo/
  gtfs/
  .venv/
  __pycache__/
  gtfs/.bus_cache.pkl
  gtfs/.arrival_patterns*.pkl
  ```
- The static data and the live feed belong to the Delhi Government and are provided by the OTD portal (with IIIT-Delhi, DTC and DIMTS). Read and follow the [OTD terms](https://otd.delhi.gov.in/terms). Do not publish raw collected data unless the terms allow it.

## Roadmap

- [x] Live bus explorer (map, stop boards, feed field views, Google Maps links)
- [x] Data collector with cleaning and a feed health log
- [x] Arrival detection with real road shapes and a quality report
- [ ] ETA dataset from arrivals, and baselines (schedule, historical average, current speed)
- [ ] ETA models (gradient boosting, LSTM/GRU) with uncertainty
- [ ] Headway, bunching and anomaly detection
- [ ] Route analytics (reliability, delay hotspots)
- [ ] Interactive dashboard