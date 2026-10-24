# DTC Delhi: live bus explorer, data collector, arrival detection and feed comparison [Work in Progress]

Terminal tools for **Delhi buses (DTC and DIMTS cluster)**, built on open **GTFS static** and **GTFS-Realtime** data from the Delhi **Open Transit Data (OTD)** portal and the **Delhi Transport Stack**.
The goal of the project is AI/ML on top of this data: ETA prediction, anomaly detection (bunching, irregular headways) and route analytics, with an interactive dashboard at the end. This project is also a submission as my third semester dissertation project for my current degree at NFSU, Gandhinagar.

> Status: work in progress. The four tools below work on the live feeds. The ML models and the dashboard are the next steps (see [Roadmap](#roadmap)).
> This is an academic project (M.Tech, AI and Data Science, public transport monitoring and ETA prediction). It is **not** an official Delhi Government, DTC or DIMTS product.

## Contents

- [What is in this repo](#what-is-in-this-repo)
- [Quick start](#quick-start)
- [Tool 1: Live bus explorer (`delhi_bus_tui.py`)](#tool-1-live-bus-explorer-delhi_bus_tuipy)
- [Tool 2: Data collector (`otd_collector.py`)](#tool-2-data-collector-otd_collectorpy)
- [Tool 3: Arrival detection (`build_arrivals.py`)](#tool-3-arrival-detection-build_arrivalspy)
- [Tool 4: Feed comparison (`compare_feeds.py`)](#tool-4-feed-comparison-compare_feedspy)
- [How the tools fit together](#how-the-tools-fit-together)
- [Two feeds: which one is used when](#two-feeds-which-one-is-used-when)
- [What we learned about the Delhi feeds](#what-we-learned-about-the-delhi-feeds)
- [Troubleshooting](#troubleshooting)
- [Security and data notes](#security-and-data-notes)
- [Roadmap](#roadmap)

## What is in this repo

| File | What it does | Use it when |
|---|---|---|
| `delhi_bus_tui.py` | Live terminal app, like a small One Delhi: bus bubbles on a map, stop boards with ETAs, reported and calculated speed, every field of the feed | You want to see and understand the live data |
| `otd_collector.py` | Downloads the live feed again and again, cleans it and saves it to disk, showing every step. Version 2 uses one feed at a time with a fallback | You want to build a dataset (run it for hours) |
| `build_arrivals.py` | Turns the saved GPS dots into **stop passing times** (when each bus passed each stop) | You have collected data and want ETA, headway or bunching analysis |
| `compare_feeds.py` | Downloads both feeds and compares their fields, buses, freshness and ids | You want to know what the two feeds give and where they differ |

## Quick start

### 1. Requirements

- Python 3.11 or newer, and [uv](https://docs.astral.sh/uv/)
- A terminal of at least **120 x 35** characters (Windows Terminal, iTerm2, Terminal.app and most Linux terminals work)
- An API key for at least one feed (step 3) and the static GTFS files (step 4)

### 2. Install with uv

```bash
# install uv first: https://docs.astral.sh/uv/
git clone https://github.com/ri5h46h/DTC_Delhi.git
cd DTC_Delhi

uv python install 3.11
uv sync                      # if the repo has a pyproject.toml
# or, if you start from an empty project:
# uv init && uv add -r requirements.txt
```

Without uv you can use the normal Python tools:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Packages used: `gtfs-realtime-bindings`, `requests`, `pandas`, `numpy`, and `windows-curses` (Windows only).

### 3. Get your API keys

You need a key for each feed you want to use. Using both gives you the automatic fallback (see [Two feeds](#two-feeds-which-one-is-used-when)).

**A. OTD Delhi key** (live bus positions, key goes in the URL)

1. Open the OTD real-time page: <https://otd.delhi.gov.in/data/realtime>
2. Fill the **Request for API Access and API Private Key** form: name, email, contact number, company or institution, usage type (for study or research choose *Non-Commercial*), purpose (for example *Research* or *Student Projects*) and a short description of what you will build.
3. Accept the Terms and Conditions and submit.
4. The site then shows your secret API key. Copy it and keep it safe.
5. Read the [OTD terms](https://otd.delhi.gov.in/terms) once. Do not share your key.

**B. Delhi Transport Stack key** (live bus positions with a speed field, key goes in the `x-api-key` header)

API page: <https://delhi.transportstack.in/data-services/datadetails/7>

1. Visit the Delhi Transpost Stack Page: <https://delhi.transportstack.in/> and create an account.
2. Navigate to Data and Services from the top menu and look for **"Delhi Buses: GTFS-RT feed - DTC / DoT"** or just click on the above given link.
3. Fill the subscribe form on the page with your details, your request should get approved in no time and you shall shortly receive an email with your api key.

### 4. Get the static GTFS files

1. Download the Delhi bus static data from <https://otd.delhi.gov.in/data/static>. [this is mostly not reachable or down]. The alternative source for the latest static GTFS Data is Transport Stack itself, visit <https://delhi.transportstack.in/data-services/datadetails/1> and the fill the form to be able to download the static data.
2. Unzip it into a folder called `gtfs/` next to the scripts. You need `routes.txt`, `trips.txt`, `stops.txt` and `stop_times.txt`. `shapes.txt` (the real road line of each route) is used by the arrival script and is recommended.

Note from the OTD documentation: the times inside `stop_times.txt` are rough estimates made by assuming a constant speed. We use the files for stop lists and route lines, not for real times.

### 5. Paste your keys in the env file

There is an example `.env.example` file available in the repo, rename it to `.env` and paste your API keys in it.
(`TSTACK_API_KEY` and `OTD_API_KEY`)

The files `delhi_bus_tui.py`,`otd_collector.py` and `compare feeds` use that env file to read the api keys from.

You can paste only one key. The tool then runs with that feed and has no fallback. **Never commit an env file that contains your keys** (see [Security and data notes](#security-and-data-notes)).

### 6. Folder layout

```
DTC_Delhi/

├── delhi_bus_tui.py        tool 1: live explorer
├── otd_collector.py        tool 2: collector
├── build_arrivals.py       tool 3: arrival detection
├── compare_feeds.py        tool 4: feed comparison
├── requirements.txt
├── .env.example            env file example
├── gtfs/                   you add this (static GTFS files), not in git
├── feed_compare_out/       created by compare_feeds.py (report and saved feed files)
└── data/                   created by the tools, not in git
    ├── clean/              one file per day: cleaned GPS pings
    ├── health/             one file per day: one line per poll
    ├── raw/                optional: untouched downloads (--save-raw)
    └── arrivals/           output of build_arrivals.py
```

### 7. Run something

```bash
uv run delhi_bus_tui.py --demo              # try the explorer with fake buses (no keys, no files)
uv run delhi_bus_tui.py                     # live explorer
uv run otd_collector.py run --demo          # try the collector with fake feeds (shows the fallback too)
uv run otd_collector.py run                 # real collection
uv run build_arrivals.py selftest           # check the arrival method on simulated buses
uv run compare_feeds.py                     # compare the two live feeds
```

The first run of the explorer and of the arrival script reads `stop_times.txt` (it is large) and saves a cache, so it can take a few minutes. Later runs start fast.

---

## Tool 1: Live bus explorer (`delhi_bus_tui.py`)

A terminal app in the spirit of the One Delhi app. It uses a live feed and your static GTFS files.

```bash
uv run delhi_bus_tui.py                          # Transport Stack feed (default)
uv run delhi_bus_tui.py --using-otd              # OTD as the main feed (calculated speed only)
uv run delhi_bus_tui.py --eta-speed calculated   # ETA uses the calculated speed (default: reported speed)
uv run delhi_bus_tui.py --no-fallback            # never switch to the other feed
uv run delhi_bus_tui.py --demo                   # fake buses, no keys or GTFS files needed (add --using-otd to see the OTD layout)
uv run delhi_bus_tui.py --refresh 15             # seconds between feed updates (minimum 10)
```

### Screens

- **Map:** live bus bubbles around any area, zoom and pan, filter by route. No map image is used: stops and buses are placed on the screen from their latitude and longitude (one character is about 100 m wide and 200 m tall at the default zoom).
- **Stop board:** choose a stop and see the upcoming buses of every route, sorted by ETA, with distance, reported speed (REAL), calculated speed (CALC), which speed the ETA used, and the vehicle number. Long route and towards names are cut so they cannot push the columns.
- **Bus fields:** every field the feed gives for a bus, plus the reported speed, the calculated speed and the speed used for the ETA.
- **Feed fields (`i`):** which fields the feed really fills, how often they are zero, and how many trip and route ids match your static data.

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

- **Route and direction.** The feed's `route_id` already carries the direction (the static data has route ids like `102AUP` and `102ADOWN`). The `DIR` column of the stop board tells how the app decided: `trip` (trip_id found in static data), `bearing`, `history` (the bus moved that way over its last pings), `nearest` (one direction is clearly closer), `only` (the route_id has just one stop list, which is reliable here), `dir?` (could not decide, treat with care).
- **Reported speed (REAL).** The number in the feed's speed field, shown in km/h assuming metres per second. Only the Transport Stack feed has it; OTD always sends 0. **Its meaning is still being checked** (see [What we learned](#what-we-learned-about-the-delhi-feeds)).
- **Calculated speed (CALC).** Distance between the bus's recent positions divided by the time, using the oldest ping that is 20 to 180 seconds old (an average, shown without clipping).
- **ETA speed.** By default the ETA uses the reported speed (the calculated speed when the feed has none). `--eta-speed calculated` uses the calculated speed. Either way the speed is clipped to 2-14 m/s so a bus standing at a red light does not give an endless ETA. Only the latest reported speed is used (no averaging).
- **ETA (a simple baseline).** Remaining distance **along the route** divided by the speed, plus a time for each stop in between, minus the age of the bus's last ping. The route line is made of straight lines between stops (the explorer does not use `shapes.txt`), so on curved roads the real road distance is a bit longer.
- **Dwell.** `DWELL_S = 12` at the top of the file is a guess of seconds lost at each stop between the bus and your stop. DTC and DIMTS buses stop only when someone boards or gets off, so this can overestimate. Set `DWELL_S = 0` to see the effect.
- **Empty feed answers.** The server sometimes sends an empty file. The app then keeps the last good data and shows a warning, instead of clearing the map.

The ETA is a baseline for comparison, not the final model. It will differ from the One Delhi app, which uses its own ETA method.

---

## Tool 2: Data collector (`otd_collector.py`)

Collects live GPS data and cleans it, while **showing every step**. It is made to run for hours and build a dataset for the ML work. The main feed is Transport Stack and OTD is the fallback.

### Commands

```bash
uv run otd_collector.py run                 # live dashboard
uv run otd_collector.py run --plain         # same steps as readable text (good for a log file: ... --plain > log.txt)
uv run otd_collector.py run --once          # one poll, then stop (a quick test of your keys)
uv run otd_collector.py run --demo          # fake feeds, no keys needed, saves to data_demo/ (includes a simulated outage)
uv run otd_collector.py run --interval 15   # seconds between polls (default 15, minimum 10)
uv run otd_collector.py run --save-raw      # also keep every untouched download in data/raw/
uv run otd_collector.py run --using-otd     # OTD as the main feed (Transport Stack becomes the fallback)
uv run otd_collector.py run --no-fallback   # never switch feeds
uv run otd_collector.py inspect             # summary of the data collected so far
```

Dashboard keys: `p` pause or resume, `f` fetch now, `+` / `-` poll faster / slower, `e` explanations on/off, `q` quit. A line at the top shows which feed is in use.

### The 5 steps of every poll (shown on screen)

| Step | What happens |
|---|---|
| 1 FETCH | Download the live file from **one** feed. Records the HTTP status, size and time taken. Errors are shown and retried. |
| 2 DECODE | Binary protobuf becomes one row per bus. An empty file (or under 20% of the usual number of vehicles) is **ignored** and retried in 5 seconds. Shows how many buses have a reported speed. |
| 3 VALIDATE | Every row is checked (table below). |
| 4 SAVE | New valid pings are appended to `data/clean/YYYY-MM-DD.csv.gz` (one file per day, Indian time). |
| 5 HEALTH | One line about the poll goes to `data/health/YYYY-MM-DD.csv`: a record of how reliable the feeds are, including which feed was used and every switch. |

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
| `source` | `tstack` or `otd`: which feed this row came from |
| `ts` | the bus's own ping time (epoch seconds) |
| `vehicle_id` | number plate style id |
| `route_id`, `trip_id`, `start_date`, `start_time` | from the feed |
| `schedule` | `trip.schedule_relationship` (`SCHEDULED` or `ADDED`) |
| `lat`, `lon` | position |
| `speed_rep` | the raw number in the feed's speed field (empty for OTD rows, because OTD always sends 0). The unit is not assumed |
| `age_s` | `fetched_at - ts` |
| `dt_s` | seconds since this bus's previous saved ping |
| `moved_m` | metres since the previous saved ping |
| `speed_calc_kmh` | speed **calculated** from positions (distance / dt) |
| `flag` | empty or `gps_jump` |
| `trip_in_static`, `route_in_static` | 1 or 0: does the id exist in your static GTFS |
| `extra` | any field the feed sends that we did not expect, as JSON (empty normally) |

Dropped on purpose: `vehicle.label` (same as the id; kept in `extra` if it ever differs).
</details>

### One feed at a time, with a fallback

Each poll calls only the active feed. After 2 problems in a row (error, rejected key, empty or tiny file) the collector switches to the other feed and tries it right away. While on the fallback, every 4th poll the main feed is tried once; when it works again the collector switches back. The fallback only exists if you pasted both keys.

### Collecting for hours

- You do **not** need 24-hour collection. Sessions of 4-5 hours work. Cover the hours that matter (morning peak about 8-11, evening peak about 17-20, some off-peak and a weekend) and write down which hours your data covers.
- On a Mac keep the laptop awake with `caffeinate`: `caffeinate -dims uv run otd_collector.py run --interval 15 --save-raw`. Plug in the charger and turn off sleep on other systems.
- A session adds to the same day file. A new session never overwrites old data. Always start from the same project folder, otherwise a new empty `data/` is created.
<!-- - The collector refuses to add rows to a file of today that has an older column layout. Rename the old `data/` folder and start again. -->
- A download is about 0.4-0.5 MB: about 80-105 MB of internet per hour at 15-20 s polling. Saved data is small (about 13 MB per hour).
- Back up `data/` once in a while. The feeds only show the present, so lost data cannot be downloaded again.
- Use `--save-raw` at least in the first weeks. If you change a cleaning rule later, you can run the cleaning again on the raw files.
- The API keys are removed from error messages before they are saved.

`inspect` shows, for every day: rows per feed, time between saved pings, reported and calculated speed with a **speed check** (does the reported speed follow the bus's movement, and in which unit), schedule_relationship against trip_id match, extra fields, flags, rows per hour and busiest routes. It also shows the feed health per feed (clean share, download time, vehicles, file age, problems, switches) and the rhythm of the feed (how far apart polls were, how many polls brought nothing new).

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
uv run build_arrivals.py run --no-trip-id            # do not use trip_id to cut runs (if it changes too often)
uv run build_arrivals.py show --vehicle DL51EV2595   # one bus step by step: pings, progress along route, arrivals
```

The real road line from `shapes.txt` is used **by default**. If `shapes.txt` or a `shape_id` column is missing, the script says so and uses straight lines between stops.
Stop the collector (or wait for a poll to finish) before running this, so the day file is not being written while it is read. If you get a read error, run it again.

### The 5 steps (printed while it runs)

1. **LOAD:** the stop list of every `route_id` (the longest trip in `stop_times.txt`) and the route line (real road shape by default). Cached after the first run.
2. **MATCH:** for every ping, how far it is from its route line and how many metres along the route it is. Pings the collector flagged as GPS jumps are skipped.
3. **RUNS:** pings are cut into runs (one trip of one bus). A new run starts after 300 s without a ping. One-off position spikes are removed. A trip that mostly moves **against** the direction of its route_id is marked `direction_ok = False` and not used. The report shows a `trip_id check` (pings per bus, route and trip_id group).
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

**arrivals:** `run_id`, `route_id`, `vehicle_id`, `trip_id`, `stop_idx` (0 is the first stop of the route_id), `stop_id`, `arrive_ts` (epoch seconds), `arrive_ist` (HH:MM:SS), `gap_s` (seconds between the two pings used), `span_m` (metres between them), `speed_kmh` (speed around the stop), `quality` (`good` gap up to 40 s, `ok` up to 90 s, `coarse` above that), `rep_speed_a` and `rep_speed_b` (the raw reported speed of the ping before and after the stop; empty for OTD rows).

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
| `--no-trip-id` | off | cut runs without using trip_id |
| `--force` | off | rebuild days that are up to date |

### Days and sessions

- The collector writes one file per Indian date. Sessions on the same day are in the same file and stay apart for analysis (a 300 s silence cuts a bus's run).
- `run` only processes days that are new, whose data changed, or whose settings changed. Other days are skipped.

### Reading the quality report

The report shows the share of pings near their route line (and the distance breakdown), routes with the lowest match, how many spikes were removed, runs moving against their route label, arrival quality, coverage, a stop-to-stop speed check, and how the reported speed compares with the speed around the stop. Use `show --vehicle ...` to follow one bus and compare an arrival time with what you know.

### Limits to remember

- Only the longest trip pattern of each route_id is used.
- If a bus waits at a stop, the interpolated time falls somewhere inside the ping gap, so the error is at most the gap.
- The self-test uses simulated buses (GPS noise about 6 m, pings every 10-30 s). Real accuracy must be checked on real data.

---

## Tool 4: Feed comparison (`compare_feeds.py`)

Downloads the OTD and the Transport Stack feed and compares them. It has its own small reader for the GTFS-Realtime file, so it also finds fields that are **not** in the standard format.

```bash
uv run compare_feeds.py                          # one snapshot of each feed
uv run compare_feeds.py --polls 4 --interval 15  # also compare how often each feed changes
uv run compare_feeds.py --offline feed_compare_out   # analyse the files saved by an earlier run (no keys needed)
uv run compare_feeds.py --samples 5              # show more sample buses
```

What it shows: the download details, the file headers and entity types (vehicle positions, trip updates, alerts), freshness of the data, a table of every field in each feed (how often it is present, how often it is zero), extra fields outside the standard, the fields that matter for this project, whether the two feeds describe the same buses and whether their values agree, whether ids match your static GTFS, sample data, and a summary. Everything is saved in `feed_compare_out/report.txt` (keys are never written there) together with the two downloaded files.

---

## How the tools fit together

```
Transport Stack feed (main) ─┐
                             ├─► otd_collector.py ──► data/clean, data/health (and data/raw)
OTD feed (fallback) ─────────┘          │
                                        │        GTFS static files (gtfs/)
                                        ▼                 │
                               build_arrivals.py ◄────────┘
                                        │
                                        ▼
        (next) ETA dataset, baselines, models, headway / bunching, dashboard

delhi_bus_tui.py  ◄── one live feed at a time (+ fallback) + gtfs/   (for looking at the live data)
compare_feeds.py  ◄── both feeds, one snapshot each                  (for comparing them)
```

## Two feeds: which one is used when

| | Delhi Transport Stack | OTD Delhi |
|---|---|---|
| Role | main feed in the explorer and the collector | fallback (or main with `--using-otd`) |
| Key sent as | `x-api-key` header | `?key=` in the URL |
| Speed field | has values (meaning being checked) | always 0 |
| Freshness | about 10 s older pings than OTD | fresher |
| `trip_id` in static GTFS | 0% | about 5-9% |

Only **one** feed is called per refresh or poll. After 2 problems in a row (error, rejected key, empty or tiny file) the tool switches to the other feed; while on the fallback, every 4th refresh the main feed is tried once, and the tool switches back when it works. `--no-fallback` turns this off.

## What we learned about the Delhi feeds

Observations from the author's own samples (October 2026). They may change over time, so check them on your data (the `i` screen of the explorer, `otd_collector.py inspect` and `compare_feeds.py` show these numbers).

- **Same buses, same fields.** In a side-by-side check both feeds had the same 3,573 buses (same ids) and the same vehicle fields. Neither has bearing, odometer, stop id, stop status, occupancy, congestion level or number plate, and neither has trip updates or alerts. The only field that differs in content is the **speed**: OTD always sends 0, Transport Stack sends values.
- **Transport Stack is older.** Its file was about 19 s older and its pings about 11 s older than OTD's for the same bus (older for 94% of buses).
- **Rhythm.** Buses report about every 30 s (median 32 s between saved pings, 90th percentile 57 s in a 30-minute Transport Stack session), and the feed file is renewed about every 30 s. About 40% of 15 s polls brought no new rows. Polling every 15-20 s catches almost every ping; 10 s brings nothing more.
- **Reported speed: open question.** In the same session the Transport Stack speed (present on every row, zero on 6%) did not agree with the speed calculated from positions: as metres per second its median for moving buses would be 36.6 km/h (90th percentile 73.4), and its correlation with the calculated speed was negative (-0.28). Until this is understood, use `--eta-speed calculated` in the explorer. The collector saves the raw value in `speed_rep` and `inspect` runs a speed check.
- **`route_id` carries the direction.** The static data has 2,554 route ids, each one direction of one route variant (names like `102AUP`, `102ADOWN`; `direction_id` is 0 everywhere). `route_id` matches the static data 100% in both feeds.
- **`trip_id` does not link to the timetable.** Transport Stack: every trip is `ADDED` and 0% of trip ids are in the static `trips.txt` (its ids look like `1529_07_05_9966`). OTD: 5-9% match (ids like `1529_7_5`), and it has both `SCHEDULED` and `ADDED` trips. `schedule_relationship` is not constant.
- **Empty answers.** About 4% of OTD polls returned an empty file (a header only, HTTP 200).
- **GPS quality is good.** About 0.03-0.05% of pings look like impossible jumps.
- **Wrong direction labels.** About 1 run in 18 (OTD sample) moves against the direction in its route_id.

## Troubleshooting

| Problem | What to do |
|---|---|
| `Paste your ... API key into ...` | Paste the key at the top of the script (see [step 5](#5-paste-your-keys-in-the-env-file)). |
| `API key rejected` or HTTP 401 / 403 | Check the key and that it is for the right feed (Transport Stack key in `TSTACK_API_KEY`, OTD key in `OTD_API_KEY`). |
| `Missing gtfs/...` | Put the static GTFS `.txt` files in `gtfs/` (or use `--gtfs`). |
| Map goes blank for a moment | The server sent an empty file. The app keeps the old data and shows a warning. |
| Header shows `(fallback)` | The main feed had problems; the other feed is in use. The tool switches back by itself. |
| Many HTTP errors during a session | Your internet dropped (laptop sleep, hotspot). The collector retries; only the gap is lost. Both feeds fail at the same time in this case. |
| Garbled screen | The terminal is too small. Use at least 120 x 35. |
| `curses` error on Windows | Install `windows-curses` (it is in `requirements.txt`). |
| First run is slow | It is reading `stop_times.txt`. The result is cached. |
| Read error in `build_arrivals.py` | The collector was writing the file. Stop it and run again. |

<!-- | `older collector` message at start | A file of today was written with an older column layout. Rename the old `data/` folder and start again. | -->

## Security and data notes

- **Never commit your API keys.** The keys sit in your env file, so keep them only in your local copy and make sure it is added in .gitignore. A key is also part of the OTD request URL, so do not paste full request URLs into public places. Tools hide the keys in error messages and saved files.
- Suggested `.gitignore` entries:

  ```
  data/
  data_demo/
  data_old/
  gtfs/
  feed_compare_out/
  .venv/
  __pycache__/
  gtfs/.bus_cache.pkl
  gtfs/.arrival_patterns*.pkl
  ```
- The static data and the live feeds belong to the Delhi Government and are provided by the OTD portal (with IIIT-Delhi, DTC and DIMTS) and the Delhi Transport Stack. Read and follow their terms. Do not publish raw collected data unless the terms allow it.

## Roadmap

- [x] Live bus explorer (map, stop boards, reported and calculated speed, Google Maps links, two feeds with fallback)
- [x] Data collector with cleaning, a feed health log and a fallback feed
- [x] Arrival detection with real road shapes and a quality report
- [x] Feed comparison tool
- [ ] Understand and validate the reported speed of the Transport Stack feed
- [ ] ETA dataset from arrivals, and baselines (schedule, historical average, current speed)
- [ ] ETA models (gradient boosting, LSTM/GRU) with uncertainty
- [ ] Headway, bunching and anomaly detection
- [ ] Route analytics (reliability, delay hotspots)
- [ ] Interactive dashboard
