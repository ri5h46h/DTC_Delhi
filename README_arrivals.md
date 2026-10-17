# build_arrivals.py - from bus pings to stop arrival times

Put `build_arrivals.py` in the same folder as `otd_collector.py` (so the folders `data/` and `gtfs/` are found).
Needs only pandas and numpy (already installed for the collector).

## Run it
    python build_arrivals.py selftest                  # 1. simulated buses with known truth: how accurate is the method?
    python build_arrivals.py run                       # 2. only NEW or CHANGED days in data/clean/ (first run reads stop_times.txt: a few minutes, then cached)
    python build_arrivals.py run --no-shapes           # straight lines between stops instead of the real road line (shapes.txt is the default now)
    python build_arrivals.py run --date 2026-10-11 --force --routes 2097,1957
    python build_arrivals.py show --vehicle DL51EV2595 # 3. one bus step by step: pings -> progress along route -> arrivals
With `uv`: `uv run build_arrivals.py run`.

## What happens (5 steps, printed while it runs)
1. LOAD     stop list of each route_id (longest trip in stop_times.txt), joined by straight lines between stops.
2. MATCH    for each ping: distance to its route line and progress (metres along the route). Pings the collector flagged as GPS jumps are skipped.
3. RUNS     pings are cut into trips per (bus, route_id, trip_id). New run if no ping for 300 s or progress falls back by more than 250 m.
            A trip that mostly moves BACKWARDS along its route_id is marked direction_ok = False and not used.
4. ARRIVALS between the two pings just before and just after a stop, the pass time is interpolated:
            time = t1 + (stop_progress - progress1) / (progress2 - progress1) * (t2 - t1)
5. SAVE     data/arrivals/<date>_arrivals.csv.gz and <date>_runs.csv.gz, then a quality report.

## Output columns
arrivals: run_id, route_id, vehicle_id, trip_id, stop_idx (0 = first stop of the route_id), stop_id, arrive_ts (epoch s), arrive_ist (HH:MM:SS),
gap_s (seconds between the two pings used), span_m (metres between them), speed_kmh (speed around the stop), quality (good <= 40 s, ok <= 90 s, coarse)
runs: run_id, vehicle_id, route_id, trip_id, off_route_share, t_start, t_end, n_pings, prog_start, prog_end, direction_ok, usable

## Days and sessions
* The collector writes one file per day (Indian date). Sessions on the same day are added to the SAME day file.
* `run` only processes days that are new, whose clean file changed, or whose settings changed (the settings of each day are saved in `<date>_settings.json`). Other days are skipped (use `--force` to redo).
* Different sessions of one day stay apart automatically: no ping for 300 s cuts a bus's run. Each day's output is rebuilt completely (all sessions of that day).
* Stop the collector (or wait for it to finish a poll) before running this, so the day file is not being written while it is read. If you get a read error, just run again.

## Settings
--vmax 20 (fastest believable speed in m/s for the spike filter)   --no-shapes   --force
--on-route 100 (metres from route line)   --max-gap 120 (drop arrivals whose two pings are further apart)   --run-gap 300

## Limits to remember (put them in the report)
* Route line = straight lines between consecutive stops, not the real road (`shapes.txt` is not used yet). Fine for timing, rough for metres.
* Only the longest trip pattern of each route_id is used. Buses on shorter variants only get arrivals for stops that are in that pattern.
* "Arrival" = the moment the bus reaches the stop position. It is not door-opening time. If a bus waits at the stop, the interpolated time falls somewhere inside the ping gap, so error is at most the gap.
* Self-test uses simulated buses (GPS noise 6 m, pings every 10-30 s). Real accuracy must be checked on real data: use `show` and compare a few arrivals with what you saw.

## What to look at after the first real run
* % of pings on route, % of runs moving against their route_id (route_id wrong or late-updated), coverage %, and the stop-to-stop speed check.
* Next steps: headways and bunching (sort arrivals by route_id + stop_idx + time and take differences), travel times between stops (ETA targets), anomaly features.
