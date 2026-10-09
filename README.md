# Delhi Live Bus Explorer (terminal app)

## Setup
    python -m venv .venv
    .venv\Scripts\activate        (Windows)   or   source .venv/bin/activate   (Linux/Mac)
    pip install -r requirements.txt
1. Open `delhi_bus_tui.py` and paste your key into `API_KEY = "PASTE_YOUR_API_KEY_HERE"`.
2. Put your GTFS files (routes, trips, stops, stop_times .txt) in a folder called `gtfs` next to the script.
3. Run `python delhi_bus_tui.py`  (first run indexes stop_times.txt and saves a cache; later runs start fast).
Try without key/files: `python delhi_bus_tui.py --demo`

## Keys
arrows pan | + / - zoom | b select bus | Enter all fields | s stop board | S board of bus's next stop
/ find stop | r route filter | x clear | i feed fields + ID match rates | f refresh | ? help | q quit

Use a terminal at least 120 x 35. Windows Terminal works well.
Tested: logic and screens offline with fake buses, and the real curses UI in a pseudo-terminal.
NOT tested: the live download (needs your key). If it fails, the red error shows at the bottom.
