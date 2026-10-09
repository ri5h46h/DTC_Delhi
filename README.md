# Delhi Live Bus Explorer TUI App (terminal app)

## Setup

    Install UV first (https://docs.astral.sh/uv/)
    uv python install 3.10
    uv init
    uv add -r requirements.txt
    uv run delhi_bus_tui.py

    or if not uv, then you can just use the basic inbuilt python-venv for creating a venv and then installing the requirements.

1. Open `.env.example` file and paste your key into `API_KEY = "<paste your Dehi OTD API Key here>"
`. And also rename it to `.env`

2. Put your GTFS files (routes, trips, stops, stop_times .txt) in a folder called `gtfs` next to the script.

3. Run `delhi_bus_tui.py`  (first run indexes stop_times.txt and saves a cache; later runs start fast).
Try without key/files: `delhi_bus_tui.py --demo`

## Keys
arrows pan | + / - zoom | b select bus | Enter all fields | s stop board | S board of bus's next stop
/ find stop | r route filter | x clear | i feed fields + ID match rates | f refresh | ? help | q quit

Use a terminal at least 120 x 35. Windows Terminal works well.
Tested: logic and screens offline with fake buses, and the real curses UI in a pseudo-terminal.
NOT tested: the live download (needs your key). If it fails, the red error shows at the bottom.
