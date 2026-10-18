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
