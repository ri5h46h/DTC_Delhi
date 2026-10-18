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
