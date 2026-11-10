#!/usr/bin/env python3
"""
train_eta_v2.py - the ETA model, version 2.

What is new compared with train_eta.py
  1. LIVE "RECENT BUSES" INPUTS: how much slower or faster than normal the buses that passed a stop pair in the last
     10 minutes (all routes) and the last 20 minutes (this route) were. This tells the model about today's traffic.
  2. HISTORY BY TIME OF DAY: a factor for every hour and for weekday against weekend, learned on the training days.
  3. STOP INPUTS: distance since the last stop, and how far the bus is through the current stop-to-stop section.
  4. THE MODEL LEARNS THE CORRECTION to the historical ETA (target = real time minus the historical ETA), not the whole time.
  5. PREDICTION RANGES: two more models give a lower and an upper guess (default 10% and 90%): "8-12 min".
  6. LEAVE-ONE-DAY-OUT TEST (--mode cv): train on all other days, test on each day, so the result does not depend on one day.
  7. A per-route check: where is the model worse than the historical ETA?

   python train_eta_v2.py                    final model: train 60% / validation 20% / test 20% in time, ranges, ablations
   python train_eta_v2.py --mode cv          leave-one-day-out test (slower)
   python train_eta_v2.py --step 10          use every 10th ping (faster)
Keep train_eta.py in the same folder (this script uses its loading code). Needs lightgbm (uv add lightgbm), else scikit-learn is used.
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from train_eta import load, ping_features, make_cases, metrics
except ImportError:
    raise SystemExit("Keep train_eta.py in the same folder as train_eta_v2.py.")

FEATURES = ["dist", "h", "v1", "v2", "v3", "v_mean3", "standing_s", "hour", "dow", "is_weekend", "v_route", "eta_route", "hist_eta", "tod_factor",
            "hist_tod", "rc10", "rc10_n", "rr20", "rr20_n", "dist_prev", "seg_frac", "prog", "rep"]
RECENT = ["rc10", "rc10_n", "rr20", "rr20_n"]
MIN_SUPPORT = 8


def day_of(ts):
    return ((np.asarray(ts) + 19800) // 86400).astype(int)