#!/usr/bin/env python3
"""
train_eta.py - the first AI model for the ETA: gradient boosting that learns from the observed arrival times.

Case      = one ping of a bus + one of the next stops ahead (like eta_eval.py).
Answer    = the real time the bus needed to reach that stop (from the arrival table).
Inputs    = distance, stops ahead, speed of the last 3 intervals, how long the bus has been standing, hour, weekday,
            the typical speed of the route, the historical travel time through the stops of the route, position on the
            route, and the raw number of the feed's speed field.
Split     = by TIME: first 60% train, next 20% validation (stops the training), last 20% test.
            The typical speeds and the historical stop-to-stop times are learned on the training part only.
Output    = test errors of the model next to the simple rules, a test without the feed number, what the model uses most,
            and the saved model in models/.

   python train_eta.py                      all days in data/arrivals
   python train_eta.py --step 10            use every 10th ping (faster)
Needs lightgbm (uv add lightgbm). If it is missing, scikit-learn's HistGradientBoostingRegressor is used instead.
"""
import argparse, json, os, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- run settings, live progress and timings
# (shared with train_eta_v2.py; none of this changes the numbers of the results)
#   --threads N : threads LightGBM may use (default 0 = LightGBM's own default = all cores, exactly as before)
#   --cool S    : pause of S seconds after every trained model so a fanless laptop can cool down (default 0 = no pause)
RUN = {"threads": 0, "cool": 0.0}
_T0 = time.time()
_MARKS, _LAST = [], [_T0]
_LIVE = {"on": sys.stderr.isatty(), "w": 0, "last": 0.0}      # the live status line is only drawn on a screen (stderr), never mixed into the normal output


_COLOR = {"out": False, "err": False}                                # colors are only used on a screen (never in a file) and can be switched off with --no-color or NO_COLOR
_KIND = {"prep": "36", "train": "1;32", "test": "1;35", "cool": "33"}     # live line: preparing data = cyan, training = green, testing = magenta, cooling pause = yellow


def _paint_live(text, kind):
    """[mm:ss] in grey, the rest in the colour of the kind of work."""
    i = text.find("] ")
    head, rest = (text[:i + 2], text[i + 2:]) if text.startswith("[") and i > 0 else ("", text)
    return (f"\033[2m{head}\033[0m" if head else "") + f"\033[{_KIND.get(kind, '0')}m{rest}\033[0m"


def _paint_line(line):
    """Colours for the printed result (only used on a screen). The text itself is never changed."""
    t = line.strip()
    if not t:
        return line
    code = None
    if t.startswith("4. TRAIN") or t.startswith("TRAIN ") or t.startswith("trained "):
        code = "1;32"                                                      # training = green
    elif t[:2] in ("1.", "2.", "3.") and t[2:3] == " ":
        code = "36"                                                        # loading / cases / learning = cyan
    elif t.startswith("=== RESULT") or t.startswith(("BY HOW FAR", "BY WHAT THE BUS", "PER-ROUTE CHECK", "PREDICTION RANGES", "WHAT THE MODEL USES MOST")):
        code = "1;35"                                                      # testing and results = magenta
    elif t.startswith("-> the AI model"):
        code = "1;32" if "lower (better)" in t else "1;31"
    elif t.startswith("RUN TIME"):
        code = "1;33"
    elif t.startswith("Saved the model"):
        code = "1;32"
    elif t.startswith("Send this whole output"):
        code = "2"
    return line if code is None else f"\033[{code}m{line}\033[0m"


def _dur(s):
    s = int(round(s))
    return f"{s} s" if s < 60 else (f"{s // 60} min {s % 60:02d} s" if s < 3600 else f"{s // 3600} h {(s % 3600) // 60:02d} min")


def _clock():
    s = int(time.time() - _T0)
    return f"{s // 60:02d}:{s % 60:02d}"


def live(text, force=False, kind="prep"):
    """One status line on the screen, overwritten by the next one and removed before any normal output."""
    if not _LIVE["on"]:
        return
    now = time.time()
    if not force and now - _LIVE["last"] < 0.4:
        return
    _LIVE["last"] = now
    try:
        cols = os.get_terminal_size(sys.stderr.fileno()).columns - 1
    except OSError:
        cols = 100
    text = text[:cols]
    shown = _paint_live(text, kind) if _COLOR["err"] else text
    sys.stderr.write("\r" + shown + " " * max(0, _LIVE["w"] - len(text)))
    sys.stderr.flush()
    _LIVE["w"] = len(text)


def live_clear():
    if _LIVE["on"] and _LIVE["w"]:
        sys.stderr.write("\r" + " " * _LIVE["w"] + "\r")
        sys.stderr.flush()
        _LIVE["w"] = 0


class _Out:
    """stdout wrapper: removes the live status line before normal text is printed, so the printed result stays clean."""
    def __init__(self, real):
        self.real = real

    def write(self, s):
        live_clear()
        if _COLOR["out"] and s and s != "\n":
            s = "\n".join(_paint_line(l) for l in s.split("\n"))
        return self.real.write(s)

    def __getattr__(self, k):
        return getattr(self.real, k)


def add_run_args(ap):
    ap.add_argument("--threads", type=int, default=0, help="threads for LightGBM (default 0 = all cores, as before). A smaller number runs cooler and slower")
    ap.add_argument("--cool", type=float, default=0.0, help="seconds to pause after every trained model so the computer can cool down (default 0)")
    ap.add_argument("--no-color", action="store_true", help="no colours on the screen (colours are also off when NO_COLOR is set or the output goes to a file)")


def setup_run(a):
    RUN["threads"], RUN["cool"] = a.threads, a.cool
    if a.threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(a.threads)      # must be set before LightGBM / scikit-learn are loaded: keeps their OpenMP code (also the prediction) at N threads
    ok = not a.no_color and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"
    _COLOR["out"], _COLOR["err"] = ok and sys.stdout.isatty(), ok and sys.stderr.isatty()
    if os.name == "nt" and (_COLOR["out"] or _COLOR["err"]):
        os.system("")                                        # lets the Windows 10+ console show colours
    if (_LIVE["on"] or _COLOR["out"]) and not isinstance(sys.stdout, _Out):
        sys.stdout = _Out(sys.stdout)
    if a.threads or a.cool:
        print(f"run settings: threads {a.threads or 'all (default)'}   pause after each model {a.cool:g} s")


class Progress:
    """Progress of a loop: 'label: 1,200/8,400 (14%), about 20 s left'. Shown live on a screen, or every 10% when the output is a file."""
    def __init__(self, label, total, kind="prep"):
        self.label, self.total, self.t0, self.next10, self.kind = label, max(1, total), time.time(), 1, kind

    def update(self, n):
        frac = n / self.total
        el = time.time() - self.t0
        left = f", about {_dur(el * (1 - frac) / frac)} left" if frac > 0.02 and el > 3 else ""
        text = f"[{_clock()}] {self.label}: {n:,}/{self.total:,} ({100 * frac:.0f}%){left}"
        if _LIVE["on"]:
            live(text, kind=self.kind)
        elif frac * 10 >= self.next10:
            self.next10 = int(frac * 10) + 1
            print(text, file=sys.stderr)

    def done(self):
        live_clear()


def note(text, kind="prep"):
    """A short live message for a step that has no loop (for example a big sort)."""
    if _LIVE["on"]:
        live(f"[{_clock()}] {text}", force=True, kind=kind)
    else:
        print(f"[{_clock()}] {text}", file=sys.stderr)


def lgb_progress(label, every=100):
    """LightGBM callback: shows tree number, validation loss, best tree so far and how close the training is to stopping. It only READS the results."""
    t0, st = time.time(), {"best": float("inf"), "bi": 0}

    def cb(env):
        it = env.iteration + 1
        res = env.evaluation_result_list
        val = res[0][2] if res else float("nan")
        if val < st["best"]:
            st["best"], st["bi"] = val, it
        text = (f"[{_clock()}] {label}: tree {it} | validation loss {val:.2f} | best {st['best']:.2f} at tree {st['bi']} | "
                f"no gain for {it - st['bi']}/50 trees (stops at 50) | {_dur(time.time() - t0)}")
        if _LIVE["on"]:
            live(text, kind="train")
        elif it % every == 0:
            print(text, file=sys.stderr)
    cb.order = 20
    return cb


def cool_down(label=""):
    s = RUN["cool"]
    if s <= 0:
        return
    end = time.time() + s
    if not _LIVE["on"]:
        print(f"[{_clock()}] cooling down {s:g} s after {label}", file=sys.stderr)
    while time.time() < end:
        live(f"[{_clock()}] cooling down after {label}: {max(0, end - time.time()):.0f} s left (--cool {s:g})", force=True, kind="cool")
        time.sleep(min(1.0, max(0.0, end - time.time())))
    live_clear()


def predict_progress(pr, X, label, chunk=100_000):
    """Same predictions as pr(X), made in chunks so the progress can be shown (every row is predicted on its own, so the numbers are identical)."""
    if len(X) <= chunk:
        note(f"predicting {label} ({len(X):,} cases) ...", kind="test")
        return pr(X)
    prog_ = Progress(f"predicting {label}", len(X), kind="test")
    out = []
    for i in range(0, len(X), chunk):
        prog_.update(i)
        out.append(np.asarray(pr(X.iloc[i:i + chunk])))
    prog_.done()
    return np.concatenate(out)


def mark(name):
    """Remember how long the step since the last mark took (printed as RUN TIME at the end)."""
    now = time.time()
    _MARKS.append((name, now - _LAST[0]))
    _LAST[0] = now


def print_times():
    live_clear()
    if not _MARKS:
        return
    print("\nRUN TIME (pauses from --cool are included in the model that was just trained)")
    for n, s in _MARKS:
        print(f"   {n:<44}{_dur(s):>14}")
    print(f"   {'total':<44}{_dur(time.time() - _T0):>14}")


MIN_REMAIN, MAX_REMAIN = 5, 1500
FEATURES = ["dist", "h", "v1", "v2", "v3", "v_mean3", "standing_s", "hour", "dow", "is_weekend", "v_route", "eta_route", "hist_eta", "prog", "rep"]


def load(folder, date):
    folder = Path(folder)
    files = sorted(folder.glob("*_pings.csv.gz"))
    if date:
        files = [f for f in files if f.name.startswith(date)]
    if not files:
        raise SystemExit(f"No *_pings.csv.gz files in {folder}. Run build_arrivals.py first.")
    P, A = [], []
    prog_ = Progress("reading the day files", len(files))
    for k, f in enumerate(files):
        prog_.update(k)
        pg = pd.read_csv(f, dtype={"vehicle_id": str, "route_id": str})
        ar = pd.read_csv(str(f).replace("_pings", "_arrivals"), dtype={"vehicle_id": str, "route_id": str})
        if "stop_prog_m" not in ar:
            raise SystemExit("The arrivals files have no stop_prog_m column. Run:  python build_arrivals.py run --force")
        pg["run_key"] = k * 10_000_000 + pg["run_id"]; ar["run_key"] = k * 10_000_000 + ar["run_id"]
        P.append(pg); A.append(ar)
    prog_.done()
    note("joining the days ...")
    return pd.concat(P, ignore_index=True), pd.concat(A, ignore_index=True), files


def ping_features(p):
    """Features that only need the pings of one run: speeds of the last intervals, standing time, hour, weekday."""
    note("speeds of the last pings of every run ...")
    p = p.sort_values(["run_key", "ts"]).reset_index(drop=True)
    g = p.groupby("run_key")
    p["v1"] = p["calc_route_kmh"]; p["v2"] = g["calc_route_kmh"].shift(1); p["v3"] = g["calc_route_kmh"].shift(2)
    p["v_mean3"] = p[["v1", "v2", "v3"]].mean(axis=1)
    still = g["prog"].diff().abs() < 15
    last_move = p["ts"].where(~still).groupby(p["run_key"]).ffill()
    p["standing_s"] = (p["ts"] - last_move).fillna(0)
    t = p["ts"] + 19800
    p["hour"] = (t % 86400) / 3600.0
    p["dow"] = ((t // 86400) + 3) % 7                      # 0 = Monday
    p["is_weekend"] = (p["dow"] >= 5).astype(int)
    return p


def make_cases(pings, arr, step, hmax):
    arr = arr.sort_values(["run_key", "stop_prog_m"]).reset_index(drop=True)
    pidx, aidx = pings.groupby("run_key").indices, arr.groupby("run_key").indices
    P_ts, P_prog = pings["ts"].values, pings["prog"].values
    A_ts, A_sp, A_si = arr["arrive_ts"].values, arr["stop_prog_m"].values, arr["stop_idx"].values
    out = {"pi": [], "h": [], "dist": [], "actual": [], "sidx": []}
    prog_ = Progress("making cases (ping + next stops)", len(aidx))
    for n_run, (key, ia) in enumerate(aidx.items()):
        if n_run % 200 == 0:
            prog_.update(n_run)
        ip = pidx.get(key)
        if ip is None or len(ip) < 3:
            continue
        ip = ip[::step]
        spr, atr, sir, prog, ts = A_sp[ia], A_ts[ia], A_si[ia], P_prog[ip], P_ts[ip]
        j0 = np.searchsorted(spr, prog + 20, side="right")
        for h in range(hmax):
            j = j0 + h
            ok = j < len(spr)
            if not ok.any():
                break
            ii, jj = np.where(ok)[0], j[ok]
            remain = atr[jj] - ts[ii]
            g = (remain >= MIN_REMAIN) & (remain <= MAX_REMAIN)
            out["pi"].append(ip[ii][g]); out["h"].append(np.full(g.sum(), h + 1)); out["dist"].append(spr[jj][g] - prog[ii][g])
            out["actual"].append(remain[g]); out["sidx"].append(sir[jj][g])
    prog_.done()
    c = pd.DataFrame({k: np.concatenate(v) if v else np.array([]) for k, v in out.items()})
    pf = pings.iloc[c["pi"].values][["ts", "route_id", "vehicle_id", "prog", "v1", "v2", "v3", "v_mean3", "standing_s", "hour", "dow", "is_weekend", "speed_rep"]].reset_index(drop=True)
    pf = pf.rename(columns={"speed_rep": "rep"})
    return pd.concat([c.drop(columns="pi").reset_index(drop=True), pf], axis=1)


def history_tables(arr, cut, v_global):
    """Median travel time between two neighbouring stops of a route, from arrivals BEFORE the cut. Returns per route: stop idx, progress, cumulative time."""
    note("stop-to-stop times of the training part ...")
    a = arr[arr["arrive_ts"] < cut].sort_values(["run_key", "stop_idx"])
    g = a.groupby("run_key")
    a = a.assign(next_t=g["arrive_ts"].shift(-1), next_i=g["stop_idx"].shift(-1))
    seg = a[a["next_i"] == a["stop_idx"] + 1].copy()
    seg["t"] = seg["next_t"] - seg["arrive_ts"]
    med = seg[seg["t"].between(3, 600)].groupby(["route_id", "stop_idx"])["t"].median()
    sp = arr.groupby(["route_id", "stop_idx"])["stop_prog_m"].first().reset_index().sort_values(["route_id", "stop_idx"])
    tables = {}
    medd = med.to_dict()
    prog_ = Progress("history tables per route", sp["route_id"].nunique())
    for n_r, (r, g2) in enumerate(sp.groupby("route_id")):
        if n_r % 50 == 0:
            prog_.update(n_r)
        idx, prog = g2["stop_idx"].values, g2["stop_prog_m"].values
        if len(idx) < 3 or np.any(np.diff(prog) < 0):
            continue
        T = np.empty(len(idx) - 1)
        for m in range(len(idx) - 1):
            T[m] = medd.get((r, idx[m]), np.nan) if idx[m + 1] == idx[m] + 1 else np.nan
            if np.isnan(T[m]):
                T[m] = (prog[m + 1] - prog[m]) / v_global                      # no data: distance / typical speed
        tables[r] = (idx, prog, np.concatenate([[0.0], np.cumsum(T)]))
    prog_.done()
    return tables


def hist_eta(cases, tables):
    out = np.full(len(cases), np.nan)
    pos = {}
    for k, r in enumerate(cases["route_id"].values):
        pos.setdefault(r, []).append(k)
    p = cases["prog"].values; si = cases["sidx"].values
    for r, rows in pos.items():
        t = tables.get(r)
        if t is None:
            continue
        idx, prog, C = t
        rows = np.array(rows)
        j = np.searchsorted(idx, si[rows])
        ok = (j < len(idx)) & (idx[np.minimum(j, len(idx) - 1)] == si[rows])
        tpos = np.interp(p[rows], prog, C)
        res = np.where(ok, C[np.minimum(j, len(idx) - 1)] - tpos, np.nan)
        out[rows] = res
    return out


def metrics(pred, actual):
    e = pred - actual
    return dict(mae=np.nanmean(np.abs(e)), med=np.nanmedian(np.abs(e)), bias=np.nanmean(e), w60=100 * np.nanmean(np.abs(e) <= 60), w120=100 * np.nanmean(np.abs(e) <= 120))


def _predict_fn(m):
    """The predict function of a LightGBM model. With --threads N it also uses N threads (the numbers are the same, only the speed differs)."""
    if RUN["threads"] > 0:
        return lambda X: m.predict(X, num_threads=RUN["threads"])
    return m.predict


def fit_model(Xtr, ytr, Xva, yva, backend, label="model"):
    if backend == "lightgbm":
        import lightgbm as lgb
        params = dict(objective="regression_l1", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.9,
                      bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=RUN["threads"])
        dtr = lgb.Dataset(Xtr, ytr); dva = lgb.Dataset(Xva, yva, reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0), lgb_progress(label)])
        live_clear()
        print(f"      trained {m.best_iteration} trees")
        cool_down(label)
        return m, _predict_fn(m)
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.1, max_iter=300, max_leaf_nodes=63, min_samples_leaf=100,
                                      early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=0)
    note(f"{label}: training with scikit-learn (no live tree counter for this backend) ...", kind="train")
    m.fit(Xtr, ytr)
    live_clear()
    print(f"      trained {m.n_iter_} trees (scikit-learn)")
    cool_down(label)
    return m, m.predict


def main():
    ap = argparse.ArgumentParser(description="Train an ETA model on observed arrival times")
    ap.add_argument("--arrivals", default="data/arrivals"); ap.add_argument("--date")
    ap.add_argument("--step", type=int, default=6, help="use every N-th ping"); ap.add_argument("--hmax", type=int, default=6, help="stops ahead (1..N)")
    ap.add_argument("--max-train", type=int, default=1_500_000); ap.add_argument("--backend", choices=["auto", "lightgbm", "sklearn"], default="auto")
    ap.add_argument("--no-ablation", action="store_true"); ap.add_argument("--out", default="models")
    add_run_args(ap)
    a = ap.parse_args()
    setup_run(a)
    backend = a.backend
    if backend == "auto":
        try:
            import lightgbm  # noqa
            backend = "lightgbm"
        except ImportError:
            backend = "sklearn"
    print(f"model: {'LightGBM' if backend == 'lightgbm' else 'scikit-learn HistGradientBoosting'}")
    pings, arr, files = load(a.arrivals, a.date)
    print(f"1. LOAD      {len(files)} day(s): {len(pings):,} pings, {len(arr):,} arrivals")
    mark("1. load the day files")
    pings = ping_features(pings)
    cases = make_cases(pings, arr, a.step, a.hmax)
    cases = cases[cases["dist"] > 0].reset_index(drop=True)
    if len(cases) < 3000:
        raise SystemExit(f"Only {len(cases):,} cases. Collect and process more data (or use a smaller --step).")
    mark("2. speeds and cases")
    c1, c2 = cases["ts"].quantile(0.6), cases["ts"].quantile(0.8)
    tr, va, te = cases[cases["ts"] < c1], cases[(cases["ts"] >= c1) & (cases["ts"] < c2)], cases[cases["ts"] >= c2]
    print(f"2. CASES     {len(cases):,} cases (a ping + one of the next 1-{a.hmax} stops): train {len(tr):,}, validation {len(va):,}, test {len(te):,}"
          f"  (test = after {pd.to_datetime(c2 + 19800, unit='s'):%d %b %H:%M} IST)")
    vt = (tr["dist"] / tr["actual"]).clip(1, 20)
    v_global = float(vt.median()); v_route = vt.groupby(tr["route_id"]).median()
    tables = history_tables(arr, c1, v_global)
    print(f"3. LEARN     typical speed {v_global:.1f} m/s ({v_global * 3.6:.0f} km/h); typical speed of {len(v_route):,} routes; stop-to-stop times of {len(tables):,} routes (training part only)")
    mark("3. learn typical times")
    for d in (tr, va, te):
        d["v_route"] = d["route_id"].map(v_route).fillna(v_global).clip(1, 20)
        d["eta_route"] = d["dist"] / d["v_route"]
        d["hist_eta"] = hist_eta(d, tables)
    mark("   inputs for train / validation / test")
    sub = tr.sample(min(len(tr), a.max_train), random_state=0) if len(tr) > a.max_train else tr
    ev = te.copy()
    ev["r_route"] = ev["eta_route"]; ev["r_hist"] = ev["hist_eta"].fillna(ev["eta_route"])
    ev["r_calc"] = ev["dist"] / (ev["v1"] / 3.6).clip(2, 14).fillna(v_global); ev["r_feed"] = ev["dist"] / ev["rep"].clip(2, 14).fillna(v_global)
    print("4. TRAIN     main model with all inputs ...")
    model, pred = fit_model(sub[FEATURES].astype("float32"), sub["actual"], va[FEATURES].astype("float32"), va["actual"], backend, "main model")
    ev["r_model"] = predict_progress(pred, ev[FEATURES].astype("float32"), "test part, main model")
    mark("4. train main model")
    rules = [("AI model (all inputs)", "r_model")]
    if not a.no_ablation:
        f2 = [f for f in FEATURES if f != "rep"]
        print("   TRAIN     second model WITHOUT the feed number (to see if it helps) ...")
        m2, p2 = fit_model(sub[f2].astype("float32"), sub["actual"], va[f2].astype("float32"), va["actual"], backend, "model without feed number")
        ev["r_model2"] = predict_progress(p2, ev[f2].astype("float32"), "test part, model without feed number"); rules.append(("AI model without feed number", "r_model2"))
        mark("   train model without feed number")
    rules += [("historical stop-to-stop times", "r_hist"), ("route typical speed", "r_route"), ("calculated speed", "r_calc"), ("feed number as speed", "r_feed")]

    note("scoring the test part ...", kind="test")
    print(f"\n=== RESULT ON THE TEST PART ({len(ev):,} cases the model never saw; error = guess minus real time) ===")
    print(f"   {'rule':<32}{'mean error':>11}{'typical error':>14}{'bias':>9}{'within 1 min':>14}{'within 2 min':>14}")
    rows = sorted(((n, metrics(ev[c].values, ev["actual"].values)) for n, c in rules), key=lambda x: x[1]["mae"])
    for n, m in rows:
        print(f"   {n:<32}{m['mae']:>9.0f} s{m['med']:>12.0f} s{m['bias']:>+7.0f} s{m['w60']:>12.0f}%{m['w120']:>13.0f}%")
    base = min(m["mae"] for n, m in rows if not n.startswith("AI"))
    mm = metrics(ev["r_model"].values, ev["actual"].values)["mae"]
    gain = 100 * (base - mm) / base
    print(f"   -> the AI model's mean error is {abs(gain):.0f}% {'lower (better)' if gain >= 0 else 'higher (worse)'} than the best simple rule ({base:.0f} s)")
    print("\nBY HOW FAR AHEAD THE STOP IS (mean error in seconds)")
    groups = [("1 stop", ev["h"] == 1), ("2 stops", ev["h"] == 2), ("3-4 stops", ev["h"].between(3, 4)), (f"5-{a.hmax} stops", ev["h"] >= 5)]
    print(f"   {'rule':<32}" + "".join(f"{n:>11}" for n, _ in groups))
    for n, c in rules:
        print(f"   {n:<32}" + "".join(f"{np.nanmean(np.abs(ev.loc[m, c] - ev.loc[m, 'actual'])):>9.0f} s" for _, m in groups))
    print("\nBY WHAT THE BUS DOES AT THE MOMENT")
    for name, m in (("standing (under 2 km/h)", ev["v1"] < 2), ("moving (5 km/h or more)", ev["v1"] >= 5)):
        if m.sum() >= 500:
            print(f"   {name}: {int(m.sum()):,} cases   " + "   ".join(f"{n}: {np.nanmean(np.abs(ev.loc[m, c] - ev.loc[m, 'actual'])):.0f} s" for n, c in rules[:3]))
    note("what the model uses most (feature importance) ...", kind="test")
    print("\nWHAT THE MODEL USES MOST")
    if backend == "lightgbm":
        imp = dict(zip(FEATURES, model.feature_importance("gain"))); tot = sum(imp.values()) or 1
        top = sorted(imp.items(), key=lambda x: -x[1])[:10]
        for k, v in top: print(f"   {k:<14}{100 * v / tot:>6.1f}%")
    else:
        from sklearn.inspection import permutation_importance
        s = ev.sample(min(len(ev), 30000), random_state=0)
        r = permutation_importance(model, s[FEATURES].astype("float32"), s["actual"], n_repeats=1, random_state=0, scoring="neg_mean_absolute_error")
        for k, v in sorted(zip(FEATURES, r.importances_mean), key=lambda x: -x[1])[:10]: print(f"   {k:<14} error grows by {v:>6.1f} s if this input is shuffled")
    note("saving the model and tables ...")
    out = Path(a.out); out.mkdir(exist_ok=True)
    (out / "eta_model.pkl").write_bytes(pickle.dumps({"backend": backend, "model": model, "features": FEATURES}))
    (out / "eta_tables.pkl").write_bytes(pickle.dumps({"v_global": v_global, "v_route": v_route.to_dict(), "hist_tables": tables}))
    (out / "eta_info.json").write_text(json.dumps({"backend": backend, "features": FEATURES, "test_mean_error_s": round(float(mm), 1), "cases": len(cases)}, indent=1))
    print(f"\nSaved the model and the learned tables in {out}/")
    mark("5. report and save")
    print_times()
    print("\nSend this whole output back.")


if __name__ == "__main__":
    main()
