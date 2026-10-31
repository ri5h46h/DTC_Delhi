#!/usr/bin/env python3
"""
speed_probe.py - what is the 'speed' field of the Transport Stack feed really?

In our data the number in that field does NOT behave like a speed (higher numbers go with buses that hardly move).
This script looks at the saved pings (data/clean) from several sides and tests some simple guesses:
   [counter]   the number grows while the bus stands still and drops when it moves  (a standing / idle time?)
   [accuracy]  the number is bigger when the GPS position jitters more               (a GPS accuracy value?)
It prints numbers and a short reading of them. It proves nothing by itself: it tells you where to look next.

   python speed_probe.py                       all days in data/clean
   python speed_probe.py --date 2026-10-25     one day
   python speed_probe.py --buses 3             show the pings of 3 buses around a stop-and-go moment
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

IST = 19800
WANT = {"ts", "vehicle_id", "source", "speed_rep", "speed_ms", "dt_s", "moved_m", "speed_calc_kmh", "age_s", "lat", "lon"}


def load(data, date):
    files = sorted((Path(data) / "clean").glob("*.csv.gz"))
    if date:
        files = [f for f in files if f.name.startswith(date)]
    if not files:
        raise SystemExit(f"No files in {data}/clean (date filter: {date}).")
    parts = []
    for f in files:
        df = pd.read_csv(f, usecols=lambda c: c in WANT, dtype={"vehicle_id": str, "source": str}, low_memory=False)
        if "speed_ms" in df and "speed_rep" not in df:
            df = df.rename(columns={"speed_ms": "speed_rep"})
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    if "source" in df:
        df = df[df["source"] == "tstack"]
    for c in ("speed_rep", "dt_s", "moved_m", "speed_calc_kmh", "age_s", "lat", "lon", "ts"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["speed_rep", "ts"]).sort_values(["vehicle_id", "ts"]).reset_index(drop=True)
    g = df.groupby("vehicle_id")
    df["rep"] = df["speed_rep"]
    df["rep_prev"] = g["rep"].shift(1)
    df["calc"] = df["speed_calc_kmh"]                         # speed of the interval that ENDS at this ping
    df["calc_next"] = g["calc"].shift(-1)                     # speed of the interval that STARTS at this ping
    df["same_bus_prev"] = df["vehicle_id"] == df["vehicle_id"].shift(1)
    df["hour"] = ((df["ts"] + IST) // 3600 % 24)
    return df, files


def main():
    ap = argparse.ArgumentParser(description="Look at the speed field of the Transport Stack feed")
    ap.add_argument("--data", default="data"); ap.add_argument("--date"); ap.add_argument("--buses", type=int, default=2)
    a = ap.parse_args()
    df, files = load(a.data, a.date)
    print(f"=== SPEED FIELD PROBE: {len(df):,} Transport Stack pings from {len(files)} file(s) ===")
    r = df["rep"]

    print("\n1. THE NUMBERS THEMSELVES")
    q = r.quantile([.01, .05, .25, .5, .75, .9, .99])
    print("   percentiles 1/5/25/50/75/90/99: " + " / ".join(f"{v:.1f}" for v in q.values) + f"   max {r.max():.2f}")
    print(f"   zero: {100 * (r == 0).mean():.1f}%   whole numbers: {100 * ((r % 1) == 0).mean():.1f}%   different values: {r.nunique():,}")
    print(f"   exactly the largest value ({r.max():g}): {100 * (r == r.max()).mean():.2f}% of rows  (a cap such as 50 would point to a limited scale)")

    s = df.sample(min(len(df), 400_000), random_state=0)
    s = s.assign(moved=s["moved_m"], dt=s["dt_s"], age=s["age_s"])
    cols = {"calculated speed before this ping": "calc", "calculated speed after this ping": "calc_next", "metres moved since the previous ping": "moved",
            "seconds since the previous ping": "dt", "age of the ping at download": "age", "hour of the day": "hour", "latitude": "lat", "longitude": "lon"}
    print("\n2. RANK CORRELATION OF THE NUMBER WITH OTHER THINGS (+1 together, -1 opposite, 0 unrelated)")
    cor = s[["rep"] + list(cols.values())].corr(method="spearman")["rep"]
    for name, c in cols.items():
        print(f"   {name:<40} {cor[c]:+.2f}")

    print("\n3. WHAT THE NUMBER LOOKS LIKE FOR BUSES MOVING AT DIFFERENT SPEEDS (calculated speed before the ping)")
    bins = [-0.01, 0.5, 2, 5, 10, 20, 30, 1e9]; names = ["0", "0-2", "2-5", "5-10", "10-20", "20-30", "30+"]
    t = df.dropna(subset=["calc"]).groupby(pd.cut(df.dropna(subset=["calc"])["calc"], bins, labels=names), observed=True)["rep"].describe(percentiles=[.25, .5, .75])
    print("   calculated km/h -> number in the feed field: median (25th-75th percentile)  [pings]")
    for k, v in t.iterrows():
        print(f"     {k:>6}: {v['50%']:6.1f}  ({v['25%']:.1f}-{v['75%']:.1f})   [{int(v['count']):,}]")

    ok = df["same_bus_prev"] & df["dt_s"].between(15, 90) & df["rep_prev"].notna()
    st = df[ok & (df["moved_m"] < 15)]
    print("\n4. [counter] WHEN THE BUS STANDS STILL (moved under 15 m between two pings 15-90 s apart): " + f"{len(st):,} pairs")
    if len(st) >= 200:
        d = st["rep"] - st["rep_prev"]
        rising, same, falling = (d > 0).mean(), (d == 0).mean(), (d < 0).mean()
        print(f"   the number rises in {100 * rising:.0f}% of the pairs, stays the same in {100 * same:.0f}%, falls in {100 * falling:.0f}%")
        up = st[d > 0]
        k = ((up["rep"] - up["rep_prev"]) / up["dt_s"]).median() if len(up) else float("nan")
        print(f"   when it rises: median rise per second {k:.4f}  (= {k * 60:.2f} per minute)")
        if rising >= 0.5 and 0.6 <= k <= 1.4:
            print("   -> it grows about 1 unit per SECOND while the bus stands: it looks like a standing time in seconds")
        elif rising >= 0.5 and 0.009 <= k <= 0.025:
            print("   -> it grows about 1 unit per MINUTE while the bus stands: it looks like a standing time in minutes")
        else:
            print("   -> no clear counter behaviour (it does not grow steadily with time while the bus stands)")
    mv = df[df["same_bus_prev"] & df["dt_s"].between(15, 90) & (df["moved_m"] > 150)]
    if len(mv):
        print(f"   when the bus moves more than 150 m: median number {mv['rep'].median():.1f}, {100 * (mv['rep'] <= 2).mean():.0f}% of them are 2 or less ({len(mv):,} pairs)")

    sl = df[df["same_bus_prev"] & df["dt_s"].between(15, 90) & (df["calc"] < 3)]
    print("\n5. [accuracy] WHEN THE BUS IS NEARLY STILL (calculated speed under 3 km/h): " + f"{len(sl):,} pairs")
    if len(sl) >= 200:
        c = sl["rep"].corr(sl["moved_m"], method="spearman")
        print(f"   rank correlation between the number and the metres the position jumped: {c:+.2f}")
        print("   -> " + ("a bigger number goes with more jumping of the position: it may be a GPS accuracy value (like HDOP)" if c >= 0.3 else
                          "no clear link with the jumping of the position"))

    print("\n6. DOES THE NUMBER STAY THE SAME FROM ONE PING TO THE NEXT OF THE SAME BUS?")
    pr = df[df["same_bus_prev"] & df["rep_prev"].notna()]
    print(f"   exactly the same value as the previous ping: {100 * (pr['rep'] == pr['rep_prev']).mean():.1f}%;  rank correlation with the previous value: {pr['rep'].corr(pr['rep_prev'], method='spearman'):+.2f}")

    print(f"\n7. EXAMPLES: pings of {a.buses} bus(es) around a moment where it starts moving after standing")
    shown = 0
    for vid, g in df.groupby("vehicle_id"):
        if len(g) < 30:
            continue
        mvd = g["moved_m"].fillna(0).values
        idx = [i for i in range(8, len(g) - 8) if mvd[i] > 150 and (mvd[i - 3:i] < 15).all()]
        if not idx:
            continue
        i = idx[0]; sub = g.iloc[i - 5:i + 5]
        print(f"   bus {vid}:")
        print("      time IST  secs  moved_m  calc_km/h  feed_number")
        for _, row in sub.iterrows():
            clk = pd.to_datetime(row["ts"] + IST, unit="s").strftime("%H:%M:%S")
            cs = "-" if pd.isna(row["calc"]) else f"{row['calc']:.0f}"
            print(f"      {clk}  {row['dt_s'] if pd.notna(row['dt_s']) else '-':>4}  {row['moved_m'] if pd.notna(row['moved_m']) else '-':>7}  {cs:>9}  {row['rep']:>11.2f}")
        shown += 1
        if shown >= a.buses:
            break
    if not shown:
        print("   (no bus with a clear stop-and-go moment found)")

    # ---------- 8-11: do different buses use the field in different ways? ----------
    def sig(v):                                   # a speed sent as whole km/h and divided by 3.6 gives numbers like 5.00, 3.33, 9.72
        f = (v * 3.6) % 1
        return (f < 0.03) | (f > 0.97)
    df["sig"] = sig(df["rep"])
    df["series"] = df["vehicle_id"].str.extract(r"^(.*?)\d{3,4}$")[0].fillna(df["vehicle_id"])
    print("\n8. THE 'WHOLE KM/H' SIGNATURE")
    print(f"   share of all numbers that are a whole number of km/h divided by 3.6 (5.00, 3.33, 9.72 ...): {100 * df['sig'].mean():.1f}%   (by pure chance about 6%)")
    print("   -> a share far above 6% means the number is a real speed in m/s that was sent as whole km/h")
    c = df.dropna(subset=["calc"])
    per = c.groupby("vehicle_id").apply(lambda g: pd.Series({"n": len(g), "rho": g["rep"].corr(g["calc"], method="spearman") if len(g) >= 40 else np.nan,
                                                             "sig": g["sig"].mean()}), include_groups=False) if len(c) else pd.DataFrame()
    per = per[per["n"] >= 40] if len(per) else per
    if len(per):
        print("\n9. BUS BY BUS: rank correlation of the number with the calculated speed (buses with at least 40 pings): " + f"{len(per):,} buses")
        edges = [-1.01, -0.5, -0.2, 0.2, 0.5, 1.01]; labs = ["below -0.5", "-0.5 to -0.2", "-0.2 to 0.2", "0.2 to 0.5", "above 0.5"]
        hist = pd.cut(per["rho"].dropna(), edges, labels=labs).value_counts().reindex(labs).fillna(0)
        print("   " + " | ".join(f"{k}: {int(v):,} buses ({100 * v / max(1, hist.sum()):.0f}%)" for k, v in hist.items()))
        print(f"   buses where {'at least 80%'} of the numbers have the whole-km/h signature: {int((per['sig'] >= 0.8).sum()):,} of {len(per):,} ({100 * (per['sig'] >= 0.8).mean():.0f}%)")
        print("   -> two clearly different groups of buses (some positive, some negative) would mean the field is used in different ways")
        print("\n10. BY NUMBER-PLATE SERIES (the letters before the last digits; maybe different devices or bus types)")
        c = c.assign(rho=c["vehicle_id"].map(per["rho"]))
        rows = []
        for ser, g in c.groupby("series"):
            if len(g) < 5000:
                continue
            rows.append((ser, g["vehicle_id"].nunique(), len(g), g["rep"].corr(g["calc"], method="spearman"), 100 * g["sig"].mean(),
                         g.loc[g["calc"] < 2, "rep"].median(), g.loc[g["calc"] > 15, "rep"].median()))
        print("   series      buses    pings   corr    whole-km/h%   number when standing   number when moving fast (>15 km/h)")
        for r in sorted(rows, key=lambda x: -x[2])[:12]:
            print(f"   {r[0]:<10} {r[1]:>6,} {r[2]:>8,}  {r[3]:+.2f}   {r[4]:>9.0f}%   {r[5]:>14.1f}   {r[6]:>22.1f}")
        print("   -> if one series has a positive correlation and a high whole-km/h share and another has a negative one, the field means different things for different series")
        print("\n11. TWO CONTRASTING BUSES (most positive and most negative correlation)")
        for label, vid in (("MOST POSITIVE", per["rho"].idxmax()), ("MOST NEGATIVE", per["rho"].idxmin())):
            g = df[df["vehicle_id"] == vid].iloc[10:24]
            print(f"   {label}: bus {vid} (correlation {per.loc[vid, 'rho']:+.2f})")
            print("      time IST  secs  moved_m  calc_km/h  feed_number  feed_number x 3.6")
            for _, row in g.iterrows():
                clk = pd.to_datetime(row["ts"] + IST, unit="s").strftime("%H:%M:%S")
                print(f"      {clk}  {row['dt_s'] if pd.notna(row['dt_s']) else '-':>4}  {row['moved_m'] if pd.notna(row['moved_m']) else '-':>7}  {row['calc'] if pd.notna(row['calc']) else float('nan'):>9.0f}  {row['rep']:>11.2f}  {row['rep'] * 3.6:>14.1f}")
    print("\nSend this whole output back, so the meaning of the number can be decided from the evidence.")


if __name__ == "__main__":
    main()
