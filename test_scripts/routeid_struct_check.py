import pandas as pd
r = pd.read_csv("gtfs/routes.txt", dtype=str); t = pd.read_csv("gtfs/trips.txt", dtype=str)
print(r.columns.tolist(), len(r), "route_ids,", r["route_short_name"].nunique(), "short names")
print(r.groupby("route_short_name")["route_id"].nunique().value_counts().head())
print(r.head(6).to_string())
if "direction_id" in t:
    print(t.groupby("route_id")["direction_id"].nunique().value_counts())

# not a main project file, just for my personal test/doubt purpose.