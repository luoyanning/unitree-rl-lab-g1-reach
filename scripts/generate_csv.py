import csv
import math
from pathlib import Path
from collections import defaultdict

OUT_DIR = Path("/Users/luoyanning/Desktop/videos/benchmark/grid_fullfield_p1_00")
EPISODE_CSV = OUT_DIR / "episode_records.csv"
BY_RING_CSV = OUT_DIR / "by_ring_summary.csv"
BY_SECTOR_CSV = OUT_DIR / "by_sector_45deg_summary.csv"

rows = list(csv.DictReader(EPISODE_CSV.open(encoding="utf-8")))

def as_bool(v):
    return str(v).strip().lower() in {"1", "true", "yes"}

def as_float(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else math.nan
    except Exception:
        return math.nan

for row in rows:
    row["distance_m"] = as_float(row["distance_m"])
    row["angle_deg"] = as_float(row["angle_deg"])
    row["reach_20cm"] = as_bool(row["reach_20cm"])
    row["stop_20cm"] = as_bool(row["stop_20cm"])
    row["fall"] = as_bool(row["fall"])
    row["final_error_m"] = as_float(row["final_error_m"])
    row["final_stop_error_m"] = as_float(row["final_stop_error_m"])
    row["path_efficiency"] = as_float(row["path_efficiency"])
    row["time_to_stop_20cm_s"] = as_float(row["time_to_stop_20cm_s"])

def sector_45deg(angle_deg):
    a = ((float(angle_deg) + 180.0) % 360.0) - 180.0
    bins = [
        (-180.0, -135.0, "[-180,-135)"),
        (-135.0,  -90.0, "[-135,-90)"),
        ( -90.0,  -45.0,  "[-90,-45)"),
        ( -45.0,    0.0,    "[-45,0)"),
        (   0.0,   45.0,     "[0,45)"),
        (  45.0,   90.0,    "[45,90)"),
        (  90.0,  135.0,   "[90,135)"),
        ( 135.0,  180.0,  "[135,180)"),
    ]
    for lo, hi, label in bins:
        if lo <= a < hi:
            return label
    return "[135,180)"

def mean(vals):
    vals = [v for v in vals if isinstance(v, float) and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan

def summarize(group_rows):
    n = max(len(group_rows), 1)
    return {
        "reach@20cm": sum(r["reach_20cm"] for r in group_rows) / n,
        "stop@20cm": sum(r["stop_20cm"] for r in group_rows) / n,
        "fall_rate": sum(r["fall"] for r in group_rows) / n,
        "final_error_m_mean": mean([r["final_error_m"] for r in group_rows]),
        "final_stop_error_m_mean": mean([r["final_stop_error_m"] for r in group_rows]),
        "path_efficiency_mean": mean([r["path_efficiency"] for r in group_rows]),
        "time_to_stop_20cm_s_mean": mean([r["time_to_stop_20cm_s"] for r in group_rows]),
    }

def fmt(v):
    if isinstance(v, float):
        return "" if not math.isfinite(v) else f"{v:.4f}"
    return v

ring_groups = defaultdict(list)
sector_groups = defaultdict(list)

for row in rows:
    ring_groups[f"{row['distance_m']:.2f}m"].append(row)
    sector_groups[sector_45deg(row["angle_deg"])].append(row)

ring_order = ["ALL"] + sorted(ring_groups.keys(), key=lambda x: float(x[:-1]))
sector_order = [
    "ALL",
    "[-180,-135)",
    "[-135,-90)",
    "[-90,-45)",
    "[-45,0)",
    "[0,45)",
    "[45,90)",
    "[90,135)",
    "[135,180)",
]

fieldnames = [
    "key",
    "reach@20cm",
    "stop@20cm",
    "fall_rate",
    "final_error_m_mean",
    "final_stop_error_m_mean",
    "path_efficiency_mean",
    "time_to_stop_20cm_s_mean",
]

with BY_RING_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for key in ring_order:
        group_rows = rows if key == "ALL" else ring_groups[key]
        row = {"key": key}
        row.update(summarize(group_rows))
        writer.writerow({k: fmt(v) for k, v in row.items()})

with BY_SECTOR_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for key in sector_order:
        group_rows = rows if key == "ALL" else sector_groups[key]
        row = {"key": key}
        row.update(summarize(group_rows))
        writer.writerow({k: fmt(v) for k, v in row.items()})

print(BY_RING_CSV)
print(BY_SECTOR_CSV)
