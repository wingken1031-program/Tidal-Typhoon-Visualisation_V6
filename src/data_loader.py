from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
from datetime import datetime
from contextlib import closing

ROOT = Path(__file__).resolve().parents[1]
TIDES_CSV = ROOT / '2022-2024_Hong_Kong_Tidal_Report.csv'
TYPHOON_CSV = ROOT / 'Typhoon_data.csv'


def _index_cols(header: List[str]) -> Dict[str, int]:
    return {h.lower().strip(): i for i, h in enumerate(header)}


def _find(header: Dict[str, int], candidates: List[str]) -> Optional[int]:
    # First try exact matches
    for c in candidates:
        i = header.get(c.lower())
        if i is not None:
            return i
    # Fallback: substring matching to tolerate units like "Height (m)"
    for c in candidates:
        cl = c.lower()
        for k, i in header.items():
            if cl in k:
                return i
    return None


def _choose_encoding(path: Path) -> str:
    encs = ['utf-8-sig', 'utf-8', 'cp950', 'big5', 'big5hkscs', 'cp936', 'cp1252', 'latin-1']
    for enc in encs:
        try:
            with open(path, 'r', encoding=enc, newline='') as f:
                reader = csv.reader(f)
                _ = next(reader)  # try to read header
            return enc
        except Exception:
            continue
    # Final fallback: permissive
    return 'latin-1'


def load_tides() -> Dict[str, List]:
    enc = _choose_encoding(TIDES_CSV)
    with open(TIDES_CSV, 'r', encoding=enc, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = _index_cols(header)

        # Identify columns (flexible names tolerated)
        dt_idx = _find(idx, ['datetime', 'date_time', 'date/time', 'date time'])
        date_idx = _find(idx, ['date'])
        time_idx = _find(idx, ['time'])
        h_idx = _find(idx, ['height', 'height(m)', 'tide', 'tide_level', 'sea_level', 'water level'])
        if h_idx is None or (dt_idx is None and (date_idx is None or time_idx is None)):
            raise ValueError('Could not infer tide time/level columns in tides CSV')

        times: List[datetime] = []
        heights: List[float] = []
        for row in reader:
            try:
                if dt_idx is not None:
                    ts_str = row[dt_idx]
                else:
                    ts_str = f"{row[date_idx]} {row[time_idx]}"
                # Remove common timezone tokens if present
                for tok in ('HKT', 'UTC', '(HKT)', '(UTC)'):
                    ts_str = ts_str.replace(tok, '')
                ts = None
                for fmt in (
                    '%d/%m/%Y %H:%M', '%m/%d/%Y %H:%M', '%d/%m/%Y %H:%M:%S',
                    '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S'
                ):
                    try:
                        ts = datetime.strptime(ts_str.strip(), fmt)
                        break
                    except Exception:
                        pass
                if ts is None:
                    continue
                h_raw = row[h_idx]
                h_raw = h_raw.replace(',', '').strip() if isinstance(h_raw, str) else h_raw
                h = float(h_raw) if h_raw not in (None, '') else None
                if h is None:
                    continue
                times.append(ts)
                heights.append(h)
            except Exception:
                continue

    # Sort by time
    order = sorted(range(len(times)), key=lambda i: times[i])
    times = [times[i] for i in order]
    heights = [heights[i] for i in order]
    return {'time': times, 'tide': heights}


def load_typhoons() -> Dict[str, List[Dict]]:
    enc = _choose_encoding(TYPHOON_CSV)
    with open(TYPHOON_CSV, 'r', encoding=enc, newline='') as f:
        reader = csv.reader(f)
        # Read up to 10 lines to find a valid header containing required fields
        header = None
        for _ in range(10):
            try:
                cand = next(reader)
            except StopIteration:
                break
            idx_cand = _index_cols(cand)
            if _find(idx_cand, ['tropical cyclone name', 'name', 'typhoon']) is not None and \
               _find(idx_cand, ['year']) is not None and \
               _find(idx_cand, ['month']) is not None and \
               _find(idx_cand, ['day']) is not None and \
               _find(idx_cand, ['time (utc)', 'hour', 'time']) is not None:
                header = cand
                idx = idx_cand
                break
        if header is None:
            raise ValueError('Could not locate header row in Typhoon CSV')

        name_i = _find(idx, ['tropical cyclone name', 'name', 'typhoon'])
        year_i = _find(idx, ['year'])
        month_i = _find(idx, ['month'])
        day_i = _find(idx, ['day'])
        hour_i = _find(idx, ['time (utc)', 'hour', 'time'])
        press_i = _find(idx, ['estimated minimum central pressure (hpa)', 'pressure', 'central pressure'])
        wind_i = _find(idx, ['estimated maximum surface winds (knot)', 'wind', 'max wind'])
        lat_i = _find(idx, ['latitude (0.01 degree n)', 'lat', 'latitude'])
        lon_i = _find(idx, ['longitude (0.01 degree e)', 'lon', 'longitude'])

        rows: List[Dict] = []
        for row in reader:
            try:
                name = row[name_i].strip()
                if not name:
                    continue
                y = int(float(row[year_i]))
                m = int(float(row[month_i]))
                d = int(float(row[day_i]))
                hour_val = row[hour_i].strip()
                # Remove potential 'Z' suffix or ':00'
                hour_val = hour_val.replace('Z', '').replace(':00', '').strip()
                hour = int(float(hour_val))
                ts = datetime(y, m, d, hour)
                wind = float(row[wind_i]) if (wind_i is not None and row[wind_i] not in ('', None)) else None
                press = float(row[press_i]) if (press_i is not None and row[press_i] not in ('', None)) else None
                lat = float(row[lat_i])/100.0 if (lat_i is not None and row[lat_i] not in ('', None)) else None
                lon = float(row[lon_i])/100.0 if (lon_i is not None and row[lon_i] not in ('', None)) else None
                rows.append({'name': name, 'time': ts, 'wind': wind, 'pressure': press, 'lat': lat, 'lon': lon})
            except Exception:
                continue

    # Compute intensity proxy across all rows
    winds = [r['wind'] for r in rows if r['wind'] is not None]
    presses = [r['pressure'] for r in rows if r['pressure'] is not None]
    w_min, w_max = (min(winds), max(winds)) if winds else (0.0, 1.0)
    p_min, p_max = (min(presses), max(presses)) if presses else (0.0, 1.0)

    def norm_w(w: Optional[float]) -> float:
        if w is None or w_max == w_min:
            return 0.0
        return (w - w_min) / (w_max - w_min)

    def norm_p(p: Optional[float]) -> float:
        if p is None or p_max == p_min:
            return 0.0
        return (p_max - p) / (p_max - p_min)

    for r in rows:
        iw = norm_w(r['wind'])
        ip = norm_p(r['pressure'])
        r['intensity'] = 0.6*iw + 0.4*ip

    # Group by name and sort by time
    by_name: Dict[str, List[Dict]] = {}
    for r in rows:
        by_name.setdefault(r['name'], []).append(r)
    for name, lst in by_name.items():
        lst.sort(key=lambda x: x['time'])
    return by_name


def tide_stats(tides: Dict[str, List]) -> Tuple[float, float]:
    if not tides['tide']:
        return 0.0, 1.0
    return float(min(tides['tide'])), float(max(tides['tide']))

