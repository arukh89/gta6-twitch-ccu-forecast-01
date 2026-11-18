"""Forecast GTA VI's launch-day Twitch viewers, hour by hour (quantiles).

Model, fitted fresh from each case file's own comparables at run time:

  For every past launch in the case, take its release-day hourly curve and
  express each hour as a log-ratio to that launch's own hour-1 print.  The
  shape of a launch day depends strongly on how big hour 1 already was:
  launches that open huge stay comparatively flat, launches that open small
  grow much more over the day (the slope of the log-ratio on log hour-1
  reaches about -0.9 by the evening hours).  Two fits are averaged.  The
  first fits a least-squares line of log-ratio against log hour-1 over
  the 18 launches whose hour-1 print is closest in log terms to this
  case's hour-1 print.  The second fits a weighted line over all the
  launches, weighting each by how rolling-midnight-like its own day was:
  a launch whose audience peaked near its hour-1 print keeps full
  weight, while one that exploded to many times its hour-1 print fades
  to nothing, since a simultaneous global unlock is not the shape of a
  rolling-midnight launch.  Both fits predict at this case's hour-1,
  clamped to 0.7 log units beyond the fitted launches' range, so an
  unprecedented hour-1 cannot extrapolate wildly.  Spreads come from the
  fits' residuals, and the averaged quantile curves are lightly smoothed
  across neighbouring hours.  Leave-one-out tests over the kit's 37
  launches guided the blend; the README records the numbers.

Only the standard library is used.  No network, no clock, no randomness:
the same case file always yields the same bytes.
"""

import calendar
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]
NEIGHBOURS = 18
SIGMA_FLOOR = 0.15
EXTRAPOLATION_LIMIT = 0.7
SHAPE_FULL = 2.5
SHAPE_ZERO = 4.0
ANALOGUE_SIGMA_FLOOR = 0.10
ANALOGUE_SPREAD_SCALE = 1.05
PERSISTENCE_SIGMA = 0.15
MIN_LAUNCHES = 5
MIN_PER_HOUR = 5


def parse_epoch(text):
    """Seconds since the epoch for an ISO-8601 UTC timestamp (fast path)."""
    clean = text[:-1] if text.endswith("Z") else text
    year = int(clean[0:4])
    month = int(clean[5:7])
    day = int(clean[8:10])
    hour = int(clean[11:13])
    minute = int(clean[14:16])
    second = int(clean[17:19]) if len(clean) > 18 else 0
    return calendar.timegm((year, month, day, hour, minute, second, 0, 0, 0))


def interp_quantile(ordered, tau):
    """Linear-interpolation quantile of an already sorted list."""
    count = len(ordered)
    if count == 1:
        return ordered[0]
    position = tau * (count - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, count - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def launch_curves(comparables):
    """Release-day curves: [(name, {hour: viewers}, log hour-1, peak ratio)].

    A comparable counts only when its release day's hour 1 is present and
    at least half the day's hours are.  The target's own pre-release
    history entry has no release day and drops out here on its own.
    The peak ratio (the day's highest hour over hour 1) measures how
    rolling-midnight-like the launch was: near 1 for a launch that
    opened big and stayed flat, in the hundreds for a launch whose
    audience detonated long after a tiny first hour.
    """
    launches = []
    for entry in comparables or []:
        try:
            day_start = parse_epoch(entry["day_start"])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        curve = {}
        for row in entry.get("hourly") or []:
            try:
                offset = (parse_epoch(row["from"]) - day_start) // 3600
                viewers = row["viewers"]
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            if 0 <= offset < 24 and viewers is not None:
                curve[offset + 1] = max(1.0, float(viewers))
        if 1 in curve and len(curve) >= 12:
            name = entry.get("name") or ""
            peak = max(curve.values()) / curve[1]
            launches.append((str(name), curve, math.log(curve[1]), peak))
    launches.sort(key=lambda item: item[0])
    return launches


def persistence_row(p1, hour):
    """Flat-median fallback: hour 1's print with a widening random-walk spread."""
    spread = PERSISTENCE_SIGMA * math.sqrt(hour - 1)
    return [p1 * math.exp(z * spread) for z in Z_SCORES]


def fit_hour(neighbours, hour, target_log_p1):
    """Blend of empirical and normal quantiles for one hour's log-ratio.

    Returns None when too few neighbours carry this hour.
    """
    points = []
    for _, curve, log_p1, _peak in neighbours:
        if hour in curve:
            points.append((log_p1, math.log(curve[hour]) - log_p1))
    if len(points) < MIN_PER_HOUR:
        return None
    xs = [point[0] for point in points]
    mean_x = sum(xs) / len(xs)
    mean_r = sum(point[1] for point in points) / len(points)
    denom = sum((x - mean_x) ** 2 for x in xs)
    slope = 0.0
    if denom > 1e-9:
        slope = sum((x - mean_x) * (r - mean_r) for x, r in points) / denom
    clamped = min(max(target_log_p1, min(xs) - EXTRAPOLATION_LIMIT),
                  max(xs) + EXTRAPOLATION_LIMIT)
    centre = mean_r + slope * (clamped - mean_x)
    residuals = sorted(r - (mean_r + slope * (x - mean_x)) for x, r in points)
    sigma = max(SIGMA_FLOOR,
                (interp_quantile(residuals, 0.75) -
                 interp_quantile(residuals, 0.25)) / 1.349)
    empirical = [centre + interp_quantile(residuals, tau) for tau in LEVELS]
    normal = [centre + z * sigma for z in Z_SCORES]
    return [0.5 * e + 0.5 * n for e, n in zip(empirical, normal)]


def shape_weight(peak):
    """How much a comparable's shape should count for a rolling-midnight day.

    Full weight when the launch peaked at most SHAPE_FULL times its
    hour-1 print, fading smoothly (cosine) to nothing at SHAPE_ZERO.
    """
    if peak <= SHAPE_FULL:
        return 1.0
    if peak >= SHAPE_ZERO:
        return 0.0
    fraction = (peak - SHAPE_FULL) / (SHAPE_ZERO - SHAPE_FULL)
    return 0.5 * (1.0 + math.cos(math.pi * fraction))


def fit_weighted_hour(launches, hour, target_log_p1):
    """Weighted line over all launches, with a residual-SD spread.

    Returns None when too few launches carry this hour.
    """
    points = []
    for _, curve, log_p1, peak in launches:
        if hour in curve:
            points.append((log_p1, math.log(curve[hour]) - log_p1,
                           shape_weight(peak)))
    if len(points) < MIN_PER_HOUR:
        return None
    weighted = [point for point in points if point[2] > 1e-9]
    if len(weighted) < MIN_PER_HOUR:
        # Too few rolling-midnight-like launches: use the most alike
        # five, unweighted, rather than trusting a near-empty fit.
        weighted = sorted(points, key=lambda point: (-point[2], point[0]))
        weighted = [(x, r, 1.0) for x, r, _ in weighted[:MIN_PER_HOUR]]
    total = sum(point[2] for point in weighted)
    if total <= 1e-9:
        return None
    mean_x = sum(w * x for x, _, w in weighted) / total
    mean_r = sum(w * r for _, r, w in weighted) / total
    denom = sum(w * (x - mean_x) ** 2 for x, _, w in weighted)
    slope = 0.0
    if denom > 1e-9:
        slope = sum(w * (x - mean_x) * (r - mean_r)
                    for x, r, w in weighted) / denom
    xs = [point[0] for point in weighted]
    clamped = min(max(target_log_p1, min(xs) - EXTRAPOLATION_LIMIT),
                  max(xs) + EXTRAPOLATION_LIMIT)
    centre = mean_r + slope * (clamped - mean_x)
    residuals = [r - (mean_r + slope * (x - mean_x))
                 for x, r, _ in weighted]
    mean_res = sum(w * r for r, (_, _, w) in zip(residuals, weighted)) / total
    variance = sum(w * (r - mean_res) ** 2
                   for r, (_, _, w) in zip(residuals, weighted)) / total
    sigma = max(ANALOGUE_SIGMA_FLOOR, math.sqrt(max(variance, 0.0)))
    sigma *= ANALOGUE_SPREAD_SCALE
    return [centre + z * sigma for z in Z_SCORES]


def forecast_log_quantiles(p1, launches):
    """Per-hour log-quantiles of the ratio to hour 1, smoothed across hours.

    Each hour averages the nearest-neighbour fit and the
    rolling-midnight-weighted fit, so neither a single odd neighbour nor
    a single odd analogue can dominate the forecast.
    """
    target_log = math.log(p1)
    neighbours = sorted(launches, key=lambda item: (abs(item[2] - target_log), item[0]))
    neighbours = neighbours[:NEIGHBOURS]
    log_rows = {}
    for hour in range(2, 25):
        local = fit_hour(neighbours, hour, target_log)
        if local is None:
            local = [math.log(max(1e-9, value / p1)) for value in persistence_row(p1, hour)]
        try:
            analogue = fit_weighted_hour(launches, hour, target_log)
        except (ArithmeticError, ValueError, TypeError):
            analogue = None
        if analogue is None:
            analogue = local
        log_rows[hour] = [0.5 * a + 0.5 * b for a, b in zip(local, analogue)]
    smoothed = {}
    for hour in range(2, 25):
        window = [h for h in (hour - 1, hour, hour + 1) if 2 <= h <= 24]
        smoothed[hour] = [sum(log_rows[h][j] for h in window) / len(window)
                          for j in range(len(LEVELS))]
    return smoothed


def round_half_up_2(value):
    """Round to 2 decimals, halves away from zero (values here are positive)."""
    return math.floor(value * 100.0 + 0.5) / 100.0


def finish_row(p1, log_row):
    """Exponentiate, round, clip to the contest's bounds, keep monotone."""
    values = []
    for log_q in log_row:
        rounded = round_half_up_2(p1 * math.exp(log_q))
        values.append(min(1e9, max(0.01, rounded)))
    for j in range(1, len(values)):
        values[j] = max(values[j], values[j - 1])
    return values


def hour_one_print(case):
    """Hour 1's print, with defensive fallbacks that never raise."""
    for row in case.get("known") or []:
        if row.get("hour") == 1 and row.get("viewers") is not None:
            return max(1.0, float(row["viewers"]))
    hourly = case.get("history_hourly") or []
    if hourly:
        last = hourly[-1].get("viewers")
        if last is not None:
            return max(1.0, float(last))
    return 1.0


def persistence_forecast(p1):
    """Whole-case fallback forecast: every hour from hour 1's print."""
    rows = []
    for hour in range(2, 25):
        log_row = [math.log(max(1e-9, value / p1))
                   for value in persistence_row(p1, hour)]
        rows.append({"hour": hour, "q": finish_row(p1, log_row)})
    return {"hours": rows}


def forecast_case(case):
    """Forecast one case, falling back to persistence if fitting is impossible."""
    p1 = hour_one_print(case)
    try:
        launches = launch_curves(case.get("comparables"))
    except (TypeError, AttributeError):
        launches = []
    if len(launches) >= MIN_LAUNCHES:
        try:
            log_rows = forecast_log_quantiles(p1, launches)
            return {"hours": [{"hour": h, "q": finish_row(p1, log_rows[h])}
                              for h in range(2, 25)]}
        except (ArithmeticError, ValueError, TypeError, KeyError):
            pass
    return persistence_forecast(p1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except (OSError, ValueError, TypeError):
            forecast = persistence_forecast(1.0)
        payload = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
