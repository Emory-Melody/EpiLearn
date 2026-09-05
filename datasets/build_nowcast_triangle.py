#!/usr/bin/env python
"""
Build a nowcasting reporting triangle from the CMU Delphi Epidata API.

EpiLearn does not redistribute this dataset, so this script regenerates it from
the original source. It writes the .npz that the nowcasting benchmark configs
(`configs/nowcast_benchmark_config.yaml`, `configs/quick_nowcast_test.yaml`)
expect, and that `NowcastTask.load_triangle()` reads.

SOURCE
------
CMU Delphi Epidata API, COVIDcast endpoint:
    https://api.delphi.cmu.edu/epidata/covidcast/
    docs: https://cmu-delphi.github.io/delphi-epidata/api/covidcast.html

The default signal is `hospital-admissions / smoothed_adj_covid19_from_claims`
(COVID-19 hospital admissions estimated from Change Healthcare insurance
claims). It is one of the few COVIDcast signals that stores a full revision
history, which is what makes a reporting triangle possible: querying with
`issues` returns every version of a value as it was reported over time.

Delphi asks API users to register for an API key for anything beyond light use:
    https://api.delphi.cmu.edu/epidata/admin/registration_form
Pass it with --api-key, or set the DELPHI_API_KEY environment variable.

Please cite Delphi and respect the terms of the underlying data source:
    https://cmu-delphi.github.io/delphi-epidata/api/covidcast_licensing.html

WHAT A REPORTING TRIANGLE IS
----------------------------
Row t is an event day; column d is a reporting delay in days. `triangle[t, d]`
is the value that had been reported for day t as of d days after day t. Early
columns of recent rows are the only ones filled in -- later reports have not
happened yet -- which is exactly the incompleteness a nowcasting model corrects.
`final_counts[t]` is the latest (most revised) value for day t: the target.

USAGE
-----
    # the dataset the shipped configs expect (~1800 days; takes a few minutes)
    python datasets/build_nowcast_triangle.py -o epilearn/data/nowcast_ready_data.npz

    # a small range first, to check connectivity
    python datasets/build_nowcast_triangle.py --start 20220101 --end 20220301 \
        -o /tmp/triangle_small.npz

The output .npz holds `triangle` (n_days, n_delays), `final_counts` (n_days,),
`delays` (n_delays,) and `time_values` (n_days,) -- the four arrays
`NowcastTask.load_triangle()` expects.
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

import numpy as np

BASE_URL = 'https://api.delphi.cmu.edu/epidata/covidcast/'


def _daterange_months(start, end, step_months=3):
    """Yield (start, end) yyyymmdd int pairs covering [start, end] in chunks."""
    cur = datetime.strptime(str(start), '%Y%m%d')
    stop = datetime.strptime(str(end), '%Y%m%d')
    while cur <= stop:
        # ~step_months of days; calendar-exact month math is unnecessary here
        nxt = min(cur + timedelta(days=30 * step_months), stop)
        yield int(cur.strftime('%Y%m%d')), int(nxt.strftime('%Y%m%d'))
        cur = nxt + timedelta(days=1)


def _get(params, api_key=None, timeout=120):
    """One Epidata request. Returns the `epidata` list (possibly empty)."""
    if api_key:
        params = dict(params, api_key=api_key)
    url = BASE_URL + '?' + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode('utf-8'))
    result = payload.get('result')
    if result == -2:                      # documented "no results" code
        return []
    if result != 1:
        raise RuntimeError('Epidata error (result=%s): %s'
                           % (result, payload.get('message')))
    return payload.get('epidata', [])


def fetch_revisions(source, signal, geo_type, geo_value, start, end,
                    api_key=None, verbose=True):
    """Fetch every reported version of every day in [start, end]."""
    rows = []
    for a, b in _daterange_months(start, end):
        # Issues must extend past the event window so late revisions are seen.
        issue_end = (datetime.strptime(str(b), '%Y%m%d')
                     + timedelta(days=120)).strftime('%Y%m%d')
        batch = _get({
            'endpoint': 'covidcast',
            'data_source': source,
            'signal': signal,
            'time_type': 'day',
            'geo_type': geo_type,
            'geo_value': geo_value,
            'time_values': '%d-%d' % (a, b),
            'issues': '%d-%s' % (a, issue_end),
        }, api_key=api_key)
        if verbose:
            print('  %d-%d: %d rows' % (a, b, len(batch)))
        rows.extend(batch)
    return rows


def build_triangle(rows, min_delay, max_delay):
    """Turn revision rows into (triangle, final_counts, delays, time_values)."""
    if not rows:
        raise SystemExit('No data returned -- check the signal name and dates.')

    delays = list(range(min_delay, max_delay + 1))
    delay_pos = {d: i for i, d in enumerate(delays)}
    times = sorted({r['time_value'] for r in rows})
    time_pos = {t: i for i, t in enumerate(times)}

    triangle = np.full((len(times), len(delays)), np.nan, dtype=np.float64)
    latest = {}                      # time_value -> (issue, value)

    for r in rows:
        t, value = r['time_value'], r['value']
        if value is None:
            continue
        # `lag` is the reporting delay in days; fall back to computing it.
        lag = r.get('lag')
        if lag is None:
            d0 = datetime.strptime(str(t), '%Y%m%d')
            d1 = datetime.strptime(str(r['issue']), '%Y%m%d')
            lag = (d1 - d0).days
        if lag in delay_pos:
            triangle[time_pos[t], delay_pos[lag]] = value
        # track the most recently issued value as ground truth
        if t not in latest or r['issue'] > latest[t][0]:
            latest[t] = (r['issue'], value)

    # A triangle is cumulative-in-delay: carry the last known report forward so
    # a missing intermediate issue does not read as "no data yet".
    for i in range(triangle.shape[0]):
        last = np.nan
        for j in range(triangle.shape[1]):
            if np.isnan(triangle[i, j]):
                triangle[i, j] = last
            else:
                last = triangle[i, j]

    final_counts = np.array([latest.get(t, (None, np.nan))[1] for t in times],
                            dtype=np.float64)

    # Drop rows with no reports at all, or no ground truth.
    keep = ~(np.isnan(triangle).all(axis=1) | np.isnan(final_counts))
    return (triangle[keep], final_counts[keep],
            np.array(delays), np.array(times)[keep])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-o', '--output', default='epilearn/data/nowcast_ready_data.npz',
                   help='where to write the .npz (default: %(default)s)')
    p.add_argument('--source', default='hospital-admissions')
    p.add_argument('--signal', default='smoothed_adj_covid19_from_claims')
    p.add_argument('--geo-type', default='nation')
    p.add_argument('--geo-value', default='us')
    p.add_argument('--start', type=int, default=20200601,
                   help='first event day, yyyymmdd (default: %(default)s)')
    p.add_argument('--end', type=int, default=None,
                   help='last event day, yyyymmdd (default: 60 days ago, so the '
                        'most recent rows have a full revision history)')
    p.add_argument('--min-delay', type=int, default=3)
    p.add_argument('--max-delay', type=int, default=30)
    p.add_argument('--api-key', default=os.environ.get('DELPHI_API_KEY'))
    args = p.parse_args(argv)

    end = args.end or int((datetime.now() - timedelta(days=60)).strftime('%Y%m%d'))

    print('Delphi Epidata: %s / %s (%s=%s), days %d-%d, delays %d-%d'
          % (args.source, args.signal, args.geo_type, args.geo_value,
             args.start, end, args.min_delay, args.max_delay))
    if not args.api_key:
        print('No API key given (--api-key / DELPHI_API_KEY). Fine for small '
              'ranges; register for one if you get rate-limited:\n'
              '  https://api.delphi.cmu.edu/epidata/admin/registration_form')

    rows = fetch_revisions(args.source, args.signal, args.geo_type,
                           args.geo_value, args.start, end, args.api_key)
    triangle, final_counts, delays, time_values = build_triangle(
        rows, args.min_delay, args.max_delay)

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, triangle=triangle, final_counts=final_counts,
             delays=delays, time_values=time_values)

    observed = 100.0 * np.mean(~np.isnan(triangle))
    print('\ntriangle %s, %.1f%% observed, %d days %d-%d'
          % (triangle.shape, observed, len(time_values),
             time_values[0], time_values[-1]))
    print('wrote %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
