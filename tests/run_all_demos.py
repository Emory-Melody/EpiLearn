#!/usr/bin/env python
"""
Run every demo script and report a pass/fail table.

This is the one command that answers "does everything in here still work?".
Each demo runs in its own interpreter with the repository root as the working
directory -- exactly how a user would run it.

Usage:
    python tests/run_all_demos.py             # every demo (~3 min on CPU)
    python tests/run_all_demos.py --list      # show what would run
    python tests/run_all_demos.py cola epi    # only demos matching these names
    python tests/run_all_demos.py --readme    # also run test_readme_examples.py
    python tests/run_all_demos.py --examples  # also run every examples/*.py (~10 min)
    python tests/run_all_demos.py --fit       # also run model_fit_tests (~3 min more)
    python tests/run_all_demos.py --all       # everything

Exits 0 only if every selected demo exits 0.

Environment:
    EPILEARN_DEMO_TIMEOUT   per-demo timeout in seconds (default: 900)
"""
import os
import subprocess
import sys
import time

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)

# Scripts in tests/ that are harnesses rather than demos.
NOT_A_DEMO = {'run_all_demos.py', 'test_readme_examples.py'}


def discover():
    """Return demo script names, task demos first, then per-model demos."""
    names = sorted(f for f in os.listdir(TESTS_DIR)
                   if f.endswith('.py') and f not in NOT_A_DEMO)
    # Task-level demos read better first; the per-model ones are all `test_*`.
    task_level = [n for n in names if not n.startswith('test_')]
    per_model = [n for n in names if n.startswith('test_')]
    return [os.path.join('tests', n) for n in task_level + per_model]


def discover_examples():
    """Every examples/*.py -- they are runnable demos too, and nothing else runs them."""
    d = os.path.join(REPO_ROOT, 'examples')
    if not os.path.isdir(d):
        return []
    return [os.path.join('examples', f)
            for f in sorted(os.listdir(d)) if f.endswith('.py')]


def run(script, timeout):
    """Run one script from the repo root. Returns (ok, seconds, tail)."""
    env = dict(os.environ)
    env['PYTHONPATH'] = REPO_ROOT + os.pathsep + env.get('PYTHONPATH', '')
    env['MPLBACKEND'] = 'Agg'                 # never try to open a window
    env.setdefault('PYTHONWARNINGS', 'ignore')

    started = time.time()
    try:
        proc = subprocess.run([sys.executable, '-u', script],
                              cwd=REPO_ROOT,   # ./datasets must resolve
                              env=env,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, time.time() - started, 'TIMEOUT after %ss' % timeout
    out = proc.stdout.decode('utf-8', 'replace')
    tail = '\n'.join(out.strip().splitlines()[-3:])
    return proc.returncode == 0, time.time() - started, tail


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    listing = '--list' in argv
    with_all = '--all' in argv
    with_readme = '--readme' in argv or with_all
    with_fit = '--fit' in argv or with_all
    with_examples = '--examples' in argv or with_all
    for flag in ('--list', '--readme', '--fit', '--examples', '--all'):
        while flag in argv:
            argv.remove(flag)
    patterns = [a.lower() for a in argv if not a.startswith('-')]
    timeout = int(os.environ.get('EPILEARN_DEMO_TIMEOUT', 900))

    scripts = discover()
    if with_readme:
        scripts.append(os.path.join('tests', 'test_readme_examples.py'))
    if with_examples:
        scripts.extend(discover_examples())
    selected = [s for s in scripts
                if not patterns or any(p in s.lower() for p in patterns)]

    if listing:
        for s in selected:
            print('  %s' % s)
        return 0
    if not selected:
        print('no demo matched %s' % patterns)
        return 1

    print('=' * 66)
    print('EpiLearn demos  (%d scripts, cwd=%s)' % (len(selected), REPO_ROOT))
    print('=' * 66)

    results = []
    for script in selected:
        sys.stdout.write('  %-26s ' % os.path.basename(script))
        sys.stdout.flush()
        ok, secs, tail = run(script, timeout)
        print('%s  %5.1fs' % ('PASS' if ok else 'FAIL', secs))
        if not ok:
            for line in tail.splitlines():
                print('        | %s' % line)
        results.append((script, ok, secs))

    if with_fit:
        sys.stdout.write('  %-26s ' % 'model_fit_tests/')
        sys.stdout.flush()
        ok, secs, tail = run(
            os.path.join('tests', 'model_fit_tests', 'run_all_tests.py'), timeout * 2)
        print('%s  %5.1fs' % ('PASS' if ok else 'FAIL', secs))
        results.append(('model_fit_tests', ok, secs))

    failed = [s for s, ok, _ in results if not ok]
    total = sum(secs for _, _, secs in results)
    print('-' * 66)
    print('%d/%d passed in %.0fs' % (len(results) - len(failed), len(results), total))
    if failed:
        print('FAILED: %s' % ', '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
