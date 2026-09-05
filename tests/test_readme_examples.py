#!/usr/bin/env python
"""
Regression test for the code on the project homepage.

Every ```python block in README.md is extracted verbatim and executed in a fresh
subprocess whose working directory is the repository root -- i.e. exactly what a
user gets when they copy the snippet out of the README and run it. If an API
changes and the homepage is not updated, this test fails.

The blocks are only rewritten in one way: `epochs=N` is lowered so the suite runs
in ~20 seconds instead of ~10 minutes (the README's forecast example alone is 50
epochs of STGCN). Blocks listed in NO_EPOCH_SCALING are exempt, because they are
already fast and their printed numbers back a claim made in the README.
Pass --full to run every block exactly as published.

Usage:
    python tests/test_readme_examples.py                  # all blocks, epochs=2
    python tests/test_readme_examples.py --full           # verbatim (~10 min, CPU)
    python tests/test_readme_examples.py --epochs 5       # custom epoch budget
    python tests/test_readme_examples.py forecast         # only matching blocks
    python tests/test_readme_examples.py --list           # show what would run

Environment:
    EPILEARN_README_TEST_EPOCHS   default epoch budget (default: 2)
    EPILEARN_README_TEST_TIMEOUT  per-block timeout in seconds (default: 1800)
"""
import os
import re
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README_PATH = os.path.join(REPO_ROOT, 'README.md')

# A block must exist for each of these, and it must contain the marker. This is
# the part that stops the README from silently losing an example (or reverting to
# a 0.0.x API): dropping the forecast snippet, or swapping `rolling_train` back
# for the removed `train_model(dataset=...)`, fails the test instead of reducing
# its coverage.
REQUIRED_MARKERS = {
    'forecast (rolling_train)': 'task.rolling_train(',
    'detection (train_model with explicit splits)': 'task.train_model(train_split=',
    'nowcast (NowcastTask)': 'NowcastTask(',
    'dataset (Dataset, not UniversalDataset)': 'from epilearn.data import Dataset',
}

FENCE_RE = re.compile(r'^```(\w+)?\s*$')
HEADING_RE = re.compile(r'^#{1,4}\s+(.*?)\s*$')
EPOCHS_RE = re.compile(r'\bepochs\s*=\s*\d+')


def slugify(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_') or 'block'


def extract_python_blocks(readme_path=None):
    """Return [(name, line_number, code), ...] for every ```python fence."""
    readme_path = readme_path or README_PATH
    with open(readme_path, encoding='utf-8') as handle:
        lines = handle.read().splitlines()

    blocks = []
    heading = 'top'
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        fence = FENCE_RE.match(line)
        if fence is None:
            found = HEADING_RE.match(line)
            if found:
                heading = found.group(1)
            idx += 1
            continue

        language = (fence.group(1) or '').lower()
        start = idx + 1
        end = start
        while end < len(lines) and not FENCE_RE.match(lines[end]):
            end += 1
        if language == 'python':
            blocks.append((slugify(heading), start + 1, '\n'.join(lines[start:end])))
        idx = end + 1

    # Disambiguate repeated headings.
    seen = {}
    named = []
    for name, lineno, code in blocks:
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = '%s_%d' % (name, seen[name])
        named.append((name, lineno, code))
    return named


def check_required_markers(blocks):
    """Return a list of human-readable problems (empty if the README is intact)."""
    joined = '\n'.join(code for _, _, code in blocks)
    return ['README.md lost its %s example (missing %r)' % (label, marker)
            for label, marker in REQUIRED_MARKERS.items() if marker not in joined]


# Blocks that already run in seconds and whose printed numbers back a claim in the
# README. Lowering their epochs would make the log contradict the documentation --
# the nowcast example is only convincing because it beats the naive baseline.
NO_EPOCH_SCALING = ('nowcast',)


def scale_epochs(name, code, epochs):
    """Lower every `epochs=N` so the homepage examples are testable quickly."""
    if epochs is None or any(tag in name for tag in NO_EPOCH_SCALING):
        return code
    return EPOCHS_RE.sub('epochs=%d' % epochs, code)


def run_block(name, code, workdir, timeout):
    """Execute one block in its own interpreter. Returns (ok, seconds, output)."""
    script = os.path.join(workdir, '%s.py' % name)
    with open(script, 'w', encoding='utf-8') as handle:
        handle.write(code + '\n')

    env = dict(os.environ)
    env['PYTHONPATH'] = REPO_ROOT + os.pathsep + env.get('PYTHONPATH', '')
    env['MPLBACKEND'] = 'Agg'          # never open a window
    env.setdefault('PYTHONWARNINGS', 'ignore')

    started = time.time()
    try:
        proc = subprocess.run([sys.executable, script],
                              cwd=REPO_ROOT,           # ./datasets must resolve
                              env=env,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, time.time() - started, 'TIMEOUT after %ss' % timeout
    output = proc.stdout.decode('utf-8', 'replace')
    return proc.returncode == 0, time.time() - started, output


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    full = '--full' in argv
    if full:
        argv.remove('--full')
    listing = '--list' in argv
    if listing:
        argv.remove('--list')

    epochs = int(os.environ.get('EPILEARN_README_TEST_EPOCHS', 2))
    if '--epochs' in argv:
        pos = argv.index('--epochs')
        epochs = int(argv[pos + 1])
        del argv[pos:pos + 2]
    if full:
        epochs = None
    timeout = int(os.environ.get('EPILEARN_README_TEST_TIMEOUT', 1800))
    patterns = [arg.lower() for arg in argv if not arg.startswith('-')]

    blocks = extract_python_blocks()
    print('=' * 78)
    print('README EXAMPLES: %s' % README_PATH)
    print('found %d python block(s); epochs=%s' % (len(blocks), epochs or 'as published'))
    print('=' * 78)

    problems = check_required_markers(blocks)
    for problem in problems:
        print('  FAIL  %s' % problem)

    selected = [b for b in blocks
                if not patterns or any(p in b[0] for p in patterns)]
    if listing:
        for name, lineno, _ in selected:
            print('  %-28s README.md:%d' % (name, lineno))
        return 1 if problems else 0

    if not selected:
        print('  FAIL  no python block matched %s' % patterns)
        return 1

    results = []
    workdir = tempfile.mkdtemp(prefix='epilearn_readme_')
    for name, lineno, code in selected:
        print('\n' + '-' * 78)
        print('%s   (README.md:%d)' % (name, lineno))
        print('-' * 78)
        ok, elapsed, output = run_block(name, scale_epochs(name, code, epochs), workdir, timeout)
        tail = output.strip().splitlines()[-15:]
        for line in tail:
            print('  | %s' % line)
        print('  %s  (%.1fs)' % ('PASS' if ok else 'FAIL', elapsed))
        results.append((name, ok, elapsed))

    print('\n' + '=' * 78)
    print('SUMMARY')
    print('=' * 78)
    for name, ok, elapsed in results:
        print('  %-6s %-28s %.1fs' % ('PASS' if ok else 'FAIL', name, elapsed))
    failed = [name for name, ok, _ in results if not ok] + problems
    passed = sum(1 for _, ok, _ in results if ok)
    print('\n  %d/%d blocks ran clean' % (passed, len(results)))
    if failed:
        print('  FAILED: %s' % ', '.join(failed))
    print('  scratch scripts kept in %s' % workdir)
    return 1 if failed else 0


def test_readme_examples():
    """pytest entry point (pytest is optional; the script runs standalone too)."""
    assert main([]) == 0


if __name__ == '__main__':
    sys.exit(main())
