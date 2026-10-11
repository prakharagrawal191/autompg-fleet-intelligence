import re
import subprocess
import sys

for test in ['_audit_science.py', 'test_ui.py', 'test_spotcheck.py', 'test_interaction_repair.py',
             'test_redesign.py']:
    r = subprocess.run([sys.executable, test], capture_output=True, text=True)
    fails = [l for l in r.stdout.splitlines() if l.startswith('FAIL')]
    print(f'===== {test} =====')
    print(f'EXIT: {r.returncode}')
    print(f'ALL_PASSED: {r.returncode == 0 and bool(re.search(r"ALL .*PASSED", r.stdout))}')
    print(f'FAIL_COUNT: {len(fails)}')
    for f in fails:
        print(f'  FAIL: {f}')
    print()
