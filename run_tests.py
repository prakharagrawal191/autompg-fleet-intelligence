import subprocess
import sys

for test in ['test_ui.py', 'test_spotcheck.py']:
    r = subprocess.run(['python', test], capture_output=True, text=True)
    fails = [l for l in r.stdout.splitlines() if 'FAIL' in l]
    print(f'===== {test} =====')
    print(f'EXIT: {r.returncode}')
    print(f'ALL_PASSED: {"ALL CHECKS PASSED" in r.stdout or "ALL SPOT CHECKS PASSED" in r.stdout}')
    print(f'FAIL_COUNT: {len(fails)}')
    for f in fails:
        print(f'  FAIL: {f}')
    print()
