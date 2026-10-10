import subprocess
import time
import urllib.request
import sys

print("Starting Streamlit app...", flush=True)
log_out = open("streamlit_out.log", "w")
log_err = open("streamlit_err.log", "w")
p = subprocess.Popen(
    ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"],
    stdout=log_out,
    stderr=log_err,
)
print(f"Streamlit PID: {p.pid}", flush=True)
time.sleep(10)
try:
    r = urllib.request.urlopen("http://localhost:8501/_stcore/health", timeout=5)
    print(f"APP HEALTH: HTTP {r.status}", flush=True)
    body = r.read().decode()
    print(f"HEALTH BODY: {body}", flush=True)
except Exception as e:
    print(f"APP HEALTH ERROR: {e}", flush=True)
    sys.stdout.flush()
time.sleep(2)
p.terminate()
p.wait()
print("Streamlit stopped.", flush=True)
