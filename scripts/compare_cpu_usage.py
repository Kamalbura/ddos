#!/usr/bin/env python3
"""
Compare CPU usage between XGBoost and TST pipelines.
- Spawns the chosen app as a subprocess
- Samples its CPU% and memory via psutil for a fixed duration
- Prints a summary

Usage examples:
  python scripts/compare_cpu_usage.py --target xgb --duration 60
  python scripts/compare_cpu_usage.py --target tst --duration 60 --profile ultra

Notes:
- Requires: psutil (pip install psutil)
- By default, synthetic capture is disabled to avoid busy loops. You can enable it by setting ALLOW_SYNTHETIC_CAPTURE=1 if you want to drive more packets.
"""
import argparse
import os
import sys
import time
import subprocess
from pathlib import Path

# Try to import psutil, provide a helpful message if missing
try:
    import psutil
except ImportError:
    print("This script requires 'psutil'. Install it with: pip install psutil")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]

XGB_MAIN = ROOT / 'xgboost_app' / 'main.py'
TST_MAIN = ROOT / 'tst_app' / 'main.py'
PROFILE_FILE = ROOT / 'config' / 'current_profile.txt'


def set_profile(profile: str):
    try:
        PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
        PROFILE_FILE.write_text(profile.strip())
        print(f"✓ Set TST profile to: {profile}")
    except Exception as e:
        print(f"Warning: could not set profile: {e}")


def spawn_target(target: str, profile: str | None):
    env = os.environ.copy()
    # Disable synthetic capture by default
    env.setdefault('ALLOW_SYNTHETIC_CAPTURE', '0')

    # Ensure unbuffered output for clean log streaming
    py = sys.executable or 'python'

    if target == 'xgb':
        cmd = [py, '-u', str(XGB_MAIN)]
        print(f"Launching XGBoost app: {' '.join(cmd)}")
        return subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    elif target == 'tst':
        if profile:
            set_profile(profile)
        cmd = [py, '-u', str(TST_MAIN)]
        print(f"Launching TST app (profile={profile or 'current'}): {' '.join(cmd)}")
        return subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    else:
        raise ValueError("target must be one of: xgb, tst")


def sample_cpu(proc: psutil.Process, seconds: int):
    print(f"Sampling CPU for {seconds}s...")
    cpu_samples = []
    rss_samples = []

    # Prime CPU percent measurement
    proc.cpu_percent(interval=None)

    for _ in range(seconds):
        # Read stdout non-blocking to avoid pipe filling
        try:
            # Drain a few lines if available
            if proc.is_running():
                pass
        except Exception:
            pass

        cpu = proc.cpu_percent(interval=1)  # % over 1s
        try:
            mem = proc.memory_info().rss / (1024 * 1024)
        except Exception:
            mem = 0.0
        cpu_samples.append(cpu)
        rss_samples.append(mem)

    return cpu_samples, rss_samples


def terminate_process(p: subprocess.Popen):
    try:
        if os.name == 'nt':
            p.terminate()
        else:
            p.send_signal(subprocess.signal.SIGINT)
            time.sleep(1)
            if p.poll() is None:
                p.terminate()
            time.sleep(1)
            if p.poll() is None:
                p.kill()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description='Compare CPU usage of XGBoost vs TST pipelines')
    ap.add_argument('--target', choices=['xgb', 'tst'], required=True, help='Which pipeline to run')
    ap.add_argument('--duration', type=int, default=60, help='Sampling duration in seconds (default: 60)')
    ap.add_argument('--profile', type=str, default=None, help='TST profile to set before running (e.g., ultra, heavy, medium, light)')
    args = ap.parse_args()

    # Launch subprocess
    p = spawn_target(args.target, args.profile)

    # Give it time to start
    time.sleep(5)

    try:
        proc = psutil.Process(p.pid)
    except psutil.Error as e:
        print(f"Failed to attach to process: {e}")
        terminate_process(p)
        sys.exit(1)

    # Sample CPU and memory
    try:
        cpu_samples, rss_samples = sample_cpu(proc, args.duration)
    finally:
        terminate_process(p)

    # Summarize
    import statistics as stats
    avg_cpu = stats.mean(cpu_samples) if cpu_samples else 0.0
    peak_cpu = max(cpu_samples) if cpu_samples else 0.0
    min_cpu = min(cpu_samples) if cpu_samples else 0.0
    avg_mem = stats.mean(rss_samples) if rss_samples else 0.0

    print("\n=== RESULTS ===")
    print(f"Target: {args.target.upper()} | Duration: {args.duration}s")
    if args.target == 'tst':
        print(f"TST Profile: {args.profile or PROFILE_FILE.read_text().strip() if PROFILE_FILE.exists() else 'unknown'}")
    print(f"CPU %%  -> avg: {avg_cpu:.1f}  min: {min_cpu:.1f}  max: {peak_cpu:.1f}")
    print(f"Memory -> avg RSS: {avg_mem:.1f} MB")

    # Quick hint for expected ranges
    if args.target == 'xgb':
        print("Expected CPU (XGBoost): ~5-15% on Pi")
    else:
        print("Expected CPU (TST ultra): ~80-95% on Pi")


if __name__ == '__main__':
    main()
