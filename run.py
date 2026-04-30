import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"
RESET  = "\033[0m"


def stream(proc, prefix, colour):
    for line in iter(proc.stdout.readline, b""):
        text = line.decode("utf-8", errors="replace").rstrip()
        if text:
            print(f"{colour}[{prefix}]{RESET} {text}", flush=True)


def main(model, fps, conf, monitor, gsi_port):
    if not Path(model).exists():
        print(f"{RED}[run] Model not found: {model}{RESET}")
        print("      Pass --model <path> or put your .pt file there.")
        sys.exit(1)

    print(f"{GREEN}[run] Starting — GSI on :{gsi_port} | YOLO: {model}{RESET}")

    gsi_cmd = [sys.executable, "recording/gsi_recorder.py", "--port", str(gsi_port)]
    cap_cmd = [
        sys.executable, "recording/capture.py",
        "--model",   model,
        "--fps",     str(fps),
        "--conf",    str(conf),
        "--monitor", str(monitor),
    ]

    procs = []

    try:
        gsi_proc = subprocess.Popen(gsi_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        procs.append(gsi_proc)
        threading.Thread(target=stream, args=(gsi_proc, "GSI", CYAN), daemon=True).start()

        time.sleep(1.5)

        cap_proc = subprocess.Popen(cap_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        procs.append(cap_proc)
        threading.Thread(target=stream, args=(cap_proc, "CAP", YELLOW), daemon=True).start()

        print(f"{GREEN}[run] Both running. Open CS2 and play! Ctrl+C to stop.{RESET}\n")

        while True:
            time.sleep(2)
            for proc, name in zip(procs, ["gsi_recorder", "capture"]):
                if proc.poll() is not None:
                    print(f"{RED}[run] {name} crashed (code {proc.returncode}){RESET}")
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print(f"\n{GREEN}[run] Shutting down...{RESET}")
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print(f"{GREEN}[run] Done. Data saved to source/gsi/ and source/yolo/{RESET}")
        print(f"      Next: python prepare.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default="models/csgo_yolo.pt")
    p.add_argument("--fps",      type=int,   default=30)
    p.add_argument("--conf",     type=float, default=0.40)
    p.add_argument("--monitor",  type=int,   default=1)
    p.add_argument("--gsi-port", type=int,   default=3000)
    args = p.parse_args()
    main(args.model, args.fps, args.conf, args.monitor, args.gsi_port)
