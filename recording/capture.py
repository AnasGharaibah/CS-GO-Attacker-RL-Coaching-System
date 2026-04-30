import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests

YOLO_MODEL_PATH = "models/csgo_yolo.pt"
TARGET_FPS      = 30
CONF_THRESHOLD  = 0.40
GSI_URL         = "http://127.0.0.1:3000"
YOLO_DIR        = Path("source/yolo")
YOLO_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "player_enemy",
    1: "player_ct",
    2: "player_t",
    3: "smoke",
    4: "flash",
    5: "fire",
    6: "bomb",
    7: "weapon",
}


def load_yolo(model_path):
    from ultralytics import YOLO
    model = YOLO(model_path)
    print(f"[capture] YOLO loaded: {model_path}")
    return model


def capture_frame(sct, monitor):
    raw = sct.grab(monitor)
    img = np.frombuffer(raw.bgra, dtype=np.uint8).reshape(raw.height, raw.width, 4)
    return img[:, :, :3]


def run_yolo(model, frame, conf, class_names):
    h, w    = frame.shape[:2]
    results = model(frame, conf=conf, verbose=False)[0]
    dets    = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        dets.append({
            "class":      class_names.get(cls_id, f"class_{cls_id}"),
            "bbox_x":     round(((x1 + x2) / 2) / w, 4),
            "bbox_y":     round(((y1 + y2) / 2) / h, 4),
            "bbox_w":     round((x2 - x1) / w, 4),
            "bbox_h":     round((y2 - y1) / h, 4),
            "confidence": round(float(box.conf[0]), 4),
        })
    return dets


def get_match_id():
    try:
        r = requests.get(f"{GSI_URL}/match_id", timeout=0.5)
        return r.json().get("match_id") if r.ok else None
    except Exception:
        return None


def get_round_phase():
    try:
        r = requests.get(f"{GSI_URL}/round_phase", timeout=0.5)
        return r.json().get("phase", "live") if r.ok else "live"
    except Exception:
        return "live"


def get_player_team():
    try:
        r = requests.get(f"{GSI_URL}/status", timeout=0.5)
        return r.json().get("team", "") if r.ok else ""
    except Exception:
        return ""


def main(model_path, fps, conf, monitor_idx):
    import mss

    model    = load_yolo(model_path)
    sct      = mss.MSS()
    monitor  = sct.monitors[monitor_idx] if monitor_idx < len(sct.monitors) else sct.monitors[0]

    print(f"[capture] {monitor['width']}x{monitor['height']} | waiting for match...")

    interval   = 1.0 / fps
    frame_id   = 0
    match_id   = None
    out_file   = None
    start_time = None

    try:
        while True:
            t0 = time.perf_counter()

            current = get_match_id()
            if current != match_id:
                if out_file and not out_file.closed:
                    out_file.close()
                    print(f"[capture] Closed {match_id}")

                match_id = current
                if match_id:
                    out_file   = open(YOLO_DIR / f"{match_id}.jsonl", "a", encoding="utf-8")
                    frame_id   = 0
                    start_time = time.perf_counter()
                    print(f"[capture] Recording → source/yolo/{match_id}.jsonl")
                else:
                    out_file = None
                    print("[capture] No active match, waiting...")

            if not match_id or out_file is None:
                time.sleep(0.5)
                continue

            if get_round_phase() == "freezetime":
                time.sleep(0.1)
                continue

            team = get_player_team()
            if team == "CT":
                time.sleep(0.5)
                continue

            frame = capture_frame(sct, monitor)
            dets  = run_yolo(model, frame, conf, CLASS_NAMES)
            ts    = time.perf_counter() - start_time

            out_file.write(json.dumps({
                "frame_id":      frame_id,
                "timestamp_sec": round(ts, 4),
                "detections":    dets,
            }) + "\n")
            out_file.flush()

            if frame_id % (fps * 5) == 0:
                enemies = sum(1 for d in dets if "player" in d["class"])
                print(f"[capture] frame {frame_id:>6} | t={ts:6.2f}s | enemies={enemies}")

            frame_id += 1
            wait = interval - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)

    except KeyboardInterrupt:
        print("\n[capture] Stopped.")
    finally:
        if out_file and not out_file.closed:
            out_file.close()
        sct.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default=YOLO_MODEL_PATH)
    p.add_argument("--fps",     type=int,   default=TARGET_FPS)
    p.add_argument("--conf",    type=float, default=CONF_THRESHOLD)
    p.add_argument("--monitor", type=int,   default=1)
    args = p.parse_args()
    main(args.model, args.fps, args.conf, args.monitor)
