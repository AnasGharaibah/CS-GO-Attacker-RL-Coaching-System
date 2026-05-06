import argparse
import sys
import time
import threading
import queue
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from rl.observation import encode_state
from rl.actions import Action, get_action_mask

YOLO_MODEL_PATH = "yolo_model/best.pt"
GSI_PORT        = 3000
DEFAULT_MODEL   = "rl/models/attacker_ppo_final"
CONF_THRESHOLD  = 0.35
TARGET_FPS      = 10

ACTION_META = {
    Action.MOVE_FORWARD:  ("MOVE FORWARD",  "#00d4ff", "Move toward the objective"),
    Action.MOVE_BACK:     ("MOVE BACK",     "#6b7280", "Retreat"),
    Action.STRAFE_LEFT:   ("STRAFE LEFT",   "#00d4ff", "Side-step left"),
    Action.STRAFE_RIGHT:  ("STRAFE RIGHT",  "#00d4ff", "Side-step right"),
    Action.CROUCH_HOLD:   ("CROUCH HOLD",   "#fbbf24", "Hold a low angle"),
    Action.PEEK_CORNER:   ("PEEK CORNER",   "#fbbf24", "Peek around corner"),
    Action.ENGAGE_ENEMY:  ("ENGAGE ENEMY",  "#ff4466", "Shoot the visible enemy"),
    Action.REPOSITION:    ("REPOSITION",    "#c084fc", "Change position"),
    Action.THROW_SMOKE:   ("THROW SMOKE",   "#6b7280", "Deploy smoke grenade"),
    Action.THROW_FLASH:   ("THROW FLASH",   "#fbbf24", "Throw flashbang"),
    Action.THROW_HE:      ("THROW HE",      "#ff8c00", "Throw HE grenade"),
    Action.PLANT_BOMB:    ("PLANT BOMB",    "#00ff88", "Plant the bomb NOW"),
    Action.RUSH_SITE:     ("RUSH SITE",     "#ff4466", "Fast push the bombsite"),
    Action.HOLD_ANGLE:    ("HOLD ANGLE",    "#fbbf24", "Hold a defensive angle"),
}

_WEAPON_MAP = {
    "ak47": "rifle", "m4a1": "rifle", "m4a1_silencer": "rifle", "m4a4": "rifle",
    "sg553": "rifle", "aug": "rifle", "famas": "rifle", "galil": "rifle",
    "awp": "sniper", "ssg08": "sniper", "g3sg1": "sniper", "scar20": "sniper",
    "glock": "pistol", "usp_silencer": "pistol", "p250": "pistol",
    "deagle": "pistol", "cz75": "pistol", "tec9": "pistol", "fiveseven": "pistol",
    "dualberettas": "pistol", "revolver": "pistol",
    "knife": "knife",
}


class GSIServer:
    """Minimal GSI receiver that runs inside inference.py — no separate server needed."""

    def __init__(self, port=GSI_PORT):
        self._port   = port
        self._player = {}
        self._map    = {}
        self._bomb   = {}
        self._last   = 0.0
        self._lock   = threading.Lock()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import json as _json

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # silence request logs

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body   = self.rfile.read(length)
                try:
                    payload = _json.loads(body)
                    with outer._lock:
                        outer._player = payload.get("player", {})
                        outer._map    = payload.get("map",    {})
                        outer._bomb   = payload.get("bomb",   {})
                        outer._last   = time.time()
                except Exception:
                    pass
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')

        try:
            HTTPServer(("0.0.0.0", self._port), Handler).serve_forever()
        except OSError as e:
            print(f"[gsi] Could not bind port {self._port}: {e}")

    def fetch(self):
        with self._lock:
            return {"player": self._player, "map": self._map, "bomb": self._bomb}

    @property
    def connected(self):
        return (time.time() - self._last) < 10.0


class YOLODetector:
    def __init__(self, model_path, conf=CONF_THRESHOLD):
        self.conf   = conf
        self._model = None
        if Path(model_path).exists():
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            print(f"[inference] YOLO loaded: {model_path}")
        else:
            print(f"[inference] YOLO weights not found at {model_path!r} — vision disabled")

    def detect(self):
        import mss
        import cv2
        with mss.MSS() as sct:
            img = np.array(sct.grab(sct.monitors[1]))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if self._model is None:
            return []

        results = self._model(frame, conf=self.conf, verbose=False)[0]
        h, w    = frame.shape[:2]
        dets    = []

        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if cls != 0:  # only enemy class
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append({
                "class":      "enemy",
                "confidence": conf,
                "bbox_x":     (x1 + x2) / 2 / w,
                "bbox_y":     (y1 + y2) / 2 / h,
                "bbox_w":     (x2 - x1) / w,
                "bbox_h":     (y2 - y1) / h,
            })
        return dets


def _weapon_cat(name):
    return _WEAPON_MAP.get(name.lower().replace("weapon_", ""), "rifle")


def _utility_from_weapons(weapons):
    for w in weapons:
        clean = w.lower().replace("weapon_", "")
        if clean == "smokegrenade":  return "smoke"
        if clean == "flashbang":     return "flash"
        if clean == "hegrenade":     return "he"
        if clean in ("molotov", "incgrenade"): return "molotov"
    return "none"


def build_state(gsi, detections, frame_id):
    player_gsi = gsi.get("player", {})
    map_gsi    = gsi.get("map",    {})
    bomb_gsi   = gsi.get("bomb",   {})

    weapons      = player_gsi.get("weapons", {})
    active_name  = next(
        (w for w, info in weapons.items() if isinstance(info, dict) and info.get("state") == "active"), ""
    )

    player = {
        "role":         "attacker",
        "position_x":  player_gsi.get("position", {}).get("x", 0) / 4096,
        "position_y":  player_gsi.get("position", {}).get("y", 0) / 4096,
        "crosshair_x": player_gsi.get("forward",  {}).get("x", 0),
        "crosshair_y": player_gsi.get("forward",  {}).get("y", 0),
        "health":      player_gsi.get("state", {}).get("health", 100),
        "armor":       player_gsi.get("state", {}).get("armor",  100),
        "alive":       player_gsi.get("state", {}).get("health", 100) > 0,
        "weapon":      _weapon_cat(active_name),
        "utility_held":_utility_from_weapons(weapons),
    }

    enemies = [
        {
            "confidence":  d["confidence"],
            "bbox_x":      d["bbox_x"],
            "bbox_y":      d["bbox_y"],
            "bbox_w":      d.get("bbox_w", 0),
            "bbox_h":      d.get("bbox_h", 0),
            "distance_est":1.0 - d["confidence"],
            "is_visible":  True,
        }
        for d in detections
    ]

    b_state   = bomb_gsi.get("state", "")
    b_planted = b_state in ("planted", "defusing")
    b_site    = bomb_gsi.get("position", "unknown")
    if b_site not in ("A", "B"):
        b_site = "unknown"

    return {
        "frame_id":       frame_id,
        "timestamp_sec":  time.time() % 60.0,
        "round_number":   map_gsi.get("round", 1),
        "map_name":       map_gsi.get("name", "de_dust2"),
        "player":         player,
        "enemies":        enemies,
        "bomb": {
            "detected":        bool(detections) or b_planted,
            "site":            b_site,
            "planted":         b_planted,
            "timer_remaining": bomb_gsi.get("countdown", -1),
        },
        "utility_active": {"smoke_count": 0, "flash_active": False, "fire_active": False},
        "game_events": {
            "kill_this_frame":   False,
            "death_this_frame":  not player["alive"],
            "assist_this_frame": False,
            "round_won":  map_gsi.get("win_team") == player_gsi.get("team"),
            "round_lost": map_gsi.get("win_team") not in (None, "", player_gsi.get("team")),
        },
    }


class HUD:
    W, H = 320, 190

    def __init__(self, q):
        self._q = q
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        import tkinter as tk
        root = tk.Tk()
        root.title("RL HUD")
        root.geometry(f"{self.W}x{self.H}+20+20")
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.88)
        root.configure(bg="#0f1117")
        root.resizable(False, False)
        root.overrideredirect(True)

        self._header = tk.Label(root, text="○ GSI", font=("Courier", 9, "bold"),
                                fg="#6b7280", bg="#0f1117")
        self._header.pack(pady=(6, 0))

        self._action = tk.Label(root, text="—", font=("Courier", 18, "bold"),
                                fg="#00d4ff", bg="#0f1117", wraplength=self.W - 20)
        self._action.pack(pady=4)

        self._desc = tk.Label(root, text="", font=("Courier", 9),
                              fg="#94a3b8", bg="#0f1117", wraplength=self.W - 20)
        self._desc.pack()

        self._info = tk.Label(root, text="", font=("Courier", 8), fg="#6b7280", bg="#0f1117")
        self._info.pack(pady=(2, 0))

        self._status = tk.Label(root, text="connecting...", font=("Courier", 8),
                                fg="#6b7280", bg="#0f1117")
        self._status.pack(pady=(4, 0))

        root.bind("<ButtonPress-1>",   lambda e: setattr(self, "_dx", e.x) or setattr(self, "_dy", e.y))
        root.bind("<B1-Motion>",       lambda e: root.geometry(f"+{root.winfo_x()+e.x-self._dx}+{root.winfo_y()+e.y-self._dy}"))
        root.bind("<Double-Button-1>", lambda e: root.destroy())
        self._dx = self._dy = 0

        self._root = root
        root.after(50, self._poll)
        root.mainloop()

    def _poll(self):
        try:
            d = self._q.get_nowait()
            meta  = ACTION_META.get(d["action"], (str(d["action"]), "#ffffff", ""))
            label, color, desc = meta
            self._action.config(text=label, fg=color)
            self._desc.config(text=desc)
            self._info.config(text=f"conf {d['confidence']:.0%}  |  enemies {d['enemies']}")
            gsi_ok = d["gsi_ok"]
            self._header.config(
                text=f"{'●' if gsi_ok else '○'} GSI  |  {d['fps']:.1f} Hz",
                fg="#00ff88" if gsi_ok else "#ff4466",
            )
            self._status.config(
                text="live" if gsi_ok else "GSI not connected",
                fg="#00ff88" if gsi_ok else "#ff4466",
            )
        except queue.Empty:
            pass
        if hasattr(self, "_root"):
            self._root.after(50, self._poll)

    def push(self, data):
        try:
            self._q.put_nowait(data)
        except queue.Full:
            pass


class InferenceEngine:
    def __init__(self, model_path, yolo_path, use_overlay=True, hz=TARGET_FPS):
        from sb3_contrib import MaskablePPO
        print("[inference] Loading model...")
        self.model    = MaskablePPO.load(model_path)
        self.gsi      = GSIServer()
        self.detector = YOLODetector(yolo_path)
        print(f"[inference] GSI listening on port {GSI_PORT} — make sure CS2 GSI config points here")
        self.hz       = hz
        self.frame_id = 0
        self.hud      = None

        if use_overlay:
            q = queue.Queue(maxsize=2)
            self.hud = HUD(q)
            time.sleep(0.5)

        print("[inference] Ready. Ctrl+C to stop.\n")
        print(f"  {'Step':>6}  {'Action':<18}  {'Conf':>6}  {'Enemies':>7}  {'GSI':>5}")
        print("  " + "─" * 48)

    def _confidence(self, obs, mask):
        import torch
        obs_t  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_t, mask_t)
            return dist.distribution.probs.squeeze(0).numpy()

    def step(self):
        t0      = time.perf_counter()
        gsi     = self.gsi.fetch()  # reads from in-process GSI server cache
        dets    = self.detector.detect()
        state   = build_state(gsi, dets, self.frame_id)
        obs     = encode_state(state).reshape(1, -1)
        mask    = get_action_mask(state["player"]["utility_held"])

        action_arr, _ = self.model.predict(obs, action_masks=mask.reshape(1, -1), deterministic=True)
        action = int(action_arr[0])

        try:
            probs = self._confidence(obs[0], mask)
            conf  = float(probs[action])
        except Exception:
            conf = 1.0

        label  = ACTION_META.get(action, (str(action),))[0]
        gsi_ok = self.gsi.connected
        fps    = 1.0 / max(time.perf_counter() - t0, 1e-6)

        print(f"  {self.frame_id:>6}  {label:<18}  {conf:>5.0%}  {len(dets):>7}  {'OK' if gsi_ok else '--':>5}")

        if self.hud:
            self.hud.push({"action": action, "confidence": conf, "enemies": len(dets), "gsi_ok": gsi_ok, "fps": fps})

        self.frame_id += 1

    def run(self):
        interval = 1.0 / self.hz
        try:
            while True:
                t = time.perf_counter()
                self.step()
                wait = interval - (time.perf_counter() - t)
                if wait > 0:
                    time.sleep(wait)
        except KeyboardInterrupt:
            print("\n[inference] Stopped.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default=DEFAULT_MODEL)
    p.add_argument("--yolo",       default=YOLO_MODEL_PATH)
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("--hz",         type=float, default=TARGET_FPS)
    args = p.parse_args()

    InferenceEngine(
        model_path  = args.model,
        yolo_path   = args.yolo,
        use_overlay = not args.no_overlay,
        hz          = args.hz,
    ).run()


if __name__ == "__main__":
    main()
