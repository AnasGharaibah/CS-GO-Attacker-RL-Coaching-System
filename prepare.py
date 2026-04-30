from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from rl.pipeline import build_episodes, load_jsonl
from rl.replay_buffer import ReplayBuffer
from rl.actions import Action

SOURCE_DIR = Path("source")
DATA_DIR   = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def episode_stats(episodes):
    if not episodes:
        return {}

    total_frames = sum(len(ep) for ep in episodes)
    wins = deaths = kills = plants = 0

    for ep in episodes:
        for frame in ep:
            ev = frame.get("game_events", {})
            if ev.get("round_won"):          wins   += 1
            if ev.get("kill_this_frame"):    kills  += 1
            if ev.get("death_this_frame"):   deaths += 1
            if frame.get("bomb", {}).get("planted"): plants += 1

    return {
        "episodes":      len(episodes),
        "total_frames":  total_frames,
        "avg_length":    round(total_frames / len(episodes), 1),
        "win_rate":      round(wins / len(episodes) * 100, 1),
        "kills":         kills,
        "deaths":        deaths,
        "plants":        plants,
        "kd":            round(kills / max(deaths, 1), 2),
    }


def buffer_stats(buf):
    _, actions, rewards, _, dones = buf.get_all()
    counts = Counter(int(a) for a in actions)
    return {
        "transitions":  len(buf),
        "episodes":     int(np.sum(dones)),
        "reward_mean":  round(float(np.mean(rewards)), 4),
        "reward_std":   round(float(np.std(rewards)),  4),
        "reward_min":   round(float(np.min(rewards)),  4),
        "reward_max":   round(float(np.max(rewards)),  4),
        "action_dist":  {
            Action(k).name: round(v / len(actions) * 100, 1)
            for k, v in sorted(counts.items()) if k < len(Action)
        },
    }


def print_report(episodes, buf, out_path, matches):
    ep  = episode_stats(episodes)
    bst = buffer_stats(buf)
    sep = "─" * 56

    print(f"\n{sep}")
    print("  DATASET REPORT")
    print(sep)
    print(f"  Matches : {len(matches)}")
    for m in matches:
        print(f"    • {m}")
    print(sep)
    print("  EPISODES")
    print(f"    Rounds       : {ep.get('episodes', 0)}")
    print(f"    Total frames : {ep.get('total_frames', 0):,}")
    print(f"    Avg length   : {ep.get('avg_length', 0)} frames")
    print(f"    Win rate     : {ep.get('win_rate', 0)}%")
    print(f"    K/D          : {ep.get('kd', 0)}")
    print(f"    Kills        : {ep.get('kills', 0)}   Deaths: {ep.get('deaths', 0)}")
    print(f"    Bomb plants  : {ep.get('plants', 0)}")
    print(sep)
    print("  REPLAY BUFFER")
    print(f"    Transitions  : {bst.get('transitions', 0):,}")
    print(f"    Episodes     : {bst.get('episodes', 0)}")
    print(f"    Reward mean  : {bst.get('reward_mean', 0):+.4f}")
    print(f"    Reward std   : {bst.get('reward_std',  0):.4f}")
    print(f"    Reward range : [{bst.get('reward_min', 0):+.2f}, {bst.get('reward_max', 0):+.2f}]")
    print(sep)
    print("  ACTION DISTRIBUTION")
    for name, pct in bst.get("action_dist", {}).items():
        bar = "█" * int(pct / 2)
        print(f"    {name:<18} {pct:>5.1f}%  {bar}")
    print(sep)
    print(f"  Saved → {out_path}")
    print(f"  Next  → python -m rl.train --steps 500000")
    print(f"{sep}\n")


def save_with_splits(buf, out_path, train_frac=0.70, val_frac=0.15):
    buf.save(out_path)

    n     = len(buf)
    idx   = np.random.permutation(n)
    t_end = int(n * train_frac)
    v_end = t_end + int(n * val_frac)

    splits_path = Path(out_path).with_suffix(".splits.json")
    with open(splits_path, "w") as f:
        json.dump({
            "total": n,
            "train": idx[:t_end].tolist(),
            "val":   idx[t_end:v_end].tolist(),
            "test":  idx[v_end:].tolist(),
        }, f)

    print(f"[prepare] Splits → {splits_path}")
    print(f"[prepare] Train: {t_end:,}  Val: {v_end - t_end:,}  Test: {n - v_end:,}")


def main(source, out, match, stats_only, role):
    gsi_dir  = Path(source) / "gsi"
    yolo_dir = Path(source) / "yolo"

    if not gsi_dir.exists():
        print(f"[prepare] {gsi_dir} not found — run 'python run.py' first.")
        return
    if not yolo_dir.exists():
        print(f"[prepare] {yolo_dir} not found — run 'python run.py' first.")
        return

    gsi_files = list(gsi_dir.glob(f"{match}*.jsonl" if match else "*.jsonl"))
    if not gsi_files:
        print(f"[prepare] No .jsonl files in {gsi_dir}")
        return

    pairs   = [(gf, yolo_dir / gf.name) for gf in sorted(gsi_files) if (yolo_dir / gf.name).exists()]
    skipped = [gf.stem for gf in gsi_files if not (yolo_dir / gf.name).exists()]

    if skipped:
        print(f"[prepare] Skipped (no YOLO match): {', '.join(skipped)}")
    if not pairs:
        print("[prepare] No matched GSI+YOLO pairs found.")
        return

    print(f"[prepare] Processing {len(pairs)} match(es)...\n")

    all_episodes = []
    match_names  = []

    for gf, yf in pairs:
        print(f"[prepare] {gf.stem}")
        gsi_lines  = load_jsonl(str(gf))
        yolo_lines = load_jsonl(str(yf))
        print(f"          GSI: {len(gsi_lines)} ticks | YOLO: {len(yolo_lines)} frames")
        eps = build_episodes(str(gf), str(yf), role=role)
        print(f"          Episodes: {len(eps)}")
        all_episodes.extend(eps)
        match_names.append(gf.stem)

    if not all_episodes:
        print("[prepare] No usable episodes. Check your data.")
        return

    print(f"\n[prepare] Building replay buffer from {len(all_episodes)} episodes...")
    buf = ReplayBuffer.build_from_episodes(all_episodes)
    print(f"[prepare] {len(buf):,} transitions")

    print_report(all_episodes, buf, out, match_names)

    if not stats_only:
        save_with_splits(buf, out)
    else:
        print("[prepare] --stats-only: nothing saved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source",     default="source")
    p.add_argument("--out",        default="data/attacker_buffer.h5")
    p.add_argument("--match",      default=None)
    p.add_argument("--role",       default="attacker", choices=["attacker", "defender"])
    p.add_argument("--stats-only", action="store_true")
    args = p.parse_args()

    main(args.source, args.out, args.match, args.stats_only, args.role)
