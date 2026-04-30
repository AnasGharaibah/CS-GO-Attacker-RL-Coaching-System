import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PLOTS_DIR = Path("rl/models/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DARK   = "#0f1117"
CARD   = "#1a1d27"
ACCENT = "#00d4ff"
GREEN  = "#00ff88"
ORANGE = "#ff8c00"
RED    = "#ff4466"
PURPLE = "#c084fc"
YELLOW = "#fbbf24"
GREY   = "#6b7280"
TEXT   = "#e2e8f0"

plt.rcParams.update({
    "figure.facecolor": DARK,
    "axes.facecolor":   CARD,
    "axes.edgecolor":   GREY,
    "axes.labelcolor":  TEXT,
    "axes.titlecolor":  TEXT,
    "xtick.color":      GREY,
    "ytick.color":      GREY,
    "text.color":       TEXT,
    "grid.color":       "#2d3147",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "legend.facecolor": CARD,
    "legend.edgecolor": GREY,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
})


def _kfmt(x, _):
    return f"{int(x/1000)}k" if x >= 1000 else str(int(x))

kfmt = FuncFormatter(_kfmt)


def load_tb(logdir):
    ea = EventAccumulator(logdir)
    ea.Reload()
    out = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        out[tag] = ([e.step for e in events], [e.value for e in events])
    return out


def smooth(values, w=5):
    if len(values) < w:
        return values
    arr    = np.array(values, dtype=float)
    pad    = np.pad(arr, (w // 2, w // 2), mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(pad, kernel, mode="valid")[:len(arr)].tolist()


def save(fig, name):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_reward(data):
    steps_r, vals_r = data.get("rollout/ep_rew_mean", ([], []))
    steps_l, vals_l = data.get("rollout/ep_len_mean", ([], []))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Episode Reward & Length", fontsize=15, fontweight="bold", y=0.98)

    ax1.plot(steps_r, vals_r, color=GREY, alpha=0.3, linewidth=0.8, label="raw")
    ax1.plot(steps_r, smooth(vals_r, 7), color=GREEN, linewidth=2.2, label="smoothed")
    ax1.axhline(0, color=RED, linewidth=0.8, linestyle=":")
    ax1.set_ylabel("Mean Episode Reward")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    ax1.xaxis.set_major_formatter(kfmt)

    ax2.plot(steps_l, vals_l, color=GREY, alpha=0.3, linewidth=0.8)
    ax2.plot(steps_l, smooth(vals_l, 7), color=ACCENT, linewidth=2.2)
    ax2.set_ylabel("Mean Episode Length (frames)")
    ax2.set_xlabel("Training Steps")
    ax2.grid(True)
    ax2.xaxis.set_major_formatter(kfmt)

    fig.tight_layout()
    save(fig, "01_reward_and_length.png")


def plot_losses(data):
    tags = {
        "Policy Loss":  ("train/policy_gradient_loss", ACCENT),
        "Value Loss":   ("train/value_loss",            ORANGE),
        "Entropy Loss": ("train/entropy_loss",          PURPLE),
        "Total Loss":   ("train/loss",                  GREEN),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("PPO Loss Components", fontsize=15, fontweight="bold")
    axes = axes.flatten()

    for ax, (label, (tag, col)) in zip(axes, tags.items()):
        if tag not in data:
            ax.set_visible(False)
            continue
        steps, vals = data[tag]
        ax.plot(steps, vals, color=GREY, alpha=0.25, linewidth=0.8)
        ax.plot(steps, smooth(vals, 7), color=col, linewidth=2.2)
        ax.set_title(label)
        ax.set_xlabel("Steps")
        ax.grid(True)
        ax.xaxis.set_major_formatter(kfmt)

    fig.tight_layout()
    save(fig, "02_ppo_losses.png")


def plot_health(data):
    tags = {
        "Approx KL Divergence": ("train/approx_kl",         ACCENT, (0, 0.05)),
        "Clip Fraction":        ("train/clip_fraction",      ORANGE, (0, 0.5)),
        "Explained Variance":   ("train/explained_variance", GREEN,  (-1, 1)),
        "FPS":                  ("time/fps",                 YELLOW, None),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Training Health Metrics", fontsize=15, fontweight="bold")
    axes = axes.flatten()

    for ax, (label, (tag, col, ylim)) in zip(axes, tags.items()):
        if tag not in data:
            ax.set_visible(False)
            continue
        steps, vals = data[tag]
        ax.plot(steps, vals, color=GREY, alpha=0.25, linewidth=0.8)
        ax.plot(steps, smooth(vals, 7), color=col, linewidth=2.2)
        ax.set_title(label)
        ax.set_xlabel("Steps")
        ax.grid(True)
        ax.xaxis.set_major_formatter(kfmt)
        if ylim:
            ax.set_ylim(*ylim)
        if "KL" in label:
            ax.axhline(0.01, color=RED, linewidth=1, linestyle="--", alpha=0.7, label="target")
            ax.legend(fontsize=8)
        if "Variance" in label:
            ax.axhline(1.0, color=GREEN, linewidth=1, linestyle=":", alpha=0.6)
            ax.axhline(0.0, color=RED,   linewidth=1, linestyle=":", alpha=0.6)

    fig.tight_layout()
    save(fig, "03_training_health.png")


def plot_eval(data, eval_npz_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Evaluation Performance", fontsize=15, fontweight="bold")

    if "eval/mean_reward" in data:
        steps, vals = data["eval/mean_reward"]
        ax1.plot(steps, vals, color=GREY, alpha=0.3, linewidth=0.8)
        ax1.plot(steps, smooth(vals, 3), color=GREEN, linewidth=2.2, marker="o", markersize=4)
        ax1.set_title("Eval Mean Reward (TB)")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Mean Reward")
        ax1.grid(True)
        ax1.xaxis.set_major_formatter(kfmt)

    if Path(eval_npz_path).exists():
        ev       = np.load(eval_npz_path)
        steps_e  = ev["timesteps"]
        mean_r   = ev["results"].mean(axis=1)
        std_r    = ev["results"].std(axis=1)

        ax2.fill_between(steps_e, mean_r - std_r, mean_r + std_r, color=ACCENT, alpha=0.15, label="±1 std")
        ax2.plot(steps_e, mean_r, color=ACCENT, linewidth=2.2, marker="o", markersize=4, label="mean reward")
        ax2.set_title("Eval Mean Reward ± Std (NPZ)")
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Mean Reward")
        ax2.legend()
        ax2.grid(True)
        ax2.xaxis.set_major_formatter(kfmt)
    else:
        ax2.text(0.5, 0.5, "evaluations.npz not found", ha="center", va="center",
                 transform=ax2.transAxes, color=GREY)

    fig.tight_layout()
    save(fig, "04_eval_performance.png")


def plot_dashboard(data):
    fig = plt.figure(figsize=(18, 10), facecolor=DARK)
    fig.suptitle("Attacker RL Training Dashboard — 500k Steps",
                 fontsize=17, fontweight="bold", color=TEXT, y=0.99)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.35)

    panels = [
        (gs[0, :2], "rollout/ep_rew_mean",         "Episode Reward",     GREEN,  None),
        (gs[0, 2:], "rollout/ep_len_mean",          "Episode Length",     ACCENT, None),
        (gs[1, 0],  "train/policy_gradient_loss",   "Policy Loss",        ORANGE, None),
        (gs[1, 1],  "train/value_loss",             "Value Loss",         PURPLE, None),
        (gs[1, 2],  "train/entropy_loss",           "Entropy",            YELLOW, None),
        (gs[1, 3],  "train/approx_kl",              "Approx KL",          RED,    None),
        (gs[2, 0],  "train/clip_fraction",          "Clip Fraction",      ACCENT, None),
        (gs[2, 1],  "train/explained_variance",     "Explained Variance", GREEN,  (-1, 1)),
        (gs[2, 2],  "eval/mean_reward",             "Eval Reward",        GREEN,  None),
        (gs[2, 3],  "time/fps",                     "FPS",                YELLOW, None),
    ]

    for spec, tag, title, col, ylim in panels:
        ax = fig.add_subplot(spec)
        ax.set_facecolor(CARD)
        ax.set_title(title, fontsize=9, pad=4)
        ax.grid(True, alpha=0.4)
        ax.xaxis.set_major_formatter(kfmt)
        ax.tick_params(labelsize=7)

        if tag in data:
            steps, vals = data[tag]
            ax.plot(steps, vals, color=GREY, alpha=0.2, linewidth=0.6)
            ax.plot(steps, smooth(vals, 5), color=col, linewidth=1.8)
            if ylim:
                ax.set_ylim(*ylim)
            if vals:
                ax.text(0.97, 0.04, f"{vals[-1]:.3f}", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=8, color=col, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color=GREY, fontsize=8)

    save(fig, "00_dashboard.png")


def find_latest_run(base):
    runs = sorted(Path(base).glob("MaskablePPO_*"), key=lambda p: int(p.name.split("_")[-1]))
    if not runs:
        raise FileNotFoundError(f"No runs found in {base}")
    return str(runs[-1])


def main(logdir):
    if logdir is None:
        logdir = find_latest_run("rl/models/tb_logs")

    print(f"\n[plot] Reading: {logdir}")
    data = load_tb(logdir)
    print(f"[plot] Tags: {list(data.keys())}")

    print("\n[plot] Generating plots...")
    plot_dashboard(data)
    plot_reward(data)
    plot_losses(data)
    plot_health(data)
    plot_eval(data, "rl/models/eval_logs/evaluations.npz")

    print(f"\n[plot] Done → {PLOTS_DIR}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default=None)
    args = p.parse_args()
    main(args.logdir)
