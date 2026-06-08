"""
Render an animated cumulative-profit comparison for the three blackjack
agents -- the LinkedIn deliverable.

Run after train_dqn.py finishes:

    python make_video.py --hands 200000 --seed 42 --out blackjack.mp4

The video shows:
  - A flat-bet basic-strategy player slowly bleeding to the house edge.
  - A Hi-Lo card counter with bet ramping pulling ahead.
  - A Dueling QR-DQN that sees the exact shoe composition tracking with
    or exceeding the counter.

Outputs an MP4 (or animated GIF as fallback) at 1080p, 30 fps.
"""

import argparse
import os
import random
import time

import numpy as np

# Set matplotlib backend BEFORE importing pyplot -- prevents any GUI
# windows from popping up when running headless.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from cardCounting import BlackjackEnv
from agents import (
    FlatBetBasicStrategyAgent,
    CountingBasicStrategyAgent,
    DQNCardCountingAgent,
)


# Brand-style color palette: high-contrast, readable on white & dark.
AGENT_COLORS = {
    "Flat-bet Basic Strategy": "#E63946",   # red  - loses to house
    "Counting Basic Strategy": "#2A9D8F",   # teal - profitable
    "DQN Card Counting":       "#F4A261",   # amber - profitable+
}


def run_agent(agent, num_hands, seed):
    """Play `num_hands` hands. Returns the per-hand profit array."""
    random.seed(seed)
    np.random.seed(seed)
    env = BlackjackEnv(num_decks=8, min_bet=50, max_bet=50)
    rewards = np.zeros(num_hands, dtype=np.float64)
    for i in range(num_hands):
        # Pre-deal true count for the bet ramp.
        pre_was_shuffled = (env.deck.needs_shuffle()
                            or len(env.deck.dealt_cards) == 0)
        if pre_was_shuffled:
            ptc = 0.0
        else:
            ptc = (env.counter.running_count
                   / max(env.counter.decks_remaining, 0.5))
        state = env.reset()
        state['pre_true_count'] = ptc
        env.bets[0] = int(agent.get_bet(state))
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.get_action(state)
            state, r, done = env.step(action)
            ep_reward += r
        rewards[i] = ep_reward
    return rewards


def animate(profit_series, out_path, num_frames=600, hold_frames=120,
            fps=30, title=None):
    """Render the cumulative-profit lines, growing from 0 hands to N.

    profit_series: list of (label, cumulative-profit array, color)
    num_frames: animation frames (the "growing" part).
    hold_frames: extra frames at the end that hold the final image.
    """
    n_hands = len(profit_series[0][1])
    # Frame i shows hands up to index ceil(n_hands * (i+1) / num_frames).
    # Slightly bias toward the start so the early variance is visible.
    progress = np.linspace(0.0, 1.0, num_frames) ** 0.85
    indices = np.maximum(1, (progress * n_hands).astype(int))

    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor('#0E1116')
    ax.set_facecolor('#0E1116')
    for spine in ax.spines.values():
        spine.set_color('#3A3F47')
        spine.set_linewidth(1.2)
    ax.tick_params(colors='#C8CDD3', labelsize=14)
    ax.grid(True, color='#21262D', alpha=0.85, linewidth=0.8)
    ax.axhline(0, color='#586069', linestyle='--', linewidth=0.9, alpha=0.7)

    ax.set_xlabel("Hands played", color='#E5EBF1', fontsize=18, labelpad=12)
    ax.set_ylabel("Cumulative profit ($)", color='#E5EBF1', fontsize=18,
                  labelpad=12)
    title_str = title or "Blackjack agents — cumulative profit"
    ax.set_title(title_str, color='#FFFFFF', fontsize=22, pad=22,
                 fontweight='bold')

    # Find global y-range so the axis is stable across the animation.
    all_y = np.concatenate([s[1] for s in profit_series])
    ymin, ymax = all_y.min(), all_y.max()
    pad = max(abs(ymin), abs(ymax)) * 0.08
    ax.set_xlim(0, n_hands)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x else "0"))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _:
                          f"${y/1000:+.0f}k" if abs(y) >= 1000 else f"${y:+.0f}"))

    lines = []
    labels_text = []
    for label, cum, color in profit_series:
        ln, = ax.plot([], [], color=color, linewidth=3.0, label=label,
                      solid_capstyle='round', solid_joinstyle='round')
        lines.append(ln)
        labels_text.append((label, cum, color))

    legend = ax.legend(loc='upper left', frameon=True, fontsize=15,
                       facecolor='#161B22', edgecolor='#3A3F47',
                       labelcolor='#E5EBF1')
    for handle in legend.get_lines():
        handle.set_linewidth(3.5)

    # Floating "current value" markers on the right side.
    end_dots = []
    end_labels = []
    for label, cum, color in profit_series:
        dot, = ax.plot([], [], 'o', color=color, markersize=10,
                       markeredgecolor='#0E1116', markeredgewidth=1.5)
        txt = ax.text(0, 0, "", color=color, fontsize=14, fontweight='bold',
                      ha='left', va='center')
        end_dots.append(dot)
        end_labels.append(txt)

    # Footer credit
    fig.text(0.5, 0.02, "8-deck H17 · 3:2 blackjack · DAS · late surrender · "
                       "DQN sees per-rank shoe composition",
             ha='center', color='#7D858F', fontsize=12, fontstyle='italic')

    def update(frame_idx):
        if frame_idx < num_frames:
            i = int(indices[frame_idx])
        else:
            i = n_hands  # Hold the final state.
        i = min(i, n_hands)            # never go past the last hand

        xs = np.arange(i)              # arange is exclusive on the right
        if i == 0:
            xs = np.array([0])
        for k, (ln, (label, cum, color), dot, txt) in enumerate(
                zip(lines, labels_text, end_dots, end_labels)):
            y = cum[:i] if i > 0 else cum[:1]
            ln.set_data(xs[:len(y)], y)
            current_val = float(cum[max(i - 1, 0)])
            cur_x = float(max(i - 1, 0))
            dot.set_data([cur_x], [current_val])
            txt.set_position((cur_x + n_hands * 0.005, current_val))
            txt.set_text(f"${current_val:+,.0f}")
        return (*lines, *end_dots, *end_labels)

    total_frames = num_frames + hold_frames
    animation = anim.FuncAnimation(
        fig, update, frames=total_frames, interval=1000 / fps, blit=False)

    # Save -- prefer mp4 (ffmpeg), fall back to gif.
    if out_path.lower().endswith('.mp4'):
        writer = anim.FFMpegWriter(fps=fps, bitrate=6000,
                                   metadata={'title': title_str})
        animation.save(out_path, writer=writer, dpi=120)
    else:
        writer = anim.PillowWriter(fps=fps)
        animation.save(out_path, writer=writer, dpi=120)

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hands', type=int, default=200_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', default='dqn_agent.pt')
    parser.add_argument('--out', default='blackjack.mp4')
    parser.add_argument('--frames', type=int, default=600,
                        help='Animated growth frames (default 600 ~ 20s at 30fps)')
    parser.add_argument('--hold-frames', type=int, default=120,
                        help='Extra frames holding the final image (default 120 = 4s)')
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    print(f"Running agents for {args.hands:,} hands each "
          f"(seed={args.seed}) ...")
    t0 = time.time()

    agents = [
        FlatBetBasicStrategyAgent(),
        CountingBasicStrategyAgent(),
        DQNCardCountingAgent(model_path=args.model),
    ]
    series = []
    for ag in agents:
        ti = time.time()
        rewards = run_agent(ag, args.hands, args.seed)
        cum = np.cumsum(rewards)
        edge = rewards.sum() / (50 * args.hands) * 100  # bets ramp; close approx
        print(f"  {ag.name}: net=${cum[-1]:+,.0f}  edge~={edge:+.2f}%  "
              f"({time.time() - ti:.1f}s)")
        series.append((ag.name, cum, AGENT_COLORS.get(ag.name, '#FFFFFF')))

    print(f"\nRendering animation -> {args.out} "
          f"({args.frames} frames + {args.hold_frames} hold @ {args.fps}fps) ...")
    animate(series, args.out, num_frames=args.frames,
            hold_frames=args.hold_frames, fps=args.fps)
    elapsed = time.time() - t0
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"Done in {elapsed:.1f}s. Output: {args.out} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
