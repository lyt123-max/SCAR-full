from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]


COLORS = {
    "panel_a": "#9FBF9F",
    "panel_b": "#ADC4FF",
    "panel_c": "#FFB3B3",
    "repr": "#222222",
    "retr": "#2155FF",
    "score": "#E03131",
    "guide": "#909090",
    "green": "#7CB342",
    "green_fill": "#EEF7E8",
    "orange": "#C97A00",
    "orange_fill": "#FFF3DE",
    "blue_fill": "#EEF4FF",
    "purple": "#7C4DFF",
    "purple_fill": "#F4EDFF",
    "red_fill": "#FFF1F1",
    "cal_fill": "#F4FBF1",
    "fusion_fill": "#F7F1FF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the CoReM-AD framework diagram.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "main-result" / "figures" / "framework_diagram",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="coremad_framework_diagram",
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.size": 9.5,
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stix",
        }
    )


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    edge: str = "#D0D7DE",
    fill: str = "white",
    lw: float = 1.2,
    radius: float = 0.22,
    alpha: float = 1.0,
    zorder: float = 1.0,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=fill,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def add_text(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 9.5,
    weight: str = "normal",
    color: str = "#111111",
    ha: str = "center",
    va: str = "center",
    zorder: float = 5,
) -> None:
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        zorder=zorder,
    )


def arrow(
    ax: plt.Axes,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = "#222222",
    lw: float = 1.4,
    ls: str = "-",
    ms: float = 12,
    zorder: float = 4,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            linestyle=ls,
            color=color,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=zorder,
        )
    )


def draw_series_icon(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    n_lines: int = 4,
    seed: int = 0,
    show_axis: bool = False,
) -> None:
    colors = ["#FF6D3A", "#4C8DFF", "#47A447", "#7C3AED", "#F4B942"]
    rng = np.random.RandomState(seed)
    xs = np.linspace(x + 0.08 * w, x + 0.92 * w, 18)
    base_levels = np.linspace(y + 0.24 * h, y + 0.78 * h, n_lines)
    for idx, level in enumerate(base_levels):
        jitter = rng.normal(scale=0.035 * h, size=xs.size)
        trend = 0.05 * h * np.sin(np.linspace(0, 2 * np.pi, xs.size) + idx)
        ys = level + jitter + trend
        ax.plot(xs, ys, color=colors[idx % len(colors)], lw=1.0, zorder=3)
    if show_axis:
        ax.plot([x + 0.08 * w, x + 0.92 * w], [y + 0.14 * h, y + 0.14 * h], color="#222222", lw=1.0, zorder=2)
        ax.plot([x + 0.92 * w, x + 0.92 * w], [y + 0.14 * h, y + 0.88 * h], color="#222222", lw=1.0, zorder=2)
        add_text(ax, x + 0.96 * w, y + 0.18 * h, "time", size=7, ha="right", va="bottom")


def draw_descriptor_icon(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = "#9CCC65",
    edge: str = "#4E6C30",
    n_blocks: int = 4,
) -> None:
    rounded_box(ax, x, y, w, h, edge="#B9C1CC", fill="white", lw=1.0, radius=0.06, zorder=2)
    pad = 0.08 * w
    inner_w = w - 2 * pad
    block_gap = 0.05 * h
    block_h = (h - 2 * pad - (n_blocks - 1) * block_gap) / n_blocks
    for idx in range(n_blocks):
        by = y + pad + idx * (block_h + block_gap)
        ax.add_patch(
            Rectangle(
                (x + pad, by),
                inner_w,
                block_h,
                facecolor=fill if idx < n_blocks - 1 else "#E8F4D7",
                edgecolor=edge,
                linewidth=0.8,
                zorder=3,
            )
        )


def draw_patch_grid(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    rows: int = 3,
    cols: int = 3,
    color: str = "#7EA8F8",
) -> None:
    rounded_box(ax, x, y, w, h, edge="#B7C5D9", fill="white", lw=0.9, radius=0.06, zorder=2)
    pad_x = 0.08 * w
    pad_y = 0.08 * h
    gap_x = 0.04 * w
    gap_y = 0.05 * h
    cell_w = (w - 2 * pad_x - (cols - 1) * gap_x) / cols
    cell_h = (h - 2 * pad_y - (rows - 1) * gap_y) / rows
    for r in range(rows):
        for c in range(cols):
            cx = x + pad_x + c * (cell_w + gap_x)
            cy = y + pad_y + r * (cell_h + gap_y)
            ax.add_patch(
                Rectangle(
                    (cx, cy),
                    cell_w,
                    cell_h,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.8 if (r + c) % 2 == 0 else 0.52,
                    linewidth=0.5,
                    zorder=3,
                )
            )


def draw_tiny_plot_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    descriptor_color: str = "#9CCC65",
    seed: int = 0,
    border: str = "#C8D0DA",
) -> None:
    rounded_box(ax, x, y, w, h, edge=border, fill="white", lw=1.0, radius=0.08, zorder=2)
    draw_series_icon(ax, x + 0.05 * w, y + 0.22 * h, 0.9 * w, 0.64 * h, n_lines=4, seed=seed)
    draw_descriptor_icon(ax, x + 0.20 * w, y + 0.06 * h, 0.60 * w, 0.16 * h, fill=descriptor_color, n_blocks=4)


def draw_input_panel(ax: plt.Axes, x0: float, y0: float, w: float, h: float) -> None:
    rounded_box(ax, x0, y0, w, h, edge=COLORS["panel_a"], fill="#FBFCFA", lw=0.9, radius=0.8)
    add_text(ax, x0 + w / 2, y0 + h - 1.3, "(a) Input and Representation Learning", size=14, weight="bold")

    input_box = (x0 + 0.8, y0 + h - 14.5, 12.0, 11.0)
    stsd_box = (x0 + 14.0, y0 + h - 14.8, 15.3, 9.0)
    state_box = (x0 + 1.6, y0 + h - 24.2, 6.7, 6.6)
    residual_box = (x0 + 10.1, y0 + h - 24.2, 7.1, 6.6)
    gate_box = (x0 + 20.4, y0 + h - 21.8, 7.7, 4.6)
    encoder_box = (x0 + 1.8, y0 + h - 31.0, 6.5, 4.0)
    patch_enc_box = (x0 + 9.8, y0 + h - 31.0, 10.0, 4.4)
    completion_box = (x0 + 22.0, y0 + h - 31.0, 7.6, 5.5)

    rounded_box(ax, *input_box, edge="#C7CCD3", fill="white", lw=0.9, radius=0.45)
    add_text(ax, input_box[0] + input_box[2] / 2, input_box[1] + input_box[3] - 1.0, "Input Window $X$", size=11, weight="bold")
    add_text(
        ax,
        input_box[0] + input_box[2] / 2,
        input_box[1] + input_box[3] - 2.2,
        "(Multivariate Time-Series Window)",
        size=8,
    )
    draw_series_icon(ax, input_box[0] + 0.6, input_box[1] + 1.3, input_box[2] - 1.2, input_box[3] - 3.8, n_lines=4, seed=11, show_axis=True)
    add_text(ax, input_box[0] + input_box[2] / 2, input_box[1] + 1.0, r"$X\ \in\ R^{L\times C}$", size=13, weight="bold")
    add_text(ax, input_box[0] + input_box[2] / 2, input_box[1] + 2.3, "...", size=14)

    rounded_box(ax, *stsd_box, edge="#C7CCD3", fill="white", lw=0.9, radius=0.45)
    add_text(
        ax,
        stsd_box[0] + stsd_box[2] / 2,
        stsd_box[1] + stsd_box[3] - 1.2,
        "Slow-Fast Temporal\nDecomposition (STSD)",
        size=10.5,
        weight="bold",
    )
    add_text(ax, stsd_box[0] + 3.6, stsd_box[1] + 5.3, "State $S$", size=10, weight="bold")
    add_text(ax, stsd_box[0] + 3.6, stsd_box[1] + 4.2, "slow-varying", size=8)
    add_text(ax, stsd_box[0] + 11.6, stsd_box[1] + 5.3, "Residual $R$", size=10, weight="bold")
    add_text(ax, stsd_box[0] + 11.6, stsd_box[1] + 4.2, "fast residual", size=8)
    draw_series_icon(ax, stsd_box[0] + 0.9, stsd_box[1] + 1.6, 5.0, 2.7, n_lines=4, seed=21)
    draw_series_icon(ax, stsd_box[0] + 9.0, stsd_box[1] + 1.6, 5.0, 2.7, n_lines=3, seed=31)
    add_text(ax, stsd_box[0] + 7.2, stsd_box[1] + 2.8, "+", size=18, weight="bold")
    add_text(ax, stsd_box[0] + stsd_box[2] / 2, stsd_box[1] + 0.9, r"$X=S+R,\quad R=X-S$", size=12.5, weight="bold")

    rounded_box(ax, *state_box, edge=COLORS["panel_a"], fill="white", lw=0.9, radius=0.35)
    add_text(ax, state_box[0] + state_box[2] / 2, state_box[1] + state_box[3] - 0.8, "State $S$", size=10, weight="bold")
    add_text(ax, state_box[0] + state_box[2] / 2, state_box[1] + state_box[3] - 2.0, "slow-varying", size=8)
    draw_series_icon(ax, state_box[0] + 0.4, state_box[1] + 0.8, state_box[2] - 0.8, 2.5, n_lines=4, seed=51)

    rounded_box(ax, *residual_box, edge="#F0B44C", fill="white", lw=0.9, radius=0.35)
    add_text(ax, residual_box[0] + residual_box[2] / 2, residual_box[1] + residual_box[3] - 0.8, "Residual $R$", size=10, weight="bold", color="#9A5D00")
    add_text(ax, residual_box[0] + residual_box[2] / 2, residual_box[1] + residual_box[3] - 2.0, "fast residual", size=8)
    add_text(ax, residual_box[0] + residual_box[2] / 2, residual_box[1] + residual_box[3] - 3.0, r"$(R=X-S)$", size=9, weight="bold")
    draw_series_icon(ax, residual_box[0] + 0.35, residual_box[1] + 0.8, residual_box[2] - 0.7, 2.5, n_lines=3, seed=61)

    rounded_box(ax, *gate_box, edge="#C7CCD3", fill="white", lw=0.9, radius=0.32)
    add_text(ax, gate_box[0] + gate_box[2] / 2, gate_box[1] + gate_box[3] - 0.9, "Gate $g$", size=10, weight="bold")
    add_text(ax, gate_box[0] + gate_box[2] / 2, gate_box[1] + gate_box[3] - 2.0, "state-conditioned", size=8)
    add_text(ax, gate_box[0] + gate_box[2] / 2, gate_box[1] + 1.0, r"$g=\sigma(f_{\mathrm{gate}}(h))$", size=10.5)

    rounded_box(ax, *encoder_box, edge="#C7CCD3", fill="white", lw=0.9, radius=0.25)
    add_text(ax, encoder_box[0] + encoder_box[2] / 2, encoder_box[1] + encoder_box[3] / 2, "State Encoder $f_{\\mathrm{state}}$\n(Temporal)", size=9.2)

    rounded_box(ax, *patch_enc_box, edge="#C7CCD3", fill="white", lw=0.9, radius=0.25)
    add_text(ax, patch_enc_box[0] + patch_enc_box[2] / 2, patch_enc_box[1] + patch_enc_box[3] / 2, "Patch Encoder\n(shared trunk + scale heads)", size=9.2)

    rounded_box(ax, *completion_box, edge="#C7CCD3", fill="white", lw=0.9, radius=0.25)
    add_text(ax, completion_box[0] + completion_box[2] / 2, completion_box[1] + completion_box[3] - 1.2, "Completion Head", size=9.5, weight="bold")
    add_text(ax, completion_box[0] + completion_box[2] / 2, completion_box[1] + completion_box[3] - 2.5, "(masked reconstruction /\nnext-patch prediction)", size=7.8)
    for idx in range(4):
        ax.add_patch(Rectangle((completion_box[0] + 0.9, completion_box[1] + 0.7 + idx * 0.6), 0.7, 0.38, facecolor="#A6C8FF", edgecolor="#5E7FC4", lw=0.6))
        ax.add_patch(Rectangle((completion_box[0] + 5.0, completion_box[1] + 0.7 + idx * 0.6), 0.7, 0.38, facecolor="#FFB3A7", edgecolor="#D9666C", lw=0.6))
    add_text(ax, completion_box[0] + completion_box[2] / 2, completion_box[1] - 0.8, "auxiliary training and\nreconstruction diagnostics", size=7.5)

    h_x = x0 + 3.0
    h_y = y0 + 4.0
    draw_descriptor_icon(ax, h_x, h_y, 0.9, 3.5, fill="#9CCC65")
    add_text(ax, h_x - 0.4, h_y + 0.3, r"$h=f_{\mathrm{state}}(S)$", size=11, weight="bold", ha="left", va="top")
    add_text(ax, h_x - 0.4, h_y - 0.8, r"$(h)$: window-level", size=8.2, weight="bold", ha="left", va="top")
    add_text(ax, h_x - 0.4, h_y - 1.7, "state descriptor", size=8.2, ha="left", va="top")

    z_x = x0 + 9.7
    z_y = y0 + 2.6
    for idx in range(4):
        ax.add_patch(Rectangle((z_x + idx * 0.9, z_y), 0.5, 3.4, facecolor="#AFC8FF", edgecolor="#4C78D0", lw=0.7))
    add_text(ax, z_x + 1.5, z_y - 0.6, r"$z_i^{(p)}$", size=11, weight="bold")
    add_text(ax, z_x + 1.5, z_y - 1.6, "patch-level content", size=8.2)
    add_text(ax, z_x + 1.5, z_y - 2.4, "representation", size=8.2)
    add_text(ax, z_x + 6.7, z_y - 0.6, "...", size=14)

    c_x = x0 + 15.0
    c_y = y0 + 2.6
    for idx in range(4):
        ax.add_patch(Rectangle((c_x + idx * 0.9, c_y), 0.5, 3.4, facecolor="#D9C1FF", edgecolor="#7C4DFF", lw=0.7))
    add_text(ax, c_x + 1.5, c_y - 0.6, r"$c_i^{(p)}$", size=11, weight="bold")
    add_text(ax, c_x + 1.5, c_y - 1.6, "patch-level context", size=8.2)
    add_text(ax, c_x + 1.5, c_y - 2.4, "representation", size=8.2)
    add_text(ax, c_x + 1.5, c_y - 3.5, "(local mean pooling over\nneighboring $z$)", size=8.0, color="#2155FF")

    mult_center = (x0 + 14.1, y0 + 10.2)
    ax.add_patch(Circle(mult_center, 0.42, edgecolor="#222222", facecolor="white", lw=1.1, zorder=4))
    add_text(ax, mult_center[0], mult_center[1] - 0.02, r"$\otimes$", size=12, weight="bold")

    arrow(ax, input_box[0] + input_box[2], input_box[1] + input_box[3] / 2, stsd_box[0], stsd_box[1] + stsd_box[3] / 2, color=COLORS["repr"])
    arrow(ax, input_box[0] + input_box[2] * 0.35, input_box[1], state_box[0] + state_box[2] / 2, state_box[1] + state_box[3], color=COLORS["repr"])
    arrow(ax, stsd_box[0] + 3.8, stsd_box[1], state_box[0] + state_box[2] / 2, state_box[1] + state_box[3], color=COLORS["repr"])
    arrow(ax, stsd_box[0] + 10.8, stsd_box[1], residual_box[0] + residual_box[2] / 2, residual_box[1] + residual_box[3], color=COLORS["repr"])
    arrow(ax, state_box[0] + state_box[2] / 2, state_box[1], encoder_box[0] + encoder_box[2] / 2, encoder_box[1] + encoder_box[3], color=COLORS["repr"])
    arrow(ax, encoder_box[0] + encoder_box[2] / 2, encoder_box[1], h_x + 0.45, h_y + 3.55, color=COLORS["repr"])
    arrow(ax, residual_box[0] + residual_box[2] / 2, residual_box[1], mult_center[0], mult_center[1] + 0.42, color=COLORS["repr"])
    arrow(ax, mult_center[0], mult_center[1] - 0.42, patch_enc_box[0] + patch_enc_box[2] / 2, patch_enc_box[1] + patch_enc_box[3], color=COLORS["repr"])
    arrow(ax, patch_enc_box[0] + patch_enc_box[2] * 0.35, patch_enc_box[1], z_x + 1.2, z_y + 3.4, color=COLORS["repr"])
    arrow(ax, patch_enc_box[0] + patch_enc_box[2] * 0.65, patch_enc_box[1], c_x + 1.2, c_y + 3.4, color=COLORS["repr"])
    arrow(ax, patch_enc_box[0] + patch_enc_box[2], patch_enc_box[1] + patch_enc_box[3] * 0.45, completion_box[0], completion_box[1] + completion_box[3] * 0.65, color=COLORS["guide"], ls=(0, (4, 3)), lw=1.1)
    arrow(ax, h_x + 1.0, h_y + 1.8, gate_box[0] + gate_box[2] / 2, gate_box[1], color=COLORS["guide"], ls=(0, (4, 3)), lw=1.1)
    arrow(ax, stsd_box[0] + stsd_box[2] * 0.75, stsd_box[1], gate_box[0] + gate_box[2] / 2, gate_box[1] + gate_box[3], color=COLORS["guide"], ls=(0, (4, 3)), lw=1.1)
    arrow(ax, gate_box[0], gate_box[1] + gate_box[3] * 0.2, mult_center[0] + 0.45, mult_center[1] + 0.1, color=COLORS["guide"], ls=(0, (4, 3)), lw=1.1)


def draw_retrieval_panel(ax: plt.Axes, x0: float, y0: float, w: float, h: float) -> None:
    rounded_box(ax, x0, y0, w, h, edge=COLORS["panel_b"], fill="#FBFCFF", lw=0.9, radius=0.8)
    add_text(ax, x0 + w / 2, y0 + h - 1.3, "(b) Two-Level State-Aware Retrieval", size=14, weight="bold")

    level1 = (x0 + 0.7, y0 + h - 27.4, w - 1.4, 19.2)
    level2 = (x0 + 0.7, y0 + 2.2, w - 1.4, 15.2)

    rounded_box(ax, *level1, edge=COLORS["panel_b"], fill="white", lw=0.8, radius=0.45)
    rounded_box(ax, *level2, edge=COLORS["panel_b"], fill="white", lw=0.8, radius=0.45)

    add_text(ax, x0 + w / 2, level1[1] + level1[3] - 1.0, "Level-1: State-Aware Window Retrieval", size=12.5, weight="bold", color=COLORS["retr"])
    add_text(ax, x0 + w / 2, level1[1] + level1[3] - 2.7, r"State bank $H^{\mathrm{tr}}=\{h_j\}$", size=12, weight="bold")
    add_text(ax, x0 + w / 2, level1[1] + level1[3] - 4.0, "normal training windows with state descriptors", size=8.5)

    q_box = (x0 + 1.1, level1[1] + 7.2, 3.2, 7.8)
    rounded_box(ax, *q_box, edge="#C7CCD3", fill="white", lw=0.9, radius=0.2)
    add_text(ax, q_box[0] + q_box[2] / 2, q_box[1] + q_box[3] - 1.1, r"$h_w$", size=13, weight="bold")
    add_text(ax, q_box[0] + q_box[2] / 2, q_box[1] + q_box[3] - 2.3, "query state", size=7.6)
    add_text(ax, q_box[0] + q_box[2] / 2, q_box[1] + q_box[3] - 3.1, "descriptor", size=7.6)
    draw_descriptor_icon(ax, q_box[0] + 0.55, q_box[1] + 0.8, 0.9, 4.1, fill="#8BC34A")
    add_text(ax, q_box[0] + q_box[2] / 2, q_box[1] + 1.6, "...", size=11)
    add_text(ax, q_box[0] + q_box[2] / 2, q_box[1] - 0.9, "Similarity\nin state space", size=8.2)

    bank = (x0 + 5.7, level1[1] + 7.2, 18.0, 9.5)
    rounded_box(ax, *bank, edge="#CDD6E4", fill="#F7F8FA", lw=0.9, radius=0.12)
    card_w = 2.8
    card_h = 3.9
    xs_top = [bank[0] + 1.0 + i * 3.2 for i in range(5)]
    xs_bot = [bank[0] + 1.2 + i * 3.8 for i in range(4)]
    y_top = bank[1] + 5.2
    y_bot = bank[1] + 1.0
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for idx, cx in enumerate(xs_top):
        draw_tiny_plot_card(ax, cx, y_top, card_w, card_h, descriptor_color="#A5D66A", seed=seeds[idx])
    for idx, cx in enumerate(xs_bot):
        draw_tiny_plot_card(ax, cx, y_bot, card_w, card_h, descriptor_color="#A5D66A", seed=seeds[idx + 5])
    add_text(ax, bank[0] + bank[2] / 2, bank[1] + 3.2, "...", size=16, color="#888888")

    sel_box = (x0 + w - 5.5, level1[1] + 7.0, 3.6, 10.8)
    rounded_box(ax, *sel_box, edge=COLORS["retr"], fill="white", lw=1.1, radius=0.22)
    for idx in range(3):
        draw_tiny_plot_card(ax, sel_box[0] + 0.5, sel_box[1] + 0.6 + idx * 3.25, 2.55, 2.7, descriptor_color="#A5D66A", seed=40 + idx, border="#A7B8E8")
    add_text(ax, sel_box[0] + sel_box[2] / 2, sel_box[1] + 0.3, r"Top-$M$", size=11, weight="bold", va="top")
    add_text(ax, sel_box[0] + sel_box[2] / 2, sel_box[1] - 0.5, "state-consistent\nwindows $J_M(w)$", size=8.6, va="top")
    add_text(ax, sel_box[0] - 1.2, sel_box[1] + sel_box[3] / 2, "Select\n top-$M$\nwindows", size=10, weight="bold", ha="center")

    arrow(ax, q_box[0] + q_box[2], q_box[1] + q_box[3] / 2, bank[0] - 0.6, bank[1] + bank[3] / 2, color=COLORS["retr"], lw=1.6)
    arrow(ax, bank[0] + bank[2], bank[1] + bank[3] / 2, sel_box[0] - 0.3, sel_box[1] + sel_box[3] / 2, color=COLORS["retr"], lw=1.6)

    legend_x = x0 + 9.0
    legend_y = level1[1] + 1.1
    rounded_box(ax, legend_x, legend_y, 9.2, 2.1, edge="#CAD3E0", fill="white", lw=0.8, radius=0.15)
    draw_series_icon(ax, legend_x + 0.25, legend_y + 0.35, 1.7, 1.3, n_lines=2, seed=91)
    add_text(ax, legend_x + 2.5, legend_y + 1.05, "multi-channel\ntime series", size=8.3, ha="left")
    draw_descriptor_icon(ax, legend_x + 5.35, legend_y + 0.55, 1.2, 0.7, fill="#A5D66A")
    add_text(ax, legend_x + 6.95, legend_y + 1.05, "state descriptor", size=8.3, ha="left")

    expand_y = level2[1] + level2[3] + 0.2
    add_text(ax, x0 + w / 2, expand_y + 0.8, "Expand selected windows into patches", size=11.5, weight="bold", color=COLORS["retr"])
    arrow(ax, x0 + w / 2 - 10.0, expand_y + 0.3, x0 + w / 2 - 10.0, level2[1] + level2[3] + 0.1, color="#8FB1FF", lw=1.8, ms=18)
    arrow(ax, x0 + w / 2, expand_y + 0.3, x0 + w / 2, level2[1] + level2[3] + 0.1, color="#8FB1FF", lw=1.8, ms=18)
    arrow(ax, x0 + w / 2 + 10.0, expand_y + 0.3, x0 + w / 2 + 10.0, level2[1] + level2[3] + 0.1, color="#8FB1FF", lw=1.8, ms=18)

    add_text(ax, x0 + w / 2, level2[1] + level2[3] - 1.0, "Level-2: Fine Context Retrieval within $J_M(w)$", size=12.5, weight="bold", color=COLORS["retr"])
    add_text(
        ax,
        x0 + w / 2,
        level2[1] + level2[3] - 3.1,
        r"Patch pool $B^{(p)}(w)$ from retrieved windows $J_M(w)$",
        size=10.5,
        weight="bold",
    )
    add_text(ax, x0 + w / 2, level2[1] + level2[3] - 4.2, r"(search restricted to $J_M(w)$)", size=8.5)

    cq_box = (x0 + 1.3, level2[1] + 4.2, 2.7, 6.5)
    rounded_box(ax, *cq_box, edge="#C7CCD3", fill="white", lw=0.9, radius=0.2)
    add_text(ax, cq_box[0] + cq_box[2] / 2, cq_box[1] + cq_box[3] - 1.0, r"$c_{w,i}^{(p)}$", size=11.5, weight="bold")
    add_text(ax, cq_box[0] + cq_box[2] / 2, cq_box[1] + cq_box[3] - 2.2, "context", size=7.4)
    add_text(ax, cq_box[0] + cq_box[2] / 2, cq_box[1] + cq_box[3] - 3.0, "query", size=7.4)
    for idx in range(4):
        ax.add_patch(Rectangle((cq_box[0] + 0.8, cq_box[1] + 0.6 + idx * 0.9), 0.9, 0.58, facecolor="#C9B3FF", edgecolor="#7C4DFF", lw=0.6))
    add_text(ax, cq_box[0] + cq_box[2] / 2, cq_box[1] - 0.7, "Similarity\nin local\ncontext space", size=8.0)

    pool = (x0 + 6.1, level2[1] + 4.0, 19.0, 6.2)
    rounded_box(ax, *pool, edge="#CDD6E4", fill="#F7F8FA", lw=0.9, radius=0.12)
    add_text(ax, pool[0] + 3.4, pool[1] + pool[3] - 0.7, r"Window $j_1$", size=9.5, weight="bold")
    add_text(ax, pool[0] + 9.6, pool[1] + pool[3] - 0.7, r"Window $j_2$", size=9.5, weight="bold")
    add_text(ax, pool[0] + 17.0, pool[1] + pool[3] - 0.7, r"Window $j_M$", size=9.5, weight="bold")
    draw_patch_grid(ax, pool[0] + 0.8, pool[1] + 0.8, 4.3, 3.7, rows=3, cols=3, color="#8AB2FF")
    draw_patch_grid(ax, pool[0] + 7.0, pool[1] + 0.8, 4.3, 3.7, rows=3, cols=3, color="#A7D87C")
    draw_patch_grid(ax, pool[0] + 14.2, pool[1] + 0.8, 4.3, 3.7, rows=3, cols=3, color="#FFAE6C")
    add_text(ax, pool[0] + 12.0, pool[1] + 2.6, "...", size=16, color="#888888")

    ref_box = (x0 + w - 5.3, level2[1] + 4.0, 3.3, 7.1)
    rounded_box(ax, *ref_box, edge=COLORS["retr"], fill="white", lw=1.1, radius=0.22)
    for idx in range(4):
        draw_patch_grid(ax, ref_box[0] + 0.65, ref_box[1] + 0.6 + idx * 1.55, 1.8, 1.15, rows=2, cols=2, color="#6EA3FF")
    add_text(ax, ref_box[0] - 1.1, ref_box[1] + ref_box[3] / 2, "Select\n top-$K$\npatches", size=10, weight="bold", ha="center")
    add_text(ax, ref_box[0] + ref_box[2] / 2, ref_box[1] - 0.45, r"Top-$K$", size=11, weight="bold", va="top")
    add_text(ax, ref_box[0] + ref_box[2] / 2, ref_box[1] - 1.25, r"local references", size=8.4, va="top")
    add_text(ax, ref_box[0] + ref_box[2] / 2, ref_box[1] - 2.15, r"$N_i^{(p)}(w)$", size=9.2, weight="bold", va="top")
    add_text(ax, ref_box[0] + ref_box[2] / 2, ref_box[1] - 2.95, "(patch-level)", size=8.0, va="top")

    arrow(ax, cq_box[0] + cq_box[2], cq_box[1] + cq_box[3] / 2, pool[0] - 0.4, pool[1] + pool[3] / 2, color=COLORS["retr"], lw=1.6)
    arrow(ax, pool[0] + pool[2], pool[1] + pool[3] / 2, ref_box[0] - 0.35, ref_box[1] + ref_box[3] / 2, color=COLORS["retr"], lw=1.6)

    add_text(ax, pool[0] + 2.1, level2[1] + 0.9, r"$j_1$", size=11, weight="bold")
    add_text(ax, pool[0] + 8.5, level2[1] + 0.9, r"$j_2$", size=11, weight="bold")
    add_text(ax, pool[0] + 14.7, level2[1] + 0.9, r"$\cdots$", size=14, weight="bold")
    add_text(ax, pool[0] + 18.2, level2[1] + 0.9, r"$j_M$", size=11, weight="bold")
    add_text(ax, x0 + w / 2, level2[1] - 0.2, "(patches from each selected window)", size=8.8)


def draw_score_box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, formula: str, note: str | None = None) -> None:
    rounded_box(ax, x, y, w, h, edge="#FF9E9E", fill="white", lw=0.9, radius=0.25)
    add_text(ax, x + w / 2, y + h - 1.6, title, size=8.8, weight="bold")
    add_text(ax, x + w / 2, y + h / 2 - 0.2, formula, size=12)
    if note:
        add_text(ax, x + w / 2, y + 1.1, note, size=8.4)


def draw_calibration_panel(ax: plt.Axes, x0: float, y0: float, w: float, h: float) -> None:
    rounded_box(ax, x0, y0, w, h, edge=COLORS["panel_c"], fill="#FFFCFC", lw=0.9, radius=0.8)
    add_text(ax, x0 + w / 2, y0 + h - 1.3, "(c) Score Calibration and Fusion", size=14, weight="bold")

    add_text(ax, x0 + 5.2, y0 + h - 5.0, "Raw Scores\n(uncalibrated)", size=11, weight="bold", color=COLORS["score"])
    add_text(ax, x0 + 14.3, y0 + h - 5.0, "Step 1:\nEmpirical CDF\nCalibration", size=11, weight="bold")
    add_text(ax, x0 + 21.2, y0 + h - 4.9, "Step 2:\nFusion", size=11, weight="bold")
    add_text(ax, x0 + 28.2, y0 + h - 5.0, "Final Output", size=11, weight="bold")

    raw_x = x0 + 0.8
    cal_x = x0 + 10.6
    fusion_x = x0 + 18.1
    final_x = x0 + 25.5
    box_w = 6.7
    cal_w = 4.3
    fusion_w = 4.2
    box_h = 5.6
    ys = [y0 + h - 15.7, y0 + h - 23.6, y0 + h - 31.5, y0 + h - 39.4]

    draw_score_box(ax, raw_x, ys[0], box_w, box_h, "State Novelty Score", r"$a_w=d(h_w,H^{tr})$")
    draw_score_box(ax, raw_x, ys[1], box_w, box_h, "Memory Score", r"$s_t^{\mathrm{mem}}=d_t^{\mathrm{mem}}$", r"(using $z$ and $N_i^{(p)}(w)$)")
    draw_score_box(ax, raw_x, ys[2], box_w, box_h, "Reconstruction-8 Score", r"$d_t^{(\mathrm{rec}\!-\!8)}$", "masked ratio 8%")
    draw_score_box(ax, raw_x, ys[3], box_w, box_h, "Reconstruction-32 Score", r"$d_t^{(\mathrm{rec}\!-\!32)}$", "masked ratio 32%")

    cal_texts = [
        r"$\hat{F}_p(a_w,\Omega)$",
        r"$\hat{F}_p(s_t^{\mathrm{mem}},\Omega)$",
        r"$\hat{F}_p(d_t^{(\mathrm{rec}\!-\!8)},\Omega)$",
        r"$\hat{F}_p(d_t^{(\mathrm{rec}\!-\!32)},\Omega)$",
    ]
    for y, label in zip(ys, cal_texts):
        rounded_box(ax, cal_x, y, cal_w, box_h, edge="#B9D8AE", fill=COLORS["cal_fill"], lw=0.9, radius=0.25)
        add_text(ax, cal_x + cal_w / 2, y + box_h / 2, label, size=11.2)
        arrow(ax, raw_x + box_w, y + box_h / 2, cal_x - 0.15, y + box_h / 2, color=COLORS["score"], lw=1.5)

    fusion_y = y0 + h - 33.2
    rounded_box(ax, fusion_x, fusion_y, fusion_w, 21.4, edge="#C9B6FF", fill=COLORS["fusion_fill"], lw=0.9, radius=0.25)
    add_text(ax, fusion_x + fusion_w / 2, fusion_y + 11.4, r"$A_t=$", size=15, weight="bold")
    add_text(ax, fusion_x + fusion_w / 2, fusion_y + 8.0, r"$\frac{1}{4}\sum_k\ \hat{F}_p(s_k,\Omega)$", size=12.2, weight="bold")
    add_text(ax, fusion_x + fusion_w / 2, fusion_y + 4.4, r"$(k\in\{a_w,$", size=9.8)
    add_text(ax, fusion_x + fusion_w / 2, fusion_y + 2.8, r"$s_t^{\mathrm{mem}},$", size=9.8)
    add_text(ax, fusion_x + fusion_w / 2, fusion_y + 1.2, r"$d_t^{(\mathrm{rec}\!-\!8)},$", size=9.8)
    add_text(ax, fusion_x + fusion_w / 2, fusion_y - 0.4, r"$d_t^{(\mathrm{rec}\!-\!32)}\})$", size=9.8)

    for y in ys:
        arrow(ax, cal_x + cal_w, y + box_h / 2, fusion_x - 0.15, y + box_h / 2, color=COLORS["score"], lw=1.5)

    final_box = (final_x, y0 + h - 33.0, 5.2, 13.2)
    rounded_box(ax, *final_box, edge="#FF9E9E", fill="white", lw=0.9, radius=0.25)
    add_text(ax, final_box[0] + final_box[2] / 2, final_box[1] + final_box[3] - 2.2, "Anomaly\nScore $A_t$", size=12.2, weight="bold", color=COLORS["score"])
    px = np.linspace(final_box[0] + 0.8, final_box[0] + final_box[2] - 0.7, 10)
    py = np.array([1.3, 3.0, 1.6, 2.2, 4.1, 3.2, 4.0, 3.3, 7.8, 4.8]) + final_box[1]
    ax.plot(px, py, color=COLORS["score"], lw=1.3, zorder=3)
    ax.plot([final_box[0] + 0.7, final_box[0] + final_box[2] - 0.6], [final_box[1] + 4.1, final_box[1] + 4.1], color="#999999", lw=0.9, ls="--")
    ax.plot([final_box[0] + 0.7, final_box[0] + 0.7], [final_box[1] + 1.1, final_box[1] + 8.5], color="#222222", lw=0.9)
    ax.plot([final_box[0] + 0.7, final_box[0] + final_box[2] - 0.7], [final_box[1] + 1.1, final_box[1] + 1.1], color="#222222", lw=0.9)
    add_text(ax, final_box[0] + 0.15, final_box[1] + 8.6, "score", size=7, ha="left", va="bottom", color="#444444")
    add_text(ax, final_box[0] + final_box[2] - 0.3, final_box[1] + 0.5, "time", size=7, ha="right", va="top", color="#444444")
    arrow(ax, fusion_x + fusion_w, fusion_y + 10.7, final_box[0] - 0.2, final_box[1] + final_box[3] / 2, color=COLORS["score"], lw=1.7)

    foot = (x0 + 4.2, y0 + 1.0, 17.6, 3.0)
    rounded_box(ax, *foot, edge="#D2D2D2", fill="white", lw=0.8, radius=0.22)
    add_text(ax, foot[0] + 1.1, foot[1] + 1.95, r"$\hat{F}_p$:", size=10.2, weight="bold", ha="left")
    add_text(ax, foot[0] + 3.0, foot[1] + 1.95, "empirical CDF with tail clipping (level $p$)", size=8.8, ha="left")
    add_text(ax, foot[0] + 1.1, foot[1] + 0.9, r"$\Omega$:", size=10.2, weight="bold", ha="left")
    add_text(ax, foot[0] + 3.0, foot[1] + 0.9, "calibration set (normal training data)", size=8.8, ha="left")


def draw_flow_legend(ax: plt.Axes, x0: float, y0: float, w: float, h: float) -> None:
    rounded_box(ax, x0, y0, w, h, edge="#D5D5D5", fill="white", lw=0.9, radius=0.25)
    items = [
        ("Representation flow", COLORS["repr"], "-"),
        ("Retrieval flow", COLORS["retr"], "-"),
        ("Scoring flow", COLORS["score"], "-"),
        ("State guidance / modulation", COLORS["guide"], (0, (4, 3))),
    ]
    x_positions = [x0 + 3.2, x0 + 15.3, x0 + 26.8, x0 + 39.3]
    for (label, color, ls), x in zip(items, x_positions):
        arrow(ax, x - 1.8, y0 + h / 2, x + 0.8, y0 + h / 2, color=color, lw=1.4, ls=ls, ms=11)
        add_text(ax, x + 1.2, y0 + h / 2, label, size=10.2, ha="left")


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{name}.png"
    pdf_path = output_dir / f"{name}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def build_figure() -> plt.Figure:
    configure_style()
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    draw_input_panel(ax, 0.8, 4.2, 31.0, 47.0)
    draw_retrieval_panel(ax, 32.4, 4.2, 35.2, 47.0)
    draw_calibration_panel(ax, 68.2, 4.2, 30.9, 47.0)
    draw_flow_legend(ax, 16.5, 0.8, 67.0, 2.6)
    return fig


def main() -> None:
    args = parse_args()
    fig = build_figure()
    paths = save_figure(fig, args.output_dir, args.name)
    for path in paths:
        print(f"[FrameworkDiagram] wrote {path}")


if __name__ == "__main__":
    main()
