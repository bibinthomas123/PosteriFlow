"""Animated README overview (docs/img/overview.gif): strain streams in ->
LeanNPE stages light up -> the posterior for rank 0, then rank 1, fills in.
Illustrative (stylized chirps and posteriors), not model output.

    python scripts/make_readme_animation.py
"""
import io
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
from PIL import Image

INK, MUTED, GRID = "#1f2430", "#6b7280", "#e5e7eb"
C0, C1, NOISE = "#3b6fd4", "#d07a2e", "#b8bec9"
HI = "#e8eefc"
OUT = Path(__file__).resolve().parents[1] / "docs/img/overview.gif"
rng = np.random.default_rng(3)

def chirp(t, tc, mc, amp):
    tau = np.clip(tc - t, 0.02, None)
    f = 4.0 * (mc / 30) ** (-5 / 8) * tau ** (-3 / 8)
    phase = 2 * np.pi * np.cumsum(f) * (t[1] - t[0])
    return (amp * (0.02 / tau) ** 0.35 * (t < tc) + amp * np.exp(-(t - tc) / 0.03) * (t >= tc)) * np.sin(phase)

t = np.linspace(0, 4, 1600)
dets = [("H1", 0.000, 1.0, 0.8), ("L1", 0.007, 0.8, 1.0), ("V1", -0.012, 0.45, 0.5)]
STRAIN = []
for name, dt, a0, a1 in dets:
    s0 = chirp(t, 2.6 + dt, 30, a0); s1 = chirp(t, 3.3 + dt, 12, 0.6 * a1)
    STRAIN.append((name, s0, s1, rng.normal(0, 0.12, t.size)))
POSTS = [dict(color=C0, label="rank 0\nsignal A", y0=2.2, mu=[0.6, 0.45], cov=[[0.006, 0.003], [0.003, 0.012]]),
         dict(color=C1, label="rank 1\nsignal B", y0=0.6, mu=[0.3, 0.55], cov=[[0.012, -0.006], [-0.006, 0.025]])]
for p in POSTS:
    pts = rng.multivariate_normal(p["mu"], p["cov"], 700)
    keep = (pts[:, 0] > 0.05) & (pts[:, 0] < 0.95) & (pts[:, 1] > 0.05) & (pts[:, 1] < 0.95)
    p["pts"] = pts[keep]

# timeline (frames at 12 fps)
T_STRAIN, T_ENC, T_RANK, T_FLOW, T_P0, T_P1, T_END = 0, 34, 44, 52, 60, 80, 112

def frame(i):
    fig = plt.figure(figsize=(12, 4.3), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12); ax.set_ylim(0, 4.3); ax.axis("off")

    # strain streaming in
    ax.text(0.25, 4.0, "Whitened strain, 4 s", fontsize=12, weight="bold", color=INK)
    ax.text(0.25, 3.72, "two signals overlap in time", fontsize=9.5, color=MUTED)
    frac = min(1.0, (i - T_STRAIN) / (T_ENC - 4))
    n = max(2, int(frac * t.size))
    x = 0.55 + t[:n] / 4 * 3.3
    for k, (name, s0, s1, nz) in enumerate(STRAIN):
        y0 = 2.95 - k * 1.0
        ax.plot(x, y0 + 0.3 * (nz + s0 + s1)[:n], color=NOISE, lw=0.6)
        ax.plot(x, y0 + 0.3 * s1[:n], color=C1, lw=1.2)
        ax.plot(x, y0 + 0.3 * s0[:n], color=C0, lw=1.2)
        ax.text(0.25, y0, name, fontsize=10, color=INK, va="center", weight="bold")
    ax.text(2.2, 0.3, "signal A (loud)", color=C0, fontsize=9, ha="right")
    ax.text(2.35, 0.3, "signal B (quiet)", color=C1, fontsize=9, ha="left")

    def arrow(x0, x1, y, on):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=16,
                                     color=INK if on else "#d1d5db", lw=1.6 if on else 1.2))
    arrow(4.0, 4.55, 2.0, i >= T_ENC - 4)

    # LeanNPE
    ax.add_patch(FancyBboxPatch((4.6, 0.55), 2.75, 3.0, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#f5f7fb", ec="#c9d1e3", lw=1.2))
    ax.text(5.975, 3.25, "LeanNPE", fontsize=13, weight="bold", color=INK, ha="center")
    k_now = 0 if i < T_P1 else 1
    rows = [("encoder", "conv stem + geometry,\nmatched-filter, polarization tokens", T_ENC),
            (f"+ rank k = {k_now}" if i >= T_RANK else "+ rank k", "query the k-th loudest signal", T_RANK),
            ("spline flow", "11-D posterior, one pass", T_FLOW)]
    for j, (h, d, t_on) in enumerate(rows):
        y = 2.7 - j * 0.78
        active = i >= t_on
        pulse = active and (i - t_on < 8 or (j == 1 and T_P1 <= i < T_P1 + 8))
        ax.add_patch(FancyBboxPatch((4.8, y - 0.5), 2.35, 0.6, boxstyle="round,pad=0.01,rounding_size=0.08",
                                    fc=HI if pulse else "white", ec=C0 if pulse else "#d5dbe8", lw=1.6 if pulse else 1))
        ax.text(4.92, y - 0.04, h, fontsize=9.5, weight="bold", color=INK if active else "#9ca3af", va="center")
        ax.text(4.92, y - 0.3, d, fontsize=7.6, color=MUTED if active else "#c4c8d0", va="center", linespacing=1.1)
    ax.text(5.975, 0.32, "seconds per event", fontsize=8.5, color=MUTED, ha="center", style="italic")

    # posteriors
    ax.text(8.15, 4.0, "Posterior per signal", fontsize=12, weight="bold", color=INK)
    for r, (p, t0) in enumerate(zip(POSTS, (T_P0, T_P1))):
        arrow(7.4, 7.95, p["y0"] + 0.65, i >= t0)
        w, h, x0, y0 = 1.6, 1.25, 8.3, p["y0"]
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.01,rounding_size=0.06", fc="white", ec=GRID))
        ax.text(x0 + w / 2, y0 - 0.12, "chirp mass", fontsize=7.5, color=MUTED, ha="center", va="top")
        ax.text(x0 - 0.08, y0 + h / 2, "distance", fontsize=7.5, color=MUTED, ha="right", va="center", rotation=90)
        if i < t0:
            continue
        g = min(1.0, (i - t0) / 14)
        q = p["pts"][: int(g * len(p["pts"]))]
        ax.scatter(x0 + q[:, 0] * w, y0 + q[:, 1] * h, s=2.2, color=p["color"], alpha=0.4, lw=0)
        if g >= 1:
            ev, evec = np.linalg.eigh(p["cov"]); ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))
            for ns in (1, 2):
                ax.add_patch(Ellipse((x0 + p["mu"][0] * w, y0 + p["mu"][1] * h), 2 * ns * np.sqrt(ev[1]) * w,
                                     2 * ns * np.sqrt(ev[0]) * h, angle=ang, fill=False, ec=p["color"], lw=1.2))
        ax.text(x0 + w + 0.12, y0 + h - 0.15, p["label"], fontsize=10, weight="bold", color=p["color"], va="top")
    if i >= T_P1 + 14:
        ax.text(10.3, 2.3, "masses, distance,\nsky, inclination,\npolarization, phase,\nmerger time, spins",
                fontsize=8.3, color=MUTED, va="top", linespacing=1.3)

    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor="white"); plt.close(fig)
    return Image.open(buf).convert("RGB")

frames = [frame(i) for i in range(T_END)]
pal = frames[-1].quantize(colors=96, method=Image.Quantize.MEDIANCUT)
q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
q += [q[-1]] * 30                       # hold the final state ~2.5 s
q[0].save(str(OUT), save_all=True,
          append_images=q[1:], duration=83, loop=0, optimize=False)
