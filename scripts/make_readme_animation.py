"""Animated README overview (docs/img/overview.gif).

Left: two black holes spiral in on a warped spacetime sheet, radiating
spiral gravitational waves; they merge and the remnant rings down.
Right: the wave is recorded in H1/L1/V1, LeanNPE runs, and the posterior
appears. Illustrative (stylized physics), not model output.

    python scripts/make_readme_animation.py
"""
import io
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs/img/overview.gif"
FPS = 10
BG, PANEL, EDGE = "#070b14", "#0f1626", "#22304a"
INK, MUTED, DIM = "#e8ecf5", "#8b95ab", "#3a465e"
CYAN, ORANGE, VIOLET = "#5cc8ff", "#ffad5c", "#a98bff"
SHEET = LinearSegmentedColormap.from_list(
    "sheet", ["#020409", "#0b1a3a", "#1d3f8a", "#3f7fd6", "#8fd3ff", "#e8f7ff"])

# ---- timeline (frames) ------------------------------------------------------
F_INSPIRAL, F_RING, F_NPE, F_POST, F_HOLD = 80, 26, 18, 22, 25
F_MERGE = F_INSPIRAL
N_FRAMES = F_INSPIRAL + F_RING + F_NPE + F_POST
T_C = F_INSPIRAL / FPS                                    # merger time [s]

# ---- source: stylized inspiral / merger / ringdown ---------------------------
C = 3.2                                                   # wave speed on the sheet
TAU_RD = 0.35                                             # ringdown e-fold [s]


def orbit(t):
    """Orbital phase, angular frequency, separation and wave amplitude at
    source time t (vectorised)."""
    t = np.asarray(t, float)
    tau = np.clip(T_C - t, 0.05, None)
    omega = 2.2 * tau ** (-3 / 8)
    phase = -2.2 * (8 / 5) * tau ** (5 / 8)               # d(phase)/dt = omega
    sep = 1.15 * tau ** (1 / 4)
    amp = 0.12 * omega ** (2 / 3)
    after = t > T_C
    w_rd = 2.2 * 0.05 ** (-3 / 8) * 1.3
    ph_c = -2.2 * (8 / 5) * 0.05 ** (5 / 8)
    phase = np.where(after, ph_c + w_rd * (t - T_C), phase)
    amp = np.where(after, 0.12 * (2.2 * 0.05 ** (-3 / 8)) ** (2 / 3) * np.exp(-(t - T_C) / TAU_RD), amp)
    amp = np.where(t < 0, 0.0, amp)
    return phase, omega, np.where(after, 0.0, sep), amp


# polar grid for a round sheet
R_MAX = 10.0
rr, th = np.meshgrid(np.linspace(0.0, R_MAX, 56), np.linspace(0, 2 * np.pi, 97))
X, Y = rr * np.cos(th), rr * np.sin(th)


def sheet(t):
    phase_s, _, sep, _ = orbit(t)
    if t <= T_C:
        bx, by = 0.5 * sep * np.cos(phase_s), 0.5 * sep * np.sin(phase_s)
        holes = [(bx, by, 1.0), (-bx, -by, 1.0)]
    else:
        holes = [(0.0, 0.0, 1.9)]
    well = np.zeros_like(X)
    for hx, hy, m in holes:
        well -= 1.0 * m / np.sqrt((X - hx) ** 2 + (Y - hy) ** 2 + 1.1)
    t_ret = t - rr / C
    ph, _, _, amp = orbit(t_ret)
    wave = amp * np.cos(2 * (th - ph)) * np.sqrt(2.0 / (rr + 1.0)) * 1.7 * (t_ret > 0)
    return well, wave, holes



# ---- detector strain ----------------------------------------------------------
t_strain = np.linspace(0, (F_INSPIRAL + F_RING) / FPS, 1400)
_, _, _, amp_s = orbit(t_strain)
ph_s = orbit(t_strain)[0]
rng = np.random.default_rng(7)
DETS = [("H1", 1.0, 0.0, 0.0), ("L1", 0.85, 0.9, 0.010), ("V1", 0.5, 2.1, -0.015)]
STRAIN = []
for name, f, dphi, dt in DETS:
    ph = np.interp(t_strain - dt, t_strain, ph_s)
    a = np.interp(t_strain - dt, t_strain, amp_s)
    STRAIN.append((name, f * a * np.cos(2 * ph + dphi) / amp_s.max(), rng.normal(0, 0.10, t_strain.size)))

POST = rng.multivariate_normal([0.58, 0.46], [[0.006, 0.0035], [0.0035, 0.013]], 650)
POST = POST[(POST.min(1) > 0.04) & (POST.max(1) < 0.96)]

CAPTIONS = [(0, "Two black holes spiral together"),
            (F_MERGE - 6, "They merge; spacetime rings down"),
            (F_MERGE + F_RING, "LeanNPE reads the strain in one pass"),
            (F_MERGE + F_RING + F_NPE, "Posterior over all 11 parameters, in seconds")]


def caption(i):
    return [c for f, c in CAPTIONS if i >= f][-1]


def frame(i):
    t = i / FPS
    fig = plt.figure(figsize=(12, 5), dpi=75)
    fig.patch.set_facecolor(BG)

    # ---- 3D merger ----
    ax = fig.add_axes([-0.06, -0.16, 0.66, 1.18], projection="3d", computed_zorder=False)
    ax.set_facecolor(BG)
    t_src = min(t, (F_INSPIRAL + F_RING) / FPS)
    well, wave, holes = sheet(t_src)
    z = well + wave
    # colour: depth sets the base tone, the wave sets the brightness, and the
    # rim fades into the background
    depth = np.clip((well + 1.9) / 1.9, 0, 1)
    rgba = SHEET(np.clip(0.22 + 0.42 * depth + 1.1 * wave, 0.04, 0.93))
    rgba[..., 3] = np.clip(1.25 - rr / R_MAX, 0, 1) ** 0.8
    ax.plot_surface(X, Y, z, facecolors=rgba, rstride=1, cstride=1, linewidth=0,
                    antialiased=False, shade=False, zorder=1)
    for hx, hy, m in holes:
        hz = -1.0 * m / np.sqrt(1.1 + 0.25 * (m < 1.5)) + 0.05
        s = 120 * m
        ax.scatter([hx], [hy], [hz], s=s * 5, color=ORANGE, alpha=0.10, lw=0, zorder=5)
        ax.scatter([hx], [hy], [hz], s=s * 2.2, color=ORANGE, alpha=0.25, lw=0, zorder=6)
        ax.scatter([hx], [hy], [hz], s=s, color="#000000", edgecolor="#ffd29e", lw=1.2, zorder=7)
    if F_MERGE - 2 <= i < F_MERGE + 6:                    # merger flash
        k = 1 - abs(i - F_MERGE - 1) / 5
        ax.scatter([0], [0], [-1.6], s=3000 * k, color="#fff1d6", alpha=0.4 * k, lw=0, zorder=8)
    ax.set_xlim(-R_MAX * 0.7, R_MAX * 0.7); ax.set_ylim(-R_MAX * 0.7, R_MAX * 0.7); ax.set_zlim(-2.6, 2.4)
    ax.view_init(elev=40, azim=-60 + 0.42 * i)
    ax.set_box_aspect((1, 1, 0.42))
    ax.set_axis_off()

    fig.text(0.03, 0.92, caption(i), color=INK, fontsize=14, weight="bold")
    fig.text(0.03, 0.87, "PosteriFlow · neural posterior estimation for gravitational waves",
             color=MUTED, fontsize=9.5)

    # ---- right: detectors ----
    a2 = fig.add_axes([0.6, 0.0, 0.4, 1.0]); a2.set_xlim(0, 4); a2.set_ylim(0, 5); a2.axis("off")
    a2.set_facecolor(BG)
    a2.text(0.15, 4.62, "Detectors", color=INK, fontsize=11, weight="bold")
    n = max(2, int(min(1.0, t / ((F_INSPIRAL + F_RING) / FPS)) * t_strain.size))
    xs = 0.55 + t_strain[:n] / t_strain[-1] * 3.25
    for k, (name, h, nz) in enumerate(STRAIN):
        y0 = 4.15 - k * 0.55
        a2.plot(xs, y0 + 0.2 * (h + nz)[:n], color=DIM, lw=0.6)
        a2.plot(xs, y0 + 0.2 * h[:n], color=CYAN, lw=1.1)
        a2.text(0.15, y0, name, color=MUTED, fontsize=9, va="center", weight="bold")

    # ---- right: LeanNPE ----
    on = i >= F_MERGE + F_RING
    a2.add_patch(FancyArrowPatch((2.0, 2.62), (2.0, 2.28), arrowstyle="-|>", mutation_scale=12,
                                 color=CYAN if on else DIM, lw=1.3))
    a2.add_patch(FancyBboxPatch((0.25, 1.62), 3.5, 0.62, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc=PANEL, ec=EDGE, lw=1.0))
    stages = ["encoder", "rank k", "spline flow"]
    for j, s in enumerate(stages):
        t_on = F_MERGE + F_RING + j * 6
        lit = i >= t_on
        x0 = 0.38 + j * 1.14
        a2.add_patch(FancyBboxPatch((x0, 1.72), 1.0, 0.42, boxstyle="round,pad=0.01,rounding_size=0.07",
                                    fc="#16305e" if lit and i - t_on < 10 else PANEL,
                                    ec=CYAN if lit else EDGE, lw=1.1))
        a2.text(x0 + 0.5, 1.93, s, color=INK if lit else DIM, fontsize=8.5, ha="center", va="center")
    a2.text(0.25, 2.4, "LeanNPE", color=INK if on else MUTED, fontsize=10, weight="bold")

    # ---- right: posterior ----
    p_on = i >= F_MERGE + F_RING + F_NPE
    a2.add_patch(FancyArrowPatch((2.0, 1.6), (2.0, 1.36), arrowstyle="-|>", mutation_scale=12,
                                 color=CYAN if p_on else DIM, lw=1.3))
    px, py, w, h = 0.55, 0.12, 1.55, 1.15
    a2.add_patch(FancyBboxPatch((px, py), w, h, boxstyle="round,pad=0.01,rounding_size=0.06", fc=PANEL, ec=EDGE))
    a2.text(px + w / 2, py + 0.06, "chirp mass", color=MUTED, fontsize=7, ha="center")
    a2.text(px + 0.06, py + h / 2, "distance", color=MUTED, fontsize=7, rotation=90, va="center")
    if p_on:
        g = min(1.0, (i - F_MERGE - F_RING - F_NPE) / 16)
        q = POST[: int(g * len(POST))]
        a2.scatter(px + q[:, 0] * w, py + q[:, 1] * h, s=2.5, color=VIOLET, alpha=0.45, lw=0)
        if g >= 1:
            cov = np.cov(POST.T); ev, evec = np.linalg.eigh(cov)
            ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))
            mu = POST.mean(0)
            for ns in (1, 2):
                a2.add_patch(Ellipse((px + mu[0] * w, py + mu[1] * h), 2 * ns * np.sqrt(ev[1]) * w,
                                     2 * ns * np.sqrt(ev[0]) * h, angle=ang, fill=False, ec="#d6c8ff", lw=1.0))
            a2.text(2.3, 1.12, "masses · distance\nsky · inclination\npolarization · phase\nmerger time · spins",
                    color=MUTED, fontsize=7.8, va="top", linespacing=1.45)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)
    return Image.open(buf).convert("RGB")


if __name__ == "__main__":
    frames = [frame(i) for i in range(N_FRAMES)]
    # palette from representative frames + large swatches of the accent
    # colours, so the small orange/violet details survive quantization
    W, H = frames[0].size
    pal = Image.new("RGB", (W, H * 4), BG)
    for k, f in enumerate((frames[F_MERGE // 2], frames[F_MERGE + 2], frames[-1])):
        pal.paste(f, (0, k * H))
    accents = [ORANGE, "#ffd29e", "#ffe4c2", "#fff1d6", VIOLET, "#d6c8ff", CYAN, INK, MUTED, "#16305e"]
    sw = W // len(accents)
    for k, c in enumerate(accents):
        pal.paste(Image.new("RGB", (sw, H), c), (k * sw, 3 * H))
    pal = pal.quantize(colors=80, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q += [q[-1]] * F_HOLD
    q[0].save(str(OUT), save_all=True, append_images=q[1:], duration=int(1000 / FPS), loop=0)
    print(f"{OUT} ({len(q)} frames, {OUT.stat().st_size / 1e6:.1f} MB)")
