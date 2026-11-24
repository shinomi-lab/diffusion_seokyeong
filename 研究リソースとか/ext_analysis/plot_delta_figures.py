"""
推薦システム単純.md 3.8 節の図 1・図 2 を作る。
  図 1: 露出の割引 δ を動かしたときの 1 ラウンドの広がり（Rモデルの実装そのもので回した結果。reccomend_system/delta_sweep.csv）
        データは diffusion_seokyeong のルートで `cargo run --release --example delta_sweep -p runner -- ../reccomend_system/delta_sweep.csv 0.3 300`
  図 2: フォロー内とフォロー外の投稿が表示枠を奪い合う玩具のシミュレーション（競合モデルの順位づけだけを取り出したもの。Rモデルではない）

実行: python3 研究リソースとか/ext_analysis/plot_delta_figures.py
出力: reccomend_system/fig_delta_single.png, fig_delta_competition.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]          # diffusion_seokyeong/
MEMO = REPO.parent / "reccomend_system"
matplotlib.rcParams["font.family"] = ["Hiragino Sans", "AppleGothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
MSG_COLORS = ["#2a78d6", "#eb6834", "#2f9e6a", "#8e5bd0"]
MSG_LABELS = ["m0（内部 0.20・外部 0.05）", "m1（0.15・0.10）", "m2（0.10・0.15）", "m3（0.05・0.20）"]


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK_2)


def fig1():
    lines = (MEMO / "delta_sweep.csv").read_text(encoding="utf-8").splitlines()
    header = lines[0].lstrip("# ")
    rows = list(csv.DictReader(l for l in lines if not l.startswith("#")))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True, facecolor=SURFACE)
    for c, tau in enumerate([0.0, 1.0]):
        for r, (key, ylabel) in enumerate([("rec_rcvd", "推薦で受信した人数（平均）"), ("share", "シェアして広げた人数（平均）")]):
            ax = axes[r, c]
            style(ax)
            for m in range(4):
                d = sorted((float(x["delta"]), float(x[f"{key}_mean"]), float(x[f"{key}_se"]))
                           for x in rows if float(x["tau"]) == tau and int(x["msg"]) == m)
                dl, mu, se = (np.array(v) for v in zip(*d))
                ax.plot(dl, mu, color=MSG_COLORS[m], lw=2.2, marker="o", ms=4, label=MSG_LABELS[m])
                ax.fill_between(dl, mu - 2 * se, mu + 2 * se, color=MSG_COLORS[m], alpha=0.18, lw=0)
            ax.set_ylim(bottom=0)
            if r == 0:
                ax.set_title(f"しきい値 τ = {tau:g}" + ("（しきい値なし）" if tau == 0 else "（反応率が基準 ē0 を下回ると推薦しない）"), color=INK, fontsize=11)
            if c == 0:
                ax.set_ylabel(ylabel, color=INK_2)
            if r == 1:
                ax.set_xlabel("露出の割引 δ（0 = 推薦なし、1 = 割引なし）", color=INK_2)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("図 1　露出の割引 δ と 1 ラウンドの広がり（Rモデルの実装による。帯は ±2 標準誤差）", color=INK, fontsize=13, y=0.995)
    fig.text(0.5, 0.005, header, ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(MEMO / "fig_delta_single.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig2():
    """玩具の競合: 利用者ごとにフォロー内 10 個・フォロー外 10 個の候補（点数は対数正規）から、
    フォロー外の点数に δ をかけて上位 5 個を表示する。フォロー外の平均点は 2 倍（δ_c = S_in / S_out = 0.5）。"""
    rng = np.random.default_rng(0)
    n_users, n_in, n_out, k = 20000, 10, 10, 5
    deltas = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(8.2, 5.2), facecolor=SURFACE)
    style(ax)
    greys = ["#104281", "#2a78d6", "#86b6ef"]
    for sigma, color in zip([0.25, 0.5, 1.0], greys):
        s_in = rng.lognormal(-sigma**2 / 2, sigma, (n_users, n_in))
        s_out = rng.lognormal(np.log(2) - sigma**2 / 2, sigma, (n_users, n_out))
        frac = []
        for d in deltas:
            s = np.concatenate([s_in, s_out * d], axis=1)
            top = np.argpartition(-s, k, axis=1)[:, :k]
            frac.append((top >= n_in).mean())
        ax.plot(deltas, frac, color=color, lw=2.4, label=f"点数のばらつき σ = {sigma:g}")
    ax.axvline(0.5, color=MUTED, ls=":", lw=1.4)
    ax.text(0.505, 0.04, "δ_c = S_in / S_out = 0.5", color=INK_2, fontsize=9)
    ax.set_xlabel("フォロー外の割引 δ（小さいほど表示されにくい）", color=INK_2)
    ax.set_ylabel("表示枠のうちフォロー外の割合", color=INK_2)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("図 2　競合モデルで予想される動き（玩具のシミュレーション。Rモデルではない）", color=INK, fontsize=12)
    fig.text(0.5, 0.005, f"利用者 {n_users} 人、フォロー内の候補 {n_in} 個・フォロー外の候補 {n_out} 個（平均点は 2 倍）、表示枠 {k}、点数は対数正規、シード 0", ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(MEMO / "fig_delta_competition.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    print("wrote fig_delta_single.png, fig_delta_competition.png")
