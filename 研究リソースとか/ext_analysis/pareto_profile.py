"""
X-slow / X-fast で使う2種類のパレート分布（FAST, SLOW）の特性を「速さ」と「裾」に
分けて分析し、ユーザー個人差の層別割合を計算する。baseline（現行の全検証で使ってきた
参照条件）も比較対象として加える。

実装の対応: runner/src/brute_force/config.rs::sample_iter
  µ ~ Pareto(scale = 1/(beta*T), shape = gamma/(beta-gamma))
  n = 1/µ  （必要ラウンド数）
  n/n_max ~ Power(alpha)  ⇒  P(n<=x) = (x/n_max)^alpha   (n_max = beta*T)

出力:
  研究リソースとか/ext_analysis/figures/pareto_fast.png
  研究リソースとか/ext_analysis/figures/pareto_slow.png
  研究リソースとか/ext_analysis/figures/pareto_compare.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

T = 5
CONDITIONS = {
    "FAST": dict(beta=0.4, gamma=0.30, color="#86b6ef"),
    "baseline": dict(beta=0.8, gamma=0.60, color="#3987e5"),
    "SLOW": dict(beta=1.4, gamma=1.05, color="#104281"),
}
for c in CONDITIONS.values():
    c["n_max"] = c["beta"] * T
    c["alpha"] = c["gamma"] / (c["beta"] - c["gamma"])

matplotlib.rcParams["font.family"] = [
    "Hiragino Sans", "Hiragino Kaku Gothic Pro", "AppleGothic", "sans-serif",
]
matplotlib.rcParams["axes.unicode_minus"] = False
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


# ---------------------------------------------------------------------------
# 1. 閉形式 P(n<=x) = (x/n_max)^alpha と、モンテカルロでの数値照合
# ---------------------------------------------------------------------------
BOUNDARIES = [1, 2, 3, 5]  # 5 = T
LAYER_LABELS = ["即時反応層 (n≤1)", "早期反応層 (1<n≤2)", "中間層 (2<n≤3)",
                "標準〜鈍感層 (3<n≤5=T)", "期間内に届かない層 (n>5=T)"]

rng = np.random.default_rng(0)
N = 200_000

print("=== 層別割合：閉形式 vs モンテカルロ照合 (N=200,000) ===")
results = {}
for name, c in CONDITIONS.items():
    n_max, alpha = c["n_max"], c["alpha"]
    # 閉形式
    cdf_closed = [min((x / n_max) ** alpha, 1.0) for x in BOUNDARIES]
    layer_closed = np.diff([0.0] + cdf_closed + [1.0])

    # モンテカルロ: mu ~ Pareto(scale=1/n_max, shape=alpha) を numpy で再現
    xm = 1.0 / n_max
    mu = xm * (1.0 + rng.pareto(alpha, N))
    n = 1.0 / mu
    cdf_mc = [np.mean(n <= x) for x in BOUNDARIES]
    layer_mc = np.diff([0.0] + cdf_mc + [1.0])

    median_n = n_max / 2 ** (1 / alpha)
    results[name] = dict(n_max=n_max, alpha=alpha, median=median_n,
                          layer_closed=layer_closed, layer_mc=layer_mc, n_samples=n)

    print(f"\n[{name}] beta={c['beta']} gamma={c['gamma']} n_max={n_max:.2f} alpha={alpha:.2f} "
          f"median_n={median_n:.3f}")
    for lab, lc, lm in zip(LAYER_LABELS, layer_closed, layer_mc):
        print(f"  {lab:<26s} 閉形式={lc*100:6.2f}%  MC={lm*100:6.2f}%  誤差={abs(lc-lm)*100:.3f}pt")

assert results["FAST"]["layer_closed"][-1] == 0.0, "FASTの期間内未達層は0のはず"

# ---------------------------------------------------------------------------
# 2. 単体図（FAST, SLOW）: PDF + CDF、層の境界を塗り分け
# ---------------------------------------------------------------------------
def single_figure(name):
    c = CONDITIONS[name]
    n_max, alpha, color = c["n_max"], c["alpha"], c["color"]
    x = np.linspace(1e-6, n_max, 500)
    pdf = alpha / n_max * (x / n_max) ** (alpha - 1)
    cdf = (x / n_max) ** alpha

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), facecolor=SURFACE)
    axes[0].plot(x, pdf, color=color, linewidth=2)
    axes[0].fill_between(x, pdf, color=color, alpha=0.15)
    axes[0].set_xlabel("必要ラウンド数 n = 1/μ")
    axes[0].set_ylabel("確率密度")
    axes[0].set_title(f"{name}: n の確率密度（n_max={n_max:.1f}, α={alpha:.1f}）", fontsize=10)

    axes[1].plot(x, cdf, color=color, linewidth=2)
    for b in BOUNDARIES:
        if b <= n_max:
            axes[1].axvline(b, color=MUTED, linewidth=0.8, linestyle="--")
    axes[1].axvline(T, color="#c94f4f", linewidth=1.4, linestyle="-", label=f"T={T}（キャンペーン期間）")
    axes[1].set_xlabel("必要ラウンド数 n = 1/μ")
    axes[1].set_ylabel("累積確率 P(n≤x)")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f"{name}: n の累積分布（CDF）", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")

    for ax in axes:
        style_axes(ax)
        ax.set_xlim(0, max(n_max, T) * 1.05)
    fig.suptitle(f"パレート分布の特性：{name}（β={c['beta']}, γ={c['gamma']}）", fontsize=12, color=INK)
    fig.tight_layout()
    path = FIGURES / f"pareto_{name.lower()}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


path_fast = single_figure("FAST")
path_slow = single_figure("SLOW")

# ---------------------------------------------------------------------------
# 3. 比較図（FAST / baseline / SLOW 重ね描き）
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
x_max = max(c["n_max"] for c in CONDITIONS.values())
x = np.linspace(1e-6, x_max, 800)

for name, c in CONDITIONS.items():
    n_max, alpha, color = c["n_max"], c["alpha"], c["color"]
    pdf = np.where(x <= n_max, alpha / n_max * (x / n_max) ** (alpha - 1), 0.0)
    cdf = np.where(x <= n_max, (x / n_max) ** alpha, 1.0)
    label = f"{name}（n_max={n_max:.0f}, α={alpha:.0f}）"
    axes[0].plot(x, pdf, color=color, linewidth=2, label=label)
    axes[1].plot(x, cdf, color=color, linewidth=2, label=label)

axes[1].axvline(T, color="#c94f4f", linewidth=1.4, label=f"T={T}（キャンペーン期間）")
axes[0].set_xlabel("必要ラウンド数 n = 1/μ")
axes[0].set_ylabel("確率密度")
axes[0].set_title("FAST / baseline / SLOW の n の確率密度", fontsize=10)
axes[1].set_xlabel("必要ラウンド数 n = 1/μ")
axes[1].set_ylabel("累積確率 P(n≤x)")
axes[1].set_ylim(0, 1.05)
axes[1].set_title("FAST / baseline / SLOW の n の累積分布（CDF）", fontsize=10)
for ax in axes:
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
fig.suptitle("反応性パラメータ（β, γ）が変えるのは「速さ」だけ — 裾の重さ α=3 は共通",
             fontsize=11.5, color=INK)
fig.tight_layout()
path_compare = FIGURES / "pareto_compare.png"
fig.savefig(path_compare, dpi=200, bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)

print(f"\n図を保存: {path_fast}, {path_slow}, {path_compare}")

# ---------------------------------------------------------------------------
# 4. Notion掲載用の表データを出力
# ---------------------------------------------------------------------------
print("\n=== Notion掲載用 層別割合表（閉形式） ===")
header = f"{'層':<28s}" + "".join(f"{n:>12s}" for n in CONDITIONS)
print(header)
for i, lab in enumerate(LAYER_LABELS):
    row = f"{lab:<28s}" + "".join(f"{results[n]['layer_closed'][i]*100:11.2f}%" for n in CONDITIONS)
    print(row)
print()
for name in CONDITIONS:
    r = results[name]
    print(f"{name}: n_max={r['n_max']:.1f}, alpha={r['alpha']:.1f}, median_n={r['median']:.3f}")
