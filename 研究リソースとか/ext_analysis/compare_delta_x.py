"""
δ_X 横断比較（baseline 0.6 / extzero 1.0 / extneg 1.4）。

baseline は 研究リソースとか/結果.pdf・claude-research.md §5.1 の「記録値のみ」で、
全1024系列の分布は手元に無い。比較できる量だけを扱い、
記録の無いところは「記録なし」と明示する。

出力:
  <repo>/研究リソースとか/delta_x_横断比較/{tables,figures}/
  <repo>/研究リソースとか/delta_x_横断比較.pdf
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import polars as pl

from analyze_ext import (
    BASE, BASE_OPT, BASE_RAM, BASE_RAM_M0, BASE_OPT_TAIL3, BASE_RAM_TAIL3,
    A4, AXIS, GRID, INK, INK_2, MUTED, SURFACE,
    draw_df, page, run, style_axes,
)

HERE = Path(__file__).resolve().parent
RESOURCES = HERE.parent
OUT = RESOURCES / "delta_x_横断比較"
TABLES, FIGURES = OUT / "tables", OUT / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
PDF_PATH = RESOURCES / "delta_x_横断比較.pdf"

# 3条件は順序付き（δ_X の大小）なので単一色相の ordinal ランプを使う。
# validate_palette.js --ordinal → ALL CHECKS PASS
COND_RAMP = {"baseline": "#86b6ef", "extzero": "#3987e5", "extneg": "#104281"}
DELTAS = {"baseline": 0.6, "extzero": 1.0, "extneg": 1.4}
NA = "記録なし"


def save_fig(f, name):
    f.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)


def write_table(d, name):
    d.write_csv(TABLES / f"{name}.csv")
    return d


# ---------------------------------------------------------------------------
# 両条件の集計を取得（analyze_ext.run は資料も再生成するが、数値の出所を一本化する）
# ---------------------------------------------------------------------------
res = {c: run(c) for c in ("extzero", "extneg")}
ez, en = res["extzero"], res["extneg"]

base_ram1 = BASE["ram1"]
base_ram10 = BASE["ram10"]
base_opt = BASE["opt_eval"]

# ---------------------------------------------------------------------------
# 表A 主要指標
# ---------------------------------------------------------------------------
def col(cond_res, base_val, fmt_="{:.4f}"):
    return fmt_.format(cond_res)


tableA = pl.DataFrame({
    "指標": [
        "δ_X", "m0 の外部刺激 s_X(m0)",
        "xact 全系列平均", "share 全系列平均", "外部/内部 の割合（全系列）",
        "RAM 1位の系列", "RAM 1位の xact", "00003 の RAM 順位",
        "OPT 評価値", "RAM単一 / OPT", "RAM上位10平均 / OPT",
        "最大化msg適用後の xact", "xact の倍率", "最大化msg適用後の割合", "割合の向上",
        "OPT・RAM の共通系列数", "末尾 m3（RAM上位10）", "末尾 m3（OPT上位10）",
    ],
    "baseline (0.6)": [
        "0.6", "0.05",
        f"{BASE['xact_all']:.4f}", f"{BASE['share_all']:.4f}", f"{BASE['ratio_all']:.4f}",
        "00003", f"{base_ram1:.2f}", "1",
        f"{base_opt:.2f}", f"{base_ram1/base_opt*100:.1f}%", f"{base_ram10/base_opt*100:.1f}%",
        f"{BASE['xact_ram']:.2f}", f"×{BASE['xact_ram']/BASE['xact_all']:.2f}",
        f"{BASE['ratio_ram']:.4f}", f"+{(BASE['ratio_ram']-BASE['ratio_all'])*100:.1f}pt",
        "1", f"{BASE_RAM_TAIL3}/10", f"{BASE_OPT_TAIL3}/10",
    ],
    "extzero (1.0)": [
        "1.0", "0.0",
        f"{ez['overall_all'][0]:.4f}", f"{ez['overall_all'][1]:.4f}", f"{ez['ratio_all']:.4f}",
        ez["ram_list"][0], f"{ez['ram_top1']:.2f}",
        str(ez["rob"].filter(pl.col("msg") == "00003")["rank"][0]),
        f"{ez['opt_eval']:.2f}", f"{ez['ram_top1']/ez['opt_eval']*100:.1f}%",
        f"{ez['ram_top10_mean']/ez['opt_eval']*100:.1f}%",
        f"{ez['overall_ram'][0]:.2f}", f"×{ez['xact_mult']:.2f}",
        f"{ez['ratio_ram']:.4f}", f"+{ez['ratio_gain_pt']:.1f}pt",
        str(len(ez["common"])), f"{ez['ram_tail3']}/10", f"{ez['opt_tail3']}/10",
    ],
    "extneg (1.4)": [
        "1.4", "-0.05",
        f"{en['overall_all'][0]:.4f}", f"{en['overall_all'][1]:.4f}", f"{en['ratio_all']:.4f}",
        en["ram_list"][0], f"{en['ram_top1']:.2f}",
        str(en["rob"].filter(pl.col("msg") == "00003")["rank"][0]),
        f"{en['opt_eval']:.2f}", f"{en['ram_top1']/en['opt_eval']*100:.1f}%",
        f"{en['ram_top10_mean']/en['opt_eval']*100:.1f}%",
        f"{en['overall_ram'][0]:.2f}", f"×{en['xact_mult']:.2f}",
        f"{en['ratio_ram']:.4f}", f"+{en['ratio_gain_pt']:.1f}pt",
        str(len(en["common"])), f"{en['ram_tail3']}/10", f"{en['opt_tail3']}/10",
    ],
})
write_table(tableA, "A_main_metrics")

# ---------------------------------------------------------------------------
# 表B / 表C 上位10 の顔ぶれ
# ---------------------------------------------------------------------------
def rank_of(d_list, m):
    return str(d_list.index(m) + 1) if m in d_list else "—"


base_ram_list = list(BASE_RAM)
base_opt_list = list(BASE_OPT)
all_ram = list(dict.fromkeys(base_ram_list + ez["ram_list"] + en["ram_list"]))
tableB = pl.DataFrame({
    "系列": all_ram,
    "baseline 順位": [rank_of(base_ram_list, m) for m in all_ram],
    "extzero 順位": [rank_of(ez["ram_list"], m) for m in all_ram],
    "extneg 順位": [rank_of(en["ram_list"], m) for m in all_ram],
    "extzero xact": [ez["rob"].filter(pl.col("msg") == m)["xact"][0] for m in all_ram],
    "extneg xact": [en["rob"].filter(pl.col("msg") == m)["xact"][0] for m in all_ram],
})
write_table(tableB, "B_ram_top10_shift")

all_opt = list(dict.fromkeys(base_opt_list + ez["opt10"] + en["opt10"]))
def opt_len(r, m):
    f = r["opt_freq"].filter(pl.col("argmax_msg") == m)
    return int(f["len"][0]) if f.height else 0


tableC = pl.DataFrame({
    "系列": all_opt,
    "baseline 順位": [rank_of(base_opt_list, m) for m in all_opt],
    "baseline 件数": [str(BASE_OPT.get(m, NA)) for m in all_opt],
    "extzero 順位": [rank_of(ez["opt10"], m) for m in all_opt],
    "extzero 件数": [opt_len(ez, m) for m in all_opt],
    "extneg 順位": [rank_of(en["opt10"], m) for m in all_opt],
    "extneg 件数": [opt_len(en, m) for m in all_opt],
})
write_table(tableC, "C_opt_top10_shift")

# ---------------------------------------------------------------------------
# 図A 主要指標の δ_X 推移
# ---------------------------------------------------------------------------
xs = [0.6, 1.0, 1.4]
series = {
    "RAM単一 / OPT (%)": [base_ram1 / base_opt * 100,
                          ez["ram_top1"] / ez["opt_eval"] * 100,
                          en["ram_top1"] / en["opt_eval"] * 100],
    "RAM上位10平均 / OPT (%)": [base_ram10 / base_opt * 100,
                                ez["ram_top10_mean"] / ez["opt_eval"] * 100,
                                en["ram_top10_mean"] / en["opt_eval"] * 100],
    "最大化msg適用の xact 倍率": [BASE["xact_ram"] / BASE["xact_all"],
                                  ez["xact_mult"], en["xact_mult"]],
    "外部/内部 割合の向上 (pt)": [(BASE["ratio_ram"] - BASE["ratio_all"]) * 100,
                                  ez["ratio_gain_pt"], en["ratio_gain_pt"]],
}
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), facecolor=SURFACE)
for ax, (name, vals) in zip(axes, series.items()):
    ax.plot(xs, vals, "-", color="#2a78d6", linewidth=2, zorder=2)
    for x, v, c in zip(xs, vals, COND_RAMP.values()):
        ax.plot(x, v, "o", color=c, markersize=10, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=3)
        ax.annotate(f"{v:,.1f}", (x, v), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(xs, [f"{d}" for d in xs], fontsize=8.5)
    ax.set_xlabel("δ_X", fontsize=8.5)
    ax.set_title(name, fontsize=9.5)
    ax.margins(y=0.24)
    style_axes(ax)
fig.suptitle("図A　δ_X（0.6 / 1.0 / 1.4）に対する主要指標の推移", fontsize=12, color=INK)
fig.tight_layout()
save_fig(fig, "figA_metrics_vs_delta")
figA = fig

# ---------------------------------------------------------------------------
# 図B 00003 と 00013 の入れ替わり
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
pairs = ["00003", "00013"]
vals = {
    "00003": [BASE_RAM["00003"], ez["rob"].filter(pl.col("msg") == "00003")["xact"][0],
              en["rob"].filter(pl.col("msg") == "00003")["xact"][0]],
    "00013": [BASE_RAM["00013"], ez["rob"].filter(pl.col("msg") == "00013")["xact"][0],
              en["rob"].filter(pl.col("msg") == "00013")["xact"][0]],
}
for m, style, dy in zip(pairs, ["-o", "--s"], [-15, 10]):
    # 2系列は δ_X=1.0 付近で値がほぼ重なるので、ラベルは上下に振り分ける
    axes[0].plot(xs, vals[m], style, linewidth=2, markersize=8,
                 color="#2a78d6" if m == "00003" else "#eb6834", label=m,
                 markeredgecolor=SURFACE, markeredgewidth=1.2)
    for x, v in zip(xs, vals[m]):
        axes[0].annotate(f"{v:,.0f}", (x, v), textcoords="offset points", xytext=(0, dy),
                         ha="center", fontsize=7.5,
                         color="#2a78d6" if m == "00003" else "#eb6834")
axes[0].set_xticks(xs, [f"{d}" for d in xs], fontsize=8.5)
axes[0].set_xlabel("δ_X")
axes[0].set_ylabel("xact（全シナリオ平均）")
axes[0].set_title("00003 と 00013 の xact", fontsize=10)
axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_2)
axes[0].margins(y=0.22)

diff = [vals["00013"][i] - vals["00003"][i] for i in range(3)]
bars = axes[1].bar(range(3), diff, 0.5, color=list(COND_RAMP.values()),
                   edgecolor=SURFACE, linewidth=1.2)
for b, v in zip(bars, diff):
    axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:+,.1f}",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK_2)
axes[1].axhline(0, color=AXIS, linewidth=1.0)
axes[1].set_xticks(range(3), [f"δ_X={d}" for d in xs], fontsize=8.5)
axes[1].set_ylabel("00013 − 00003 の xact 差")
axes[1].set_title("00013 が 00003 を上回る差", fontsize=10)
axes[1].margins(y=0.25)
for ax in axes:
    style_axes(ax)
fig.suptitle("図B　主系列 00003 と 00013 の入れ替わり", fontsize=12, color=INK)
fig.tight_layout()
save_fig(fig, "figB_00003_vs_00013")
figB = fig

# ---------------------------------------------------------------------------
# 図C extzero と extneg の全1024系列の対応（2条件のみ）
# ---------------------------------------------------------------------------
joined = (ez["rob"].select(["msg", "xact"]).rename({"xact": "extzero"})
          .join(en["rob"].select(["msg", "xact"]).rename({"xact": "extneg"}), on="msg"))
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURFACE)
axes[0].scatter(joined["extzero"], joined["extneg"], s=13, color="#2a78d6",
                alpha=0.45, edgecolors="none", label="1024 系列")
lim = max(joined["extzero"].max(), joined["extneg"].max()) * 1.05
axes[0].plot([0, lim], [0, lim], "--", color=MUTED, linewidth=1.2, label="y = x")
top = joined.filter(pl.col("msg").is_in(en["ram_list"][:5]))
axes[0].scatter(top["extzero"], top["extneg"], s=48, color="#eb6834",
                edgecolors=SURFACE, linewidths=1.2, zorder=3, label="extneg RAM 上位5")
for m, a, b in zip(top["msg"], top["extzero"], top["extneg"]):
    axes[0].annotate(m, (a, b), textcoords="offset points", xytext=(7, 0),
                     fontsize=7.5, color=INK_2, va="center")
axes[0].set_xlabel("extzero (δ_X=1.0) の xact")
axes[0].set_ylabel("extneg (δ_X=1.4) の xact")
axes[0].set_title("系列ごとの xact 対応（全1024系列）", fontsize=10)
axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2)

y = np.arange(10)[::-1]
rl = en["ram_list"]
axes[1].barh(y + 0.22, [en["rob"].filter(pl.col("msg") == m)["xact"][0] for m in rl],
             0.26, color=COND_RAMP["extneg"], label="extneg (1.4)")
axes[1].barh(y, [ez["rob"].filter(pl.col("msg") == m)["xact"][0] for m in rl],
             0.26, color=COND_RAMP["extzero"], label="extzero (1.0)")
axes[1].barh(y - 0.22, [BASE_RAM.get(m, np.nan) for m in rl],
             0.26, color=COND_RAMP["baseline"], label="baseline (0.6) 記録値")
for i, m in enumerate(rl):
    if m not in BASE_RAM:
        axes[1].text(12, y[i] - 0.22, NA, va="center", fontsize=6.5, color=MUTED)
axes[1].set_yticks(y, rl, fontsize=8.5)
axes[1].set_xlabel("xact（全シナリオ平均）")
axes[1].set_title("extneg RAM 上位10 の3条件比較", fontsize=10)
axes[1].legend(frameon=False, fontsize=8, loc="upper center",
               bbox_to_anchor=(0.5, -0.13), ncol=3, labelcolor=INK_2)
for ax in axes:
    style_axes(ax)
fig.suptitle("図C　extzero と extneg の系列別対応、および3条件の比較", fontsize=12, color=INK)
fig.tight_layout()
save_fig(fig, "figC_cross")
figC = fig

# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
corr = np.corrcoef(joined["extzero"].to_numpy(), joined["extneg"].to_numpy())[0, 1]

with PdfPages(PDF_PATH) as pdf:
    page(pdf, "δ_X 横断比較（0.6 / 1.0 / 1.4）", [
        (None,
         "外部軸の差別化度 δ_X だけを変えた3条件を並べた資料。\n"
         "内部軸（δ_S=0.6, s_S=0.20/0.15/0.10/0.05）と平均刺激（内外とも 0.125）、\n"
         "その他のパラメータは3条件で共通しており、w の較正も同一である。"),
        ("■ 3条件",
         "baseline (δ_X=0.6) : s_X = 0.05 / 0.10 / 0.15 / 0.20   … 結果.pdf の記録値\n"
         "extzero  (δ_X=1.0) : s_X = 0.0  / 0.0833 / 0.1667 / 0.25  … m0 の外部刺激が 0\n"
         "extneg   (δ_X=1.4) : s_X = -0.05 / 0.0667 / 0.1833 / 0.30 … m0 の外部刺激が負\n"
         "いずれも s_X の平均は 0.125。グラフは test/twitter_dataset.txt（81,306ノード）、\n"
         "T=5・L=4（1024系列）、512シナリオ、拡散試行100回。"),
        ("■ baseline の扱い（重要）",
         "baseline の実データは削除済みで、手元にあるのは 結果.pdf・claude-research.md §5.1 の\n"
         "記録値だけである。したがって baseline については\n"
         "  ・RAM 上位10 と OPT 頻度上位10 の顔ぶれと値\n"
         "  ・全体像の平均値（xact / share / 割合）と最大化msg適用後の値\n"
         "  ・OPT 評価値\n"
         "しか比較できない。全1024系列の分布や、任意の系列の値は取り出せないため、\n"
         "表・図で該当する箇所は「記録なし」と明示している。\n"
         "extzero と extneg は全1024系列が手元にあるので、両者だけの比較（図C左）は\n"
         "「2条件のみ」と断ったうえで載せている。"),
        ("表A　主要指標の3条件比較", tableA),
    ])

    page(pdf, "1. 上位系列の顔ぶれの変化", [
        ("表B　RAM 上位10 の順位（3条件、いずれかで上位10に入った系列）", tableB),
        (None,
         f"・baseline で1位だった 00003 は extzero で {tableA['extzero (1.0)'][7]} 位、"
         f"extneg で {tableA['extneg (1.4)'][7]} 位。\n"
         f"・extzero・extneg とも RAM 1位は 00013。\n"
         "・「extzero xact」「extneg xact」列は全1024系列から取れるため、上位10圏外の系列でも値がある。\n"
         "  baseline は記録値のみのため順位列は上位10以外「—」になる。"),
    ])

    page(pdf, "2. OPT 頻度上位の変化", [
        ("表C　OPT 頻度上位10 の順位と件数（3条件）", tableC),
        (None,
         "・件数は argmax になったシナリオ数（全512）。\n"
         "・baseline の件数は記録のある上位10系列のみ。それ以外は「記録なし」。\n"
         "・extzero / extneg の件数は全系列から数えているため、0 件の系列も値を持つ。"),
    ])

    for f in (figA, figB, figC):
        pdf.savefig(f)

    page(pdf, "3. 図の読み方", [
        ("図A", "δ_X を横軸に、対OPT比（2種）・xact 倍率・割合の向上幅を並べたもの。\n"
                "点の色は条件（薄い→濃いが δ_X の小→大）。"),
        ("図B", "左：主系列 00003 と 00013 の xact が δ_X でどう動くか。\n"
                "右：両者の差（00013 − 00003）。正なら 00013 が上回る。"),
        ("図C", "左：extzero と extneg の全1024系列を対応させた散布図（2条件のみ）。\n"
                "      破線は y=x。上側にある系列は extneg のほうが大きいことを示す。\n"
                "右：extneg の RAM 上位10 について3条件の xact を並べたもの。\n"
                "      baseline に記録の無い系列は棒を描かず「記録なし」と注記している。"),
        (None,
         f"・extzero と extneg の系列別 xact の相関係数は {corr:.4f}（1024系列）。\n"
         f"・図C左で y=x より上に位置する系列は "
         f"{int((joined['extneg'] > joined['extzero']).sum())} / 1024 系列。"),
    ])

for f in (figA, figB, figC):
    plt.close(f)

print("\n=== 横断比較 ===")
print(tableA)
print(f"extzero-extneg 相関 = {corr:.4f}")
print(f"PDF: {PDF_PATH}")
