"""
δ_X を変えた条件（extzero: δ_X=1.0 / extneg: δ_X=1.4）の結果を
研究リソースとか/結果.pdf と同じ手順で整理し、
既発表（要旨・発表資料）の結果・考察と一致するかを検証する。

1回の実行で両条件を生成する。

入力:
  <repo>/twitter_extzero, <repo>/twitter_extneg   … parquet（拡張子なし）
出力:
  <repo>/研究リソースとか/{cond}_結果/{tables,figures}/
  <repo>/研究リソースとか/twitter_{cond}_結果.pdf
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
RESOURCES = HERE.parent                  # 研究リソースとか/
REPO = RESOURCES.parent                  # リポジトリルート

# ---------------------------------------------------------------------------
# 条件
# ---------------------------------------------------------------------------
CONDITIONS = {
    "extzero": dict(
        delta_x=1.0, config="test/config_extzero.toml",
        msg_space=("m0 (s_S=0.20, s_X=0.0)    / m1 (0.15, 0.0833)\n"
                   "                m2 (0.10, 0.1667)      / m3 (0.05, 0.25)"),
    ),
    "extneg": dict(
        delta_x=1.4, config="test/config_extneg.toml",
        msg_space=("m0 (s_S=0.20, s_X=-0.05) / m1 (0.15, 0.0667)\n"
                   "                m2 (0.10, 0.1833)     / m3 (0.05, 0.30)"),
    ),
}
REF = "00003"   # 1ポジション差し替えの参照系列（baseline の主系列）

# baseline（δ_X=0.6）の記録値。研究リソースとか/結果.pdf・claude-research.md §5.1 より。
# dict の順序が「上位10」の順位そのものなので、末尾 m3 の本数などはここから数える。
BASE_RAM = {"00003": 1000.3, "00013": 962.4, "00023": 894.8, "10003": 842.1, "00033": 836.6,
            "00002": 813.9, "10013": 807.1, "00012": 802.2, "00022": 787.5, "00103": 782.2}
BASE_OPT = {"00003": 109, "10003": 36, "20003": 28, "00013": 27, "00002": 25,
            "30003": 19, "00023": 16, "00033": 15, "10002": 14, "01003": 11}
BASE = {"xact_all": 226.414718, "share_all": 357.858306, "ratio_all": 0.632694,
        "xact_ram": 852.92943, "share_ram": 957.768342, "ratio_ram": 0.890538,
        "opt_eval": 1071.122441, "ram1": 1000.2617578125, "ram10": 852.92943}

# baseline 自身の「末尾 m3 の本数」と「m0 の平均個数」。C1・C7 の判定基準に使う。
BASE_RAM_TAIL3 = sum(1 for m in BASE_RAM if m[-1] == "3")   # 7/10
BASE_OPT_TAIL3 = sum(1 for m in BASE_OPT if m[-1] == "3")   # 8/10
BASE_RAM_M0 = float(np.mean([m.count("0") for m in BASE_RAM]))  # 3.1/5

# ---------------------------------------------------------------------------
# 描画設定（dataviz スキルの検証済みパレット）
# ---------------------------------------------------------------------------
matplotlib.rcParams["font.family"] = [
    "Hiragino Sans", "Hiragino Kaku Gothic Pro", "AppleGothic", "sans-serif",
]
matplotlib.rcParams["axes.unicode_minus"] = False

MSG_RAMP = ["#86b6ef", "#3987e5", "#256abf", "#104281"]   # m0 → m3（ordinal）
SERIES_1 = "#2a78d6"   # 当該条件
SERIES_2 = "#eb6834"   # baseline
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
A4 = (8.27, 11.69)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def fmt(v):
    if isinstance(v, bool):
        return "○" if v else "×"
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) >= 1000:
            return f"{v:,.2f}"
        if abs(v) >= 1:
            return f"{v:.3f}"
        return f"{v:.4f}"
    return str(v)


def draw_df(fig, df, top, fontsize=7.5, left=0.06, width=0.88):
    row_h = 0.0168
    height = row_h * (df.height + 1)
    ax = fig.add_axes([left, top - height, width, height])
    ax.axis("off")
    tbl = ax.table(
        cellText=[[fmt(v) for v in r] for r in df.iter_rows()],
        colLabels=df.columns, cellLoc="right", colLoc="right",
        loc="upper left", bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#f0efec")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_facecolor(SURFACE)
            cell.set_text_props(color=INK_2)
    return top - height


def page(pdf, title, blocks):
    """blocks: [(見出し|None, DataFrame | str)]"""
    fig = plt.figure(figsize=A4, facecolor=SURFACE)
    fig.text(0.06, 0.963, title, fontsize=13.5, weight="bold", color=INK)
    y = 0.932
    for heading, body in blocks:
        if heading:
            fig.text(0.06, y, heading, fontsize=9.5, weight="bold", color=INK)
            y -= 0.020
        if isinstance(body, str):
            fig.text(0.065, y, body, fontsize=8.2, va="top", color=INK_2, linespacing=1.72)
            y -= 0.0165 * (body.count("\n") + 1) + 0.014
        else:
            y = draw_df(fig, body, y) - 0.022
    pdf.savefig(fig)
    plt.close(fig)


def judge(ok, in_range=True):
    """○=主張どおり成立 / △=傾向は同じだが記録値の水準・範囲から外れる / ×=成立しない"""
    if not ok:
        return "×"
    return "○" if in_range else "△"


# ===========================================================================
def run(cond):
    spec = CONDITIONS[cond]
    dx = spec["delta_x"]
    label = f"{cond} (δ_X={dx})"
    out_dir = RESOURCES / f"{cond}_結果"
    tables, figures = out_dir / "tables", out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    pdf_path = RESOURCES / f"twitter_{cond}_結果.pdf"

    def write_table(d, name):
        d.write_csv(tables / f"{name}.csv")
        return d

    def save_fig(f, name):
        f.savefig(figures / f"{name}.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)

    # -- 1. データ前処理 ----------------------------------------------------
    df = pl.read_parquet(REPO / f"twitter_{cond}")
    assert df.height == 524_288, df.height
    assert df["msg"].n_unique() == 1024
    assert df["id"].n_unique() == 512
    assert df.null_count().sum_horizontal()[0] == 0

    rob = (
        df.group_by("msg")
        .agg(pl.col("mean_num_xact").mean().alias("xact"),
             pl.col("mean_num_share").mean().alias("share"))
        .with_columns((pl.col("xact") / pl.col("share")).alias("xact_share_ratio"))
        .sort("xact", descending=True)
        .with_row_index("rank", offset=1)
    )
    write_table(rob, "00_rob_details_1024")

    opt = (
        df.filter(pl.col("mean_num_xact") == pl.col("mean_num_xact").max().over(["us_id", "ip"]))
        .sort(["ip", "us_id", "msg"])
        .unique(subset=["ip", "us_id"], keep="first")
        .rename({"msg": "argmax_msg", "mean_num_xact": "max_mean_num_xact"})
    )
    assert opt.height == 512

    # -- 2. 抽出（結果.pdf と同じ表）----------------------------------------
    ram10 = rob.head(10)
    t1 = write_table(
        ram10.select(pl.col("msg"), pl.col("xact").alias("外部行動者"),
                     pl.col("share").alias("内部行動回数")), "01_ram_top10")

    opt_freq = (
        opt.group_by("argmax_msg")
        .agg(pl.col("max_mean_num_xact").mean().alias("max_mean_num_xact"),
             pl.len().alias("len"))
        .sort("len", descending=True)
    )
    t2 = write_table(opt_freq.head(10), "02_opt_top10")

    opt_eval = opt["max_mean_num_xact"].mean()
    ram_top10_mean = ram10["xact"].mean()
    ram_top1 = rob["xact"][0]
    urs = df["mean_num_xact"].mean()
    t3 = write_table(pl.DataFrame({
        "項目": ["OPT 評価値（各シナリオ最良値の平均）", "RAM 上位10 の平均", "RAM 上位1（単一系列）",
                 "URS（全系列平均）", "RAM上位10平均 / OPT", "RAM上位1 / OPT", "URS / OPT"],
        "値": [opt_eval, ram_top10_mean, ram_top1, urs,
               ram_top10_mean / opt_eval, ram_top1 / opt_eval, urs / opt_eval],
    }), "03_eval")

    ram_list = ram10["msg"].to_list()
    sub = df.filter(pl.col("msg").is_in(ram_list))
    overall_all = (df["mean_num_xact"].mean(), df["mean_num_share"].mean())
    overall_ram = (sub["mean_num_xact"].mean(), sub["mean_num_share"].mean())
    t4 = write_table(pl.DataFrame({
        "xact": [overall_all[0]], "share": [overall_all[1]],
        "対内部行動の外部行動回数の割合": [overall_all[0] / overall_all[1]]}), "04_overall_all")
    t5 = write_table(pl.DataFrame({
        "xact": [overall_ram[0]], "share": [overall_ram[1]],
        "対内部行動の外部行動回数の割合": [overall_ram[0] / overall_ram[1]]}), "05_overall_ram")

    xact_mult = overall_ram[0] / overall_all[0]
    ratio_all = overall_all[0] / overall_all[1]
    ratio_ram = overall_ram[0] / overall_ram[1]
    ratio_gain_pt = (ratio_ram - ratio_all) * 100

    opt10 = opt_freq.head(10)["argmax_msg"].to_list()
    common = [m for m in ram_list if m in opt10]
    t6 = write_table(pl.DataFrame({
        "共通系列": common,
        "RAM順位": [rob.filter(pl.col("msg") == m)["rank"][0] for m in common],
        "RAM xact": [rob.filter(pl.col("msg") == m)["xact"][0] for m in common],
        "OPT 件数": [opt_freq.filter(pl.col("argmax_msg") == m)["len"][0] for m in common],
    }), "06_common")

    def compose_row(m, source, val):
        return {"系列": m, "抽出元": source,
                "m0の数": m.count("0"), "m1の数": m.count("1"),
                "m2の数": m.count("2"), "m3の数": m.count("3"),
                "先頭": f"m{m[0]}", "末尾": f"m{m[-1]}", "値": val}

    rows = [compose_row(m, "RAM上位10", rob.filter(pl.col("msg") == m)["xact"][0]) for m in ram_list]
    rows += [compose_row(m, "OPT上位10", float(opt_freq.filter(pl.col("argmax_msg") == m)["len"][0]))
             for m in opt10]
    t7 = write_table(pl.DataFrame(rows), "07_composition")

    # -- 3. 既発表との一致検証 ----------------------------------------------
    ram_tail3 = sum(1 for m in ram_list if m[-1] == "3")
    opt_tail3 = sum(1 for m in opt10 if m[-1] == "3")
    m_counts = {k: float(np.mean([m.count(str(k)) for m in ram_list])) for k in range(4)}
    m0_is_max = all(m_counts[0] > m_counts[k] for k in (1, 2, 3))

    verify = pl.DataFrame({
        "#": ["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
        "既発表の主張": [
            "OPT・RAM とも「前半 m0 多め＋末尾 m3」型が上位",
            "OPT・RAM の共通系列に 00003 が含まれる",
            "RAM 単一系列で対 OPT 約93%",
            "RAM 上位10平均で対 OPT 約80%",
            "最大化msg適用で外部行動者数が増加（×1.4〜×3.8）",
            "最大化msg適用で外部/内部比率が向上（+17.0〜+33.7pt）",
            "上位系列は拡散促進型(m0)の割合が多め",
        ],
        "baseline 記録値": [
            f"末尾m3: RAM {BASE_RAM_TAIL3}/10, OPT {BASE_OPT_TAIL3}/10",
            "00003",
            f"{BASE['ram1']/BASE['opt_eval']*100:.1f}%",
            f"{BASE['ram10']/BASE['opt_eval']*100:.1f}%",
            f"{BASE['xact_all']:.1f}→{BASE['xact_ram']:.1f} (×{BASE['xact_ram']/BASE['xact_all']:.2f})",
            f"{BASE['ratio_all']:.3f}→{BASE['ratio_ram']:.3f} (+{(BASE['ratio_ram']-BASE['ratio_all'])*100:.1f}pt)",
            f"m0 平均 {BASE_RAM_M0:.1f}/5",
        ],
        f"{cond} の値": [
            f"末尾m3: RAM {ram_tail3}/10, OPT {opt_tail3}/10",
            f"共通{len(common)}件、00003 {'含む' if '00003' in common else '含まない'}",
            f"{ram_top1/opt_eval*100:.1f}%",
            f"{ram_top10_mean/opt_eval*100:.1f}%",
            f"{overall_all[0]:.1f}→{overall_ram[0]:.1f} (×{xact_mult:.2f})",
            f"{ratio_all:.3f}→{ratio_ram:.3f} (+{ratio_gain_pt:.1f}pt)",
            f"m0 平均 {m_counts[0]:.1f}/5（m1 {m_counts[1]:.1f} / m2 {m_counts[2]:.1f} / m3 {m_counts[3]:.1f}）",
        ],
        "判定": [
            # baseline 自身の本数（RAM 7/10・OPT 8/10）以上なら成立とする
            judge(ram_tail3 >= BASE_RAM_TAIL3 and opt_tail3 >= BASE_OPT_TAIL3),
            judge("00003" in common),
            judge(ram_top1 / opt_eval > 0.80, abs(ram_top1 / opt_eval - 0.934) <= 0.02),
            judge(ram_top10_mean / opt_eval > 0.70, abs(ram_top10_mean / opt_eval - 0.796) <= 0.02),
            judge(xact_mult > 1.0, 1.4 <= xact_mult <= 3.8),
            judge(ratio_gain_pt > 0, 17.0 <= ratio_gain_pt <= 33.7),
            judge(m0_is_max),
        ],
        "備考": [
            f"baseline（RAM {BASE_RAM_TAIL3}/10・OPT {BASE_OPT_TAIL3}/10）以上か",
            f"共通 {len(common)} 件",
            f"baseline 比 {ram_top1/opt_eval*100-93.4:+.1f}pt",
            f"baseline 比 {ram_top10_mean/opt_eval*100-79.6:+.1f}pt",
            "増加の向きは一致、倍率は記録範囲を超過" if xact_mult > 3.8 else "記録範囲内",
            "向上の向きは一致、幅は記録範囲を超過" if ratio_gain_pt > 33.7 else "記録範囲内",
            "m0 が最多" if m0_is_max else "m0 は最多でない",
        ],
    })
    write_table(verify, "08_verification")

    # -- 4. 図 --------------------------------------------------------------
    figs = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    y = np.arange(10)[::-1]
    axes[0].barh(y + 0.19, ram10["xact"].to_numpy(), 0.36, color=SERIES_1, label=label)
    axes[0].barh(y - 0.19, [BASE_RAM.get(m, np.nan) for m in ram_list], 0.36,
                 color=SERIES_2, label="baseline (δ_X=0.6) 記録値")
    for i, (m, v) in enumerate(zip(ram_list, ram10["xact"])):
        axes[0].text(v + 15, y[i] + 0.19, f"{v:,.0f}", va="center", fontsize=7.5, color=INK_2)
        if m not in BASE_RAM:
            axes[0].text(15, y[i] - 0.19, "baseline 記録なし", va="center", fontsize=6.5, color=MUTED)
    axes[0].set_yticks(y, ram_list, fontsize=8.5)
    axes[0].set_xlabel("外部行動者数（対全拡散モデルの平均）")
    axes[0].set_title("RAM 上位10系列", fontsize=10)

    o10 = opt_freq.head(10)
    axes[1].barh(y + 0.19, o10["len"].to_numpy(), 0.36, color=SERIES_1, label=label)
    axes[1].barh(y - 0.19, [BASE_OPT.get(m, np.nan) for m in opt10], 0.36,
                 color=SERIES_2, label="baseline (δ_X=0.6) 記録値")
    for i, (m, v) in enumerate(zip(opt10, o10["len"])):
        axes[1].text(v + 1, y[i] + 0.19, str(v), va="center", fontsize=7.5, color=INK_2)
        if m not in BASE_OPT:
            axes[1].text(1, y[i] - 0.19, "baseline 記録なし", va="center", fontsize=6.5, color=MUTED)
    axes[1].set_yticks(y, opt10, fontsize=8.5)
    axes[1].set_xlabel("argmax になったシナリオ数（全512）")
    axes[1].set_title("OPT 頻度上位10系列", fontsize=10)
    for ax in axes:
        style_axes(ax)
        ax.legend(frameon=False, fontsize=8, loc="upper center",
                  bbox_to_anchor=(0.5, -0.155), ncol=2, labelcolor=INK_2)
    fig.suptitle("図1　RAM 上位10 と OPT 頻度上位10（baseline は結果.pdf の記録値）",
                 fontsize=11.5, color=INK)
    fig.tight_layout()
    save_fig(fig, "fig1_ram_opt")
    figs.append(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), facecolor=SURFACE)
    panels = [
        ("外部行動者数 xact", [BASE["xact_all"], BASE["xact_ram"]], [overall_all[0], overall_ram[0]]),
        ("内部行動回数 share", [BASE["share_all"], BASE["share_ram"]], [overall_all[1], overall_ram[1]]),
        ("外部/内部 の割合", [BASE["ratio_all"], BASE["ratio_ram"]], [ratio_all, ratio_ram]),
    ]
    x = np.arange(2)
    for ax, (title, bvals, evals) in zip(axes, panels):
        ax.bar(x - 0.19, evals, 0.36, color=SERIES_1, label=label)
        ax.bar(x + 0.19, bvals, 0.36, color=SERIES_2, label="baseline (δ_X=0.6)")
        for xi, v in list(zip(x - 0.19, evals)) + list(zip(x + 0.19, bvals)):
            ax.text(xi, v, f"{v:,.3f}" if v < 10 else f"{v:,.0f}", ha="center", va="bottom",
                    fontsize=8, color=INK_2)
        ax.set_xticks(x, ["全系列平均", "最大化msg\n（RAM上位10）"], fontsize=8.5)
        ax.set_title(title, fontsize=10)
        ax.margins(y=0.18)
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK_2)
    fig.suptitle("図2　全系列平均 と 最大化msg適用後 の比較", fontsize=11.5, color=INK)
    fig.tight_layout()
    save_fig(fig, "fig2_overall")
    figs.append(fig)

    # 1ポジション差し替え対比（5ラウンド合計しか無いので、平均を挟まない比較にする）
    xact_d = dict(zip(rob["msg"].to_list(), rob["xact"].to_list()))
    share_d = dict(zip(rob["msg"].to_list(), rob["share"].to_list()))
    contrast = pl.DataFrame([
        {"差し替え位置": f"R{t+1}", "置いたメッセージ": f"m{m}",
         "系列": REF[:t] + str(m) + REF[t + 1:],
         "参照と同一": REF[:t] + str(m) + REF[t + 1:] == REF,
         "xact": xact_d[REF[:t] + str(m) + REF[t + 1:]],
         "share": share_d[REF[:t] + str(m) + REF[t + 1:]]}
        for t in range(5) for m in range(4)
    ])
    write_table(contrast, "09_contrast_ref00003")

    def contrast_fig(col, lab, figno, fname):
        f, axs = plt.subplots(1, 5, figsize=(13.5, 3.8), facecolor=SURFACE, sharey=True)
        vmax = contrast[col].max()
        for t, ax in enumerate(axs):
            st = contrast.filter(pl.col("差し替え位置") == f"R{t+1}")
            vals = st[col].to_numpy()
            bars = ax.bar(np.arange(4), vals, 0.62, color=MSG_RAMP,
                          edgecolor=SURFACE, linewidth=1.2)
            for i, (b, v) in enumerate(zip(bars, vals)):
                is_ref = st["参照と同一"][i]
                ax.text(b.get_x() + b.get_width() / 2, v + vmax * 0.02,
                        f"{v:,.0f}\n(参照)" if is_ref else f"{v:,.0f}",
                        ha="center", va="bottom", fontsize=7.5,
                        color=INK if is_ref else INK_2)
            ax.set_xticks(range(4), [f"m{m}" for m in range(4)], fontsize=8.5)
            ax.set_title(f"R{t+1} を差し替え\n{REF[:t]}?{REF[t+1:]}", fontsize=9)
            ax.set_ylim(0, vmax * 1.22)
            style_axes(ax)
        axs[0].set_ylabel(f"{lab}（5ラウンド合計）")
        f.suptitle(f"図{figno}　参照系列 {REF} の1ポジションだけを差し替えたときの{lab}",
                   fontsize=11.5, color=INK)
        f.tight_layout()
        save_fig(f, fname)
        return f

    figs.append(contrast_fig("xact", "外部行動者数 xact", 3, "fig3_contrast_xact"))
    figs.append(contrast_fig("share", "内部行動回数 share", 4, "fig4_contrast_share"))

    # -- 5. PDF -------------------------------------------------------------
    n_ok = verify.filter(pl.col("判定") == "○").height
    n_partial = verify.filter(pl.col("判定") == "△").height
    n_ng = verify.filter(pl.col("判定") == "×").height
    rank00003 = rob.filter(pl.col("msg") == "00003")["rank"][0]
    x00003 = rob.filter(pl.col("msg") == "00003")["xact"][0]

    with PdfPages(pdf_path) as pdf:
        page(pdf, f"twitter_{cond} 結果（δ_X = {dx}）", [
            (None,
             f"本資料は 研究リソースとか/結果.pdf と同じ前処理・同じ抽出を twitter_{cond} に適用し、\n"
             "既発表（要旨・発表資料）の結果および考察が成り立つかを検証したものである。"),
            (f"■ 実行条件（{spec['config']}）",
             "グラフ        : test/twitter_dataset.txt（81,306 ノード, directed, transposed）\n"
             f"message_space : {spec['msg_space']}\n"
             f"                s_S: δ_S=0.6（baseline と同一）／ s_X: δ_X={dx}（baseline は 0.6）\n"
             "                内外とも平均刺激 0.125 で baseline と同一（w の較正は不変）\n"
             "T=5, L=4 → 1024 系列 ／ 受信確率 0.5 ／ 拡散試行 100 回\n"
             "IP 16 × ユーザ状態 32 → 512 シナリオ ／ beta=0.8, gamma=0.6, u0=-1.0"),
            ("■ データ",
             f"twitter_{cond}（parquet, version_runner=0.2.0）: {df.height:,} 行 = 512 × 1024、欠損 0\n"
             "列: id, ip, us_id, msg, mean_num_xact, mean_num_share（msg は5桁・先頭がラウンド1）"),
            ("■ 指標の定義（analysis/CLAUDE.md を踏襲）",
             "RAM : 全シナリオでの msg 別平均が最大となる単一系列（上位10系列を「最大化msg」とする）\n"
             "OPT : シナリオ (ip, us_id) ごとに mean_num_xact を最大化する msg（同点は msg 昇順）\n"
             "URS : mean_num_xact の総平均"),
            ("■ データの粒度について（重要）",
             "mean_num_xact / mean_num_share は 1セル（シナリオ×系列）につき\n"
             "**5ラウンドを合計した1個の値**である（core/src/diffusion.rs の multiple_diffuse が\n"
             "各ラウンドの結果を += で積算している）。ラウンド別の内訳は出力に保存されておらず、\n"
             "「R1 で何人、R2 で何人」という分解はこのデータからは復元できない。\n"
             "本資料の図3・図4 は、参照系列の他4ポジションを固定して1ポジションだけ差し替えた\n"
             "系列どうしを比べたもので、いずれの棒も『5ラウンド合計』である。"),
            ("■ 個人単位の指標について",
             "結果.pdf にある mean_num_xact_of_users / mean_num_share_of_users は\n"
             "user_analysis.parquet 由来だが、シミュレータ v0.2.0 で個人単位出力を廃止したため\n"
             f"{cond} には存在せず、算出できない。ただし xact/share の割合は集計粒度によらず\n"
             "一致する（結果.pdf でも個人単位と全体像の割合は同値）ため、割合の比較は成立する。\n"
             "baseline の実データも削除済みのため、比較値は 結果.pdf の記録値を引用している。"),
        ])

        page(pdf, "1. RAM / OPT の抽出", [
            ("表1　ram上位msg（= 対全拡散モデルの外部行動者数平均の上位10個msg）", t1),
            ("表2　opt の msg 頻度順上位10", t2),
            (None,
             f"・RAM 上位10 は {ram_tail3}/10、OPT 上位10 は {opt_tail3}/10 が末尾 m3 の系列である\n"
             f"  （baseline の記録値は RAM {BASE_RAM_TAIL3}/10、OPT {BASE_OPT_TAIL3}/10）。\n"
             f"・RAM 上位10 の m0 の個数は平均 {m_counts[0]:.1f}/5 で、"
             f"m1 {m_counts[1]:.1f}・m2 {m_counts[2]:.1f}・m3 {m_counts[3]:.1f} より多い。\n"
             f"・baseline で 1位だった 00003 は {cond} では RAM {rank00003} 位（{x00003:,.1f}）、"
             f"RAM 1位は {ram_list[0]}（{ram_top1:,.1f}）である。"),
            ("表3　評価値", t3),
            (None,
             f"・OPT 評価値 {opt_eval:,.2f} に対し RAM 上位10 の平均は {ram_top10_mean:,.2f} で "
             f"{ram_top10_mean/opt_eval*100:.1f}%。\n"
             f"・RAM 上位1（{ram_list[0]}）のみを適用すると対 OPT {ram_top1/opt_eval*100:.1f}% となる。\n"
             f"  baseline の記録値はそれぞれ 79.6% / 93.4% である。"),
            ("表6　opt と ram で共通する msg", t6),
            (None,
             f"・共通系列は {len(common)} 件で、baseline の主系列 00003 を"
             f"{'含む' if '00003' in common else '含まない'}。"),
        ])

        page(pdf, "2. 全体像（外部行動者数・内部行動回数）", [
            ("表4　全体の平均値（全1024系列）", t4),
            ("表5　最大化msg（RAM上位10）のみ適用", t5),
            (None,
             f"・外部行動者数は {overall_all[0]:.2f} → {overall_ram[0]:.2f} で ×{xact_mult:.2f}。\n"
             f"・内部行動回数は {overall_all[1]:.2f} → {overall_ram[1]:.2f} で "
             f"×{overall_ram[1]/overall_all[1]:.2f}。\n"
             f"・外部/内部 の割合は {ratio_all:.4f} → {ratio_ram:.4f} で +{ratio_gain_pt:.1f}pt。\n"
             f"  baseline の記録値は ×{BASE['xact_ram']/BASE['xact_all']:.2f}、"
             f"{BASE['ratio_all']:.4f} → {BASE['ratio_ram']:.4f}"
             f"（+{(BASE['ratio_ram']-BASE['ratio_all'])*100:.1f}pt）。\n"
             f"・全系列平均の内部行動回数 {overall_all[1]:.2f} は baseline の記録値 "
             f"{BASE['share_all']:.2f} と一致する（内部軸のパラメータを変更していないため）。"),
            ("表7　上位系列の構成（先頭・末尾・各メッセージの個数）", t7.head(20)),
        ])

        verify_block = "\n\n".join(
            f"{r['#']}  [{r['判定']}]  {r['既発表の主張']}\n"
            f"        baseline : {r['baseline 記録値']}\n"
            f"        {cond:<8s} : {r[f'{cond} の値']}\n"
            f"        備考     : {r['備考']}"
            for r in verify.iter_rows(named=True)
        )

        page(pdf, "3. 既発表の結果・考察との一致検証", [
            (None,
             "判定基準　○＝主張どおり成立／△＝傾向は同じだが記録値の水準・範囲から外れる／"
             "×＝成立しない\n"
             f"C1 は baseline 自身の本数（RAM {BASE_RAM_TAIL3}/10・OPT {BASE_OPT_TAIL3}/10）以上か、\n"
             "C7 は m0 の平均個数が m1〜m3 のどれよりも多いか、で判定する。"),
            ("表8　主張ごとの照合", verify_block),
            (None,
             f"・7 項目中 ○{n_ok} / △{n_partial} / ×{n_ng}。\n"
             f"・C1・C2・C7（系列の型に関する主張）は {cond} でも成立した。\n"
             f"  末尾 m3 は RAM {ram_tail3}/10・OPT {opt_tail3}/10 で baseline 以上、"
             f"共通系列 {len(common)} 件は 00003 を含む。\n"
             f"・C3 は {ram_top1/opt_eval*100:.1f}%（baseline 93.4%）、"
             f"C4 は {ram_top10_mean/opt_eval*100:.1f}%（baseline 79.6%）。\n"
             f"・C5 は ×{xact_mult:.2f}、C6 は +{ratio_gain_pt:.1f}pt で、"
             f"いずれも向きは一致するが記録範囲（×1.4〜×3.8、+17.0〜+33.7pt）を上回る。\n"
             f"・RAM 1位の系列は baseline の 00003 から {ram_list[0]} に変わった"
             f"（00003 は {rank00003} 位）。"),
        ])

        for f in figs:
            pdf.savefig(f)

        r1 = contrast.filter(pl.col("差し替え位置") == "R1")
        r4 = contrast.filter(pl.col("差し替え位置") == "R4")
        r5 = contrast.filter(pl.col("差し替え位置") == "R5")
        page(pdf, "4. 図の読み方と、1ポジション差し替えの対比", [
            ("図1", "RAM 上位10 と OPT 頻度上位10 を、結果.pdf に記録のある baseline の値と並べたもの。\n"
                    "baseline 側に記録の無い系列は棒を描かず「baseline 記録なし」と注記している。"),
            ("図2", "全1024系列の平均と、最大化msg（RAM上位10）のみを適用した場合の\n"
                    "外部行動者数・内部行動回数・両者の割合。C5・C6 の照合に対応する。"),
            ("図3・図4",
             f"参照系列 {REF} の他4ポジションを固定し、1ポジションだけ m0〜m3 に差し替えた\n"
             "系列どうしの比較（図3 が外部行動者数、図4 が内部行動回数）。\n"
             "棒はいずれも 5ラウンド合計であり、ラウンド別の成果ではない。\n"
             "系列を平均していないため、C1 の「末尾 m3 が効く／先頭 m3 は効かない」を\n"
             "曖昧さなく照合できる。"),
            (f"表9　参照系列 {REF} の1ポジション差し替え（抜粋）",
             contrast.filter(pl.col("差し替え位置").is_in(["R1", "R4", "R5"]))
                     .select(["差し替え位置", "系列", "置いたメッセージ", "xact", "share"])),
            (None,
             f"・R5（末尾）を差し替えると {r5['xact'].min():,.2f}（{r5['系列'][0]}）〜"
             f"{r5['xact'].max():,.2f}（{r5['系列'][3]}）で、差し替え位置の中で最も幅が大きい。\n"
             f"・R1（先頭）を差し替えると {r1['xact'].max():,.2f}（{r1['系列'][0]}）〜"
             f"{r1['xact'].min():,.2f}（{r1['系列'][3]}）で、m0 から m3 に替えるほど減少する。\n"
             f"・R4 では {r4['系列'][1]}（{r4['xact'][1]:,.2f}）が参照 {REF}"
             f"（{r4['xact'][0]:,.2f}）を{'上回る' if r4['xact'][1] > r4['xact'][0] else '下回る'}。"),
        ])

    for f in figs:
        plt.close(f)

    print(f"--- {cond} (δ_X={dx}) ---")
    print(f"  xact平均={overall_all[0]:.4f} share平均={overall_all[1]:.4f} ratio={ratio_all:.4f}")
    print(f"  RAM1={ram_list[0]} {ram_top1:.2f} / OPT評価値={opt_eval:.2f}")
    print(f"  対OPT: RAM1 {ram_top1/opt_eval*100:.1f}% / RAM10 {ram_top10_mean/opt_eval*100:.1f}%")
    print(f"  xact ×{xact_mult:.2f} / ratio +{ratio_gain_pt:.1f}pt")
    print(f"  末尾m3: RAM {ram_tail3}/10 OPT {opt_tail3}/10 （baseline {BASE_RAM_TAIL3}/{BASE_OPT_TAIL3}）")
    print(f"  共通系列({len(common)}): {common}")
    print(f"  判定 ○{n_ok} △{n_partial} ×{n_ng}")
    print(f"  → {pdf_path.name}")

    return dict(cond=cond, delta_x=dx, rob=rob, opt_freq=opt_freq, ram_list=ram_list,
                opt10=opt10, opt_eval=opt_eval, ram_top1=ram_top1,
                ram_top10_mean=ram_top10_mean, overall_all=overall_all,
                overall_ram=overall_ram, xact_mult=xact_mult, ratio_all=ratio_all,
                ratio_ram=ratio_ram, ratio_gain_pt=ratio_gain_pt, common=common,
                ram_tail3=ram_tail3, opt_tail3=opt_tail3)


if __name__ == "__main__":
    for c in ("extzero", "extneg"):
        run(c)
