# CLAUDE.md — `analysis/`

`twitter.ipynb` が行う統計処理のアーキテクチャ定義。
このファイルは **ノートブック `twitter.ipynb` のセルだけ** を根拠に、そこで実行されている
統計処理の構造・用語・算出式を記述したもの。

> 前提: `analysis/` 配下に別実装（パッケージ等）があっても、この定義は
> `twitter.ipynb` 単体を仕様とみなす。両者が食い違う場合、ここではノートブックを正とする。
>
> 参照の分担: このファイルは **統計処理の実装**。研究の背景・結果の数値・考察・発表資料
> （要旨 / スライド / ポスター / 論文）の執筆はリポジトリルートの `../claude-research.md` を参照。

---

## 1. 目的

Rust シミュレータが総当たり探索で出力した parquet を読み、
**「どのメッセージ系列 `msg` を送るべきか」という意思決定ポリシーの良し悪し** を、
2 つの観点で評価する。

1. **ポリシー価値評価** … シナリオ全体での取引期待値。OPT / RAM / URS を比較。
2. **ユーザ行動集計** … 個々のユーザの取引数・シェア数が、RAM 適用でどう変わるか。

---

## 2. 入力データ

| 変数 | ファイル | 読み方 | 列 | 行数 |
|---|---|---|---|---|
| `df` | `twitch_DE.parquet` | `pl.read_parquet`（eager）| `id, ip, us_id, msg, mean_num_xact` | 524,288 |
| `user_analysis` | `twitch_DE_user_analysis.parquet` | `pl.scan_parquet`（**lazy 必須**）| `ip, us_id, msg, user_id, num_xact_of_users, num_share_of_users, distance_from_ip` | 巨大（2^32 行超）|

> **⚠️ シミュレータ v0.2.0 以降の出力には `user_analysis.parquet` が存在しない**
>
> Rust 側で個人単位の明細出力を廃止し（実行時間とディスク使用量の大半を占めていたため）、
> 代わりに result 側へ **`mean_num_share`**（T ラウンド合計シェア回数の MC 平均）を追加した。
> 新スキーマは `id, ip, us_id, msg, mean_num_xact, mean_num_share`。
> 新旧の判別は parquet の KV メタデータ `version_runner`（0.1.x か 0.2.0 以降か）で行う。
>
> - **トラック2の aggregate 粒度**は `mean_num_share` で直接代替できる
>   （従来 `group_by(ip, us_id, msg).sum(num_share_of_users)` で作っていた値と一致する。
>   整数で合計してから 1 度だけ割るぶん、丸め誤差はむしろ小さい）。
> - **トラック2の individual 粒度**（1 人あたりの平均行動回数）は新しい出力では算出できない。
> - このノートブックが読んでいる既存の `twitch_DE_user_analysis.parquet` は v0.1.x の
>   出力なのでそのまま読める。過去の結果は失われていない。

### 列の意味

| 列 | 意味 |
|---|---|
| `id` | シナリオ id（0..511）。`(ip, us_id)` の組に対応 |
| `ip` | 情報源ノード id。16 種 |
| `us_id` | ユーザ状態分布 id。32 種（16 × 32 = 512 シナリオ）|
| `msg` | メッセージ系列。`num_rounds = 5` 桁、各桁 `0..3` の文字列（例 `"00003"`）。全 `4^5 = 1024` 通り。id 0 = 内部刺激最強、id 3 = 外部刺激最強 |
| `mean_num_xact` | そのシナリオ×`msg` での取引回数のモンテカルロ平均。**大きいほど良い**（目的量）|
| `num_xact_of_users` / `num_share_of_users` | 個々のユーザの取引数 / シェア数。行動が非ゼロのユーザ行のみ格納 |
| `distance_from_ip` | ユーザと `ip` のグラフ距離（本ノートブックでは未使用）|

`df` は 512 シナリオ × 1024 msg = 全数。`user_analysis` はさらにユーザ次元が付く。

---

## 3. 中心概念 — 3 つのベンチマークポリシー

「シナリオ `(ip, us_id)` を観測できる度合い」が異なる 3 戦略を比較する。

| 記号 | 定義（ノートブック上の算出）| 観測前提 |
|---|---|---|
| **OPT** | シナリオごとに `mean_num_xact` を最大化する `msg`（`lf_opt`）| シナリオを完全観測（上限）|
| **RAM** | 全シナリオでの `msg` 別平均 `mean_num_xact` が最大の `msg`（`lf_rob`）。上位 10 件の集合を `ram_msg_list` とする | シナリオを観測しない固定戦略 |
| **URS** | 一様ランダムに `msg` を選ぶ = `mean_num_xact` の総平均（`lf_urs`）| 情報なし（下限）|

### `eval_df`（cell 3・5）が出す 8 指標

| key | 定義 |
|---|---|
| `ev_by_opt` | OPT が選んだ `msg` を全シナリオに反実仮想適用し、`(ip,us_id)` 平均 → さらに総平均（`lf_opt_mod`）|
| `mean_of_opt_values` | 各シナリオで OPT が達成した最適値 `max_mean_num_xact` の平均 |
| `ev_by_ram` | `msg` 別平均 `mean_num_xact` の最大値 |
| `ev_by_urs` | `mean_num_xact` の総平均 |
| `ram ev / opt ev` | `ev_by_ram / ev_by_opt` |
| `urs ev / opt ev` | `ev_by_urs / ev_by_opt` |
| `ram ev / opt mean` | `ev_by_ram / mean_of_opt_values` |
| `urs ev / opt mean` | `ev_by_urs / mean_of_opt_values` |

`ev_by_opt` と `mean_of_opt_values` の違い:
- `mean_of_opt_values` = 各シナリオで「正しい `msg`」を当てたときの値（真の上限）
- `ev_by_opt` = 「最適な `msg` 群のどれかを、シナリオを知らずに撃つ」ときの期待値（反実仮想）

---

## 4. アーキテクチャ

処理は **2 つの独立トラック** と、それを繋ぐ **1 つのブリッジ** からなる。

```
                       twitch_DE.parquet  (df, 小さい・eager)
                                │
        ┌───────────────────────┴───────────────────────────┐
        ▼ トラック1: ポリシー価値評価 (result レベル)         │
        │                                                   │
        │  lf_opt          (ip,us_id) ごとの argmax msg      │
        │    ├─► eval_df    OPT/RAM/URS の値 + 4 比率  (cell 3,5)
        │    ├─► lf_opt_list   argmax msg の頻度      (cell 17)
        │    └─► opt_len_mean  argmax msg 別の平均値  (cell 19)
        │                                                   │
        │  lf_rob_details  msg 別平均 mean_num_xact 降順 (cell 7,8)
        │    ├─► cell 9   総平均 (= URS)                     │
        │    ├─► cell 10  上位10平均                         │
        │    └─► ram_msg_list  上位10 msg           ─────────┼──┐  ★ブリッジ
        │                                                   │  │
        ▼                                                   │  │
   twitch_DE_user_analysis.parquet (user_analysis, 巨大・lazy)│  │
        │                                                   │  │
        ▼ トラック2: ユーザ行動集計 (user レベル)            ◄──┘
        │   scope = all（全 msg） / ram（ram_msg_list に絞る）
        │   grain = individual（非ゼロ行を直接平均）
        │         / aggregate（(ip,us_id,msg) で sum → msg で mean → mean）
        │
        ├─ individual × all   (cell 21,22)   0.109 / 0.151 / 0.724
        ├─ individual × ram   (cell 24)      0.152 / 0.163 / 0.937
        ├─ aggregate  × all   (cell 26,27)   40.87 / 56.45 / 0.724
        └─ aggregate  × ram   (cell 28,30)   87.83 / 93.74 / 0.937
```

### トラック1 — ポリシー価値評価（`df` レベル）

| 中間物 | 定義 | cell |
|---|---|---|
| `lf_opt` | `mean_num_xact == max().over(["us_id","ip"])` で filter → `unique(["ip","us_id","mean_num_xact"])` → 列 `[us_id, ip, argmax_msg, max_mean_num_xact]`。同点は 1 つに落とす | 3 |
| `lf_opt_mod` | `df` を `argmax_msg` で join → `(ip_right,us_id_right)` で `mean_num_xact` 平均 → `lf_opt` に戻し join → 総平均。`ev_by_opt`, `mean_of_opt_values` を得る | 3, 4 |
| `lf_rob` | `group_by("msg").agg(mean)` の最大値 = `ev_by_ram` | 3 |
| `lf_urs` | `mean_num_xact` 総平均 = `ev_by_urs` | 3 |
| `eval_df` | 上記を横結合 → 4 比率を付与 → `transpose` して `key`/`value` の 2 列 | 3, 5 |
| `lf_rob_details` | `group_by("msg").agg(mean).sort(desc)`。1024 行。RAM の基礎統計 | 7, 8 |
| `ram_msg_list` | `lf_rob_details[:10]["msg"]`。RAM 対象 msg 集合（**ブリッジ出力**）| 13, 14 |
| `lf_opt_list` | `lf_opt` を `argmax_msg` で `group_by().len()` → どの msg が何シナリオで最適だったか | 17 |
| `opt_len_mean` | `argmax_msg` 別に `max_mean_num_xact` 平均と件数 | 19 |

`df` は 52 万行と小さいので eager で処理してよい。

### トラック2 — ユーザ行動集計（`user_analysis` レベル）

2 軸で 4 通りを出す。指標は毎回 `mean_num_xact_of_users` / `mean_num_share_of_users` /
その比 `xact_share_ratio`。

**軸1 — scope**
- `all` … 全 `msg`
- `ram` … `filter(pl.col("msg").is_in(ram_msg_list))`（**collect 前に適用**）

**軸2 — grain（集計粒度）**
- `individual`（cell 21/24）… 非ゼロのユーザ行に対して `num_xact_of_users` / `num_share_of_users` を直接平均
- `aggregate`（cell 26/28）… `group_by(["ip","us_id","msg"]).sum()` → `group_by("msg").mean()` → 列平均

比率 `xact_share_ratio` は grain によらず一致する（`all` で 0.723865、`ram` で 0.936866）。
値の水準は grain で変わるが、xact と share が同じ集計を通るため比は不変。

`user_analysis` は 2^32 行を超えるため:
- 必ず `pl.scan_parquet`（LazyFrame）。全件 `collect` しない
- `scope="ram"` の `is_in` フィルタは `collect` 前に push down する
- `collect(engine="streaming")` を使う（cell 26/28）

---

## 5. データフロー（依存順）

```
1. df = read_parquet(twitch_DE)                                  … cell 1
2. lf_opt ← df                                                   … cell 3
3. eval_df ← lf_opt, df   (ev_by_opt/ram/urs + 4 ratio)          … cell 3,5
4. lf_rob_details ← df    (msg 別平均, 降順)                       … cell 7,8
5. ram_msg_list ← lf_rob_details[:10]                            … cell 13   ★
6. lf_opt_list / opt_len_mean ← lf_opt   (argmax msg の分布)      … cell 17,19
7. user_analysis = scan_parquet(...user_analysis)               … cell 11
8. individual×all  ← user_analysis                              … cell 21,22
9. individual×ram  ← user_analysis, ram_msg_list                … cell 24
10. aggregate×all  ← user_analysis                              … cell 26,27
11. aggregate×ram  ← user_analysis, ram_msg_list                … cell 28,30
```

`ram_msg_list`（手順 5）がトラック1からトラック2への唯一の受け渡し。

---

## 6. セル対応表

| cell | 生成物 | 種別 |
|---|---|---|
| 1 | `df`, `df.csv`, parquet メタ表示 | 入力 |
| 3 | `lf_opt`, `lf_opt_mod`, `lf_rob`, `lf_urs`, `eval_df` | トラック1 |
| 4 | `lf_opt_mod` の再掲（日本語コメント付き検算）| トラック1 |
| 5 | `eval_df` 表示 | トラック1 |
| 7,8 | `lf_rob_details` | トラック1 |
| 9 | `lf_rob_details` 総平均（= URS）| トラック1 |
| 10 | `lf_rob_details` 上位10平均 | トラック1 |
| 11 | `user_analysis`（scan）| 入力 |
| 13,14 | `ram_msg_list`, `ram_msg_us_an` | ★ブリッジ |
| 16 | `opt`（`lf_opt` を最適値降順）| トラック1 |
| 17 | `lf_opt_list`（argmax msg 頻度）| トラック1 |
| 19 | `opt_len_mean` | トラック1 |
| 21,22 | individual × all の xact/share/比率 | トラック2 |
| 24 | individual × ram | トラック2 |
| 26,27 | aggregate × all | トラック2 |
| 28 | `ram_xact_share`（ram, msg 別）| トラック2 |
| 30 | aggregate × ram の平均・比率 | トラック2 |

---

## 7. 再現・改修時の注意

- `pl.concat(..., how="horizontal")` は将来 `horizontal_extend` が必要になる旨の
  DeprecationWarning が出る（cell 3,22,24,27,30）。1 行同士の横結合なので機能上は問題ない。
- `df.write_csv("df.csv")`（cell 1）は `msg` の先頭ゼロ（`"00003"`）を読み戻すと
  数値化される。`msg` を厳密に扱うなら parquet のまま保持する。
- RAM 集合は「`lf_rob_details` 上位 **10**」固定（cell 13 の `[:10]`）。閾値化や k 変更をするなら
  ここを起点にする。
- OPT の同点処理は `unique(["ip","us_id","mean_num_xact"])` → その後 join で 1 つに絞られる。
  `mean_num_xact` が完全一致する複数 `msg` があると、どれが残るかは非決定的。
- `user_analysis` を触る処理は LazyFrame のまま組み立て、`collect(engine="streaming")` は
  最後の 1 回。`scope="ram"` は必ず `collect` 前に `is_in` フィルタ。
