# 2026-10-26 推薦チャネル（Rモデル）の実装

## 依頼と決定
- 依頼：`reccomend_system/推薦システム単純.md` の3節で定義した推薦チャネル（Rモデル）を diffusion_seokyeong の Rust に入れる。設計仕様は `研究リソースとか/`、ログはこの `log/` に置く。
- 決定：
  - 推薦はオフが既定で、オフのときは既存の結果とビット一致させる（`diffuse` の引数は変えず、`diffuse_with_rec` を足した）。
  - 推薦の乱数は、拡散と同じ系列ごとの `SmallRng` から引く（オフでは1つも引かないので一致は保たれる）。
  - 起点と1歩目は、シェア者の出辺を1回の乱数で選んでまとめた（分布は同じ）。
  - 出力は parquet に2列（`mean_num_rec_rcvd`、`mean_num_xact_rec`）と KV メタデータ `recommender` を足した。core・runner は 0.3.0 に上げた。
  - 仮定：$\bar e_0=0$ なら推薦しない。
  - `CLAUDE.md` と `研究リソースとか/claude-research.md` は、方針（自発的に更新しない）により触っていない。

## 実施
- 手順0：変更前のバイナリで、小さな設定（`log/work/config_small.toml`：congress、IP 2、ユーザ集合 2、T = 3、$p=0.1$、MC 20）の parquet を記録した（`log/work/before_off.parquet`）。
- core：`core/src/recommend.rs` を新設（`RecParams`、`Recommender`、`base_share_rate`、`two_step_walk`）。`core/src/diffusion.rs` に `diffuse_with_rec` を足し、`multiple_diffuse` と `monte_carlo_average` に `rec` 引数を足した。`DiffusionResult`・`Aggregator`・`Workspace` を拡張した。
- runner：`Config` に `#[serde(default)] recommender` を足し、`validate()` に範囲の検査を足した。$\bar e_0$ はユーザ状態集合ごとに計算して `JobChunk` に持たせる。parquet の列と KV メタデータを足した。
- 設定例：`test/config_rec.toml`。
- 文書：`研究リソースとか/推薦チャネル_実装仕様.md`。

## 確認
- `cargo test --workspace`：32件すべて成功（core 21 = 既存13 + 追加8、runner 11 = 既存8 + 追加3）。
- 変更前と変更後（推薦なし）の parquet：既存6列が256行すべて完全に一致（`id, ip, us_id, msg` で並べ替えて比較）。追加2列はすべて 0。
- 推薦オン（同じ小さな設定、先頭のメッセージ別に平均した T = 3 ラウンド合計）：

  | ε | 先頭 m0 の外部行動 | 先頭 m3 の外部行動 | 先頭 m0 の推薦経由の受信 |
  | --- | --- | --- | --- |
  | 0（なし） | 2.25 | 1.75 | 0 |
  | 0.5 | 34.21 | 8.12 | 98.78 |
  | 1.0 | 90.00 | 23.14 | 234.29 |
  | 2.0 | 170.50 | 54.76 | 424.64 |

  ε を上げるほど到達が増え、内部刺激の強い m0 から始める系列ほど伸びが大きい（元資料 4.2節の「強弱の差が広がる」と同じ向き）。
- `test/config_rec.toml` を縮めた設定（ENGB、IP 1、ユーザ集合 1、T = 2）でも読み込み・実行できた。

## つまずき
- S-04 で、`DiGraphB::from(&[(…)])` がラベルを 0 から詰め直すことを忘れて、ラベル 10〜13 を使い失敗した。ラベルを 6〜9 に直した。
- ENGB の確認で値がすべて 0 になった。抽選された IP（6608）のフォロワーが0人だったため。推薦量はシェア者の出次数に比例するので、フォロワーのいない IP には推薦が付かない。仕様書の限界に書いた。

## 次
- 図1（δ を変えたときの動き）を、この実装で描き直す（`reccomend_system/推薦システム単純.md` 3.8節）。
- 実データ（別 PC の ENGB・Twitter など）で、典型系列 × ε・θ・τ のスクリーニング（元資料 8節）。
- ビューア側の実装（`information_diffusion_veiwer/recommender-v3-design.md`）。
- `log/work/` は作業用の一時ファイル（git に入れない）。不要になったら消してよい。
