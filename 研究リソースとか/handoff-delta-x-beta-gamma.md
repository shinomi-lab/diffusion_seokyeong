# SNS拡散シミュレーション — δ_X検証・v0.2.0リファクタ・β/γ設計 — 引き継ぎブリーフィング

> この文書は前のセッションから引き継いだブリーフィングです。
> 以下の内容をそのまま前提として、「次にやること」から作業を続けてください。
> 改めて説明を求めず、ここに書かれた文脈を踏まえてすぐ続行してください。

## 目標

`xa-promote`（Rust製SNS拡散シミュレータ）を使って、研究の中心的主張
**「拡散を促す投稿を前半に多め、外部行動を促す投稿を最後に置く順序が効果的」**
を、メッセージ空間のパラメータ（δ_X＝外部軸の差別化度）を変えても成り立つか検証してきた。
現在は、次の一手として **意思決定モデルの反応性パラメータ（β, γ）を調整し、
「反応の速さ」と「感度の裾の重さ」を操作して同じ順序仮説を検証する**、
その config 設計の途中で会話が中断した。

リポジトリ: `/Users/hongseokyeong/Desktop/rustモデル/diffusion_seokyeong`
（git管理下。他PCとはこのリポジトリの同期で連携している）

## 背景・前提

### ドメインモデルの要点（実装ベース）

- `core/src/decision.rs`: 効用更新則 `u' = w*u + (1-w)*s`。`DecisionAny`（内部行動＝シェア、
  何度でもON/OFFを往復）と`DecisionOnce`（外部行動＝取引、生涯1回のみ）の2種類。
- `core/src/message.rs`: `MessageSpace`は`TryFrom<Vec<Message>>`で構築（刺激値`[-1,1]`を検証）。
  メッセージ系列は`seq_by_index`/`seq_into`/`index_to_string`で処理、**桁0＝ラウンド1（先頭）**。
- `core/src/diffusion.rs`: `Workspace`構造体でバッファ再利用。`diffuse()`内の
  `(rcvd[j]==false) & (prop_prob > rng.random())` の **`&`（非短絡）は意図的に維持**
  （`&&`にすると乱数消費順序が変わり、過去の実行結果とのビット一致が崩れるため）。
- `runner/src/brute_force/config.rs`: `DecisionParameters::sample_iter`が
  **反応性パラメータの核心**（詳細は後述）。`Config::validate()`で
  `gamma < beta`・確率範囲・刺激値範囲・`log2(L)*T<=64`等を検証、違反はpanicでなく
  分かりやすいエラーで停止する。

### 反応性パラメータ β, γ の数式（今回導出・数値検証済み）

`sample_iter`の実装:
```rust
let scale = 1.0 / (self.beta * (num_rounds as f64));
let shape = self.gamma / (self.beta - self.gamma);
Pareto::<f64>::new(scale, shape)...
```
responsiveness `µ ~ Pareto(x_m = 1/(β·T), α = γ/(β−γ))`、
`w = (s̄/(s̄−u0))^µ`、必要ラウンド数 `n = 1/µ`。

導出した閉じた式（数値シミュレーションで小数3桁まで一致確認済み）:

| 量 | 式 | 現行 (β=0.8, γ=0.6, T=5) |
|---|---|---|
| **最大必要ラウンド** | `n_max = β·T` | 4.00 |
| **裾の重さ（α）** | `α = γ/(β−γ)` | 3.00 |
| 平均必要ラウンド | `E[n] = n_max·α/(α+1)` | 3.00 |
| 中央値 | `n_max / 2^(1/α)` | 3.18 |

**2つの独立な操作軸に分離できる**：
- β だけが「速さ」を決める → `β = n_max / T`
- γ は β と組で「裾」を決める → `γ = α·β/(1+α)`
- **γ単独ではなく (n_max, α) を先に決めてβ,γに変換する**のが正しい設計手順。
- **γ < β は`Config::validate`が強制**。PDF#2（`メッセージ空間と反応性パラメータ_解説.pdf`,
  `内部軸の平均刺激を上げた場合_シミュレーション.pdf`）が示唆した「γを2〜3に上げる」は
  β=0.8のままでは不可能（β,γを同時に上げる必要がある）。
- 現行のE[n]=3.0（正しい式で算出）なので、**PDF#2が懸念した内部飽和はそもそも
  実装コードでは起きていない**ことを前セッションで実証済み
  （`内部飽和現象の検証_確定版.pdf`参照）。

### 累積発火率シミュレーション結果（m0, s=0.20を毎ラウンド受け続けた場合）

| 条件 | β | γ | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|---|
| n_max=2, α=1.5 | 0.4 | 0.24 | 0.479 | **1.000** | 1.000 | 1.000 | 1.000 |
| n_max=2, α=3 | 0.4 | 0.30 | 0.230 | **1.000** | 1.000 | 1.000 | 1.000 |
| n_max=4, α=1.5 | 0.8 | 0.48 | 0.169 | 0.480 | 0.882 | 1.000 | 1.000 |
| **n_max=4, α=3（現行）** | 0.8 | 0.60 | 0.029 | 0.231 | 0.779 | 1.000 | 1.000 |
| n_max=4, α=8 | 0.8 | 0.711 | 0.000 | 0.020 | 0.512 | 1.000 | 1.000 |
| n_max=7, α=3 | 1.4 | 1.05 | 0.005 | 0.043 | 0.145 | 0.344 | **0.672** |
| n_max=7, α=8 | 1.4 | 1.244 | 0.000 | 0.000 | 0.006 | 0.059 | **0.347** |

読み取れること：
- **α は「R1で動く少数層の厚さ」をほぼ直読みで決める**（活性層の偏りの操作子）
- **n_max=2は完全に飽和**（R2で全員）。差別化の窓が消える
- **n_max=7はT=5を超え**期間内に飽和しない
- 現行n_max=4は「Tの中でちょうど飽和する」作業点

## これまでに下した決定

- **PDF#2「内部飽和」説の否定と原因特定**: 独立検証スクリプトのPareto scaleパラメータが
  実装（`1/(beta*num_rounds)`）と異なり（当該資料は`1/beta`を使用、T=5倍の差）誤った結論に
  至った可能性が高いと判明。実装コードを直接呼ぶ`runner/examples/saturation_probe.rs`で
  s̄=0.125時点で最強メッセージの1発発火率は約3%と実証（飽和には程遠い）。
- **代用グラフの使用**: ENGB等の実データがチェックアウトに無いため、
  `test/congress_network_graph.txt`（475ノード、有向）で代用。ユーザー確認済み。
- **δ_X拡張2条件の追加**: `test/config_extzero.toml`（δ_X=1.0、m0の外部刺激=0）、
  `test/config_extneg.toml`（δ_X=1.4、m0の外部刺激=-0.05）。両方とも
  `s̄_X=0.125`を維持しbaselineとw較正を同一に保つ設計（δの効果だけを単離）。
- **出力スキーマの変更（v0.2.0）**: `user_analysis.parquet`（個人単位、370MB/条件）を廃止、
  `result.parquet`に`mean_num_share`列を追加。理由：実行時間・ディスクの大半を占めていたため。
  **旧バイナリとの照合でmean_num_xactがビット単位で完全一致**することを検証済み
  （乱数消費順序を変えなかったため）。
- **Rustワークスペース全体の改修**（監査5項目、全て対応）:
  1. parquet書き込みエラーの黙殺を修正（`try_join!`のResultを`?`で伝播）
  2. `.unwrap()`18箇所を`anyhow`+`Context`でエラーメッセージ付きに変更
  3. `Config::validate()`追加（`gamma<beta`、確率範囲、`ip_sampling.size<=ノード数`等）
  4. メッセージ系列文字列の二重生成を`digits()`共通イテレータで解消
  5. テスト0→21件（`core`13件、`runner`8件）
  - `core`/`runner`のバージョンを0.1.1→**0.2.0**に上げ（出力parquetのKVメタデータで新旧判別可能）
- **ジョブのチャンク化と自動並列度調整**: `--chunk-size`オプション追加、
  `effective_chunk_size()`で系列数と並列度に応じて自動縮小。
  **注意**: 最初の実装では系列数が少ない設定（例：T=2, L=4→16系列）で
  全系列が1チャンクに収まり並列度を失うバグがあり、ベンチマークで発見・修正した。
- **図3/4の設計をユーザー指摘で修正**: 当初「ラウンド位置×メッセージのヒートマップ」を
  作ったが、ユーザーから「各セルは事後的な5ラウンド合計であって、各ラウンドの状態を
  示すのは不可能では」と指摘され、正しい指摘と認めて撤回。
  `multiple_diffuse`は`+=`で5ラウンドを積算するのみでラウンド別内訳は保存されないため。
  → **「参照系列00003の他4ポジションを固定し、1ポジションだけm0〜m3に差し替える」対比図**
  に置き換えた（平均を挟まないので曖昧さがない）。データ粒度の注記もPDF冒頭に追加。
- **C1・C7の判定基準の誤りを修正**: 当初C1「RAM/OPT上位10が10/10末尾m3」を○の条件にしたが、
  `結果.pdf`のbaseline記録値を数えると**baseline自身がRAM 7/10・OPT 8/10で10/10を満たさない**
  過剰に厳しい基準だった。→ **baselineの本数以上なら○**に変更。C7も
  「m0の平均個数2.5個以上」という恣意的閾値をやめ、**「m0がm1〜m3のどれよりも多い」**という
  相対基準に変更。この修正に伴いextneg資料も作り直した（数値は前回と完全一致、判定根拠のみ変更）。
- **共通スクリプト化**: `analyze_extneg.py`を`analyze_ext.py`に一般化、
  `CONDITIONS`辞書で条件を切替。1回の実行で複数条件を生成する設計。
- **δ_X横断比較は独立した別資料にする**: extzero結果資料とは別に
  `delta_x_横断比較.pdf`を作成（ユーザー選択）。baselineは記録値のみで全1024系列を
  持たないため、比較可能な範囲を明示し、記録の無い箇所は「記録なし」と注記。
- **作図パレット**: `dataviz`スキルの検証済みパレットを使用。
  m0..m3（順序付き4値）は単一色相ordinalランプ`#86b6ef/#3987e5/#256abf/#104281`、
  2条件比較はcategorical slot1`#2a78d6`/slot2`#eb6834`、3条件（δ_X横断）は
  ordinalランプの3段`#86b6ef/#3987e5/#104281`。いずれも
  `node scripts/validate_palette.js`で検証済み（ALL CHECKS PASS）。
- **「データは数値・統計・図のみ、解釈・考察は書かない」という制約**が
  一部の資料（`twitter_extneg_データ資料.pdf`、後に破棄）で課された。
  現行の`twitter_{cond}_結果.pdf`と`delta_x_横断比較.pdf`は
  「事実ベースの短い説明」（結果.pdfと同じ粒度）を許容する方針に戻っている。

## 現在の状態

### 完了

- `研究リソースとか/内部飽和現象の検証_確定版.pdf`（7ページ）＋
  `研究リソースとか/internal_saturation_check/`（configs/scripts/outputs一式）
- `core/src/decision.rs`, `core/src/diffusion.rs`, `core/src/message.rs`,
  `runner/src/brute_force.rs`, `runner/src/brute_force/config.rs`, `util/src/lib.rs`
  の改修完了。`cargo build --release`警告ゼロ、`cargo test --workspace`全21件PASS。
  `core`/`runner`は**バージョン0.2.0**。
- `runner/examples/saturation_probe.rs`（意思決定モデル単体の診断バイナリ、
  weight/utilityのgetterを`core/src/decision.rs`に追加して利用）
- `test/config_extzero.toml`, `test/config_extneg.toml`（両方とも
  `graph.path = "ENGB_edges.csv"`のまま。実データ配置後にそのまま使える）
- `twitter_extzero`, `twitter_extneg`（リポジトリ直下、拡張子なしparquet、
  `test/twitter_dataset.txt`81,306ノードで別PC実行済み、`version_runner=0.2.0`、
  524,288行=512シナリオ×1024系列、欠損0）
- `研究リソースとか/twitter_extzero_結果.pdf`（9ページ、最新版・C1/C7修正反映済み）
- `研究リソースとか/twitter_extneg_結果.pdf`（9ページ、最新版・C1/C7修正反映済み・図3/4修正済み）
- `研究リソースとか/delta_x_横断比較.pdf`（6ページ）
- `研究リソースとか/ext_analysis/analyze_ext.py`（両条件生成する共通スクリプト）
- `研究リソースとか/ext_analysis/compare_delta_x.py`（横断比較スクリプト）
- `研究リソースとか/{extzero,extneg}_結果/{tables,figures}/`、
  `研究リソースとか/delta_x_横断比較/{tables,figures}/`（CSV・PNG一式）

### 進行中・未着手（ここから続ける）

- **β/γ（反応性パラメータ）を動かした新条件のconfig設計** ← 直前の応答で提案し、
  会話が中断した箇所。まだconfigファイルは1つも作っていない、実行もしていない。
- スクリーニング用の`saturation_probe.rs`をβ,γ掃引に拡張する作業も未着手。

## 核心データ・提案内容（原文保存）

### 提案した実験条件（次にconfig化する候補）

メッセージ空間は**baseline（`test/config.toml`、δ_S=δ_X=0.6）に固定**し、
β,γだけを動かす。現行条件がA・Bの中心を兼ねるので、新規は5＋2＝7条件。

**A. 速さのスイープ（軸対称、α=3固定）**

| ID | n_max | β | γ | 狙い |
|---|---|---|---|---|
| R-fast | 2 | 0.4 | 0.30 | 飽和側。順序効果が消えるか |
| （現行） | 4 | 0.8 | 0.60 | 参照 |
| R-slow | 7 | 1.4 | 1.05 | 未飽和側。末尾m3が1発では足りなくなるか |

**B. 裾のスイープ（軸対称、n_max=4固定）**

| ID | α | β | γ | 狙い |
|---|---|---|---|---|
| T-heavy | 1.5 | 0.8 | 0.48 | 少数の超高感度層が牽引 |
| （現行） | 3 | 0.8 | 0.60 | 参照 |
| T-light | 8 | 0.8 | 0.7111 | ほぼ均質。全員が同じ蓄積を要する |

**C. 軸非対称（本命）** — `internal_parameters`と`external_parameters`は
別々に指定可能。PDF#1が「外部行動の起こしにくさはs̄_Xではなくbeta/gammaで
表現するしかない」と指摘した点を直接検証できる。

| ID | 内部 (β,γ) | 外部 (β,γ) | 予測 |
|---|---|---|---|
| **X-slow** | 0.4, 0.30（n_max=2 速い）| 1.4, 1.05（n_max=7 鈍い）| リーチは即座に完成し変換だけが蓄積を要する → **順序効果が最大化**するはず |
| **X-fast** | 1.4, 1.05（鈍い）| 0.4, 0.30（速い）| いつm3を打っても即変換でき、リーチ構築が間に合わない → **順序効果が消える**はず |

C（X-slow / X-fast）はどちらかが外れれば「拡散多め→外部寄り」の因果説明が
間違っていることになる、最も強い検証形。**優先度は C > A(R-slow) > B**。

### 進め方の提案（未実施）

1. **決定モデル単体でスクリーニング**（数秒）— `runner/examples/saturation_probe.rs`を
   β,γ掃引に拡張し、上の発火率表を実コードで全条件ぶん出す。退化条件
   （差別化の窓が消えるもの）を先に落とす。
2. **残った条件だけフル実行** — `研究リソースとか/ext_analysis/analyze_ext.py`に
   条件を1行足すだけで、末尾m3の本数・m0の平均個数・共通系列・R5差し替えの効き幅・
   対OPT比がそのまま出る。

### baseline記録値（δ_X=0.6、`結果.pdf`・`claude-research.md`§5.1より、Twitter）

| 指標 | 値 |
|---|---|
| xact全系列平均 | 226.4147 |
| share全系列平均 | 357.8583 |
| RAM1位 | 00003 (1000.26) |
| RAM上位10平均 | 852.93 |
| OPT評価値 | 1071.12 |
| RAM単一/OPT | 93.4% |
| RAM上位10平均/OPT | 79.6% |
| 最大化msg適用後xact | 852.93（×3.77）|
| 割合向上 | 0.6327→0.8905（+25.8pt）|
| RAM上位10末尾m3 | 7/10 |
| OPT上位10末尾m3 | 8/10 |

RAM上位10（baseline）: `00003,00013,00023,10003,00033,00002,10013,00012,00022,00103`
OPT上位10（baseline）: `00003,10003,20003,00013,00002,30003,00023,00033,10002,01003`

### extzero（δ_X=1.0）・extneg（δ_X=1.4）の主要数値（3条件横断表、実測済み）

| 指標 | baseline (0.6) | extzero (1.0) | extneg (1.4) |
|---|---|---|---|
| xact全系列平均 | 226.4147 | 215.1383 | 227.7539 |
| share全系列平均 | 357.8583 | 357.8583 | 357.8583 |
| 外部/内部の割合（全系列） | 0.6327 | 0.6012 | 0.6364 |
| RAM1位の系列 | 00003 | 00013 | 00013 |
| RAM1位のxact | 1000.26 | 1098.56 | 1241.75 |
| 00003のRAM順位 | 1 | 2 | 4 |
| OPT評価値 | 1071.12 | 1219.08 | 1400.79 |
| RAM単一/OPT | 93.4% | 90.1% | 88.6% |
| RAM上位10平均/OPT | 79.6% | 78.8% | 78.4% |
| xactの倍率（最大化msg適用） | ×3.77 | ×4.46 | ×4.82 |
| 割合の向上 | +25.8pt | +57.8pt | +87.7pt |
| OPT・RAMの共通系列数 | 1 | 6 | 7 |
| 末尾m3（RAM上位10） | 7/10 | 9/10 | 10/10 |
| 末尾m3（OPT上位10） | 8/10 | 9/10 | 10/10 |

extzero・extneg間の系列別xact相関（全1024系列）: **0.9886**

### config_extzero.toml / config_extneg.tomlのmessage_space（実データ）

```toml
# config_extzero.toml（δ_X=1.0）
message_space = [
{ internal_stimulus = 0.20, external_stimulus = 0.0    },
{ internal_stimulus = 0.15, external_stimulus = 0.0833 },
{ internal_stimulus = 0.10, external_stimulus = 0.1667 },
{ internal_stimulus = 0.05, external_stimulus = 0.25   },
]
num_rounds = 5
receipt_probability = 0.5
internal_parameters = { beta = 0.8, gamma = 0.6, initial_utility = -1.0 }
external_parameters = { beta = 0.8, gamma = 0.6, initial_utility = -1.0 }
graph = { path = "ENGB_edges.csv", directed = true, transposed = true, format = "csv" }
ip_sampling = { size = 16, seed = 0 }
user_sampling = { size = 32, seed = 0 }
diffusion_sample_size = 100
```

```toml
# config_extneg.toml（δ_X=1.4）
message_space = [
{ internal_stimulus = 0.20, external_stimulus = -0.05  },
{ internal_stimulus = 0.15, external_stimulus = 0.0667 },
{ internal_stimulus = 0.10, external_stimulus = 0.1833 },
{ internal_stimulus = 0.05, external_stimulus = 0.30   },
]
# 以下 config_extzero.toml と同一
```

いずれも `s̄_S=s̄_X=0.125`（baselineと一致、w較正は不変）、δ_S=0.6は共通。

### 出力parquetスキーマ（v0.2.0、現行）

```
id:u64, ip:u64, us_id:u64, msg:Utf8, mean_num_xact:f64, mean_num_share:f64
```
1行＝1セル（シナリオ×系列）。`mean_num_xact`/`mean_num_share`は共に
**Tラウンド合計**のモンテカルロ平均（ラウンド別内訳は存在しない）。
個人単位の明細（`user_analysis.parquet`、`num_xact_of_users`等）はv0.2.0で廃止済み。

### 重要なコード変更（そのまま貼付）

`core/src/diffusion.rs`の該当箇所（非短絡`&`を維持する理由のコメント込み）:
```rust
// NOTE: 非短絡の `&` を意図的に残している。`&&` にすると受信済みノードに対して
// 乱数を引かなくなり高速化できるが、乱数列がずれて過去の実行結果との
// 再現性（ビット一致）が失われるため変更しない。
if (ws.rcvd[j] == false) & (prop_prob > rng.random()) {
```

`runner/src/brute_force/config.rs`の`sample_iter`（反応性パラメータの核心）:
```rust
pub fn sample_iter<D: From<DecisionBase>>(
    &self,
    num_rounds: u32,
    num_nodes: usize,
    mean_stimulus: f64,
    rng: &mut SmallRng,
) -> impl IntoIterator<Item = D> {
    let scale = 1.0 / (self.beta * (num_rounds as f64));
    let shape = self.gamma / (self.beta - self.gamma);
    Pareto::<f64>::new(scale, shape)
        .expect("scale / shape の妥当性は Config::validate で検証済み")
        .sample_iter(rng)
        .take(num_nodes)
        .map(move |responsiveness| {
            let weight = initialize_weight(mean_stimulus, self.initial_utility, responsiveness);
            DecisionBase::new(weight, self.initial_utility).into()
        })
}
```

`Config::validate()`の主な制約（`runner/src/brute_force/config.rs`）:
- `0 < gamma < beta`、`initial_utility ∈ [-1,1]`
- `receipt_probability ∈ [0,1]`、`num_rounds >= 1`、`diffusion_sample_size >= 1`
- `1 <= ip_sampling.size <= graph.node_count()`
- `1 <= L <= 16`、各刺激値`[-1,1]`、`log2(L)*num_rounds <= 64`

### 関連ドキュメント（参照先）

- `CLAUDE.md`（ルート）: crate構成・ドメインモデル・出力スキーマ・CLI・不変条件
- `analysis/CLAUDE.md`: Python分析（RAM/OPT/URS定義、`twitch_DE.parquet`前提の
  ノートブック仕様、v0.2.0での個人単位廃止の影響を追記済み）
- `研究リソースとか/claude-research.md`: 研究の一次資料（背景・先行研究・結果・考察）
- `研究リソースとか/結果.pdf`: baseline実データの生の数値表（今回分析の参照フォーマット）
- `研究リソースとか/メッセージ空間と反応性パラメータ_解説.pdf`（PDF#1）:
  w正規化・δ・zの意味、s̄が吸収される仕組み
- `研究リソースとか/内部軸の平均刺激を上げた場合_シミュレーション.pdf`（PDF#2）:
  誤ったPareto scaleで「内部飽和」を主張した資料（前セッションで反証済み）

## 未解決・要確認

- β/γ候補7条件（A2+B2+C2、現行含めると7水準）のうち、**どこまでフル実行するか**未決定。
  提案は「スクリーニング→絞り込み→フル実行」の2段階だが、実行環境の計算資源次第で
  条件数を減らす必要があるかもしれない。
- スクリーニング用`saturation_probe.rs`のβ,γ掃引対応は未実装（現状は`test/config.toml`の
  β=0.8,γ=0.6固定で7水準のs̄のみ掃引する設計）。
- グラフは`test/twitter_dataset.txt`を使うか、`test/congress_network_graph.txt`
  （軽量・前セッションの代用グラフ）で先にスクリーニングするか未決定。
- 実データ（ENGB/Twitch/Twitter）は依然としてこのチェックアウトには無い
  （`twitter_extzero`/`twitter_extneg`は別PCで生成し持ち込んだもの）。

## 次にやること

1. β/γ候補（上記A/B/C、計7条件）を確定し、`test/config_beta_*.toml`のような
   命名でconfigファイルを作成する（message_spaceはbaselineのまま、
   `internal_parameters`/`external_parameters`のbeta,gammaだけ変える）。
2. `runner/examples/saturation_probe.rs`をβ,γ掃引に対応させ、
   7条件の累積発火率・末尾m3効果を意思決定モデル単体で先にスクリーニングする。
3. スクリーニングで有望な条件（優先度: C＝X-slow/X-fast > A＝R-slow > B）を選び、
   フル総当たりシミュレーションを実行する。
4. `研究リソースとか/ext_analysis/analyze_ext.py`の`CONDITIONS`辞書に
   新条件を追加し、既存フォーマット（RAM/OPT/評価値/全体像/C1〜C7照合/
   1ポジション差し替え対比図）で結果資料を生成する。

## 制約・好み

- 出力言語は日本語。技術文書・数値表は簡潔に、解釈と事実を明確に分ける。
- 図は`dataviz`スキルの検証済みパレットを使い、`validate_palette.js`で
  事前検証してから使う（m0-m3はordinalランプ、条件比較はcategorical/ordinal使い分け）。
- PDFは`matplotlib.backends.backend_pdf.PdfPages`で作成（追加インストール不要の方針）。
  日本語フォントは`Hiragino Sans`指定（macOS）。
- **乱数の再現性（ビット一致）を破る変更は事前に確認を取る**（過去に一度、
  `&`→`&&`の変更是非をユーザーに確認し、「変更しない」の判断を得ている）。
- 大規模な実行・破壊的操作（config削除、大きい中間ファイルの削除等）は
  必ず事前確認する（過去にoutputs/ディレクトリ削除の可否を確認した実績あり）。
- Rustコードの変更は最小限・サージカルに。既存のコメント密度・命名規則に合わせる。
- git commit/pushはユーザーの明示的な指示があるときのみ行う（現状まだcommitしていない
  変更が複数ある可能性が高いので、続行時に`git status`で確認すること）。
