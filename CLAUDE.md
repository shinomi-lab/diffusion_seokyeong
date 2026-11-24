# CLAUDE.md — `diffusion_seokyeong`（Rust シミュレータ）

`xa-promote`: グラフ上の情報拡散をシミュレートし、**メッセージ系列の総当たり探索**の結果を
parquet に書き出す Rust ワークスペース。

> スコープ: このファイルは Rust 側（`core` / `runner` / `util` / ルート `src`）の
> アーキテクチャを定義する。`analysis/`（Python の統計処理）は対象外で、
> そちらの設計は `analysis/CLAUDE.md` を参照。
>
> **更新ポリシー**: 新しい進捗・資料ができても、**Claude がこの md を自発的に更新しない**。
> 更新はユーザーが直接依頼したときだけ行う。記述と現状のずれに気づいても、更新せず報告にとどめる。
>
> **参照の分担**:
> - **開発（コード・アーキテクチャ・実行）** → このファイル（`CLAUDE.md` / `analysis/CLAUDE.md`）。
> - **研究作業（要旨・論文・発表資料・ポスター・考察の執筆）** → `研究リソースとか/claude-research.md`。
>   研究の全体像と、作業別に読むべき資料のルーティング表がある。作業前にそこで該当資料を特定すること。
>   技術的な detail（crate 構成、データ構造、parquet スキーマ等）が必要なときだけ
>   `研究リソースとか/claude-research.md` からこのファイルを参照し、
>   コード分析に不要なトークンを使わないこと。
> - **開発ログ** → `log/`（段階ごとに `YYYY-MM-DD_<題>.md` を新しく作る。`log/work/` は作業用の一時ファイル）。

---

## 1. ワークスペース構成

ルート `Cargo.toml` は package `xa-promote`（バイナリ）兼 workspace。

| crate | 役割 | 依存 |
|---|---|---|
| `core` | ドメインモデル（意思決定・メッセージ・ユーザ状態・拡散カーネル・推薦チャネル）。副作用なしの純粋ロジック。版 0.3.0 | `graph-lib`, `rand`, `approx`, `serde` |
| `util` | 汎用ヘルパー: 決定論的 RNG サンプリング（`RngSampling`）、グラフ設定の deserialize（`GraphConfig`）。版 0.1.0 | `graph-lib`, `rand`, `serde` |
| `runner` | 総当たり探索の実行エンジン。TOML 設定読込 → ジョブ生成 → 並列実行 → parquet 書き出し。版 0.3.0 | `core`, `util`, `graph-lib`, `rand`, `rand_distr`, `serde(_with)`, `toml`, `anyhow`, `itertools`, `tokio`, `parquet`, `arrow-*`, `clap` |
| ルート `src/main.rs` | CLI エントリ。`clap` で 1 サブコマンド `brute-force` を `runner::brute_force::start` に委譲。版 0.1.0 | `runner`, `tokio`, `clap`, `anyhow` |

依存方向: `main → runner → {core, util}`、`core → graph-lib`（`core` は版 0.3.0）。
外部依存 `graph-lib` は git tag 固定（`v0.6.1`、`shinomi-lab/graph-lib`）。

---

## 2. ドメインモデル（`core/`）

### 2.1 意思決定 — `decision.rs`

ユーザの「ある行動をするか」を、刺激の指数移動平均で表す。

```
DecisionBase { weight w, utility u }
  update(stimulus):  u ← w·u + (1−w)·stimulus        // stimulus ∈ [−1, 1]
  decided():         u > 0      (u ≈ 0 は false)
```

`w` が大きいほど過去の慣性が強く、決定が変わりにくい。

2 つのラッパでライフサイクルが異なる:

| 型 | 挙動 | 用途 |
|---|---|---|
| `DecisionAny` | `update` のたびに `decided` を再評価（何度でも true↔false）| **内部決定 = 再シェアするか**（毎ラウンド判定）|
| `DecisionOnce` | 一度 `decided` になると `done`、以後の `update` では `decided=false` に固定 | **外部決定 = 取引するか**（一度だけ。不可逆イベント）|

### 2.2 メッセージ — `message.rs`

```
Message { internal_stimulus, external_stimulus }      // それぞれ内部/外部決定への刺激
MessageSpace(Vec<Message>)                             // L 種のメッセージ。id 0..L−1
```

- 設定上の慣習: `id 0` = 内部刺激最強（シェアを促す）、`id L−1` = 外部刺激最強（取引を促す）。
- `seq_by_index(index, T)`: `index`（`0..L^T`）を L 進で分解し、長さ `T` の
  メッセージ参照列と 16 進コード文字列を返す。**桁 j = round j+1**、桁 0（＝ `index % L`）が
  文字列の先頭かつラウンド 1。以前あった `msgs_ids.reverse()` は削除済み（コミット `344199a`）。
- `seq_into(index, T, out)`: 同じ分解を行うが**文字列を作らず**、呼び出し側の
  `Vec<&Message>` に詰め直す（バッファ再利用用）。`index_to_string` は文字列だけを返す。
  3 つとも共通の基数 L 分解イテレータ `digits()` の上に実装されている。
- `MessageSpace` は `TryFrom<Vec<Message>>` で構築し、`L ≤ 16` と各刺激値が `[-1, 1]` に
  入っていることを**構築時に検証**する（違反は `MessageSpaceError`）。
- `seq_size(T) = L^T`。T に依存する制約 `log2(L)·T ≤ 64` は `Config::validate` が検証する。

### 2.3 ユーザ状態 — `user.rs`

```
UserState { internal_decision: DecisionAny, external_decision: DecisionOnce }
  contact(message):  internal_decision.update(msg.internal_stimulus)
                     external_decision.update(msg.external_stimulus)
```

### 2.4 拡散カーネル — `diffusion.rs`

`diffuse(graph, prop_prob, ws, ip, message, rng)` — **1 メッセージ 1 回の連鎖**
（中身は `diffuse_with_rec(…, rec = None, a_t, rng)`。推薦チャネルは §2.5）:

- グラフの辺の向き = メッセージ伝播方向（follow/followee の逆）。
- `ip` ノードを起点にラウンド制の BFS。各ステップで「今シェア中」のノード `i` が、
  未受信の後続 `j` へ確率 `prop_prob`（= `receipt_probability`）で伝播。
- 受信すると `ws.states[j].contact(message)`:
  - `external_decision.decided()` なら **取引**（`num_xact` を +1）
  - 後続を持ち、かつ `internal_decision.decided()` なら **次ラウンドでシェア**
    （`num_share` を +1、`next` に追加）
- `rcvd` は `diffuse` 呼び出しごとに `fill(false)` でリセット
  （各ラウンド、連鎖は `ip` から再スタート）。
- 戻り値 `DiffusionResult { num_xact, num_share, total_step, num_rec_rcvd, num_rec_miss, num_xact_rec, e_hat_end }`（後ろの 4 つは推薦チャネル用。オフでは 0）。
- 受信判定 `(rcvd[j] == false) & (prop_prob > rng.random())` の `&` は**非短絡のまま意図的に
  残している**。`&&` にすると受信済みノードで乱数を消費しなくなり乱数列がずれ、
  過去の実行結果との再現性が失われるため。

`multiple_diffuse(…, messages: &[M], rec: Option<&Recommender>, rng, result: &mut Aggregator<usize>)`:
- メッセージ列を順に `diffuse`。**ユーザ状態は列全体で共有**（効用が前ラウンドから持ち越される
  ＝ 時間ダイナミクスの本体）。`num_xact` / `num_share` の両方を T ラウンド合計として積算。
  推薦がオンなら、R3 の事前分布の平均 a_t もラウンドを跨いで持ち回す。

`monte_carlo_average(…, initial_user_states, ip, messages, rec, sample_size, rng, ws) -> Aggregator<f64>`:
- 毎サンプル `ws.states` を `initial_user_states` で上書きしてから `multiple_diffuse` を回し、
  合計を `sample_size` で割る。
- `mean_num_xact` = 「T ラウンド合計取引数」のモンテカルロ平均。
- `mean_num_share` = 「T ラウンド合計シェア回数」のモンテカルロ平均。
- `mean_num_rec_rcvd` / `mean_num_xact_rec` = 推薦経由の受信数・外部行動数（推薦オフでは 0）。
- **個人単位のベクトルは持たない**（集計はネットワーク全体のスカラーのみ）。

`Workspace` — ノード数サイズのバッファ（`states` / `rcvd` / `curr` / `next`）をまとめて持つ
作業領域。`Workspace::new(node_count)` で 1 度だけ確保し、以降は使い回す
（`diffuse` が `rcvd`/`curr`/`next` をリセットする。`ws.states` の初期化は
`monte_carlo_average` の責務）。

`shortest_distances_from(graph, ip)` — `ip` からの BFS ホップ距離。`distance_from_ip` 列の
廃止に伴い `runner` からは使っていないが、`core` の公開 API として残してある。

### 2.5 推薦チャネル（Rモデル）— `recommend.rs`（2026-10-26、core 0.3.0）

定義は `../reccomend_system/推薦システム単純.md` の3節、実装仕様は `研究リソースとか/推薦チャネル_実装仕様.md`。

- 各波のフォロー経路の処理のあと（`mem::swap` の前）に、推薦量 B_s = ε·δ·φ(ê/ē0)·p·Σ_{i∈S_s} d_i を
  丸めた数だけ、フォロー外へ届ける。宛先は確率 θ で 2 歩のランダムウォーク（シェア者の出辺を 1 本選び、
  その先の後続を 1 人）、残りは全ノードから一様。受信済み・IP・行き止まりは空振り。
- 推薦で届いた人は、フォロー経路と**同じ** `UserState::contact` で受信する（更新則は変えない）。
- 反応率 ê = (シェアを決めた人数 + k·a_t)/(受信した人数 + k)（ベータ–二項分布の事後平均。推薦で届いた人も数え、
  フォロワー 0 人のシェアも数える）。ē0 は `base_share_rate`（ユーザ状態集合ごとに `runner` が 1 回計算）。
- `RecParams { eps, delta, theta, tau, phi_max, k, mu }`（TOML の `[recommender]`、§3.1）。
  `Recommender::new` は ε·δ = 0 なら `None` を返し、推薦の分岐に入らない。
- 乱数は拡散と同じ `SmallRng` から「丸め 1 回 → 枠ごとに θ の判定 → 出辺 → 2 歩目」の順に引く。
  **推薦オフでは 1 つも引かないので、既存の結果とビット一致する**（テスト S-01・S-02）。

---

## 3. 実行オーケストレーション（`runner/src/brute_force.rs`）

### 3.1 設定 `Config`（TOML、`brute_force/config.rs`）— `test/config.toml` が雛形

`test/config.toml`（baseline）が設定ファイルの基本形。**他の TOML はすべてこの雛形からの差分**なので、
各キーの意味を以下で押さえておけば、どの TOML も読める（各 TOML の中身はここに転記しない）。

```toml
message_space = [ { internal_stimulus = .., external_stimulus = .. }, ... ]   # L 種
num_rounds = 5
receipt_probability = 0.5
internal_parameters = { beta = .., gamma = .., initial_utility = .. }
external_parameters = { beta = .., gamma = .., initial_utility = .. }
graph = { path = "..", directed = .., transposed = .., format = ".." }
ip_sampling = { size = .., seed = .. }
user_sampling = { size = .., seed = .. }
diffusion_sample_size = 100
```

| キー | 型 | 意味・読み方 | 制約（`Config::validate`）|
|---|---|---|---|
| `message_space` | 配列（L 個）| IP が選べるメッセージの集合。各要素は **(内部刺激, 外部刺激)**＝内部決定（シェア）と外部決定（取引）への刺激。並び順が id（0..L−1）で、`msg` 文字列の各桁になる。`0` を内部寄り、`L−1` を外部寄りに並べるのが慣習。刺激の**平均** `s̄` が重みの較正に使われ（§3.2）、**平均まわりの散らばり**（差別化度 δ）がメッセージ間の効き方の差を決める | 各刺激 ∈ [−1, 1]、1 ≤ L ≤ 16 |
| `num_rounds` | 整数 | 投稿のラウンド数 T。系列長（`L^T` 通り）であり、**反応性の較正基準でもある**（`scale = 1/(β·T)`）。T だけ変えるとモデル全体の較正が変わる点に注意 | T ≥ 1、`log2(L)·T ≤ 64` |
| `receipt_probability` | 実数 | 隣接ノードへの伝播（受信）確率 p。独立カスケード（IC）の辺確率 | 0 ≤ p ≤ 1 |
| `internal_parameters` | `{beta, gamma, initial_utility}` | **内部決定（シェア）**の反応性の分布。`beta` が**速さ**（最大必要ラウンド `n_max = β·T`）、`gamma` は β と組で**裾の重さ**（`α = γ/(β−γ)`）を決める。`initial_utility` は全ユーザの初期効用 u0（負＝最初は動かない）| 0 < γ < β、u0 ∈ [−1, 1] |
| `external_parameters` | 同上 | **外部決定（取引）**の反応性。内部と**別々に指定できる**ので、軸ごとに速さ・裾を変えた条件（軸非対称）が作れる | 同上 |
| `graph` | `GraphConfig` | 拡散するネットワーク。`path`＝エッジリスト（TOML からの相対ではなく**実行時のカレント基準**）、`directed`＝有向/無向、`transposed`＝読み込み時に各行の向きを反転、`format`＝`edgelist`/`csv`/`tsv`（**省略時は `edgelist`**、`directed`/`transposed` は必須）。メッセージは**辺の向きに流れる**ので、follow 関係のデータは `transposed = true` で読む。詳細は §6 | ファイルが存在すること（無ければ path 入りのエラー）|
| `ip_sampling` | `{size, seed}` | 情報提供者（IP）ノードを全ノードから `size` 個選ぶ。**シナリオ数の片方の次元** | 1 ≤ size ≤ ノード数 |
| `user_sampling` | `{size, seed}` | ユーザ状態集合（各ノードの `w` を反応性分布から引いたもの）を `size` 通り作る。**シナリオ数のもう片方の次元** | size ≥ 1 |
| `diffusion_sample_size` | 整数 | 1 セル（シナリオ×系列）あたりのモンテカルロ試行数。`mean_num_*` はこの平均 | ≥ 1 |
| `recommender` | 表（省略可）| 推薦チャネル（§2.5）。`eps`・`delta`・`theta`・`tau`・`phi_max`・`k`・`mu`。**表ごと省略すれば推薦なし**（既定 ε = 0）。一部だけ書けば残りは既定値、綴りの誤りはエラー | ε ≥ 0、δ・θ・μ ∈ [0, 1]、τ ≥ 0、φ_max ≥ τ、k > 0 |

- シナリオ数 = `user_sampling.size × ip_sampling.size`、1 シナリオあたり `L^T` 行（§3.4）。
- `seed` は各サンプリングを決定論的にする（`SmallRng::seed_from_u64`）。
- 他の TOML との**差分の軸**（雛形のどのキーを動かしているか）は §4 の一覧を参照。

### 3.2 パラメータの意味 — 応答性のサンプリング

`DecisionParameters::sample_iter`:
- ノードごとの **responsiveness** を `Pareto(scale = 1/(beta·T), shape = gamma/(beta−gamma))` から引く。
- `weight = (mean_stimulus / (mean_stimulus − initial_utility))^responsiveness`
  （底 < 1 なので responsiveness が大きいほど `weight` 小 → 慣性が弱く早く決定する）。
- `DecisionBase::new(weight, initial_utility)` を生成。
- `mean_stimulus` は `MessageSpace` の内部/外部それぞれの平均刺激。

### 3.3 ジョブ生成と並列実行 `JobManager`

```
producer task:
  chunk = effective_chunk_size(L^T, --chunk-size, max_job_size)
  for us_id in 0..user_sampling.size:            // サンプルしたユーザ状態集合
    user_states = sample_user_states(...)         // ノード数ぶんの UserState
    for ip in ips (= ip_sampling で選んだノード):
      for start in (0..L^T).step_by(chunk):
        rngs = chunk 個の SmallRng を seq_index 昇順に引く  // 乱数列の互換性を保つ
        spawn JobChunk::compute  (semaphore で最大 max_job_size 並列)
      model_id += 1          // = シナリオ id (dm_id)

JobChunk::compute:               // 連続する chunk 本の系列を 1 タスクで処理
  ws = Workspace::new(num_nodes)   // チャンク内で使い回す
  for seq_index in start..start+chunk:
    message_space.seq_into(seq_index, T, &mut msgs)   // 文字列は作らない
    agg = monte_carlo_average(graph, receipt_probability, user_states, ip,
                              msgs, diffusion_sample_size, rng, &mut ws)
  → Vec<JobResult> を channel 送信（チャンクにつき 1 回）

consumer/writer task (tokio::spawn):
  JobResult を列バッファに詰め（msg 文字列はここでのみ生成）、
  row_buffer_size 行ごとに Arrow RecordBatch を parquet に flush（ZSTD 圧縮）
```

- `model_id`（= `dm_id` = 出力 `id`）は `(us_id, ip)` の組ごとに +1 →
  `0 .. user_sampling.size × ip_sampling.size − 1` がシナリオ id。
- producer / consumer は `mpsc::channel` と `try_join!` で連結。両タスクのエラーは
  `run()` から `anyhow::Error` として伝播する（以前は writer のエラーを握りつぶしていた）。
- `effective_chunk_size` はワーカ 1 つあたり 4 チャンク程度になるよう `--chunk-size` を
  上限として自動的に縮める。**系列数が少ない設定で全系列が 1 チャンクに入り
  並列度を失うのを防ぐため**（例: L=4, T=2 の 16 系列 × 並列 8 → チャンク 1）。

### 3.4 出力

`output_path` に 1 ファイルだけ書く（個人単位の `user_analysis.parquet` は廃止）。

| ファイル | 列（Arrow schema）|
|---|---|
| result（`output_path`）| `id:u64, ip:u64, us_id:u64, msg:Utf8, mean_num_xact:f64, mean_num_share:f64, mean_num_rec_rcvd:f64, mean_num_xact_rec:f64` |

- 1 行 = 1 セル（シナリオ × メッセージ系列）。行数は
  `user_sampling.size × ip_sampling.size × L^T` ちょうど。
- `mean_num_xact` / `mean_num_share` はいずれも **T ラウンド合計** のモンテカルロ平均。
  `mean_num_share` は旧 `user_analysis.parquet` を `(ip, us_id, msg)` で
  `sum(num_share_of_users)` した値と一致する（整数で合計してから 1 度だけ割るので、
  旧実装より丸め誤差が小さい）。
- `mean_num_rec_rcvd` / `mean_num_xact_rec` は推薦経由の受信数・外部行動数（T ラウンド合計の平均。推薦オフでは 0）。
- parquet KV メタデータに `version_core` / `version_runner` / `version` / `recommender`（`RecParams` の表記）。
  推薦チャネルの追加（列が 8 列に）に合わせて `core` / `runner` は **0.3.0** に上げてある
  （0.2.0 は 6 列）。出力ファイルが新旧どちらの形式かはこのメタデータで判別できる。

---

## 4. CLI / 実行

```bash
cargo build --release
./target/release/xa-promote brute-force <config.toml> <output.parquet> <max_job_size> \
    [--row-buffer-size 100] [--chunk-size 128] [--zstd-level 3]
```

- `max_job_size` … 同時実行チャンク数（semaphore の permit 数）。
- `--chunk-size` … 1 タスクが担当する系列数の**上限**。実際の値は並列度に合わせて
  自動的に縮む（`effective_chunk_size`）。
- 生成物: `<output.parquet>` のみ。
- 設定例:
  - `test/config.toml` — **雛形（baseline）**。キーの意味は §3.1。
    δ_S = δ_X = 0.6、`message_space` 4 種、`num_rounds=5`、内外とも β=0.8, γ=0.6、`ip_sampling.size=16`、`user_sampling.size=32`。
  - 他の TOML は雛形からの**差分の軸**だけを動かす（中身は各ファイルを読むこと）:

    | TOML | 動かすキー | 動かし方 |
    |---|---|---|
    | `config_extzero.toml` | `message_space` | 外部軸の差別化度 δ_X = 1.0（外部刺激ちょうど 0 のメッセージを含む）|
    | `config_extneg.toml` | `message_space` | δ_X = 1.4（外部刺激が**負**のメッセージを含む）|
    | `config_xslow.toml` | `internal_parameters` / `external_parameters` | 内部を**速く**（β=0.4,γ=0.30）、外部を**鈍く**（β=1.4,γ=1.05）|
    | `config_xfast.toml` | 同上 | `xslow` の逆（内部が鈍く、外部が速い）|
    | `config_rec.toml` | `recommender` | 推薦チャネルをオン（ε = 1、他は既定値）。表ごと消すと `config.toml` と同じ結果 |

    - extzero/extneg は `s̄_S = s̄_X = 0.125` を baseline と揃えてあり、w の較正は変わらない（δ の効果だけを単離）。
    - xslow/xfast は `message_space` が baseline のまま。速い/鈍いは n_max=β·T = 2.0 / 7.0、α=γ/(β−γ)=3.0 で共通（動かすのは**速さだけ**）。
    - どの TOML も `graph.path = "ENGB_edges.csv"`（`format="csv"`, `directed=true`, `transposed=true`）が書かれている。
      実データは別 PC 側にあり、**実際の実行ではパス・向きが書き換えられる**（ENGB の X-slow/X-fast は `directed=false`）。
- テスト用グラフ（`test/`）:
  - `congress_network_graph.txt`（475 ノード・有向、edgelist）、`twitter_dataset.txt`（81,306 ノード、edgelist）
  - `musae_ENGB_edges.csv`（Twitch-ENGB、7,126 ノード・35,324 辺、**ヘッダー無し**の csv。`format="csv"` で読める）
  - `100random.txt`（100 ノード・400 辺のランダム有向グラフ、edgelist、seed=42。可視化は `../plot_100random.py`）

設定は読み込み直後に `Config::validate()` で検証される（`gamma < beta`、確率の範囲、
刺激値が `[-1, 1]`、`ip_sampling.size ≤ ノード数`、`log2(L)·T ≤ 64` など）。
違反時は panic ではなく原因の分かるエラーで停止する。

### 次元の対応（`test/config.toml` の場合）

```
L = 4, T = 5           → L^T          = 1024 メッセージ系列
user_sampling.size = 32, ip_sampling.size = 16 → 512 シナリオ (id 0..511)
result 行数            = 512 × 1024   = 524,288
```

（`analysis/twitch_DE.parquet` の形状と一致。実際の実行では別グラフ・別 config が使われている。）

---

## 5. 不変条件・注意点

- `core` は I/O・並列なし。乱数は `rand` のみ、`SmallRng::seed_from_u64` で決定論的
  （`Environment.diffusion_seed = 0` 固定、各 sampling は config の `seed`）。
- `DecisionOnce` により、各ユーザの「取引」はシミュレーション中に**高々 1 回**。
- `mean_num_xact` は T ラウンドの合計であって 1 ラウンドあたりではない。
- `msg` 文字列はラウンド昇順（先頭 = ラウンド 1）。`analysis` 側もこの順で解釈する。
- グラフの辺の向きは「メッセージが流れる向き」。`transposed = true` で follow 関係を反転して読む。
  `directed = false` は無向として読む（相互関係のデータ向け）。**向きの設定だけで結果は桁違いに変わりうる**
  （ENGB の X-slow は有向/無向で URS が約 91 倍違った）ので、有向の過去結果と無向の結果は比較しない（§6）。
- 出力の `id`（`dm_id`）は `(us_id, ip)` ペアの通し番号。`msg` は含まない。
- parquet のバッチ flush 単位は `--row-buffer-size`（既定 100 **行**）。channel 容量は
  `max_job_size`（1 チャンクにつき 1 回送信するため）。
- **乱数の再現性**: 各ジョブ用 `SmallRng` は `seq_index` 昇順に引かれる。チャンク化しても
  この順序は保たれるため、`mean_num_xact` は旧実装（v0.1.x）とビット単位で一致する。
  `diffuse` の受信判定を短絡評価にしないのもこの再現性を守るため（§2.4）。
- **推薦オフのビット一致**: `[recommender]` が無い（ε·δ = 0）とき、推薦の処理は乱数を 1 つも引かない。
  変更前（core 0.2.0）の `diffuse` をテスト内に写した `diffuse_reference` と、結果・乱数の消費量まで比べている（S-01）。
  2026-10-26 に小さな設定で、変更前後の parquet の既存 6 列が全行一致することも確認した（`log/2026-10-26_recommender.md`）。
- **個人単位の出力は無い**。per-user の `num_{xact,share}_of_users` / `distance_from_ip` は
  v0.2.0 で廃止した。集計粒度の値は `mean_num_share` で代替できるが、
  「1 人あたり平均行動回数」は算出できない（`analysis/CLAUDE.md` 参照）。
- edition = 2024（Rust 1.85+ が必要）。
- テスト: `cargo test --workspace`（ワークスペース全体を指定しないと `core` / `runner` の
  テストが走らない）。現在 32 件（`core` 21 + `runner` 11。推薦チャネルの S-01〜S-07 を含む）。`core` のテストは `graph-lib` の
  `From<&[(L,L)]> for DiGraphB` で小さなグラフを組むのでファイル I/O 不要。

---

## 6. グラフ読み込みの仕様（graph-lib v0.6.1 のソース確認＋`graph_csv_probe` で実測）

`util::GraphConfig` → `graph_lib::io::ParseBuilder` → `GraphB`（`Di(DiGraphB)` / `Ud(UndiGraphB)`）。
`core` / `runner` は `Graph` トレイト越しにしか触らないので、**有向/無向の切替にコード修正は要らない**（TOML の `directed` だけ）。

| 項目 | 挙動 |
|---|---|
| 区切り・コメント | `edgelist`＝空白/タブ・`#` と `%`、`csv`＝`,`・`#` のみ、`tsv`＝タブ・`#` のみ |
| 区切りの不一致 | 例: 空白区切りのファイルを `csv` で読むと panic（誤読みせず fail-fast）|
| ヘッダー行 | `from,to` のような非数値の行があると panic。**ヘッダー無しにしておく** |
| 3 列目以降 | 無視（先頭 2 列＝送信元・宛先のみ使用）。CRLF・前後の空白は trim で吸収 |
| ノード ID | **ソート済みの相異なるラベルを 0..n に詰め直す**。`node_count` = 登場したラベル数（最大 ID+1 ではない）。次数 0 の孤立ノードは存在しない扱い。内部 index は生の ID と一致しない |
| `transposed` | 各行を (u,v)→(v,u) に反転してから辺を追加 |
| `directed=false` | `UndiGraphB`：`add_edge(u,v)` が **(u,v) と (v,u) の両方**を挿入。`edge_count` は内部の辺数÷2。同じ組の重複行は 1 本に畳まれる。**`transposed` は無向では no-op** |

- 有向では `edge_count == ファイルの行数`（重複行が無ければ）。無向では `行数 − 逆向き重複行数`。
  `musae_ENGB_edges.csv` は逆向き重複行が 0 なので、有向でも無向でも 35,324 辺・7,126 ノード。
- 1 行ずつ `unwrap` でパースするため、不正行はエラーではなく **panic** になる（graph-lib 側の仕様）。

---

## 7. config → 出力 → 分析資料の対応

| config | 実行 | 出力（v0.2.0 parquet）| 分析結果の置き場所（`研究リソースとか/`）|
|---|---|---|---|
| `config.toml` | 各ネットワークで baseline | `analysis/twitch_DE.parquet`（DE。他は別 PC）| `結果.pdf`（生数値）|
| `config_extzero.toml` | Twitter（81,306 ノード、別 PC）| リポジトリ直下 `twitter_extzero` | `extzero_結果/`、`twitter_extzero_結果.pdf` |
| `config_extneg.toml` | 同上 | `twitter_extneg` | `extneg_結果/`、`twitter_extneg_結果.pdf` |
| （extzero/extneg の比較）| — | — | `delta_x_横断比較/`、`delta_x_横断比較.pdf` |
| `config_xslow.toml` / `config_xfast.toml` | Twitch-ENGB（`directed=false`）と Congress（有向）、別 PC | `ENGB_xslow/xfast/baseline.parquet` ほか | `xslow_xfast_engb/`、`xslow_xfast_統合まとめ/` |

分析スクリプトは `研究リソースとか/ext_analysis/`（`analysis/CLAUDE.md` §8）。
X-slow/X-fast の分析スクリプトはリポジトリに無く、成果物（CSV/PDF）のみ。

---

## 8. 診断バイナリ（`runner/examples/`）

| ファイル | 目的 | 実行 |
|---|---|---|
| `saturation_probe.rs` | 意思決定モデル単体の診断。実コードの `DecisionParameters::sample_iter` で 20 万人をサンプルし、s̄ を振って発火率・重み分布を stdout に CSV 2 ブロックで出力（β=0.8, γ=0.6 固定。内部飽和説の反証に使用）| `cargo run --release --example saturation_probe -p runner` |
| `rec_crosscheck.rs` | 推薦チャネルの参照側の突き合わせ（ビューアの T-W09）。ビューアの `wave_crosscheck` が書いた母集団の CSV を読み、同じ設定で 1 ラウンドを試行数ぶん回して、シェア・外部行動・推薦経由の受信の平均と標準誤差を出す | `cargo run --release --example rec_crosscheck -p runner -- グラフ 母集団.csv IP p 平均内部刺激 内部刺激 外部刺激 [試行数]` |
| `graph_csv_probe.rs` | `GraphConfig` 経由のグラフ読み込みの検証（23 チェック: 基本形式・`transposed`・csv/edgelist の区別・ヘッダー・3 列目・疎な ID・CRLF・`directed=false`・実ファイル `test/musae_ENGB_edges.csv`）| **リポジトリルートで** `cargo run --release --example graph_csv_probe -p runner`（実ファイル節が相対パス `test/...` を使うため）|

`load(path, directed, transposed, format)` の第 2 引数が `directed`。新しいグラフファイルを読めるか確かめるときは、
この関数に渡して `node_count` / `edge_count` を行数と突き合わせる。
