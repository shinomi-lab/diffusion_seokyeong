pub mod config;

use std::{sync::Arc, time::Instant};

use anyhow::Context;
use itertools::izip;

use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};

use parquet::{
    arrow::AsyncArrowWriter,
    basic::{Compression, ZstdLevel},
    file::metadata::KeyValue,
    file::properties::WriterProperties,
};
use rand::{SeedableRng, rngs::SmallRng, seq::IteratorRandom};
use tokio::{
    fs::{self, File},
    sync::mpsc,
    try_join,
};

use core::{
    diffusion::{Workspace, monte_carlo_average},
    message::{Message, MessageSpace},
    user::UserState,
};
use graph_lib::prelude::{Graph, GraphB};
use util::RngSampling;

use config::{Config, DecisionParameters};

struct Environment {
    num_nodes: usize,
    num_rounds: u32,
    internal_parameters: DecisionParameters,
    external_parameters: DecisionParameters,
    user_sampling: RngSampling,
    ip_sampling: RngSampling,
    graph: Arc<GraphB>,
    receipt_probability: f64,
    diffusion_sample_size: usize,
    message_space: MessageSpace,
    diffusion_seed: u64,
}

impl Environment {
    pub fn from_config(
        Config {
            graph,
            num_rounds,
            internal_parameters,
            external_parameters,
            user_sampling,
            ip_sampling,
            message_space,
            receipt_probability,
            diffusion_sample_size,
        }: Config,
    ) -> Self {
        Self {
            num_nodes: graph.node_count(),
            num_rounds,
            internal_parameters,
            external_parameters,
            user_sampling,
            ip_sampling,
            graph: Arc::new(graph),
            receipt_probability,
            diffusion_sample_size,
            message_space,
            diffusion_seed: 0,
        }
    }

    /// Samples all user states from distributions derived by `DecisionParameters`.
    fn sample_user_states(
        &self,
        internal_mean_stimulus: f64,
        external_mean_stimulus: f64,
        mut internal_rng: SmallRng,
        mut external_rng: SmallRng,
    ) -> Vec<UserState> {
        let internal_decisions = self.internal_parameters.sample_iter(
            self.num_rounds,
            self.num_nodes,
            internal_mean_stimulus,
            &mut internal_rng,
        );
        let external_decisions = self.external_parameters.sample_iter(
            self.num_rounds,
            self.num_nodes,
            external_mean_stimulus,
            &mut external_rng,
        );

        izip!(internal_decisions, external_decisions)
            .map(|(internal_decision, external_decision)| {
                UserState::new(internal_decision, external_decision)
            })
            .collect()
    }
}

struct JobManager {
    semaphore: Arc<tokio::sync::Semaphore>,
    num_permits: usize,
    env: Arc<Environment>,
    tx: mpsc::Sender<Vec<JobResult>>,
    rx: mpsc::Receiver<Vec<JobResult>>,
}

impl JobManager {
    fn new(env: Environment, num_permits: usize, buffer_size: usize) -> Self {
        let (tx, rx) = mpsc::channel(buffer_size);
        Self {
            semaphore: Arc::new(tokio::sync::Semaphore::new(num_permits)),
            num_permits,
            env: Arc::new(env),
            tx,
            rx,
        }
    }

    async fn run(
        self,
        output_path: PathBuf,
        zstd_level: i32,
        batch_size: usize,
        chunk_size: usize,
        version: &'static str,
    ) -> anyhow::Result<()> {
        let started = Instant::now();
        let mut rx = self.rx;
        let env = self.env.clone();
        let level = ZstdLevel::try_new(zstd_level)
            .with_context(|| format!("zstd レベル {zstd_level} が不正です"))?;

        let result_file = File::create(&output_path)
            .await
            .with_context(|| format!("出力先 {} を作成できません", output_path.display()))?;

        // ---- writer task ----
        let rx_handle = tokio::spawn(async move {
            let result_schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::UInt64, false),
                Field::new("ip", DataType::UInt64, false),
                Field::new("us_id", DataType::UInt64, false),
                Field::new("msg", DataType::Utf8, false),
                Field::new("mean_num_xact", DataType::Float64, false),
                Field::new("mean_num_share", DataType::Float64, false),
            ]));

            let props = Arc::new(
                WriterProperties::builder()
                    .set_compression(Compression::ZSTD(level))
                    .build(),
            );

            let mut result_writer =
                AsyncArrowWriter::try_new(result_file, result_schema.clone(), Some((*props).clone()))
                    .context("parquet ライタを初期化できません")?;

            let mut dm_id_col: Vec<u64> = Vec::new();
            let mut ip_col: Vec<u64> = Vec::new();
            let mut us_id_col: Vec<u64> = Vec::new();
            let mut msg_col: Vec<String> = Vec::new();
            let mut mean_xact_col: Vec<f64> = Vec::new();
            let mut mean_share_col: Vec<f64> = Vec::new();

            let mut rows_in_batch = 0usize;
            let mut total_rows = 0usize;

            macro_rules! flush {
                () => {{
                    let batch = RecordBatch::try_new(
                        result_schema.clone(),
                        vec![
                            Arc::new(UInt64Array::from(std::mem::take(&mut dm_id_col))) as ArrayRef,
                            Arc::new(UInt64Array::from(std::mem::take(&mut ip_col))),
                            Arc::new(UInt64Array::from(std::mem::take(&mut us_id_col))),
                            Arc::new(StringArray::from(std::mem::take(&mut msg_col))),
                            Arc::new(Float64Array::from(std::mem::take(&mut mean_xact_col))),
                            Arc::new(Float64Array::from(std::mem::take(&mut mean_share_col))),
                        ],
                    )
                    .context("RecordBatch を構築できません")?;
                    result_writer
                        .write(&batch)
                        .await
                        .context("parquet バッチの書き込みに失敗しました")?;
                }};
            }

            while let Some(results) = rx.recv().await {
                for result in results {
                    // 系列コード文字列はここでのみ1行1回作る（compute 側では作らない）。
                    let msg_str = env
                        .message_space
                        .index_to_string(result.msg_seq_index, env.num_rounds);

                    dm_id_col.push(result.dm_id as u64);
                    ip_col.push(result.ip as u64);
                    us_id_col.push(result.us_id as u64);
                    msg_col.push(msg_str);
                    mean_xact_col.push(result.mean_num_xact);
                    mean_share_col.push(result.mean_num_share);

                    rows_in_batch += 1;
                    total_rows += 1;
                }

                if rows_in_batch >= batch_size {
                    flush!();
                    rows_in_batch = 0;
                    println!(
                        "[{:?}] {} rows written ({:.0} rows/s)",
                        started.elapsed(),
                        total_rows,
                        total_rows as f64 / started.elapsed().as_secs_f64().max(1e-9),
                    );
                }
            }

            // 残りのバッチを書き出す
            if !dm_id_col.is_empty() {
                flush!();
            }

            for kv in [
                KeyValue::new(
                    "version_core".to_string(),
                    Some(core::get_version().to_string()),
                ),
                KeyValue::new(
                    "version_runner".to_string(),
                    Some(env!("CARGO_PKG_VERSION").to_string()),
                ),
                KeyValue::new("version".to_string(), Some(version.to_string())),
            ] {
                result_writer.append_key_value_metadata(kv);
            }
            result_writer
                .finish()
                .await
                .context("parquet のクローズに失敗しました")?;

            Ok::<(), anyhow::Error>(())
        });

        // ---- producer task ----
        let tx_handle = tokio::spawn(async move {
            let internal_mean_stimulus = self.env.message_space.internal_mean_stimulus();
            let external_mean_stimulus = self.env.message_space.external_mean_stimulus();

            let user_rngs = self
                .env
                .user_sampling
                .sample_with(|rng| (SmallRng::from_rng(rng), SmallRng::from_rng(rng)));

            let ips = self
                .env
                .ip_sampling
                .custom(|rng, size| (0..self.env.num_nodes).choose_multiple(rng, size));

            let mut rng = SmallRng::seed_from_u64(self.env.diffusion_seed);
            let num_seqs = self.env.message_space.seq_size(self.env.num_rounds);

            // チャンクを大きくしすぎると並列度を失う（系列数が少ない設定では
            // 全系列が1タスクに収まってしまい、ワーカが1つしか動かない）。
            // ワーカ1つあたり数個のチャンクが行き渡るように上限を自動で絞る。
            let chunk_size = effective_chunk_size(num_seqs, chunk_size, self.num_permits);
            println!(
                "系列数 {num_seqs} / チャンクサイズ {chunk_size} / 並列 {}",
                self.num_permits
            );

            let mut model_id = 0;
            for (us_id, user_states) in user_rngs
                .into_iter()
                .map(|(irng, erng)| {
                    Arc::new(self.env.sample_user_states(
                        internal_mean_stimulus,
                        external_mean_stimulus,
                        irng,
                        erng,
                    ))
                })
                .enumerate()
            {
                for &ip in &ips {
                    let mut start_index = 0u64;
                    while start_index < num_seqs {
                        let len = chunk_size.min((num_seqs - start_index) as usize);
                        // 乱数はチャンク化前と同じく seq_index 昇順に引く。
                        // これにより出力が従来実装とビット単位で一致する。
                        let rngs: Vec<SmallRng> =
                            (0..len).map(|_| SmallRng::from_rng(&mut rng)).collect();

                        let job = JobChunk {
                            env: self.env.clone(),
                            user_states: user_states.clone(),
                            dm_id: model_id,
                            us_id,
                            ip,
                            start_index,
                            rngs,
                            tx: self.tx.clone(),
                        };
                        let permit = self
                            .semaphore
                            .clone()
                            .acquire_owned()
                            .await
                            .context("セマフォの取得に失敗しました")?;
                        tokio::spawn(async move {
                            job.compute().await;
                            drop(permit);
                        });

                        start_index += len as u64;
                    }
                    model_id += 1;
                }
            }
            Ok::<(), anyhow::Error>(())
        });

        let (writer_res, producer_res) = try_join!(rx_handle, tx_handle)?;
        // writer / producer 双方のエラーを握りつぶさずに伝播する。
        producer_res?;
        writer_res?;
        Ok(())
    }
}

/// 実際に使うチャンクサイズを決める。
///
/// `--chunk-size` は上限として扱い、ワーカ1つあたり `CHUNKS_PER_WORKER` 個程度の
/// チャンクが行き渡るように縮小する。こうしないと、系列数が少ない設定
/// （例: L=4, T=2 で 16 系列）で全系列が1チャンクに入ってしまい、
/// 並列実行されなくなる。
fn effective_chunk_size(num_seqs: u64, max_chunk_size: usize, num_permits: usize) -> usize {
    /// ワーカ1つあたりのチャンク数の目安。多すぎると spawn の回数が減らず、
    /// 少なすぎると終盤の負荷が偏る。
    const CHUNKS_PER_WORKER: u64 = 4;

    let target_chunks = (num_permits as u64).saturating_mul(CHUNKS_PER_WORKER).max(1);
    // num_seqs を target_chunks 個に分けたときの1チャンクの長さ（切り上げ）
    let balanced = num_seqs.div_ceil(target_chunks).max(1);
    balanced.min(max_chunk_size as u64) as usize
}

/// 連続する系列インデックスのまとまりを1タスクで処理する単位。
///
/// 1系列ごとに `tokio::spawn` していた頃と比べ、spawn・セマフォ・チャネル操作の
/// 回数が `chunk_size` 分の1になり、`Workspace` もチャンク内で使い回せる。
struct JobChunk {
    env: Arc<Environment>,
    user_states: Arc<Vec<UserState>>,
    dm_id: usize,
    us_id: usize,
    ip: usize,
    start_index: u64,
    rngs: Vec<SmallRng>,
    tx: mpsc::Sender<Vec<JobResult>>,
}

#[derive(Debug, Clone, Copy)]
struct JobResult {
    dm_id: usize,
    ip: usize,
    us_id: usize,
    msg_seq_index: u64,
    mean_num_xact: f64,
    mean_num_share: f64,
}

impl JobChunk {
    async fn compute(mut self) {
        let results = {
            let env = &self.env;
            // チャンク全体で1組だけ確保して使い回す。
            let mut ws = Workspace::new(env.num_nodes);
            let mut msgs: Vec<&Message> = Vec::with_capacity(env.num_rounds as usize);
            let mut results = Vec::with_capacity(self.rngs.len());

            for (offset, rng) in self.rngs.iter_mut().enumerate() {
                let msg_seq_index = self.start_index + offset as u64;
                env.message_space
                    .seq_into(msg_seq_index, env.num_rounds, &mut msgs);

                let agg = monte_carlo_average(
                    &env.graph,
                    env.receipt_probability,
                    &self.user_states,
                    self.ip,
                    &msgs,
                    env.diffusion_sample_size,
                    rng,
                    &mut ws,
                );

                results.push(JobResult {
                    dm_id: self.dm_id,
                    ip: self.ip,
                    us_id: self.us_id,
                    msg_seq_index,
                    mean_num_xact: agg.mean_num_xact,
                    mean_num_share: agg.mean_num_share,
                });
            }
            results
        };

        // 送信に失敗するのは writer が先に落ちた場合だけ。その場合は writer 側の
        // エラーが run() から返るので、ここでは panic せず黙って終える。
        let _ = self.tx.send(results).await;
    }
}

use std::path::PathBuf;

use clap::Args;

#[derive(Args)]
pub struct MyArgs {
    /// Path to the configuration file (TOML format)
    config_path: PathBuf,
    /// Path to the output file (.parquet file)
    output_path: PathBuf,
    /// How many chunks of jobs can be run in parallel
    max_job_size: usize,
    /// How many rows are buffered before flushing a parquet batch
    #[clap(long, default_value = "100")]
    row_buffer_size: usize,
    /// How many message sequences one task processes at once
    #[clap(long, default_value = "128")]
    chunk_size: usize,
    /// Zstd compression level
    #[clap(long, default_value = "3")]
    zstd_level: i32,
}

#[cfg(test)]
mod tests {
    use super::effective_chunk_size;

    /// 系列数が少ないときは、全系列が1チャンクに収まって並列度が消えてはいけない。
    #[test]
    fn keeps_parallelism_when_few_sequences() {
        // L=4, T=2 → 16 系列。ワーカ8つなら1チャンク1系列まで縮む。
        assert_eq!(effective_chunk_size(16, 128, 8), 1);
        // 16 系列 / ワーカ2つ → 4チャンク/ワーカ狙いで 16/8 = 2
        assert_eq!(effective_chunk_size(16, 128, 2), 2);
    }

    /// 系列数が多いときは --chunk-size を上限としてまとめ、spawn 回数を減らす。
    #[test]
    fn batches_when_many_sequences() {
        // L=4, T=5 → 1024 系列。ワーカ8つ → 1024/32 = 32
        assert_eq!(effective_chunk_size(1024, 128, 8), 32);
        // 上限を超えないこと
        assert_eq!(effective_chunk_size(1_000_000, 128, 8), 128);
    }

    /// 0 除算やゼロ長チャンクを作らない。
    #[test]
    fn never_returns_zero() {
        assert_eq!(effective_chunk_size(1, 128, 8), 1);
        assert_eq!(effective_chunk_size(1, 1, 1), 1);
        assert!(effective_chunk_size(0, 128, 8) >= 1);
    }
}

pub async fn start(args: MyArgs, version: &'static str) -> anyhow::Result<()> {
    let t = std::time::Instant::now();
    anyhow::ensure!(args.chunk_size >= 1, "--chunk-size は 1 以上である必要があります");
    anyhow::ensure!(args.max_job_size >= 1, "max_job_size は 1 以上である必要があります");

    let buf = fs::read_to_string(&args.config_path)
        .await
        .with_context(|| format!("設定ファイル {} を読めません", args.config_path.display()))?;
    let config: Config = toml::from_str(&buf)
        .with_context(|| format!("設定ファイル {} の解析に失敗しました", args.config_path.display()))?;
    config
        .validate()
        .with_context(|| format!("設定ファイル {} の内容が不正です", args.config_path.display()))?;
    let env = Environment::from_config(config);

    // channel capacity は同時実行チャンク数に合わせる（1チャンクにつき1回送信する）。
    let manager = JobManager::new(env, args.max_job_size, args.max_job_size);
    manager
        .run(
            args.output_path,
            args.zstd_level,
            args.row_buffer_size,
            args.chunk_size,
            version,
        )
        .await?;
    println!("総実行時間: {:?}", t.elapsed());
    Ok(())
}
