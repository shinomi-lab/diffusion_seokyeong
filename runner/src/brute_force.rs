mod config;

use std::{collections::HashMap, sync::Arc, time::Instant};

use itertools::izip;

use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};

use parquet::{
    arrow::AsyncArrowWriter,
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
    file::metadata::KeyValue,
    errors::ParquetError,
};
use rand::{SeedableRng, rngs::SmallRng, seq::IteratorRandom};
use tokio::{
    fs::{self, File},
    sync::mpsc,
    try_join,
};

use core::{diffusion::{monte_carlo_average, shortest_distances_from, Aggregator}, message::MessageSpace, user::UserState};
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
    // ★変更1: user_analysis.parquet を出力するかどうか（config.toml で指定）
    output_user_analysis: bool,
    // ★変更2: IP を固定リストで指定する場合（config.toml で指定、None ならランダム）
    ip_indices: Option<Vec<usize>>,
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
            output_user_analysis,   // ★変更1
            ip_indices,             // ★変更2
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
            output_user_analysis,   // ★変更1
            ip_indices,             // ★変更2
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
    env: Arc<Environment>,
    tx: mpsc::Sender<JobResult>,
    rx: mpsc::Receiver<JobResult>,
}

impl JobManager {
    fn new(env: Environment, num_permits: usize, buffer_size: usize) -> Self {
        let (tx, rx) = mpsc::channel(buffer_size);
        Self {
            semaphore: Arc::new(tokio::sync::Semaphore::new(num_permits)),
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
        version: &'static str,
    ) {
        let started = Instant::now();
        let mut rx = self.rx;
        let env = self.env.clone();
        let level = ZstdLevel::try_new(zstd_level).unwrap();

        let output_dir = output_path.parent().unwrap_or_else(|| std::path::Path::new(""));
        let result_path = output_path.clone();
        let user_analysis_path = output_dir.join("user_analysis.parquet");

        let result_file = File::create(result_path).await.unwrap();
        // ★変更1: user_analysis を出力しない設定なら、ファイル自体を作らない
        let user_analysis_file = if env.output_user_analysis {
            Some(File::create(user_analysis_path).await.unwrap())
        } else {
            None
        };

        // ---- writer task ----
        let rx_handle = tokio::spawn(async move {
            let result_schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::UInt64, false),
                Field::new("ip", DataType::UInt64, false),
                Field::new("us_id", DataType::UInt64, false),
                Field::new("msg", DataType::Utf8, false),
                Field::new("mean_num_xact", DataType::Float64, false),
                // ★変更3: 内部行動の累積回数（メッセージ系列全体・モンテカルロ平均）
                Field::new("mean_num_share", DataType::Float64, false),
            ]));

            let user_analysis_schema = Arc::new(Schema::new(vec![
                Field::new("ip", DataType::UInt64, false),
                Field::new("us_id", DataType::UInt64, false),
                Field::new("msg", DataType::Utf8, false),
                Field::new("user_id", DataType::UInt64, false),
                Field::new("num_xact_of_users", DataType::Float64, false),
                Field::new("num_share_of_users", DataType::Float64, false),
                Field::new("distance_from_ip", DataType::UInt64, true),
            ]));

            let props = Arc::new(
                WriterProperties::builder()
                    .set_compression(Compression::ZSTD(level))
                    .build(),
            );

            let mut result_writer = AsyncArrowWriter::try_new(
                result_file,
                result_schema.clone(),
                Some((*props).clone()),
            )
            .unwrap();

            // ★変更1: writer も Option 化（出力しないなら作らない）
            let mut user_analysis_writer = user_analysis_file.map(|f| {
                AsyncArrowWriter::try_new(
                    f,
                    user_analysis_schema.clone(),
                    Some((*props).clone()),
                )
                .unwrap()
            });

            // --- result.parquet 用バッファ ---
            let mut result_dm_id_col: Vec<u64> = Vec::new();
            let mut result_ip_col: Vec<u64> = Vec::new();
            let mut result_us_id_col: Vec<u64> = Vec::new();
            let mut result_msg_col: Vec<String> = Vec::new();
            let mut result_mean_xact_col: Vec<f64> = Vec::new();
            // ★変更3: 内部行動の列バッファ
            let mut result_mean_share_col: Vec<f64> = Vec::new();

            // --- user_analysis.parquet 用バッファ ---
            let mut user_ip_col: Vec<u64> = Vec::new();
            let mut user_us_id_col: Vec<u64> = Vec::new();
            let mut user_msg_col: Vec<String> = Vec::new();
            let mut user_user_id_col: Vec<u64> = Vec::new();
            let mut user_xact_col: Vec<f64> = Vec::new();
            let mut user_share_col: Vec<f64> = Vec::new();
            let mut user_distance_col: Vec<Option<u64>> = Vec::new();

            let mut jobs_processed_in_batch = 0usize;
            let mut total_jobs = 0usize;

            // バッチをファイルに書き出すクロージャ相当の処理はインラインで実行
            while let Some(result) = rx.recv().await {
                let msg_str = env.message_space.index_to_string(result.msg_seq_index, env.num_rounds);

                result_dm_id_col.push(result.dm_id as u64);
                result_ip_col.push(result.ip as u64);
                result_us_id_col.push(result.us_id as u64);
                result_msg_col.push(msg_str.clone());
                result_mean_xact_col.push(result.mean_num_xact);
                // ★変更3
                result_mean_share_col.push(result.mean_num_share);

                // user_action_results は compute 側で非ゼロのみに絞り込み済み
                // （output_user_analysis=false のときは compute 側で空 Vec になっている）
                for user_res in result.user_action_results {
                    user_ip_col.push(result.ip as u64);
                    user_us_id_col.push(result.us_id as u64);
                    user_msg_col.push(msg_str.clone());
                    user_user_id_col.push(user_res.user_id as u64);
                    user_xact_col.push(user_res.num_xact);
                    user_share_col.push(user_res.num_share);
                    user_distance_col.push(user_res.distance_from_ip.map(|d| d as u64));
                }

                jobs_processed_in_batch += 1;
                total_jobs += 1;

                if jobs_processed_in_batch >= batch_size {
                    let result_batch = RecordBatch::try_new(
                        result_schema.clone(),
                        vec![
                            Arc::new(UInt64Array::from(std::mem::take(&mut result_dm_id_col))) as ArrayRef,
                            Arc::new(UInt64Array::from(std::mem::take(&mut result_ip_col))),
                            Arc::new(UInt64Array::from(std::mem::take(&mut result_us_id_col))),
                            Arc::new(StringArray::from(std::mem::take(&mut result_msg_col))),
                            Arc::new(Float64Array::from(std::mem::take(&mut result_mean_xact_col))),
                            // ★変更3
                            Arc::new(Float64Array::from(std::mem::take(&mut result_mean_share_col))),
                        ],
                    ).unwrap();
                    result_writer.write(&result_batch).await.unwrap();

                    // ★変更1: 出力する設定のときだけ user_analysis を書く
                    if let Some(writer) = user_analysis_writer.as_mut() {
                        let user_analysis_batch = RecordBatch::try_new(
                            user_analysis_schema.clone(),
                            vec![
                                Arc::new(UInt64Array::from(std::mem::take(&mut user_ip_col))) as ArrayRef,
                                Arc::new(UInt64Array::from(std::mem::take(&mut user_us_id_col))),
                                Arc::new(StringArray::from(std::mem::take(&mut user_msg_col))),
                                Arc::new(UInt64Array::from(std::mem::take(&mut user_user_id_col))),
                                Arc::new(Float64Array::from(std::mem::take(&mut user_xact_col))),
                                Arc::new(Float64Array::from(std::mem::take(&mut user_share_col))),
                                Arc::new(UInt64Array::from(std::mem::take(&mut user_distance_col))),
                            ],
                        ).unwrap();
                        writer.write(&user_analysis_batch).await.unwrap();
                    }

                    jobs_processed_in_batch = 0;
                    println!(
                        "[{:?}] {} jobs written ({:.0} jobs/s)",
                        started.elapsed(),
                        total_jobs,
                        total_jobs as f64 / started.elapsed().as_secs_f64().max(1e-9),
                    );
                }
            }

            // 残りのバッチを書き出す
            if !result_dm_id_col.is_empty() {
                let result_batch = RecordBatch::try_new(
                    result_schema.clone(),
                    vec![
                        Arc::new(UInt64Array::from(result_dm_id_col)) as ArrayRef,
                        Arc::new(UInt64Array::from(result_ip_col)),
                        Arc::new(UInt64Array::from(result_us_id_col)),
                        Arc::new(StringArray::from(result_msg_col)),
                        Arc::new(Float64Array::from(result_mean_xact_col)),
                        // ★変更3
                        Arc::new(Float64Array::from(result_mean_share_col)),
                    ],
                ).unwrap();
                result_writer.write(&result_batch).await.unwrap();

                // ★変更1
                if let Some(writer) = user_analysis_writer.as_mut() {
                    let user_analysis_batch = RecordBatch::try_new(
                        user_analysis_schema.clone(),
                        vec![
                            Arc::new(UInt64Array::from(user_ip_col)) as ArrayRef,
                            Arc::new(UInt64Array::from(user_us_id_col)),
                            Arc::new(StringArray::from(user_msg_col)),
                            Arc::new(UInt64Array::from(user_user_id_col)),
                            Arc::new(Float64Array::from(user_xact_col)),
                            Arc::new(Float64Array::from(user_share_col)),
                            Arc::new(UInt64Array::from(user_distance_col)),
                        ],
                    ).unwrap();
                    writer.write(&user_analysis_batch).await.unwrap();
                }
            }

            let metadata = vec![
                KeyValue::new("version_core".to_string(), Some(core::get_version().to_string())),
                KeyValue::new("version_runner".to_string(), Some(env!("CARGO_PKG_VERSION").to_string())),
                KeyValue::new("version".to_string(), Some(version.to_string())),
            ];
            for kv in metadata.iter() {
                result_writer.append_key_value_metadata(kv.clone());
            }
            result_writer.finish().await.unwrap();
            // ★変更1: writer があるときだけ finish
            if let Some(mut writer) = user_analysis_writer {
                for kv in metadata {
                    writer.append_key_value_metadata(kv);
                }
                writer.finish().await.unwrap();
            }

            Ok::<(), ParquetError>(())
        });

        // ---- producer task ----
        let tx_handle = tokio::spawn(async move {
            let internal_mean_stimulus = self.env.message_space.internal_mean_stimulus();
            let external_mean_stimulus = self.env.message_space.external_mean_stimulus();

            let user_rngs = self
                .env
                .user_sampling
                .sample_with(|rng| (SmallRng::from_rng(rng), SmallRng::from_rng(rng)));

            // ★変更2: config.toml に ip_indices があれば固定リストを使い、
            //          なければ従来通りランダムサンプリングする
            let ips: Vec<usize> = match &self.env.ip_indices {
                Some(fixed) => {
                    // 範囲チェック（範囲外の index があれば即座に分かるように panic）
                    for &ip in fixed {
                        assert!(
                            ip < self.env.num_nodes,
                            "ip_indices の {} は範囲外です (num_nodes = {})",
                            ip,
                            self.env.num_nodes
                        );
                    }
                    println!("IP を固定リストで指定: {:?}", fixed);
                    fixed.clone()
                }
                None => self
                    .env
                    .ip_sampling
                    .custom(|rng, size| (0..self.env.num_nodes).choose_multiple(rng, size)),
            };

            // IP ごとの最短距離を一度だけ計算してキャッシュする（従来は全ジョブで再計算していた）
            let distances_by_ip: HashMap<usize, Arc<Vec<Option<usize>>>> = ips
                .iter()
                .map(|&ip| (ip, Arc::new(shortest_distances_from(&self.env.graph, ip))))
                .collect();

            let mut rng = SmallRng::seed_from_u64(self.env.diffusion_seed);
            let num_seqs = self.env.message_space.seq_size(self.env.num_rounds);

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
                    let distances = distances_by_ip[&ip].clone();
                    for seq_index in 0..num_seqs {
                        let job = Job {
                            graph: self.env.graph.clone(),
                            model: DiffusionModel {
                                id: model_id,
                                receipt_probability: self.env.receipt_probability,
                                user_states: user_states.clone(),
                                user_state_id: us_id,
                                ip,
                            },
                            distances: distances.clone(),
                            msg_seq_index: seq_index,
                            diffusion_sample_size: self.env.diffusion_sample_size,
                            rng: SmallRng::from_rng(&mut rng),
                            tx: self.tx.clone(),
                        };
                        let env = self.env.clone();
                        let permit = self.semaphore.clone().acquire_owned().await.unwrap();
                        tokio::spawn(async move {
                            job.compute(env).await;
                            drop(permit);
                        });
                    }
                    model_id += 1;
                }
            }
        });
        try_join!(rx_handle, tx_handle).unwrap();
    }
}

#[derive(Clone)]
struct DiffusionModel {
    id: usize,
    receipt_probability: f64,
    user_states: Arc<Vec<UserState>>,
    user_state_id: usize,
    ip: usize,
}

struct Job {
    graph: Arc<GraphB>,
    model: DiffusionModel,
    distances: Arc<Vec<Option<usize>>>,
    msg_seq_index: u64,
    diffusion_sample_size: usize,
    rng: SmallRng,
    tx: mpsc::Sender<JobResult>,
}

#[derive(Debug, Clone)]
struct UserActionResult {
    num_share: f64,
    num_xact: f64,
    user_id: usize,
    distance_from_ip: Option<usize>,
}

#[derive(Debug, Clone)]
struct JobResult {
    dm_id: usize,
    ip: usize,
    us_id: usize,
    msg_seq_index: u64,
    mean_num_xact: f64,
    // ★変更3: 内部行動の累積回数（系列全体・モンテカルロ平均）
    mean_num_share: f64,
    pub user_action_results: Vec<UserActionResult>,
}

impl Job {
    async fn compute(mut self, env: Arc<Environment>) {
        let (msgs, _) = env
            .message_space
            .seq_by_index(self.msg_seq_index, env.num_rounds);

        let result: Aggregator<f64> = monte_carlo_average(
            &self.graph,
            self.model.receipt_probability,
            &self.model.user_states,
            self.model.ip,
            &msgs,
            self.diffusion_sample_size,
            &mut self.rng,
        );

        // ★変更1: user_analysis を出力しない設定なら、個人単位の集計自体をスキップ
        //          （filter + collect のコストと channel ペイロードを削減 = 高速化）
        let user_action_results: Vec<UserActionResult> = if env.output_user_analysis {
            // 非ゼロのノードだけを残す（writer 側のフィルタを前倒し、channel ペイロードを削減）
            result
                .num_xact_of_users
                .iter()
                .enumerate()
                .zip(result.num_share_of_users.iter())
                .filter(|&((_, &num_xact), &num_share)| num_xact != 0.0 || num_share != 0.0)
                .map(|((user_id, &num_xact), &num_share)| UserActionResult {
                    user_id,
                    num_xact,
                    num_share,
                    distance_from_ip: self.distances[user_id],
                })
                .collect()
        } else {
            Vec::new()
        };

        let job_result = JobResult {
            dm_id: self.model.id,
            ip: self.model.ip,
            us_id: self.model.user_state_id,
            msg_seq_index: self.msg_seq_index,
            mean_num_xact: result.mean_num_xact,
            // ★変更3
            mean_num_share: result.mean_num_share,
            user_action_results,
        };

        self.tx.send(job_result).await.unwrap();
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
    /// How many jobs can be run in parallel
    max_job_size: usize,
    /// How many rows (jobs) are buffered before flushing a parquet batch
    #[clap(long, default_value = "100")]
    row_buffer_size: usize,
    /// Zstd compression level
    #[clap(long, default_value = "3")]
    zstd_level: i32,
}

pub async fn start(args: MyArgs, version: &'static str) -> anyhow::Result<()> {
    let t = std::time::Instant::now();
    let buf = fs::read_to_string(args.config_path).await?;
    let config = toml::from_str(&buf)?;
    let env = Environment::from_config(config);

    // channel capacity は並列ジョブ数に余裕を持たせ、batch flush 単位は row_buffer_size に統一
    let manager = JobManager::new(env, args.max_job_size, args.max_job_size.max(args.row_buffer_size));
    manager.run(args.output_path, args.zstd_level, args.row_buffer_size, version).await;
    println!("総実行時間: {:?}", t.elapsed());
    Ok(())
}