mod config;

use std::{env, sync::Arc};
use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray, UInt64Array};
use itertools::{Itertools, izip};
use parquet::{
    arrow::AsyncArrowWriter,
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
    format::KeyValue,
};
use rand::{SeedableRng, rngs::SmallRng, seq::IteratorRandom};
use tokio::{
    fs::{self, File},
    sync::mpsc,
    try_join,
};

use core::{diffusion::{monte_carlo_average, Aggregator}, message::MessageSpace, user::UserState};
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

    async fn run(self,output_path: PathBuf, zstd_level: i32, version: &'static str) {
        let mut rx = self.rx;
        let env = self.env.clone();
        let level = ZstdLevel::try_new(zstd_level).unwrap();

        
        let output_dir = output_path.parent().unwrap_or_else(|| std::path::Path::new(""));
        let result_path = output_path.clone();
        let user_analysis_path = output_dir.join("user_analysis.parquet");

        let result_file = File::create(result_path).await.unwrap();
        let user_analysis_file = File::create(user_analysis_path).await.unwrap();
        // spawns the writer task
        let rx_handle = tokio::spawn(async move {
            let mut results = Vec::new();
            while let Some(result) = rx.recv().await {
                results.push(result);
            }
            let (dm_id_col, ip_col, us_id_col, ms_col, xact_col) = results.clone()
                .into_iter()
                .map(|result| {
                    (
                        result.dm_id as u64,
                        result.ip as u64,
                        result.us_id as u64,
                        env.message_space
                            .index_to_string(result.msg_seq_index, env.num_rounds),
                        result.mean_num_xact
                    )
                })
                .multiunzip::<(Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>)>();

            let batch = RecordBatch::try_from_iter(vec![
                ("id", Arc::new(UInt64Array::from(dm_id_col)) as ArrayRef),
                ("ip", Arc::new(UInt64Array::from(ip_col.clone())) as ArrayRef),
                ("us_id", Arc::new(UInt64Array::from(us_id_col.clone())) as ArrayRef),
                ("msg", Arc::new(StringArray::from(ms_col.clone())) as ArrayRef),
                ("mean_num_xact", Arc::new(Float64Array::from(xact_col))),
            ])
            .unwrap();
            
            let mut writer = AsyncArrowWriter::try_new(
                result_file,
                batch.schema(),
                Some(
                    WriterProperties::builder()
                        .set_compression(Compression::ZSTD(level))
                        .build(),
                ),
            )
            .unwrap();
            writer.write(&batch).await.unwrap();

            // appends version information into the metadata
            writer.append_key_value_metadata(KeyValue::new(
                "version_core".to_string(),
                core::get_version().to_string(),
            ));
            writer.append_key_value_metadata(KeyValue::new(
                "version_runner".to_string(),
                env!("CARGO_PKG_VERSION").to_string(),
            ));
            writer.append_key_value_metadata(KeyValue::new(
                "version".to_string(),
                version.to_string(),
            ));
            writer.finish().await.unwrap();

            // --- `user_level_results.parquet` 用のベクター ---
            // 1. 最終的な列となる、空のベクターを初期化
            let mut ip_col: Vec<u64> = Vec::new();
            let mut us_id_col: Vec<u64> = Vec::new();
            let mut ms_col: Vec<String> = Vec::new();
            let mut user_id_col: Vec<u64> = Vec::new();
            let mut xact_users_col: Vec<f64> = Vec::new();
            let mut share_col: Vec<f64> = Vec::new();

        // 2. 全てのJobResultをループで処理
            for result in results { // `result`は1つのシミュレーション結果
                let msg_col =env.message_space
                            .index_to_string(result.msg_seq_index, env.num_rounds);
                // 内部のユーザーリストをループ処理
                for user_res in result.user_action_results {
                    // ユーザー1人分の行を追加するたびに、
                    // シミュレーション情報も「毎回」追加（複製）する
                    // 「多」の方のデータを追加
                    user_id_col.push(user_res.user_id as u64);
                    xact_users_col.push(user_res.num_xact);
                    share_col.push(user_res.num_share);
                    ip_col.push(result.ip as u64);
                    us_id_col.push(result.us_id as u64);
                    ms_col.push(msg_col.clone()); // 文字列はcloneが必要
                }
            }

            // このループが終わると、全ての `_col` ベクターは
            // 「全ユーザーの総数」と等しい、同じ長さになります。
            let batch2 = RecordBatch::try_from_iter(vec![
                ("ip", Arc::new(UInt64Array::from(ip_col.clone())) as ArrayRef),
                ("us_id", Arc::new(UInt64Array::from(us_id_col.clone())) as ArrayRef),
                ("msg", Arc::new(StringArray::from(ms_col.clone())) as ArrayRef),
                ("user_id", Arc::new(UInt64Array::from(user_id_col)) as ArrayRef),
                ("num_xact_of_users", Arc::new(Float64Array::from(xact_users_col)) as ArrayRef),
                ("num_share_of_users", Arc::new(Float64Array::from(share_col)) as ArrayRef),
            ]).expect("Failed to create RecordBatch from flattened data");


            let mut writer2 = AsyncArrowWriter::try_new(
                user_analysis_file,
                batch2.schema(),
                Some(
                    WriterProperties::builder()
                        .set_compression(Compression::ZSTD(level))
                        .build(),
                ),
            )
            .unwrap();
            writer2.write(&batch2).await.unwrap();

            // appends version information into the metadata for the second file
            writer2.append_key_value_metadata(KeyValue::new(
                "version_core".to_string(),
                core::get_version().to_string(),
            ));
            writer2.append_key_value_metadata(KeyValue::new(
                "version_runner".to_string(),
                env!("CARGO_PKG_VERSION").to_string(),
            ));
            writer2.append_key_value_metadata(KeyValue::new(
                "version".to_string(),
                version.to_string(),
            ));
            writer2.finish().await.unwrap();
        });

        // spawns the job tasks
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

}
#[derive(Debug, Clone)]
struct JobResult {
    dm_id: usize,
    ip: usize,
    us_id: usize,
    msg_seq_index: u64,
    mean_num_xact: f64,
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
            &self.model.user_states.clone(),
            self.model.ip,
            &msgs,
            self.diffusion_sample_size,
            &mut self.rng,
        );
        let user_action_results: Vec<UserActionResult> = result.num_xact_of_users.iter()
            .enumerate()
            .zip(result.num_share_of_users.iter())
            .map(|((user_id,num_xact), num_share)|{
                UserActionResult{
                    user_id,
                    num_xact: *num_xact,
                    num_share: *num_share,
                }
            })
            .collect();

        let job_result = JobResult {
            dm_id: self.model.id,
            ip: self.model.ip,
            us_id: self.model.user_state_id,
            msg_seq_index: self.msg_seq_index,
            mean_num_xact: result.mean_num_xact,
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
    /// How many rows can be buffered in the writer thread
    #[clap(long, default_value = "100")]
    row_buffer_size: usize,
    /// Zstd compression level
    #[clap(long, default_value = "3")]
    zstd_level: i32,
}

pub async fn start(args: MyArgs, version: &'static str) -> anyhow::Result<()> {
    let buf = fs::read_to_string(args.config_path).await?;
    let config = toml::from_str(&buf)?;
    let env = Environment::from_config(config);

    let manager = JobManager::new(env, args.max_job_size, args.row_buffer_size);
    manager.run(args.output_path, args.zstd_level, version).await;
    Ok(())
}
