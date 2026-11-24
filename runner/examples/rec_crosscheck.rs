//! T-W09（ビューア `recommender-v3-design.md` 10 章）: ビューアの方式 wave との統計的な一致を確かめる参照側。
//! ビューアの `sim-api/examples/wave_crosscheck.rs` が書いた母集団の CSV（w_int, w_ext, u_int, u_ext）を読み、
//! 同じグラフ・IP・設定で推薦チャネル付きの 1 ラウンドを試行数ぶん回して、平均と標準誤差を出す。
//! 実行（リポジトリのルートで）:
//!   cargo run --release --example rec_crosscheck -p runner -- グラフ.txt 母集団.csv IP p 平均内部刺激 内部刺激 外部刺激 [試行数]
use graph_lib::prelude::{Graph, GraphB};
use rand::{SeedableRng, rngs::SmallRng};
use util::GraphConfig;

use core::{
    decision::{DecisionAny, DecisionBase, DecisionOnce},
    diffusion::{Workspace, monte_carlo_average},
    message::Message,
    recommend::{RecParams, Recommender},
    user::UserState,
};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize| -> anyhow::Result<f64> { Ok(a[i].parse()?) };
    let (path, pop_csv) = (&a[0], &a[1]);
    let ip: usize = a[2].parse()?;
    let (p, mean_int) = (arg(3)?, arg(4)?);
    let msg = Message {
        internal_stimulus: arg(5)?,
        external_stimulus: arg(6)?,
    };
    let trials: u64 = a.get(7).map_or(Ok(2000), |s| s.parse())?;

    let cfg: GraphConfig = toml::from_str(&format!(
        "path = {path:?}\ndirected = true\ntransposed = true\nformat = \"edgelist\"\n"
    ))?;
    let graph: GraphB = cfg.try_into()?;
    let states: Vec<UserState> = std::fs::read_to_string(pop_csv)?
        .lines()
        .skip(1)
        .map(|l| {
            let v: Vec<f64> = l.split(',').map(|x| x.parse().unwrap()).collect();
            UserState::new(
                DecisionAny::from(DecisionBase::new(v[0], v[2])),
                DecisionOnce::from(DecisionBase::new(v[1], v[3])),
            )
        })
        .collect();
    anyhow::ensure!(
        states.len() == graph.node_count(),
        "母集団の人数がグラフのノード数と違います"
    );

    // ビューアの RecommenderConfig::default と同じ値
    let params = RecParams {
        eps: 1.0,
        delta: 1.0,
        tau: 0.0,
        phi_max: 3.0,
        k: 10.0,
        mu: 1.0,
    };
    let rec = Recommender::new(params, &states, mean_int).expect("推薦が有効な設定");
    let mut ws = Workspace::new(graph.node_count());
    let mut sum = [0f64; 3];
    let mut sq = [0f64; 3];
    for t in 0..trials {
        // 試行数 1 の monte_carlo_average = 初期状態から 1 ラウンド（a_1 = ē0）
        let mut rng = SmallRng::seed_from_u64(t ^ 0xC0FFEE);
        let r = monte_carlo_average(
            &graph,
            p,
            &states,
            ip,
            &[msg],
            Some(&rec),
            1,
            &mut rng,
            &mut ws,
        );
        for (k, v) in [r.mean_num_share, r.mean_num_xact, r.mean_num_rec_rcvd]
            .into_iter()
            .enumerate()
        {
            sum[k] += v;
            sq[k] += v.powi(2);
        }
    }
    let n = trials as f64;
    println!(
        "# graph={path} ip={ip} p={p} ebar0={} trials={trials}",
        rec.ebar0
    );
    for (k, name) in ["share", "xact", "rec_rcvd"].iter().enumerate() {
        let mean = sum[k] / n;
        let se = ((sq[k] / n - mean * mean).max(0.0) / n).sqrt();
        println!("{name},{mean},{se}");
    }
    Ok(())
}
