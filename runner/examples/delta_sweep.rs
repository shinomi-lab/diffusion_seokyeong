//! 推薦の露出の割引 δ を動かしたときの 1 ラウンドの広がり（`reccomend_system/推薦システム単純.md` 3.8 節の図 1 のデータ）。
//! `test/config.toml` の較正（β, γ, u0, T）と同じ方法で母集団を作り、δ × τ × メッセージ m0〜m3 で、
//! 推薦経由の受信数とシェア数の平均・標準誤差を CSV に書く。推薦の実装は `core::recommend`（本番と同じ）。
//! 乱数は (IP, 母集団, 試行) ごとに固定し、δ・τ・メッセージが違っても同じ乱数を使う（差が見やすくなる）。
//! 実行（リポジトリのルートで）:
//!   cargo run --release --example delta_sweep -p runner -- 出力.csv [p] [試行数]
use std::fmt::Write;

use rand::{SeedableRng, rngs::SmallRng, seq::IteratorRandom};
use runner::brute_force::config::Config;

use core::{
    decision::{DecisionAny, DecisionOnce},
    diffusion::{Workspace, monte_carlo_average},
    recommend::{RecParams, Recommender},
    user::UserState,
};
use graph_lib::prelude::Graph;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let out = a.first().expect("CSV の出力先を指定してください");
    let p: f64 = a.get(1).map_or(Ok(0.05), |s| s.parse())?;
    let trials: u64 = a.get(2).map_or(Ok(300), |s| s.parse())?;

    // test/config.toml のグラフだけ congress に差し替える（実データは別 PC）
    let toml = std::fs::read_to_string("test/config.toml")?
        .replace(
            "path = \"ENGB_edges.csv\"",
            "path = \"test/congress_network_graph.txt\"",
        )
        .replace("format=\"csv\"", "format=\"edgelist\"");
    let cfg: Config = toml::from_str(&toml)?;
    cfg.validate()?;
    let n = cfg.graph.node_count();
    let (mean_int, mean_ext) = (
        cfg.message_space.internal_mean_stimulus(),
        cfg.message_space.external_mean_stimulus(),
    );

    // IP は ip_sampling と同じ方法で選ぶ。母集団は user_sampling の最初の 2 組
    let ips = cfg
        .ip_sampling
        .custom(|rng, size| (0..n).choose_multiple(rng, size));
    let user_rngs = cfg
        .user_sampling
        .sample_with(|rng| (SmallRng::from_rng(rng), SmallRng::from_rng(rng)));
    let mut populations: Vec<Vec<UserState>> = Vec::new();
    for (mut irng, mut erng) in user_rngs.into_iter().take(2) {
        let internal = cfg.internal_parameters.sample_iter::<DecisionAny>(
            cfg.num_rounds,
            n,
            mean_int,
            &mut irng,
        );
        let external = cfg.external_parameters.sample_iter::<DecisionOnce>(
            cfg.num_rounds,
            n,
            mean_ext,
            &mut erng,
        );
        populations.push(
            internal
                .into_iter()
                .zip(external)
                .map(|(i, e)| UserState::new(i, e))
                .collect(),
        );
    }
    let num_msgs = cfg.message_space.len();
    let ebar0 = core::recommend::base_share_rate(&populations[0], mean_int);

    let mut csv = String::new();
    writeln!(
        csv,
        "# graph=congress n={n} p={p} eps=1 k=10 phi_max=3 mu=1 ips={} populations={} trials={trials} ebar0={ebar0}",
        ips.len(),
        populations.len()
    )?;
    writeln!(
        csv,
        "tau,delta,msg,rec_rcvd_mean,rec_rcvd_se,share_mean,share_se"
    )?;

    let mut ws = Workspace::new(n);
    // 1 設定（τ, δ, メッセージ）の全 (IP, 母集団, 試行) の平均と標準誤差
    let mut run = |tau: f64, delta: f64, off: bool, m: usize| -> anyhow::Result<[f64; 4]> {
        let params = RecParams {
            eps: 1.0,
            delta,
            tau,
            ..RecParams::default()
        };
        let (mut s, mut s2, mut cnt) = ([0f64; 2], [0f64; 2], 0f64);
        for (pi, states) in populations.iter().enumerate() {
            let rec = if off {
                None
            } else {
                Recommender::new(params, states, mean_int)
            };
            let (msgs, _) = cfg.message_space.seq_by_index(m as u64, 1);
            for (ii, &ip) in ips.iter().enumerate() {
                for t in 0..trials {
                    let mut rng =
                        SmallRng::seed_from_u64(((pi * 1000 + ii) as u64) * 1_000_003 + t);
                    let r = monte_carlo_average(
                        &cfg.graph,
                        p,
                        states,
                        ip,
                        &msgs,
                        rec.as_ref(),
                        1,
                        &mut rng,
                        &mut ws,
                    );
                    for (k, v) in [r.mean_num_rec_rcvd, r.mean_num_share]
                        .into_iter()
                        .enumerate()
                    {
                        s[k] += v;
                        s2[k] += v * v;
                    }
                    cnt += 1.0;
                }
            }
        }
        let f = |k: usize| {
            let mean = s[k] / cnt;
            (mean, ((s2[k] / cnt - mean * mean).max(0.0) / cnt).sqrt())
        };
        let (a, b) = (f(0), f(1));
        Ok([a.0, a.1, b.0, b.1])
    };

    let deltas: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
    for tau in [0.0, 1.0] {
        for &delta in &deltas {
            for m in 0..num_msgs {
                let r = run(tau, delta, false, m)?;
                writeln!(csv, "{tau},{delta},{m},{},{},{},{}", r[0], r[1], r[2], r[3])?;
            }
        }
    }
    // 整合性: δ = 0 は推薦なし（rec = None）と完全に一致する
    for m in 0..num_msgs {
        let on0 = run(0.0, 0.0, false, m)?;
        let off = run(0.0, 1.0, true, m)?;
        assert_eq!(on0, off, "δ = 0 が推薦なしと一致しない（m{m}）");
    }
    std::fs::write(out, csv)?;
    println!("wrote {out}（δ = 0 は推薦なしと一致: OK）ē0 = {ebar0}");
    Ok(())
}
