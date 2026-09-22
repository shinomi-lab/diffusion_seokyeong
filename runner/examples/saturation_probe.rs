//! 内部軸の平均刺激 s̄ を連続的に動かしたとき、意思決定モデルが
//! 「1回（または2回）の受信でどれだけ発火するか」を、実装コード
//! （core::decision, runner::brute_force::config::DecisionParameters）
//! をそのまま使って直接測定する診断ツール。
//!
//! 「研究リソースとか/内部軸の平均刺激を上げた場合_シミュレーション.pdf」の
//! Fig.1/2/4 に対応する数値を、独立の近似実装ではなく本物のコードで再現する。
//!
//! 実行: cargo run --release --example saturation_probe
//! 標準出力に2つのCSVブロックを続けて出す（呼び出し側でリダイレクトして分割する）。

use core::decision::{DecisionAny, DecisionBase};
use core::message::Message;
use rand::{SeedableRng, rngs::SmallRng};
use runner::brute_force::config::DecisionParameters;

const N_USERS: usize = 200_000;
const NUM_ROUNDS: u32 = 5; // test/config.toml の num_rounds に合わせる
const DELTA: f64 = 0.6; // test/config.toml の message_space が再現する δ（PDF#1 参照）
const Z: [f64; 4] = [1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0]; // m0(内部最強)..m3(内部最弱)
const S_BAR_LEVELS: [f64; 7] = [0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5];

/// test/config.toml と同一の beta/gamma/initial_utility を持つ DecisionParameters を構築。
/// DecisionParameters のフィールドは private だが Deserialize を実装しているため、
/// TOML 経由でそのまま構築する（実装コードの改変なし）。
fn make_params() -> DecisionParameters {
    toml::from_str("initial_utility = -1.0\nbeta = 0.8\ngamma = 0.6\n")
        .expect("failed to build DecisionParameters")
}

fn messages_for(s_bar: f64) -> Vec<Message> {
    Z.iter()
        .map(|&z| Message {
            internal_stimulus: s_bar * (1.0 + DELTA * z),
            external_stimulus: s_bar * (1.0 + DELTA * -z),
        })
        .collect()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

fn main() {
    let params = make_params();

    println!("# fire_rates");
    println!("s_bar,message_id,internal_stimulus,fire_rate_1shot,fire_rate_2shot");

    // weight_stats のヘッダ・データは全 s_bar のループが終わった後にまとめて出力する
    let mut weight_rows: Vec<(f64, f64, f64, f64, f64)> = Vec::new();

    for &s_bar in S_BAR_LEVELS.iter() {
        // s̄ ごとに独立・決定論的な乱数列で 200,000 人のベース意思決定モデルをサンプリング。
        // sample_iter は core::decision::DecisionBase をそのまま返せる
        // （std の `impl<T> From<T> for T` により D = DecisionBase が使える）。
        let mut rng = SmallRng::seed_from_u64(0);
        let bases: Vec<DecisionBase> = params
            .sample_iter::<DecisionBase>(NUM_ROUNDS, N_USERS, s_bar, &mut rng)
            .into_iter()
            .collect();

        // w の分布統計（このメッセージに依存しない、s̄ だけで決まる値）
        let mut weights: Vec<f64> = bases.iter().map(|b| b.weight()).collect();
        weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let w_mean = weights.iter().sum::<f64>() / weights.len() as f64;
        let w_median = percentile(&weights, 0.5);
        let w_p10 = percentile(&weights, 0.10);
        let w_p90 = percentile(&weights, 0.90);
        weight_rows.push((s_bar, w_median, w_mean, w_p10, w_p90));

        let messages = messages_for(s_bar);
        for (mid, m) in messages.iter().enumerate() {
            let mut fired_1 = 0usize;
            let mut fired_2 = 0usize;
            for base in &bases {
                let mut d1 = DecisionAny::from(base.clone());
                d1.update(m.internal_stimulus);
                if d1.decided() {
                    fired_1 += 1;
                }
                // 2回目も同じ刺激を受け続けたと仮定（PDF#2 の「2回目までに発火」に対応）
                d1.update(m.internal_stimulus);
                if d1.decided() {
                    fired_2 += 1;
                }
            }
            let rate1 = fired_1 as f64 / bases.len() as f64;
            let rate2 = fired_2 as f64 / bases.len() as f64;
            println!(
                "{s_bar},{mid},{},{rate1},{rate2}",
                m.internal_stimulus
            );
        }
    }

    println!("# weight_stats_table");
    println!("s_bar,w_median,w_mean,w_p10,w_p90");
    for (s_bar, w_median, w_mean, w_p10, w_p90) in weight_rows {
        println!("{s_bar},{w_median},{w_mean},{w_p10},{w_p90}");
    }
}
