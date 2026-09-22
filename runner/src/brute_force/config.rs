use anyhow::{Context, bail, ensure};
use graph_lib::prelude::{Graph, GraphB};
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Pareto};
use serde::Deserialize;
use serde_with::{TryFromInto, serde_as};
use util::{GraphConfig, RngSampling};

use core::{
    decision::DecisionBase,
    message::{Message, MessageSpace},
};

#[serde_as]
#[derive(Deserialize)]
pub struct Config {
    #[serde_as(as = "TryFromInto<GraphConfig>")]
    pub graph: GraphB,
    pub internal_parameters: DecisionParameters,
    pub external_parameters: DecisionParameters,
    pub user_sampling: RngSampling,
    pub ip_sampling: RngSampling,
    #[serde_as(as = "TryFromInto<Vec<Message>>")]
    pub message_space: MessageSpace,
    pub num_rounds: u32,
    pub receipt_probability: f64,
    pub diffusion_sample_size: usize,
}

impl Config {
    /// 設定値の整合性を検証する。
    ///
    /// 型レベルでは表現できない制約（`gamma < beta`、確率の範囲、サンプル数と
    /// ノード数の関係、`L^T` が u64 に収まること）をここでまとめて確認し、
    /// 実行の途中で不可解な panic を起こす代わりに、原因の分かるエラーで止める。
    /// 個々のメッセージの刺激値が \[-1, 1\] に入っているかは `MessageSpace` の
    /// 構築時（deserialize 時）に既に検証されている。
    pub fn validate(&self) -> anyhow::Result<()> {
        self.internal_parameters
            .validate()
            .context("internal_parameters が不正です")?;
        self.external_parameters
            .validate()
            .context("external_parameters が不正です")?;

        ensure!(
            (0.0..=1.0).contains(&self.receipt_probability),
            "receipt_probability は [0, 1] である必要があります (実際: {})",
            self.receipt_probability
        );
        ensure!(
            self.num_rounds >= 1,
            "num_rounds は 1 以上である必要があります"
        );
        ensure!(
            self.diffusion_sample_size >= 1,
            "diffusion_sample_size は 1 以上である必要があります"
        );
        ensure!(
            self.user_sampling.size >= 1,
            "user_sampling.size は 1 以上である必要があります"
        );

        let num_nodes = self.graph.node_count();
        ensure!(
            self.ip_sampling.size >= 1,
            "ip_sampling.size は 1 以上である必要があります"
        );
        ensure!(
            self.ip_sampling.size <= num_nodes,
            "ip_sampling.size ({}) がグラフのノード数 ({}) を超えています。\
             この場合サンプリングは黙って少ない数の IP しか返しません",
            self.ip_sampling.size,
            num_nodes
        );

        // seq_size() が計算する L^T が u64 に収まること。
        let l = self.message_space.len() as f64;
        ensure!(
            l.log2() * self.num_rounds as f64 <= 64.0,
            "log2(L) * num_rounds <= 64 を満たす必要があります \
             (L = {}, num_rounds = {} では系列数が u64 に収まりません)",
            self.message_space.len(),
            self.num_rounds
        );

        Ok(())
    }
}

/// gamma < beta
#[derive(Deserialize)]
pub struct DecisionParameters {
    initial_utility: f64,
    beta: f64,
    gamma: f64,
}

impl DecisionParameters {
    /// `sample_iter` が要求する前提を検証する。
    ///
    /// これを満たさないと `Pareto::new` が失敗し、原因の分かりにくい panic になる。
    fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            (-1.0..=1.0).contains(&self.initial_utility),
            "initial_utility は [-1, 1] である必要があります (実際: {})",
            self.initial_utility
        );
        ensure!(
            self.beta > 0.0,
            "beta は正である必要があります (実際: {})",
            self.beta
        );
        ensure!(
            self.gamma > 0.0,
            "gamma は正である必要があります (実際: {})",
            self.gamma
        );
        if self.gamma >= self.beta {
            bail!(
                "gamma < beta である必要があります (実際: gamma = {}, beta = {})。\
                 パレート分布の shape = gamma / (beta - gamma) が発散します",
                self.gamma,
                self.beta
            );
        }
        Ok(())
    }

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
}

fn initialize_weight(mean_stimulus: f64, initial_utility: f64, responsiveness: f64) -> f64 {
    (mean_stimulus / (mean_stimulus - initial_utility)).powf(responsiveness)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト用の最小構成。`graph` はリポジトリ同梱の小さなグラフを使う。
    fn config_toml(extra: &str) -> String {
        format!(
            r#"
message_space = [
{{ internal_stimulus = 0.20, external_stimulus = 0.05 }},
{{ internal_stimulus = 0.05, external_stimulus = 0.20 }},
]
num_rounds = 2
receipt_probability = 0.5
graph = {{ path = "../test/congress_network_graph.txt", directed = true, transposed = true, format = "edgelist" }}
ip_sampling = {{ size = 2, seed = 0 }}
user_sampling = {{ size = 2, seed = 0 }}
diffusion_sample_size = 5
{extra}
"#
        )
    }

    fn parse(extra: &str) -> anyhow::Result<Config> {
        Ok(toml::from_str::<Config>(&config_toml(extra))?)
    }

    const OK_PARAMS: &str = r#"
internal_parameters = { beta = 0.8, gamma = 0.6, initial_utility = -1.0 }
external_parameters = { beta = 0.8, gamma = 0.6, initial_utility = -1.0 }
"#;

    #[test]
    fn accepts_valid_config() {
        let config = parse(OK_PARAMS).expect("パースできるはず");
        config.validate().expect("検証を通るはず");
    }

    #[test]
    fn rejects_gamma_not_less_than_beta() {
        let config = parse(
            r#"
internal_parameters = { beta = 0.6, gamma = 0.6, initial_utility = -1.0 }
external_parameters = { beta = 0.8, gamma = 0.6, initial_utility = -1.0 }
"#,
        )
        .unwrap();
        let err = config.validate().unwrap_err();
        assert!(
            format!("{err:#}").contains("gamma < beta"),
            "gamma >= beta を説明するエラーであること: {err:#}"
        );
    }

    /// 刺激値が定義域外の場合は deserialize の時点で弾かれる。
    #[test]
    fn rejects_out_of_range_stimulus_at_parse_time() {
        let toml = config_toml(OK_PARAMS).replace("0.20", "1.50");
        // Config は Debug を要求しないよう unwrap_err を使わずに取り出す。
        let err = match toml::from_str::<Config>(&toml) {
            Ok(_) => panic!("定義域外の刺激値が受理されてしまった"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("[-1, 1]"),
            "定義域を説明するエラーであること: {err}"
        );
    }

    #[test]
    fn rejects_ip_sampling_larger_than_node_count() {
        let mut toml = config_toml(OK_PARAMS);
        toml = toml.replace("ip_sampling = { size = 2", "ip_sampling = { size = 100000");
        let config = toml::from_str::<Config>(&toml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(
            format!("{err:#}").contains("ノード数"),
            "ノード数超過を説明するエラーであること: {err:#}"
        );
    }

    #[test]
    fn rejects_receipt_probability_out_of_range() {
        let toml = config_toml(OK_PARAMS).replace("receipt_probability = 0.5", "receipt_probability = 1.5");
        let config = toml::from_str::<Config>(&toml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(format!("{err:#}").contains("receipt_probability"), "{err:#}");
    }
}
