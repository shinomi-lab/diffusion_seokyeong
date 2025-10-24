use graph_lib::prelude::GraphB;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Pareto};
use serde::Deserialize;
use serde_with::{FromInto, TryFromInto, serde_as};
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
    #[serde_as(as = "FromInto<Vec<Message>>")]
    pub message_space: MessageSpace,
    pub num_rounds: u32,
    pub receipt_probability: f64,
    pub diffusion_sample_size: usize,
}

/// gamma < beta
#[derive(Deserialize)]
pub struct DecisionParameters {
    initial_utility: f64,
    beta: f64,
    gamma: f64,
}

impl DecisionParameters {
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
            .unwrap()
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
