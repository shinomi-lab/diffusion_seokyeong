use std::{fs::File, io, path::PathBuf};

use graph_lib::{
    io::ParseOption,
    prelude::{DiGraphB, GraphB, UndiGraphB},
};

use rand::{SeedableRng, rngs::SmallRng};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct RngSampling {
    pub size: usize,
    pub seed: u64,
}

impl RngSampling {
    pub fn sample_with<T, F: FnMut(&mut SmallRng) -> T>(&self, mut f: F) -> Vec<T> {
        let mut rng = SmallRng::seed_from_u64(self.seed);
        (0..self.size).map(|_| f(&mut rng)).collect()
    }

    pub fn custom<T, F: FnMut(&mut SmallRng, usize) -> T>(&self, mut f: F) -> T {
        let mut rng = SmallRng::seed_from_u64(self.seed);
        f(&mut rng, self.size)
    }
}

#[derive(Deserialize)]
pub struct GraphConfig {
    path: PathBuf,
    directed: bool,
    transposed: bool,
}

impl TryFrom<GraphConfig> for GraphB {
    type Error = io::Error;

    fn try_from(value: GraphConfig) -> Result<Self, Self::Error> {
        let builder = graph_lib::io::ParseBuilder::new(
            File::open(&value.path)
                .expect(&format!("{} is not found.", &value.path.to_string_lossy())),
            graph_lib::io::DataFormat::EdgeList,
            ParseOption {
                transposed: value.transposed,
                ..ParseOption::default()
            },
        );
        if value.directed {
            Ok(GraphB::Di(builder.parse::<DiGraphB>()?))
        } else {
            Ok(GraphB::Ud(builder.parse::<UndiGraphB>()?))
        }
    }
}
