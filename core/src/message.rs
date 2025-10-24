use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Message {
    pub internal_stimulus: f64,
    pub external_stimulus: f64,
}

pub struct MessageSpace(Vec<Message>);

impl From<Vec<Message>> for MessageSpace {
    fn from(value: Vec<Message>) -> Self {
        Self(value)
    }
}

impl MessageSpace {
    pub fn internal_mean_stimulus(&self) -> f64 {
        let l = self.0.len() as f64;
        self.0.iter().map(|m| m.internal_stimulus).sum::<f64>() / l
    }

    pub fn external_mean_stimulus(&self) -> f64 {
        let l = self.0.len() as f64;
        self.0.iter().map(|m| m.external_stimulus).sum::<f64>() / l
    }

    #[inline]
    fn from_id(&self, id: u16) -> &Message {
        &self.0[id as usize]
    }

    #[inline]
    pub fn seq_size(&self, num_rounds: u32) -> u64 {
        (self.0.len() as u64).pow(num_rounds)
    }

    /// This space size $L$ and max round $T$ must satisfy $L <= 16$ and $\log_2 L \le 64/T$.
    pub fn seq_by_index(&self, mut index: u64, num_rounds: u32) -> (Vec<&Message>, String) {
        let num_rounds = num_rounds as usize;
        let mut msgs_ids = vec![0; num_rounds];
        let mut code = vec!['\0'; num_rounds];
        let l = self.0.len() as u64;
        for j in 0..num_rounds {
            let i = (index % l) as u16;
            index /= l;
            msgs_ids[j] = i;
            code[j] = char::from_digit(i as u32, 16).unwrap();
        }
        msgs_ids.reverse();
        (
            msgs_ids.into_iter().map(|id| self.from_id(id)).collect(),
            String::from_iter(code),
        )
    }

    pub fn index_to_string(&self, mut index: u64, num_rounds: u32) -> String {
        let num_rounds = num_rounds as usize;
        // let mut msgs_ids = vec![0; num_rounds];
        let mut code = vec!['\0'; num_rounds];
        let l = self.0.len() as u64;
        for j in 0..num_rounds {
            let i = (index % l) as u16;
            index /= l;
            // msgs_ids[j] = i;
            code[j] = char::from_digit(i as u32, 16).unwrap();
        }
        // msgs_ids.reverse();
        String::from_iter(code)
    }
}
