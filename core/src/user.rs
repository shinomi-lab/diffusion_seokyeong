use std::fmt::Display;

use crate::decision::{DecisionAny, DecisionOnce};
use crate::message::Message;

#[derive(Clone)]
pub struct UserState {
    pub internal_decision: DecisionAny,
    pub external_decision: DecisionOnce,
}

impl Display for UserState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "US[{}, {}]",
            self.internal_decision, self.external_decision
        )?;
        Ok(())
    }
}

impl UserState {
    pub fn new(internal_decision: DecisionAny, external_decision: DecisionOnce) -> Self {
        Self {
            internal_decision,
            external_decision,
        }
    }

    pub fn contact(&mut self, message: &Message) {
        self.internal_decision.update(message.internal_stimulus);
        self.external_decision.update(message.external_stimulus);
    }
}
