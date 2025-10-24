use std::fmt::Display;

use approx::relative_eq;

#[derive(Clone)]
pub struct DecisionOnce {
    inner: DecisionBase,
    decided: bool,
    done: bool,
}

impl Display for DecisionOnce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "D1({:.4},{:+.4},{:>5})",
            self.inner.weight, self.inner.utility, self.decided
        )?;
        Ok(())
    }
}

impl From<DecisionBase> for DecisionOnce {
    fn from(inner: DecisionBase) -> Self {
        Self {
            decided: false,
            done: false,
            inner,
        }
    }
}

impl DecisionOnce {
    pub fn update(&mut self, stimulus: f64) {
        self.inner.update(stimulus);
        if !self.done {
            self.decided = self.inner.decided();
            if self.decided {
                self.done = true;
            }
        } else {
            self.decided = false;
        }
    }

    pub fn decided(&self) -> bool {
        self.decided
    }
}

#[derive(Clone)]
pub struct DecisionAny {
    inner: DecisionBase,
    decided: bool,
}

impl Display for DecisionAny {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "D*({:.4},{:+.4},{:>5})",
            self.inner.weight, self.inner.utility, self.decided
        )?;
        Ok(())
    }
}

impl From<DecisionBase> for DecisionAny {
    fn from(inner: DecisionBase) -> Self {
        Self {
            inner,
            decided: false,
        }
    }
}

impl DecisionAny {
    pub fn update(&mut self, stimulus: f64) {
        self.inner.update(stimulus);
        self.decided = self.inner.decided();
    }

    pub fn decided(&self) -> bool {
        self.decided
    }
}

#[derive(Clone)]
pub struct DecisionBase {
    weight: f64,
    utility: f64,
}

impl DecisionBase {
    pub fn new(weight: f64, utility: f64) -> Self {
        Self { weight, utility }
    }

    /// `stimulus` must be in \[-1, 1\] range.
    fn update(&mut self, stimulus: f64) {
        self.utility = self.weight * self.utility + (1.0 - self.weight) * stimulus;
    }

    fn decided(&self) -> bool {
        if relative_eq!(&self.utility, &0.0) {
            false
        } else {
            self.utility > 0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DecisionAny, DecisionBase, DecisionOnce};

    #[test]
    fn test_update() {
        let w = 0.2;
        let mut any = DecisionAny::from(DecisionBase::new(w, -1.0));
        let mut once = DecisionOnce::from(DecisionBase::new(w, -1.0));
        any.update(0.8);
        once.update(0.2);
        println!("any: {}", any.inner.utility);
        println!("once:{}", once.inner.utility);
    }
}
