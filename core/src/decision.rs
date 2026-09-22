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

    /// 診断・分析用の読み取り専用アクセサ（挙動には影響しない）。
    pub fn weight(&self) -> f64 {
        self.inner.weight()
    }

    /// 診断・分析用の読み取り専用アクセサ（挙動には影響しない）。
    pub fn utility(&self) -> f64 {
        self.inner.utility()
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
    ///
    /// この契約は `MessageSpace` の構築時（`core::message`）と config のバリデーション
    /// （`runner::brute_force::config`）で守られる。ここではデバッグビルドでのみ再確認する。
    fn update(&mut self, stimulus: f64) {
        debug_assert!(
            (-1.0..=1.0).contains(&stimulus),
            "stimulus must be in [-1, 1], got {stimulus}"
        );
        self.utility = self.weight * self.utility + (1.0 - self.weight) * stimulus;
    }

    fn decided(&self) -> bool {
        // 0 との相対比較は実質「ちょうど 0 か」の判定になる（u ≈ 0 は未決定として扱う）。
        if relative_eq!(&self.utility, &0.0) {
            false
        } else {
            self.utility > 0.0
        }
    }

    /// 効用更新の重み w（診断・分析用の読み取り専用アクセサ。挙動には影響しない）。
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// 現在の効用 u（診断・分析用の読み取り専用アクセサ。挙動には影響しない）。
    pub fn utility(&self) -> f64 {
        self.utility
    }
}

#[cfg(test)]
mod tests {
    use super::{DecisionAny, DecisionBase, DecisionOnce};

    /// 効用更新則 u' = w*u + (1-w)*s のとおりに動くこと。
    #[test]
    fn updates_utility_by_exponential_moving_average() {
        let mut d = DecisionAny::from(DecisionBase::new(0.2, -1.0));
        d.update(0.8);
        // 0.2 * (-1.0) + 0.8 * 0.8 = 0.44
        assert!((d.utility() - 0.44).abs() < 1e-12);
        assert!(d.decided());
    }

    /// DecisionAny は毎回再評価されるので true と false を何度でも行き来する。
    #[test]
    fn decision_any_can_toggle_repeatedly() {
        let mut d = DecisionAny::from(DecisionBase::new(0.0, -1.0));
        // w = 0 なので u' はそのまま直近の刺激になる。
        d.update(0.5);
        assert!(d.decided());
        d.update(-0.5);
        assert!(!d.decided());
        d.update(0.5);
        assert!(d.decided(), "一度 false になっても再び発火できる");
    }

    /// DecisionOnce は生涯1回しか decided を返さない（外部行動の不可逆性）。
    #[test]
    fn decision_once_fires_at_most_once() {
        let mut d = DecisionOnce::from(DecisionBase::new(0.0, -1.0));
        d.update(0.5);
        assert!(d.decided(), "初回は発火する");
        d.update(0.5);
        assert!(!d.decided(), "2回目以降は同じ刺激でも発火しない");
        d.update(-0.5);
        assert!(!d.decided());
        d.update(0.5);
        assert!(!d.decided());
    }

    /// 効用がちょうど 0 のときは未決定として扱う。
    #[test]
    fn exactly_zero_utility_is_not_decided() {
        let mut d = DecisionAny::from(DecisionBase::new(0.0, -1.0));
        d.update(0.0);
        assert_eq!(d.utility(), 0.0);
        assert!(!d.decided());
    }
}
