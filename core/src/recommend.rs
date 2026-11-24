//! 推薦チャネル（Rモデル）。定義は `reccomend_system/推薦システム単純.md` の3節、
//! 実装仕様は `研究リソースとか/推薦チャネル_実装仕様.md`。
//!
//! 各波のフォロー経路のあとに、反応率 ê の良さに応じた数だけフォロー外へ届ける。
//! `eps * delta = 0` なら何もせず乱数も引かないので、推薦なしの結果とビット一致する。

use graph_lib::prelude::{Graph, GraphB};
use rand::Rng;
use serde::Deserialize;

use crate::user::UserState;

/// 推薦のパラメータ（3.1節の表）。TOML の `[recommender]` からそのまま読む。
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RecParams {
    /// 推薦の強さ ε（0 で推薦なし）。
    pub eps: f64,
    /// フォロー外の露出の割引 δ（R4。1 で割引なし）。
    pub delta: f64,
    /// しきい値 τ（基準 ē0 の何倍に届けば推薦するか）。
    pub tau: f64,
    /// 推薦量の倍率の上限 φ_max。
    pub phi_max: f64,
    /// 擬似カウント k（事前分布の強さ）。
    pub k: f64,
    /// R3：アカウント記憶の持続 μ（1 で記憶なし）。
    pub mu: f64,
}

impl Default for RecParams {
    fn default() -> Self {
        Self {
            eps: 0.0,
            delta: 1.0,
            tau: 0.0,
            phi_max: 3.0,
            k: 10.0,
            mu: 1.0,
        }
    }
}

impl RecParams {
    /// 推薦が働くか。`eps * delta = 0` なら推薦量は常に 0 なので、処理ごと飛ばす。
    pub fn is_on(&self) -> bool {
        self.eps * self.delta > 0.0
    }

    /// 反応の良さ φ(x)。基準の τ 倍に届かなければ 0、届けば x（上限 φ_max）。
    pub fn phi(&self, x: f64) -> f64 {
        if x < self.tau {
            0.0
        } else {
            x.min(self.phi_max)
        }
    }
}

/// 1つのユーザ状態集合に対する推薦の設定。基準の反応率 ē0 は集合ごとに決まる。
#[derive(Debug, Clone, Copy)]
pub struct Recommender {
    pub params: RecParams,
    /// 基準の反応率 ē0。
    pub ebar0: f64,
}

impl Recommender {
    /// 推薦が働かない設定なら `None` を返す（呼び出し側は推薦なしの経路を通る）。
    pub fn new(
        params: RecParams,
        states: &[UserState],
        internal_mean_stimulus: f64,
    ) -> Option<Self> {
        params.is_on().then(|| Self {
            params,
            ebar0: base_share_rate(states, internal_mean_stimulus),
        })
    }

    /// R3 の更新 a_{t+1} = μ a_t + (1 − μ) ê_end。
    pub fn next_prior(&self, a_t: f64, e_hat_end: f64) -> f64 {
        self.params.mu * a_t + (1.0 - self.params.mu) * e_hat_end
    }
}

/// 基準の反応率 ē0：平均的なメッセージを初めて受け取ったときにシェアする人の割合。
///
/// 更新則 u' = w·u + (1 − w)·s を初期効用に1回当てたとき u' > 0 になる人を数える。
/// `DecisionBase::decided` と同じく、ちょうど 0 は「決めない」とする。
pub fn base_share_rate(states: &[UserState], internal_mean_stimulus: f64) -> f64 {
    if states.is_empty() {
        return 0.0;
    }
    let n = states
        .iter()
        .filter(|s| {
            let d = &s.internal_decision;
            let u = d.weight() * d.utility() + (1.0 - d.weight()) * internal_mean_stimulus;
            !approx::relative_eq!(u, 0.0) && u > 0.0
        })
        .count();
    n as f64 / states.len() as f64
}

/// 近くへの推薦の宛先を選ぶ。
///
/// シェア者の出辺から1本を一様に選び（起点を出次数に比例して選び、その後続を一様に
/// 選ぶのと同じ）、行き先 l の後続をさらに1人一様に選ぶ。l に後続が無ければ空振り。
/// `deg_cum[k]` は `sharers[..=k]` の出次数の累積和。乱数は「出辺 → 2歩目」の順に引く。
pub(crate) fn two_step_walk<R: Rng>(
    graph: &GraphB,
    sharers: &[usize],
    deg_cum: &[usize],
    rng: &mut R,
) -> Option<usize> {
    let total = *deg_cum.last()?;
    if total == 0 {
        return None;
    }
    let r = rng.random_range(0..total);
    let idx = deg_cum.partition_point(|&c| c <= r);
    let offset = r - if idx == 0 { 0 } else { deg_cum[idx - 1] };
    let l = *graph.successors(sharers[idx]).nth(offset)?;

    let dl = graph.successor_count(l);
    if dl == 0 {
        return None;
    }
    graph.successors(l).nth(rng.random_range(0..dl)).copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use graph_lib::prelude::DiGraphB;
    use rand::{SeedableRng, rngs::SmallRng};

    /// S-04: 起点はシェア者の出次数に比例して選ばれる。
    /// シェア者 0 の後続は 2, 3, 4（出次数 3）、シェア者 1 の後続は 5（出次数 1）で、
    /// 2〜5 の後続はそれぞれ 6〜9 の1人だけなので、終点から通った出辺が分かる。
    /// （graph-lib はラベルを詰め直すので、ラベルは 0 から隙間なく振る。）
    #[test]
    fn s04_origin_is_proportional_to_out_degree() {
        let edges: Vec<(usize, usize)> = vec![
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
            (4, 8),
            (5, 9),
        ];
        let graph = GraphB::Di(DiGraphB::from(&edges[..]));
        let sharers = [0, 1];
        let deg_cum = [3, 4];
        let mut rng = SmallRng::seed_from_u64(0);
        let mut counts = [0usize; 4];
        let trials = 40_000;
        for _ in 0..trials {
            let j = two_step_walk(&graph, &sharers, &deg_cum, &mut rng).unwrap();
            counts[j - 6] += 1;
        }
        // どの出辺も 1/4。シェア者 0 経由（6〜8）は合わせて 3/4 になる。
        for c in counts {
            let f = c as f64 / trials as f64;
            assert!((f - 0.25).abs() < 0.01, "{counts:?}");
        }
    }

    /// 2歩目の先が行き止まりなら空振り（None）。
    #[test]
    fn two_step_walk_misses_at_dead_end() {
        let edges: Vec<(usize, usize)> = vec![(0, 1)];
        let graph = GraphB::Di(DiGraphB::from(&edges[..]));
        let mut rng = SmallRng::seed_from_u64(0);
        assert_eq!(two_step_walk(&graph, &[0], &[1], &mut rng), None);
    }

    /// S-06: R3 の更新式。μ = 1 なら記憶なし（a_t のまま）、μ = 0 なら直前の ê そのもの。
    #[test]
    fn s06_account_memory_update() {
        let rec = |mu| Recommender {
            params: RecParams {
                eps: 1.0,
                mu,
                ..RecParams::default()
            },
            ebar0: 0.1,
        };
        assert_eq!(rec(1.0).next_prior(0.1, 0.4), 0.1);
        assert_eq!(rec(0.0).next_prior(0.1, 0.4), 0.4);
        assert!((rec(0.5).next_prior(0.1, 0.3) - 0.2).abs() < 1e-12);
    }

    /// φ はしきい値未満で 0、以上では x（上限 φ_max）。
    #[test]
    fn phi_threshold_and_cap() {
        let p = RecParams {
            tau: 1.0,
            phi_max: 2.0,
            ..RecParams::default()
        };
        assert_eq!(p.phi(0.99), 0.0);
        assert_eq!(p.phi(1.5), 1.5);
        assert_eq!(p.phi(5.0), 2.0);
    }
}
