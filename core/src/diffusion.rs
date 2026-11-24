use std::{borrow::Borrow, mem};

use graph_lib::prelude::{Graph, GraphB};
use rand::Rng;

use crate::{
    message::Message,
    recommend::{Recommender, two_step_walk},
    user::UserState,
};

use std::collections::VecDeque;
#[allow(dead_code)]
pub struct DiffusionResult {
    pub num_xact: usize,
    pub num_share: usize,
    /// カスケードが何段（何ホップ分の波）進んだか。診断用に保持している。
    pub total_step: usize,
    /// 推薦経路で受信した人数（推薦なしなら 0。以下同じ）。
    pub num_rec_rcvd: usize,
    /// 推薦の空振り（宛先が受信済み・行き止まり）の数。
    pub num_rec_miss: usize,
    /// 推薦経路で受信して外部行動した人数（`num_xact` の内数）。
    pub num_xact_rec: usize,
    /// ラウンド終了時点の反応率 ê（R3 の更新に使う）。
    pub e_hat_end: f64,
}

/// 1ジョブ分の拡散計算で使い回す作業バッファ。
///
/// 従来は `diffuse` / `monte_carlo_average` の呼び出しごとにノード数サイズの
/// `Vec` を確保していた（1ジョブあたり数百回）。これを1組にまとめて
/// 「チャンクごとに1回確保 → 以後は使い回し」にするためのもの。
pub struct Workspace {
    /// 各モンテカルロ試行の開始時に初期状態からコピーし直すユーザ状態。
    states: Vec<UserState>,
    /// そのカスケードで既にメッセージを受け取ったか。
    rcvd: Vec<bool>,
    /// 現在シェア中のノード。
    curr: Vec<usize>,
    /// 次の段でシェアするノード。
    next: Vec<usize>,
    /// 推薦の起点に使う、その波のシェア者の控え（`curr` は pop で空になるため）。
    sharers: Vec<usize>,
    /// `sharers` の出次数の累積和。
    deg_cum: Vec<usize>,
}

impl Workspace {
    pub fn new(node_count: usize) -> Self {
        Self {
            states: Vec::with_capacity(node_count),
            rcvd: vec![false; node_count],
            curr: Vec::new(),
            next: Vec::new(),
            sharers: Vec::new(),
            deg_cum: Vec::new(),
        }
    }
}

/// Diffuses a `message` in a `graph` with `ip` node and the user states held in `ws`.
/// The direction of the egdes in the `graph` is iterpreted as message transition, i.e., the opposite of follow/followee relation.
///
/// 呼び出し側は事前に `ws.states` を対象ネットワークのユーザ状態で満たしておくこと
/// （`monte_carlo_average` は各試行の先頭でこれを行う）。`rcvd` / `curr` / `next` は
/// この関数がリセットするので、同じ `Workspace` を何度でも渡してよい。
pub fn diffuse<R: Rng>(
    graph: &GraphB,
    prop_prob: f64,
    ws: &mut Workspace,
    ip: usize,
    message: &Message,
    rng: &mut R,
) -> DiffusionResult {
    diffuse_with_rec(graph, prop_prob, ws, ip, message, None, 0.0, rng)
}

/// 推薦チャネル付きの `diffuse`。`rec` が `None` なら `diffuse` と同じ処理で、乱数も同じだけ引く。
///
/// `a_t` は R3 の事前分布の平均（R3 を使わなければ ē0）。各波のフォロー経路のあとに、
/// 推薦量 B_s = ε δ φ(ê/ē0) p Σ_{i∈S_s} d_i を丸めた数だけ宛先を選んで届ける。
pub fn diffuse_with_rec<R: Rng>(
    graph: &GraphB,
    prop_prob: f64,
    ws: &mut Workspace,
    ip: usize,
    message: &Message,
    rec: Option<&Recommender>,
    a_t: f64,
    rng: &mut R,
) -> DiffusionResult {
    // 直接組み立てた設定でも、ε δ = 0 なら推薦の分岐に入らない（丸めの乱数も引かない）。
    let rec = rec.filter(|r| r.params.is_on());
    let mut num_xact = 0;
    let mut num_share = 0;
    let mut num_rec_rcvd = 0;
    let mut num_rec_miss = 0;
    let mut num_xact_rec = 0;
    // そのラウンドでここまでにシェアを決めた人数（フォロワー0人も含む）と受信した人数。IP は含まない。
    let mut cum_share = 0usize;
    let mut cum_rcvd = 0usize;

    // バッファは確保し直さず、内容だけリセットして使い回す。
    ws.curr.clear();
    ws.next.clear();
    ws.rcvd.fill(false);
    ws.curr.push(ip);
    ws.rcvd[ip] = true; // ignore contact and force into share for the IP node

    let mut total_step = 0;

    loop {
        if ws.curr.is_empty() {
            break;
        }
        let mut n_curr_sharing = 0;
        let mut n_curr_acting = 0;

        if rec.is_some() {
            ws.sharers.clear();
            ws.sharers.extend_from_slice(&ws.curr);
        }

        while let Some(i) = ws.curr.pop() {
            // check message sharing event from `i` to `j`.
            //  If and only if `j` has not received message, `i` may share message to `j` following `prop_prob`.
            for &j in graph.successors(i) {
                // NOTE: 非短絡の `&` を意図的に残している。`&&` にすると受信済みノードに対して
                // 乱数を引かなくなり高速化できるが、乱数列がずれて過去の実行結果との
                // 再現性（ビット一致）が失われるため変更しない。
                if (ws.rcvd[j] == false) & (prop_prob > rng.random()) {
                    let state = &mut ws.states[j];
                    ws.rcvd[j] = true;
                    state.contact(message);
                    cum_rcvd += 1;
                    if state.internal_decision.decided() {
                        cum_share += 1;
                    }

                    if state.external_decision.decided() {
                        n_curr_acting += 1;
                    }

                    if graph.successors(j).len() > 0 {
                        if state.internal_decision.decided() {
                            n_curr_sharing += 1;
                            ws.next.push(j);
                        }
                    }
                }
            }
        }

        if let Some(rec) = rec {
            let p = &rec.params;
            let e_hat = (cum_share as f64 + p.k * a_t) / (cum_rcvd as f64 + p.k);
            // 仮定: ē0 = 0（平均的なメッセージで誰もシェアしない母集団）なら推薦しない。
            let x = if rec.ebar0 > 0.0 { e_hat / rec.ebar0 } else { 0.0 };

            ws.deg_cum.clear();
            let mut deg_total = 0usize;
            for &i in &ws.sharers {
                deg_total += graph.successor_count(i);
                ws.deg_cum.push(deg_total);
            }

            let b = p.eps * p.delta * p.phi(x) * prop_prob * deg_total as f64;
            let n = b.floor() as usize + (rng.random::<f64>() < b.fract()) as usize;
            for _ in 0..n {
                // 宛先はシェア者のフォロワーのフォロワー（2歩のランダムウォーク）だけ。行き止まりなら None
                let target = two_step_walk(graph, &ws.sharers, &ws.deg_cum, rng);
                // IP は最初から受信済みなので、ここで自動的に除かれる。
                let Some(j) = target.filter(|&j| !ws.rcvd[j]) else {
                    num_rec_miss += 1;
                    continue;
                };
                let state = &mut ws.states[j];
                ws.rcvd[j] = true;
                state.contact(message);
                num_rec_rcvd += 1;
                cum_rcvd += 1;

                if state.external_decision.decided() {
                    n_curr_acting += 1;
                    num_xact_rec += 1;
                }
                if state.internal_decision.decided() {
                    cum_share += 1;
                    if graph.successor_count(j) > 0 {
                        n_curr_sharing += 1;
                        ws.next.push(j);
                    }
                }
            }
        }

        // swap memories
        mem::swap(&mut ws.curr, &mut ws.next);

        num_share += n_curr_sharing;
        num_xact += n_curr_acting;
        total_step += 1
    }

    let e_hat_end = match rec {
        Some(rec) => (cum_share as f64 + rec.params.k * a_t) / (cum_rcvd as f64 + rec.params.k),
        None => 0.0,
    };

    DiffusionResult {
        num_share,
        num_xact,
        total_step,
        num_rec_rcvd,
        num_rec_miss,
        num_xact_rec,
        e_hat_end,
    }
}

/// 1セル（シナリオ×メッセージ系列）あたりの集計値。
///
/// 個人単位の明細は出力しないため、ネットワーク全体の2つの合計値だけを持つ。
pub struct Aggregator<V> {
    /// T ラウンド合計の外部行動者数。
    pub mean_num_xact: V,
    /// T ラウンド合計の累積シェア回数。
    pub mean_num_share: V,
    /// T ラウンド合計の、推薦経路で受信した人数（推薦なしなら 0）。
    pub mean_num_rec_rcvd: V,
    /// T ラウンド合計の、推薦経路で受信して外部行動した人数（`mean_num_xact` の内数）。
    pub mean_num_xact_rec: V,
}

pub fn multiple_diffuse<M: Borrow<Message>, R: Rng>(
    graph: &GraphB,
    receipt_probability: f64,
    ws: &mut Workspace,
    ip: usize,
    messages: &[M],
    rec: Option<&Recommender>,
    rng: &mut R,
    result: &mut Aggregator<usize>,
) {
    // R3 の事前分布の平均。1ラウンド目は ē0 から始め、ラウンドを跨いで持ち越す。
    let mut a_t = rec.map_or(0.0, |r| r.ebar0);
    for message in messages {
        let res = diffuse_with_rec(
            graph,
            receipt_probability,
            ws,
            ip,
            message.borrow(),
            rec,
            a_t,
            rng,
        );
        result.mean_num_xact += res.num_xact;
        result.mean_num_share += res.num_share;
        result.mean_num_rec_rcvd += res.num_rec_rcvd;
        result.mean_num_xact_rec += res.num_xact_rec;
        if let Some(rec) = rec {
            a_t = rec.next_prior(a_t, res.e_hat_end);
        }
    }
}

pub fn monte_carlo_average<M: Borrow<Message>, R: Rng>(
    graph: &GraphB,
    receipt_probability: f64,
    initial_user_states: &[UserState],
    ip: usize,
    messages: &[M],
    rec: Option<&Recommender>,
    sample_size: usize,
    rng: &mut R,
    ws: &mut Workspace,
) -> Aggregator<f64> {
    let mut result = Aggregator::<usize> {
        mean_num_xact: 0,
        mean_num_share: 0,
        mean_num_rec_rcvd: 0,
        mean_num_xact_rec: 0,
    };
    for _ in 0..sample_size {
        // 毎試行ごとに初期状態へ戻す。`clear` + `extend_from_slice` なので
        // 2回目以降は確保済み容量を再利用する（アロケーションが起きない）。
        ws.states.clear();
        ws.states.extend_from_slice(initial_user_states);
        multiple_diffuse(
            graph,
            receipt_probability,
            ws,
            ip,
            messages,
            rec,
            rng,
            &mut result,
        );
    }
    Aggregator {
        mean_num_xact: result.mean_num_xact as f64 / sample_size as f64,
        mean_num_share: result.mean_num_share as f64 / sample_size as f64,
        mean_num_rec_rcvd: result.mean_num_rec_rcvd as f64 / sample_size as f64,
        mean_num_xact_rec: result.mean_num_xact_rec as f64 / sample_size as f64,
    }
}

/// IPから全ユーザーまでの最短距離（ホップ数）をBFSで計算
pub fn shortest_distances_from(graph: &GraphB, ip: usize) -> Vec<Option<usize>> {
    let n = graph.node_count();
    let mut dist = vec![None; n];
    dist[ip] = Some(0);

    let mut queue: VecDeque<usize> = VecDeque::with_capacity(n / 2);
    queue.push_back(ip);

    while let Some(u) = queue.pop_front() {
        for &v in graph.successors(u) {
            if dist[v].is_none() {
                dist[v] = Some(dist[u].unwrap() + 1);
                queue.push_back(v);
            }
        }
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::{DecisionAny, DecisionBase, DecisionOnce};
    use crate::recommend::RecParams;
    use graph_lib::prelude::DiGraphB;
    use rand::{SeedableRng, rngs::SmallRng};

    /// 0 -> 1 -> 2、および 1 -> 3（3 は後続を持たない葉）。
    fn line_graph() -> GraphB {
        let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 2), (1, 3)];
        GraphB::Di(DiGraphB::from(&edges[..]))
    }

    /// 受け取れば必ず両方の決定が発火するユーザ状態（w=0 なので u' = stimulus > 0）。
    fn always_firing_states(n: usize) -> Vec<UserState> {
        (0..n)
            .map(|_| {
                UserState::new(
                    DecisionAny::from(DecisionBase::new(0.0, -1.0)),
                    DecisionOnce::from(DecisionBase::new(0.0, -1.0)),
                )
            })
            .collect()
    }

    fn run_once(graph: &GraphB, states: &[UserState], ws: &mut Workspace) -> DiffusionResult {
        let msg = Message {
            internal_stimulus: 0.5,
            external_stimulus: 0.5,
        };
        let mut rng = SmallRng::seed_from_u64(0);
        ws.states.clear();
        ws.states.extend_from_slice(states);
        diffuse(graph, 1.0, ws, 0, &msg, &mut rng)
    }

    /// prop_prob = 1.0 なら IP から到達可能な全ノードが受信し、外部行動する。
    /// 一方 num_share は「後続を持つノード」しか数えない（葉ノード 3 は除外）。
    #[test]
    fn counts_reachable_nodes_and_excludes_leaves_from_share() {
        let graph = line_graph();
        let states = always_firing_states(4);
        let mut ws = Workspace::new(4);
        let res = run_once(&graph, &states, &mut ws);

        // IP(0) を除く 1, 2, 3 が受信して外部行動する。
        assert_eq!(res.num_xact, 3);
        // シェアに数えられるのは後続を持つ 1 のみ（2 と 3 は葉）。
        assert_eq!(res.num_share, 1);
    }

    /// 同じ Workspace を使い回しても、毎回新しい Workspace を使った場合と一致する
    /// （rcvd / curr / next のリセット漏れ検出）。
    #[test]
    fn workspace_reuse_matches_fresh_workspace() {
        let graph = line_graph();
        let states = always_firing_states(4);

        let mut reused = Workspace::new(4);
        let first = run_once(&graph, &states, &mut reused);
        let second = run_once(&graph, &states, &mut reused);

        let mut fresh = Workspace::new(4);
        let fresh_res = run_once(&graph, &states, &mut fresh);

        assert_eq!(first.num_xact, fresh_res.num_xact);
        assert_eq!(first.num_share, fresh_res.num_share);
        assert_eq!(second.num_xact, fresh_res.num_xact);
        assert_eq!(second.num_share, fresh_res.num_share);
    }

    /// 決定論的な設定なら、モンテカルロ平均は単発実行の値と一致する。
    #[test]
    fn monte_carlo_average_matches_single_run() {
        let graph = line_graph();
        let states = always_firing_states(4);
        let mut ws = Workspace::new(4);
        let single = run_once(&graph, &states, &mut ws);

        let msgs = vec![Message {
            internal_stimulus: 0.5,
            external_stimulus: 0.5,
        }];
        let mut rng = SmallRng::seed_from_u64(0);
        let agg = monte_carlo_average(&graph, 1.0, &states, 0, &msgs, None, 10, &mut rng, &mut ws);

        assert_eq!(agg.mean_num_xact, single.num_xact as f64);
        assert_eq!(agg.mean_num_share, single.num_share as f64);
    }

    /// 変更前（推薦チャネル導入前、core 0.2.0）の `diffuse` をそのまま写したもの。S-01 の基準。
    fn diffuse_reference<R: Rng>(
        graph: &GraphB,
        prop_prob: f64,
        ws: &mut Workspace,
        ip: usize,
        message: &Message,
        rng: &mut R,
    ) -> (usize, usize, usize) {
        let mut num_xact = 0;
        let mut num_share = 0;
        ws.curr.clear();
        ws.next.clear();
        ws.rcvd.fill(false);
        ws.curr.push(ip);
        ws.rcvd[ip] = true;
        let mut total_step = 0;
        loop {
            if ws.curr.is_empty() {
                break;
            }
            let mut n_curr_sharing = 0;
            let mut n_curr_acting = 0;
            while let Some(i) = ws.curr.pop() {
                for &j in graph.successors(i) {
                    if (ws.rcvd[j] == false) & (prop_prob > rng.random()) {
                        let state = &mut ws.states[j];
                        ws.rcvd[j] = true;
                        state.contact(message);
                        if state.external_decision.decided() {
                            n_curr_acting += 1;
                        }
                        if graph.successors(j).len() > 0 {
                            if state.internal_decision.decided() {
                                n_curr_sharing += 1;
                                ws.next.push(j);
                            }
                        }
                    }
                }
            }
            mem::swap(&mut ws.curr, &mut ws.next);
            num_share += n_curr_sharing;
            num_xact += n_curr_acting;
            total_step += 1
        }
        (num_xact, num_share, total_step)
    }

    /// 乱数で作る有向グラフ（自己ループなし）と、重みがばらついたユーザ状態。
    fn random_setup(seed: u64, n: usize, m: usize) -> (GraphB, Vec<UserState>) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut edges = Vec::with_capacity(m);
        while edges.len() < m {
            let (a, b) = (rng.random_range(0..n), rng.random_range(0..n));
            if a != b {
                edges.push((a, b));
            }
        }
        let graph = GraphB::Di(DiGraphB::from(&edges[..]));
        let states = (0..graph.node_count())
            .map(|_| {
                UserState::new(
                    DecisionAny::from(DecisionBase::new(rng.random_range(0.0..0.4), -1.0)),
                    DecisionOnce::from(DecisionBase::new(rng.random_range(0.0..0.4), -1.0)),
                )
            })
            .collect();
        (graph, states)
    }

    /// S-01: 推薦なしの `diffuse` / `diffuse_with_rec(None)` は、変更前の実装と
    /// 結果も乱数の消費量もビット単位で一致する（prop_prob < 1、複数シード、複数ラウンド）。
    #[test]
    fn s01_off_matches_pre_change_implementation() {
        let msgs = [
            Message { internal_stimulus: 0.20, external_stimulus: 0.05 },
            Message { internal_stimulus: 0.05, external_stimulus: 0.20 },
            Message { internal_stimulus: 0.15, external_stimulus: 0.10 },
        ];
        for seed in 0..20 {
            let (graph, states) = random_setup(seed, 80, 400);
            let n = graph.node_count();
            let (mut ws_a, mut ws_b) = (Workspace::new(n), Workspace::new(n));
            ws_a.states.extend_from_slice(&states);
            ws_b.states.extend_from_slice(&states);
            let mut rng_a = SmallRng::seed_from_u64(seed + 100);
            let mut rng_b = rng_a.clone();
            for msg in &msgs {
                let a = diffuse(&graph, 0.3, &mut ws_a, 0, msg, &mut rng_a);
                let b = diffuse_reference(&graph, 0.3, &mut ws_b, 0, msg, &mut rng_b);
                assert_eq!((a.num_xact, a.num_share, a.total_step), b, "seed {seed}");
                assert_eq!((a.num_rec_rcvd, a.num_rec_miss, a.num_xact_rec), (0, 0, 0));
            }
            assert_eq!(rng_a.random::<u64>(), rng_b.random::<u64>(), "乱数の消費量が同じ");
        }
    }

    /// S-02: ε > 0 でも δ = 0（または ε = 0）なら推薦なしと完全に一致し、乱数も引かない。
    #[test]
    fn s02_zero_budget_draws_no_random_numbers() {
        let msg = Message { internal_stimulus: 0.20, external_stimulus: 0.05 };
        for params in [
            RecParams { eps: 1.0, delta: 0.0, ..RecParams::default() },
            RecParams { eps: 0.0, ..RecParams::default() },
        ] {
            assert!(Recommender::new(params, &[], 0.125).is_none());
            // 直接組み立てても推薦の分岐に入らないこと。
            let rec = Recommender { params, ebar0: 0.5 };
            let (graph, states) = random_setup(7, 80, 400);
            let n = graph.node_count();
            let (mut ws_a, mut ws_b) = (Workspace::new(n), Workspace::new(n));
            ws_a.states.extend_from_slice(&states);
            ws_b.states.extend_from_slice(&states);
            let mut rng_a = SmallRng::seed_from_u64(1);
            let mut rng_b = rng_a.clone();
            let a = diffuse_with_rec(&graph, 0.3, &mut ws_a, 0, &msg, Some(&rec), 0.5, &mut rng_a);
            let b = diffuse_reference(&graph, 0.3, &mut ws_b, 0, &msg, &mut rng_b);
            assert_eq!((a.num_xact, a.num_share, a.total_step), b);
            assert_eq!(rng_a.random::<u64>(), rng_b.random::<u64>());
        }
    }

    /// S-03: 推薦の宛先は 2 ホップ先（シェア者のフォロワーのフォロワー）だけ。受信済みの人・IP にはならず、
    /// 届かなかった枠は空振りに数える。0 → 1 → 2 で、1 は受信してもシェアしないので、フォロー経路は 1 までしか届かない。
    /// 推薦は 0 の 2 ホップ先の 2 にだけ届く（1 は受信済み）。
    #[test]
    fn s03_two_hop_recommendation_reaches_beyond_a_non_sharer() {
        let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 2)];
        let graph = GraphB::Di(DiGraphB::from(&edges[..]));
        // 1 は内部の重み w = 1（効用が動かないのでシェアしない）。ほかは受け取れば必ず両方が成立する。
        let states: Vec<UserState> = [0.0, 1.0, 0.0]
            .iter()
            .map(|&w| {
                UserState::new(
                    DecisionAny::from(DecisionBase::new(w, -1.0)),
                    DecisionOnce::from(DecisionBase::new(0.0, -1.0)),
                )
            })
            .collect();
        let params = RecParams { eps: 20.0, ..RecParams::default() };
        let rec = Recommender::new(params, &states, 0.5).unwrap();
        let msg = Message { internal_stimulus: 0.5, external_stimulus: 0.5 };

        let mut ws = Workspace::new(3);
        for seed in 0..50 {
            ws.states.clear();
            ws.states.extend_from_slice(&states);
            let mut rng = SmallRng::seed_from_u64(seed);
            let res = diffuse_with_rec(&graph, 1.0, &mut ws, 0, &msg, Some(&rec), rec.ebar0, &mut rng);
            // フォロー経路で 1、推薦で 2 に 1 回だけ届く。残りの枠（18 個以上）は受信済みで空振り。
            assert!(ws.rcvd[1] && ws.rcvd[2], "seed {seed}");
            assert_eq!((res.num_rec_rcvd, res.num_xact_rec), (1, 1), "seed {seed}");
            // 受信すれば必ず外部行動するので、外部行動数 = 受信者数（IP を除く）。二重に数えると崩れる。
            assert_eq!(res.num_xact, 2, "seed {seed}");
            assert!(res.num_rec_miss >= 10, "seed {seed}: {}", res.num_rec_miss);
        }
    }

    /// S-05: しきい値 τ を越えないメッセージには、推薦が1人も付かない（オーディション）。
    #[test]
    fn s05_high_threshold_blocks_recommendation() {
        let (graph, states) = random_setup(3, 80, 400);
        let params = RecParams { eps: 5.0, tau: 50.0, phi_max: 50.0, ..RecParams::default() };
        let rec = Recommender::new(params, &states, 0.125).unwrap();
        let msg = Message { internal_stimulus: 0.20, external_stimulus: 0.05 };
        let mut ws = Workspace::new(graph.node_count());
        for seed in 0..20 {
            ws.states.clear();
            ws.states.extend_from_slice(&states);
            let mut rng = SmallRng::seed_from_u64(seed);
            let res = diffuse_with_rec(&graph, 0.3, &mut ws, 0, &msg, Some(&rec), rec.ebar0, &mut rng);
            assert_eq!(res.num_rec_rcvd, 0);
            assert_eq!(res.num_rec_miss, 0);
        }
    }

    /// 外部行動は DecisionOnce なので、複数ラウンド回しても1人1回しか数えられない。
    #[test]
    fn external_action_counted_at_most_once_per_user() {
        let graph = line_graph();
        let states = always_firing_states(4);
        let mut ws = Workspace::new(4);

        let msg = Message {
            internal_stimulus: 0.5,
            external_stimulus: 0.5,
        };
        let msgs = vec![msg, msg, msg];
        let mut rng = SmallRng::seed_from_u64(0);
        let agg = monte_carlo_average(&graph, 1.0, &states, 0, &msgs, None, 1, &mut rng, &mut ws);

        // 3ラウンド流しても、外部行動できるのは IP 以外の 3 人が1回ずつ。
        assert_eq!(agg.mean_num_xact, 3.0);
        // シェアは毎ラウンド再評価されるので 3 ラウンド分積み上がる。
        assert_eq!(agg.mean_num_share, 3.0);
    }
}
