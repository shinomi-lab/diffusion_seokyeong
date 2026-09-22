use std::{borrow::Borrow, mem};

use graph_lib::prelude::{Graph, GraphB};
use rand::Rng;

use crate::{message::Message, user::UserState};

use std::collections::VecDeque;
#[allow(dead_code)]
pub struct DiffusionResult {
    pub num_xact: usize,
    pub num_share: usize,
    /// カスケードが何段（何ホップ分の波）進んだか。診断用に保持している。
    pub total_step: usize,
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
}

impl Workspace {
    pub fn new(node_count: usize) -> Self {
        Self {
            states: Vec::with_capacity(node_count),
            rcvd: vec![false; node_count],
            curr: Vec::new(),
            next: Vec::new(),
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
    let mut num_xact = 0;
    let mut num_share = 0;

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

        // swap memories
        mem::swap(&mut ws.curr, &mut ws.next);

        num_share += n_curr_sharing;
        num_xact += n_curr_acting;
        total_step += 1
    }

    DiffusionResult {
        num_share,
        num_xact,
        total_step,
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
}

pub fn multiple_diffuse<M: Borrow<Message>, R: Rng>(
    graph: &GraphB,
    receipt_probability: f64,
    ws: &mut Workspace,
    ip: usize,
    messages: &[M],
    rng: &mut R,
    result: &mut Aggregator<usize>,
) {
    for message in messages {
        let res = diffuse(
            graph,
            receipt_probability,
            ws,
            ip,
            message.borrow(),
            rng,
        );
        result.mean_num_xact += res.num_xact;
        result.mean_num_share += res.num_share;
    }
}

pub fn monte_carlo_average<M: Borrow<Message>, R: Rng>(
    graph: &GraphB,
    receipt_probability: f64,
    initial_user_states: &[UserState],
    ip: usize,
    messages: &[M],
    sample_size: usize,
    rng: &mut R,
    ws: &mut Workspace,
) -> Aggregator<f64> {
    let mut result = Aggregator::<usize> {
        mean_num_xact: 0,
        mean_num_share: 0,
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
            rng,
            &mut result,
        );
    }
    Aggregator {
        mean_num_xact: result.mean_num_xact as f64 / sample_size as f64,
        mean_num_share: result.mean_num_share as f64 / sample_size as f64,
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
        let agg = monte_carlo_average(&graph, 1.0, &states, 0, &msgs, 10, &mut rng, &mut ws);

        assert_eq!(agg.mean_num_xact, single.num_xact as f64);
        assert_eq!(agg.mean_num_share, single.num_share as f64);
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
        let agg = monte_carlo_average(&graph, 1.0, &states, 0, &msgs, 1, &mut rng, &mut ws);

        // 3ラウンド流しても、外部行動できるのは IP 以外の 3 人が1回ずつ。
        assert_eq!(agg.mean_num_xact, 3.0);
        // シェアは毎ラウンド再評価されるので 3 ラウンド分積み上がる。
        assert_eq!(agg.mean_num_share, 3.0);
    }
}
