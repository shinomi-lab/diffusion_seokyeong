use std::{borrow::Borrow, mem};

use graph_lib::prelude::{Graph, GraphB};
use rand::Rng;

use crate::{message::Message, user::UserState};

use std::collections::VecDeque;
#[allow(dead_code)]
pub struct DiffusionResult {
    pub num_xact: usize,
    pub num_share: usize,
    pub total_step: usize,
}

/// Diffuses a `message` in a `graph` with `ip` node and `user_states`.
/// The direction of the egdes in the `graph` is iterpreted as message transition, i.e., the opposite of follow/followee relation.
pub fn diffuse<'a, R: Rng>(
    graph: &GraphB,
    prop_prob: f64,
    user_states: &mut [UserState],
    ip: usize,
    message: &Message,
    rng: &mut R,
    num_share_of_users: &mut [usize],
    num_xact_of_users: &mut [usize],
) -> DiffusionResult {
    let mut next = Vec::new();
    let mut num_xact = 0;
    let mut num_share = 0;
    let mut curr = vec![ip]; // users about to share
    let mut rcvd = vec![false; graph.node_count()];
    rcvd[ip] = true; // ignore contact and force into share for the IP node

    let mut total_step = 0;

    loop {
        if curr.len() == 0 {
            break;
        }
        let mut n_curr_sharing = 0;
        let mut n_curr_acting = 0;

        while let Some(i) = curr.pop() {
            // check message sharing event from `i` to `j`.
            //  If and only if `j` has not received message, `i` may share message to `j` following `prop_prob`.
            for &j in graph.successors(i) {
                if (rcvd[j] == false) & (prop_prob > rng.random()) {
                    let state = &mut user_states[j];
                    rcvd[j] = true;
                    state.contact(message);

                    if state.external_decision.decided() {
                        n_curr_acting += 1;
                        num_xact_of_users[j] += 1;
                    }

                    if graph.successors(j).len() > 0 {
                        if state.internal_decision.decided() {
                            n_curr_sharing += 1;
                            next.push(j);
                            num_share_of_users[j] += 1;
                        }
                    }
                }
            }
        }

        // swap memories
        mem::swap(&mut curr, &mut next);

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

pub struct Aggregator<V> {
    pub mean_num_xact: V,
    pub num_share_of_users: Vec<V>,
    pub num_xact_of_users: Vec<V>,
}

pub fn multiple_diffuse<M: Borrow<Message>, R: Rng>(
    graph: &GraphB,
    receipt_probability: f64,
    user_states: &mut [UserState],
    ip: usize,
    messages: &[M],
    rng: &mut R,
    result: &mut Aggregator<usize>,
) {
    for message in messages {
        let res = diffuse(
            graph,
            receipt_probability,
            user_states,
            ip,
            message.borrow(),
            rng,
            &mut result.num_share_of_users,
            &mut result.num_xact_of_users,
        );
        result.mean_num_xact += res.num_xact;

    }
}

pub fn monte_carlo_average<M: Borrow<Message>, R: Rng>(
    graph: &GraphB,
    receipt_probability: f64,
    initial_user_states: &Vec<UserState>,
    ip: usize,
    messages: &[M],
    sample_size: usize,
    rng: &mut R,
) -> Aggregator<f64> {
    let mut result = Aggregator::<usize> {
        mean_num_xact: 0,
        num_share_of_users: vec![0; graph.node_count()],
        num_xact_of_users: vec![0; graph.node_count()],
    };
    for _ in 0..sample_size {
        multiple_diffuse(
            graph,
            receipt_probability,
            &mut initial_user_states.clone(),
            ip,
            messages,
            rng,
            &mut result,
        );
    }
    Aggregator {
        mean_num_xact: result.mean_num_xact as f64 / sample_size as f64,
        num_share_of_users: result.num_share_of_users.iter().map(|&x| x as f64 / sample_size as f64).collect(),
        num_xact_of_users: result.num_xact_of_users.iter().map(|&x| x as f64 / sample_size as f64).collect(),
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
