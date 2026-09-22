use serde::Deserialize;
use std::fmt;

#[derive(Debug, Deserialize, Clone, Copy)]
pub struct Message {
    pub internal_stimulus: f64,
    pub external_stimulus: f64,
}

/// `MessageSpace` の構築時に弾かれる不正なメッセージ集合。
#[derive(Debug, Clone, PartialEq)]
pub enum MessageSpaceError {
    /// メッセージが1つも無い。
    Empty,
    /// メッセージ種別数 L が 16 を超えている（系列コードが16進1桁で表せない）。
    TooManyMessages { len: usize },
    /// 刺激値が定義域 \[-1, 1\] の外。
    StimulusOutOfRange {
        index: usize,
        axis: &'static str,
        value: f64,
    },
}

impl fmt::Display for MessageSpaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "message_space が空です"),
            Self::TooManyMessages { len } => write!(
                f,
                "message_space の要素数は 16 以下である必要があります (実際: {len})"
            ),
            Self::StimulusOutOfRange { index, axis, value } => write!(
                f,
                "message_space[{index}] の {axis}_stimulus が定義域 [-1, 1] の外です (実際: {value})"
            ),
        }
    }
}

impl std::error::Error for MessageSpaceError {}

#[derive(Debug)]
pub struct MessageSpace(Vec<Message>);

impl TryFrom<Vec<Message>> for MessageSpace {
    type Error = MessageSpaceError;

    /// メッセージ集合を検証したうえで `MessageSpace` を作る。
    ///
    /// `L <= 16`（`seq_by_index` の16進コード化の制約）と、各刺激値が
    /// `DecisionBase::update` の要求する定義域 \[-1, 1\] に入っていることを確認する。
    /// 系列長 T に依存する制約（`log2(L) * T <= 64`）は T を知る呼び出し側で検証する。
    fn try_from(value: Vec<Message>) -> Result<Self, Self::Error> {
        if value.is_empty() {
            return Err(MessageSpaceError::Empty);
        }
        if value.len() > 16 {
            return Err(MessageSpaceError::TooManyMessages { len: value.len() });
        }
        for (index, m) in value.iter().enumerate() {
            for (axis, v) in [
                ("internal", m.internal_stimulus),
                ("external", m.external_stimulus),
            ] {
                if !(-1.0..=1.0).contains(&v) {
                    return Err(MessageSpaceError::StimulusOutOfRange {
                        index,
                        axis,
                        value: v,
                    });
                }
            }
        }
        Ok(Self(value))
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

    /// メッセージ種別数 L。
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    fn from_id(&self, id: u16) -> &Message {
        &self.0[id as usize]
    }

    /// 系列の総数 `L^T`。
    ///
    /// 呼び出し側は `log2(L) * num_rounds <= 64`（u64 に収まること）を事前に検証すること。
    #[inline]
    pub fn seq_size(&self, num_rounds: u32) -> u64 {
        (self.0.len() as u64).pow(num_rounds)
    }

    /// `index` を基数 L で分解し、ラウンド順（桁 j = ラウンド j+1）のメッセージ id を返す。
    ///
    /// `seq_by_index` / `seq_into` / `index_to_string` はすべてこの分解を共有する。
    #[inline]
    fn digits(&self, mut index: u64, num_rounds: u32) -> impl Iterator<Item = u16> {
        let l = self.0.len() as u64;
        (0..num_rounds).map(move |_| {
            let i = (index % l) as u16;
            index /= l;
            i
        })
    }

    /// `out` にラウンド順のメッセージ参照列を詰める（`out` の確保を呼び出し側で使い回せる）。
    ///
    /// 系列コード文字列は作らないので、文字列が要らない経路ではこちらを使う。
    pub fn seq_into<'a>(&'a self, index: u64, num_rounds: u32, out: &mut Vec<&'a Message>) {
        out.clear();
        out.extend(self.digits(index, num_rounds).map(|id| self.from_id(id)));
    }

    /// `index` に対応するメッセージ参照列と16進コード文字列を返す。
    ///
    /// 文字列が不要な場合は `seq_into` を使うこと。
    pub fn seq_by_index(&self, index: u64, num_rounds: u32) -> (Vec<&Message>, String) {
        let mut msgs = Vec::with_capacity(num_rounds as usize);
        self.seq_into(index, num_rounds, &mut msgs);
        (msgs, self.index_to_string(index, num_rounds))
    }

    /// `index` に対応する16進コード文字列（**先頭がラウンド1**）。
    pub fn index_to_string(&self, index: u64, num_rounds: u32) -> String {
        self.digits(index, num_rounds)
            .map(|id| char::from_digit(id as u32, 16).expect("L <= 16 は構築時に検証済み"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn space(len: usize) -> MessageSpace {
        MessageSpace::try_from(
            (0..len)
                .map(|i| Message {
                    internal_stimulus: i as f64 / 100.0,
                    external_stimulus: -(i as f64) / 100.0,
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    /// 桁の並びは「先頭＝ラウンド1」。コミット 344199a で reverse を外した仕様を固定する。
    #[test]
    fn index_to_string_puts_round1_first() {
        let s = space(4);
        // index = 3 * 4^4 = 768 → ラウンド1..4 が m0、ラウンド5 が m3 → "00003"
        assert_eq!(s.index_to_string(3 * 4u64.pow(4), 5), "00003");
        // index = 3 → ラウンド1 が m3、以降 m0 → "30000"
        assert_eq!(s.index_to_string(3, 5), "30000");
        assert_eq!(s.index_to_string(0, 5), "00000");
    }

    /// seq_into / seq_by_index / index_to_string が同じ分解を共有していること。
    #[test]
    fn seq_into_agrees_with_index_to_string() {
        let s = space(4);
        let mut buf = Vec::new();
        for index in 0..s.seq_size(3) {
            s.seq_into(index, 3, &mut buf);
            let code = s.index_to_string(index, 3);
            assert_eq!(buf.len(), 3);
            for (round, ch) in code.chars().enumerate() {
                let id = ch.to_digit(16).unwrap() as usize;
                // 同じメッセージを指していること（刺激値で同定）
                assert_eq!(buf[round].internal_stimulus, s.0[id].internal_stimulus);
            }
            let (msgs, code2) = s.seq_by_index(index, 3);
            assert_eq!(code, code2);
            assert_eq!(msgs.len(), buf.len());
        }
    }

    /// seq_into はバッファを使い回しても前回の中身を残さない。
    #[test]
    fn seq_into_resets_buffer() {
        let s = space(4);
        let mut buf = Vec::new();
        s.seq_into(0, 5, &mut buf);
        assert_eq!(buf.len(), 5);
        s.seq_into(0, 2, &mut buf);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn rejects_out_of_range_stimulus() {
        let err = MessageSpace::try_from(vec![Message {
            internal_stimulus: 1.5,
            external_stimulus: 0.0,
        }])
        .unwrap_err();
        assert_eq!(
            err,
            MessageSpaceError::StimulusOutOfRange {
                index: 0,
                axis: "internal",
                value: 1.5
            }
        );

        // 負の刺激そのものは正当（test/config_extneg.toml が使う）
        assert!(
            MessageSpace::try_from(vec![Message {
                internal_stimulus: 0.20,
                external_stimulus: -0.05,
            }])
            .is_ok()
        );
    }

    #[test]
    fn rejects_empty_and_too_many() {
        assert_eq!(MessageSpace::try_from(vec![]).unwrap_err(), MessageSpaceError::Empty);
        let many: Vec<Message> = (0..17)
            .map(|_| Message {
                internal_stimulus: 0.0,
                external_stimulus: 0.0,
            })
            .collect();
        assert_eq!(
            MessageSpace::try_from(many).unwrap_err(),
            MessageSpaceError::TooManyMessages { len: 17 }
        );
    }
}
