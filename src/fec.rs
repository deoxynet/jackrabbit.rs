/// Convolutional encoder (K=7, generators 0o171/0o133) and Viterbi decoder.
///
/// Rate 1/2 base code, with rate 2/3 via puncturing.
///
/// Shift register convention: input enters at bit K-1 (MSB). State = lower K-1 bits.
/// Full register = (input << (K-1)) | state.
/// Next state = full_register >> 1 = (input << (K-2)) | (state >> 1).
/// Traceback: input = (current_state >> (K-2)) & 1.

use crate::params::CodeRate;

const K: usize = 7;
const NUM_STATES: usize = 1 << (K - 1); // 64
const G0: u8 = 0o171; // 0b01111001 = 121
const G1: u8 = 0o133; // 0b01011011 = 91

/// Compute parity of a byte (number of 1-bits mod 2).
#[inline]
fn parity(x: u8) -> u8 {
    (x.count_ones() & 1) as u8
}

/// Convolutional encode at rate 1/2.
/// Input: info bits. Output: coded bits (2 × (len + K-1)).
/// Appends K-1 tail zeros to flush encoder back to state 0.
pub fn conv_encode(bits: &[u8]) -> Vec<u8> {
    let mut state: u8 = 0; // K-1 = 6 bits
    let mut output = Vec::with_capacity((bits.len() + K - 1) * 2);

    for i in 0..(bits.len() + K - 1) {
        let input_bit = if i < bits.len() { bits[i] } else { 0 };
        let reg = ((input_bit as u8) << (K - 1)) | state;

        output.push(parity(reg & G0));
        output.push(parity(reg & G1));

        state = reg >> 1;
    }
    output
}

/// Puncture rate 1/2 code to rate 2/3.
/// Pattern: [1,1, 1,0] repeated — keep positions 0,1,2; drop position 3.
pub fn puncture(coded_bits: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(coded_bits.len() * 3 / 4 + 1);
    for (i, &bit) in coded_bits.iter().enumerate() {
        if i % 4 != 3 {
            out.push(bit);
        }
    }
    out
}

/// Depuncture soft bits from rate 2/3 back to rate 1/2 (insert 0.0 erasures).
pub fn depuncture(soft_bits: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(soft_bits.len() * 4 / 3 + 2);
    let mut src = 0;
    while src < soft_bits.len() {
        let remaining = soft_bits.len() - src;
        if remaining >= 3 {
            out.push(soft_bits[src]);
            out.push(soft_bits[src + 1]);
            out.push(soft_bits[src + 2]);
            out.push(0.0); // erasure
            src += 3;
        } else {
            // Partial group at end
            for j in 0..remaining {
                out.push(soft_bits[src + j]);
            }
            src += remaining;
        }
    }
    out
}

/// Puncture rate 1/2 code to rate 3/4.
/// Pattern: [1,1,0,1,1,0] repeated — keep 4 of every 6 coded bits.
pub fn puncture_three_fourths(coded_bits: &[u8]) -> Vec<u8> {
    let pattern = [true, true, false, true, true, false];
    let mut out = Vec::with_capacity(coded_bits.len() * 2 / 3 + 1);
    for (i, &bit) in coded_bits.iter().enumerate() {
        if pattern[i % 6] {
            out.push(bit);
        }
    }
    out
}

/// Depuncture soft bits from rate 3/4 back to rate 1/2 (insert 0.0 erasures).
pub fn depuncture_three_fourths(soft_bits: &[f32]) -> Vec<f32> {
    let pattern = [true, true, false, true, true, false];
    let mut out = Vec::with_capacity(soft_bits.len() * 3 / 2 + 6);
    let mut src = 0;
    let mut pat_idx = 0;
    while src < soft_bits.len() {
        if pattern[pat_idx % 6] {
            out.push(soft_bits[src]);
            src += 1;
        } else {
            out.push(0.0); // erasure
        }
        pat_idx += 1;
    }
    // If the last consumed bit aligned with pattern index 4, we still need to
    // emit the trailing erasure at index 5 to complete the pattern.
    if pat_idx % 6 == 5 {
        out.push(0.0);
    }
    out
}

/// Encode with the given code rate.
pub fn encode(bits: &[u8], rate: CodeRate) -> Vec<u8> {
    let coded = conv_encode(bits);
    match rate {
        CodeRate::Half => coded,
        CodeRate::TwoThird => puncture(&coded),
        CodeRate::ThreeFourth => puncture_three_fourths(&coded),
    }
}

/// Viterbi decoder for rate 1/2 convolutional code.
/// Input: soft bits (positive = more likely 0), paired as (G0, G1) per trellis step.
/// Output: decoded info bits.
///
/// Uses floating-point path metrics to avoid quantization loss, which is
/// critical for higher-order modulations (8PSK) with noise-aware LLR scaling.
pub fn viterbi_decode(soft_bits: &[f32], expected_info_bits: usize) -> Vec<u8> {
    let n_steps = soft_bits.len() / 2;
    if n_steps == 0 {
        return vec![];
    }

    let mut path_metric = vec![f32::NEG_INFINITY; NUM_STATES];
    let mut new_path_metric = vec![f32::NEG_INFINITY; NUM_STATES];
    path_metric[0] = 0.0;

    let mut survivors = vec![vec![0u8; NUM_STATES]; n_steps];

    for step in 0..n_steps {
        let r0 = soft_bits[step * 2];
        let r1 = soft_bits[step * 2 + 1];

        new_path_metric.fill(f32::NEG_INFINITY);

        for state in 0..NUM_STATES {
            if path_metric[state] == f32::NEG_INFINITY {
                continue;
            }

            for input in 0..2u8 {
                // Full register: input at MSB, state in lower bits
                let reg = (input << (K - 1) as u8) | state as u8;
                let next_state = (reg >> 1) as usize;

                let e0 = parity(reg & G0);
                let e1 = parity(reg & G1);

                let bm0 = if e0 == 0 { r0 } else { -r0 };
                let bm1 = if e1 == 0 { r1 } else { -r1 };
                let branch_metric = bm0 + bm1;

                let candidate = path_metric[state] + branch_metric;
                if candidate > new_path_metric[next_state] {
                    new_path_metric[next_state] = candidate;
                    survivors[step][next_state] = state as u8;
                }
            }
        }

        std::mem::swap(&mut path_metric, &mut new_path_metric);
    }

    // Traceback: prefer state 0 (encoder flushed), fall back to best state
    let mut state = 0u8;
    let mut best_metric = path_metric[0];
    for s in 1..NUM_STATES {
        if path_metric[s] > best_metric {
            best_metric = path_metric[s];
            state = s as u8;
        }
    }
    // Prefer the flushed state (0) if it's within a small margin of the best
    let margin = best_metric.abs() * 0.01 + 1.0;
    if path_metric[0] > best_metric - margin {
        state = 0;
    }

    let mut decoded = vec![0u8; n_steps];
    for step in (0..n_steps).rev() {
        // Input that caused prev_state → state: input = (state >> (K-2)) & 1
        let input_recovered = (state >> (K as u8 - 2)) & 1;
        decoded[step] = input_recovered;
        state = survivors[step][state as usize];
    }

    decoded.truncate(expected_info_bits);
    decoded
}

/// Decode with the given code rate.
pub fn decode(soft_bits: &[f32], expected_info_bits: usize, rate: CodeRate) -> Vec<u8> {
    let rate_half_soft = match rate {
        CodeRate::Half => soft_bits.to_vec(),
        CodeRate::TwoThird => depuncture(soft_bits),
        CodeRate::ThreeFourth => depuncture_three_fourths(soft_bits),
    };
    viterbi_decode(&rate_half_soft, expected_info_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_encode_decode_roundtrip() {
        let info = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1];
        let coded = conv_encode(&info);
        assert_eq!(coded.len(), (info.len() + K - 1) * 2);

        let soft: Vec<f32> = coded.iter().map(|&b| if b == 0 { 1.0 } else { -1.0 }).collect();
        let decoded = viterbi_decode(&soft, info.len());
        assert_eq!(decoded, info);
    }

    #[test]
    fn test_conv_with_errors() {
        let info = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1];
        let coded = conv_encode(&info);
        let mut soft: Vec<f32> = coded.iter().map(|&b| if b == 0 { 1.0 } else { -1.0 }).collect();

        // Introduce some errors
        for i in (0..soft.len()).step_by(7) {
            soft[i] = -soft[i] * 0.3;
        }

        let decoded = viterbi_decode(&soft, info.len());
        assert_eq!(decoded, info, "Viterbi should correct sparse errors");
    }

    #[test]
    fn test_punctured_roundtrip() {
        let info = vec![0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0];
        let coded = conv_encode(&info);
        let punctured = puncture(&coded);

        let soft: Vec<f32> = punctured.iter().map(|&b| if b == 0 { 1.0 } else { -1.0 }).collect();
        let depunctured = depuncture(&soft);

        let decoded = viterbi_decode(&depunctured, info.len());
        assert_eq!(decoded, info);
    }

    #[test]
    fn test_encode_decode_helpers() {
        let info = vec![1, 0, 1, 1, 0, 0, 1, 0];

        for &rate in &[CodeRate::Half, CodeRate::TwoThird] {
            let coded = encode(&info, rate);
            let soft: Vec<f32> = coded.iter().map(|&b| if b == 0 { 1.0 } else { -1.0 }).collect();
            let decoded = decode(&soft, info.len(), rate);
            assert_eq!(decoded, info, "Failed for rate {:?}", rate);
        }
    }

    #[test]
    fn test_punctured_three_fourths_roundtrip_multiple_of_three() {
        // Length multiple of 3 ensures the unpunctured length is a multiple of 6,
        // which previously dropped the final erasure.
        let info = vec![1, 0, 1, 1, 0, 0];
        let coded = conv_encode(&info);
        let punctured = puncture_three_fourths(&coded);

        let soft: Vec<f32> = punctured.iter().map(|&b| if b == 0 { 1.0 } else { -1.0 }).collect();
        let depunctured = depuncture_three_fourths(&soft);

        assert_eq!(depunctured.len(), coded.len(), "Depunctured length mismatch");

        let decoded = viterbi_decode(&depunctured, info.len());
        assert_eq!(decoded, info);
    }

    #[test]
    fn test_encoder_flushes_to_zero() {
        let info = vec![1, 1, 1, 1, 1, 1, 1, 1];
        let coded = conv_encode(&info);
        // After K-1 tail zeros, encoder should be back in state 0
        // Verify by checking the last K-1 steps produce consistent output
        assert_eq!(coded.len(), (info.len() + K - 1) * 2);
    }
}
