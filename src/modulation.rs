/// BPSK, QPSK, and 8PSK modulation and soft demodulation.

use num_complex::Complex32;
use std::f32::consts::PI;

use crate::params::Modulation;

/// Modulate bits into complex symbols.
pub fn modulate(bits: &[u8], modulation: Modulation) -> Vec<Complex32> {
    match modulation {
        Modulation::Bpsk => modulate_bpsk(bits),
        Modulation::Qpsk => modulate_qpsk(bits),
        Modulation::Psk8 => modulate_8psk(bits),
        Modulation::Qam16 => modulate_16qam(bits),
        Modulation::Qam64 => modulate_64qam(bits),
        Modulation::Qam256 => modulate_256qam(bits),
        Modulation::Qam1024 => modulate_1024qam(bits),
    }
}

/// Soft-demodulate complex symbols into soft bits (LLR-like, positive = more likely 0).
pub fn soft_demodulate(symbols: &[Complex32], modulation: Modulation) -> Vec<f32> {
    match modulation {
        Modulation::Bpsk => soft_demod_bpsk(symbols),
        Modulation::Qpsk => soft_demod_qpsk(symbols),
        Modulation::Psk8 => soft_demod_8psk(symbols),
        Modulation::Qam16 => soft_demod_16qam(symbols),
        Modulation::Qam64 => soft_demod_64qam(symbols),
        Modulation::Qam256 => soft_demod_256qam(symbols),
        Modulation::Qam1024 => soft_demod_1024qam(symbols),
    }
}

// --- BPSK ---
// 0 → +1, 1 → -1

fn modulate_bpsk(bits: &[u8]) -> Vec<Complex32> {
    bits.iter()
        .map(|&b| {
            let s = if b == 0 { 1.0 } else { -1.0 };
            Complex32::new(s, 0.0)
        })
        .collect()
}

fn soft_demod_bpsk(symbols: &[Complex32]) -> Vec<f32> {
    // LLR ≈ 2 * Re(y) for BPSK with unit-power noise
    symbols.iter().map(|s| 2.0 * s.re).collect()
}

// --- QPSK ---
// Gray-coded: bits[0],bits[1] → I,Q
// 00 → (+1,+1)/√2, 01 → (+1,-1)/√2, 10 → (-1,+1)/√2, 11 → (-1,-1)/√2

const QPSK_SCALE: f32 = std::f32::consts::FRAC_1_SQRT_2;

fn modulate_qpsk(bits: &[u8]) -> Vec<Complex32> {
    assert!(bits.len() % 2 == 0, "QPSK needs even number of bits");
    bits.chunks_exact(2)
        .map(|chunk| {
            let i = if chunk[0] == 0 { 1.0 } else { -1.0 };
            let q = if chunk[1] == 0 { 1.0 } else { -1.0 };
            Complex32::new(i * QPSK_SCALE, q * QPSK_SCALE)
        })
        .collect()
}

fn soft_demod_qpsk(symbols: &[Complex32]) -> Vec<f32> {
    let mut soft = Vec::with_capacity(symbols.len() * 2);
    for s in symbols {
        // LLR for bit 0 ≈ 2√2 * Re(y), bit 1 ≈ 2√2 * Im(y)
        let scale = 2.0 * std::f32::consts::SQRT_2;
        soft.push(scale * s.re);
        soft.push(scale * s.im);
    }
    soft
}

// --- 8PSK ---
// Phase 1 stub: operates as QPSK (2 bits/symbol).
// Phase 2 will implement Gray-coded 8PSK with 3 bits/symbol on the unit circle.

fn modulate_8psk(bits: &[u8]) -> Vec<Complex32> {
    // Phase 1 stub: identical to QPSK (2 bits per symbol)
    assert!(bits.len() % 2 == 0, "8PSK stub needs even number of bits");
    bits.chunks_exact(2)
        .map(|chunk| {
            let i = if chunk[0] == 0 { 1.0 } else { -1.0 };
            let q = if chunk[1] == 0 { 1.0 } else { -1.0 };
            Complex32::new(i * QPSK_SCALE, q * QPSK_SCALE)
        })
        .collect()
}

fn soft_demod_8psk(symbols: &[Complex32]) -> Vec<f32> {
    // Phase 1 stub: identical to QPSK (2 soft bits per symbol)
    let mut soft = Vec::with_capacity(symbols.len() * 2);
    let scale = 2.0 * std::f32::consts::SQRT_2;
    for s in symbols {
        soft.push(scale * s.re);
        soft.push(scale * s.im);
    }
    soft
}

// --- 16QAM ---
// Gray-coded 4×4 constellation: 4 bits per symbol.
// Each axis uses Gray mapping: b0,b1 → {+3,+1,-1,-3}/√10
//   b0=0 → positive, b0=1 → negative
//   b1=0 → outer (mag 3), b1=1 → inner (mag 1)
// Normalized to unit average power (scale by 1/√10).

fn qam16_amplitude(b0: u8, b1: u8) -> f32 {
    let sign = if b0 == 0 { 1.0 } else { -1.0 };
    let mag = if b1 == 0 { 3.0 } else { 1.0 };
    sign * mag
}

fn modulate_16qam(bits: &[u8]) -> Vec<Complex32> {
    assert!(bits.len() % 4 == 0, "16QAM needs bits divisible by 4");
    let d = 10.0f32.sqrt();
    bits.chunks_exact(4)
        .map(|c| {
            let i = qam16_amplitude(c[0], c[1]);
            let q = qam16_amplitude(c[2], c[3]);
            Complex32::new(i / d, q / d)
        })
        .collect()
}

fn soft_demod_16qam(symbols: &[Complex32]) -> Vec<f32> {
    let d = 10.0f32.sqrt();
    let scale = 4.0 / d;
    let threshold = 2.0 / d;
    let mut soft = Vec::with_capacity(symbols.len() * 4);
    for s in symbols {
        // I-axis: b0 (MSB, sign) and b1 (LSB, inner/outer)
        soft.push(scale * s.re);
        soft.push(scale * (s.re.abs() - threshold));
        // Q-axis: b2 (MSB, sign) and b3 (LSB, inner/outer)
        soft.push(scale * s.im);
        soft.push(scale * (s.im.abs() - threshold));
    }
    soft
}

// --- 64QAM ---
// Gray-coded 8×8 constellation: 6 bits per symbol (3 I + 3 Q).
// Each axis uses Gray mapping: b0,b1,b2 → {+7,+5,+3,+1,-1,-3,-5,-7}/√42
//   b0=0 → positive, b0=1 → negative
//   b1,b2 Gray: 00→7, 01→5, 11→3, 10→1
// Normalized to unit average power (scale by 1/√42).

fn qam64_amplitude(b0: u8, b1: u8, b2: u8) -> f32 {
    let sign = if b0 == 0 { 1.0 } else { -1.0 };
    let mag = match (b1, b2) {
        (0, 0) => 7.0,
        (0, 1) => 5.0,
        (1, 1) => 3.0,
        (1, 0) => 1.0,
        _ => unreachable!(),
    };
    sign * mag
}

fn modulate_64qam(bits: &[u8]) -> Vec<Complex32> {
    assert!(bits.len() % 6 == 0, "64QAM needs bits divisible by 6");
    let d = 42.0f32.sqrt();
    bits.chunks_exact(6)
        .map(|c| {
            let i = qam64_amplitude(c[0], c[1], c[2]);
            let q = qam64_amplitude(c[3], c[4], c[5]);
            Complex32::new(i / d, q / d)
        })
        .collect()
}

fn soft_demod_64qam(symbols: &[Complex32]) -> Vec<f32> {
    let d = 42.0f32.sqrt();
    let scale = 4.0 / d;
    let t2 = 2.0 / d;
    let t4 = 4.0 / d;
    let mut soft = Vec::with_capacity(symbols.len() * 6);
    for s in symbols {
        // I-axis: b0 (MSB/sign), b1 (outer/inner group), b2 (within group)
        soft.push(scale * s.re);
        soft.push(scale * (s.re.abs() - t4));
        soft.push(scale * ((s.re.abs() - t4).abs() - t2));
        // Q-axis: b3 (MSB/sign), b4 (outer/inner group), b5 (within group)
        soft.push(scale * s.im);
        soft.push(scale * (s.im.abs() - t4));
        soft.push(scale * ((s.im.abs() - t4).abs() - t2));
    }
    soft
}

// --- 256QAM ---
// Gray-coded 16×16 constellation: 8 bits per symbol (4 I + 4 Q).
// Each axis uses Gray mapping: b0,b1,b2,b3 → {±15,±13,±11,±9,±7,±5,±3,±1}/√170
//   b0=0 → positive, b0=1 → negative
//   b1,b2,b3 Gray: 000→15, 001→13, 011→11, 010→9, 110→7, 111→5, 101→3, 100→1
// Normalized to unit average power (scale by 1/√170).

fn qam256_amplitude(b0: u8, b1: u8, b2: u8, b3: u8) -> f32 {
    let sign = if b0 == 0 { 1.0 } else { -1.0 };
    let mag = match (b1, b2, b3) {
        (0, 0, 0) => 15.0,
        (0, 0, 1) => 13.0,
        (0, 1, 1) => 11.0,
        (0, 1, 0) => 9.0,
        (1, 1, 0) => 7.0,
        (1, 1, 1) => 5.0,
        (1, 0, 1) => 3.0,
        (1, 0, 0) => 1.0,
        _ => unreachable!(),
    };
    sign * mag
}

fn modulate_256qam(bits: &[u8]) -> Vec<Complex32> {
    assert!(bits.len() % 8 == 0, "256QAM needs bits divisible by 8");
    let d = 170.0f32.sqrt();
    bits.chunks_exact(8)
        .map(|c| {
            let i = qam256_amplitude(c[0], c[1], c[2], c[3]);
            let q = qam256_amplitude(c[4], c[5], c[6], c[7]);
            Complex32::new(i / d, q / d)
        })
        .collect()
}

fn soft_demod_256qam(symbols: &[Complex32]) -> Vec<f32> {
    let d = 170.0f32.sqrt();
    let scale = 4.0 / d;
    let t2 = 2.0 / d;
    let t4 = 4.0 / d;
    let t8 = 8.0 / d;
    let mut soft = Vec::with_capacity(symbols.len() * 8);
    for s in symbols {
        // I-axis: b0 (sign), b1 (half), b2 (quarter), b3 (eighth)
        soft.push(scale * s.re);
        soft.push(scale * (s.re.abs() - t8));
        soft.push(scale * ((s.re.abs() - t8).abs() - t4));
        soft.push(scale * (((s.re.abs() - t8).abs() - t4).abs() - t2));
        // Q-axis: b4 (sign), b5 (half), b6 (quarter), b7 (eighth)
        soft.push(scale * s.im);
        soft.push(scale * (s.im.abs() - t8));
        soft.push(scale * ((s.im.abs() - t8).abs() - t4));
        soft.push(scale * (((s.im.abs() - t8).abs() - t4).abs() - t2));
    }
    soft
}

// --- 1024QAM ---
// Gray-coded 32×32 constellation: 10 bits per symbol (5 I + 5 Q).
// Each axis uses Gray mapping: b0,b1,b2,b3,b4 → {±31,±29,...,±3,±1}/√682
//   b0=0 → positive, b0=1 → negative
//   b1..b4 Gray: 0000→31, 0001→29, 0011→27, ..., 1000→1
// Normalized to unit average power (scale by 1/√682).

fn qam1024_amplitude(b0: u8, b1: u8, b2: u8, b3: u8, b4: u8) -> f32 {
    let sign = if b0 == 0 { 1.0 } else { -1.0 };
    let mag = match (b1, b2, b3, b4) {
        (0, 0, 0, 0) => 31.0,
        (0, 0, 0, 1) => 29.0,
        (0, 0, 1, 1) => 27.0,
        (0, 0, 1, 0) => 25.0,
        (0, 1, 1, 0) => 23.0,
        (0, 1, 1, 1) => 21.0,
        (0, 1, 0, 1) => 19.0,
        (0, 1, 0, 0) => 17.0,
        (1, 1, 0, 0) => 15.0,
        (1, 1, 0, 1) => 13.0,
        (1, 1, 1, 1) => 11.0,
        (1, 1, 1, 0) => 9.0,
        (1, 0, 1, 0) => 7.0,
        (1, 0, 1, 1) => 5.0,
        (1, 0, 0, 1) => 3.0,
        (1, 0, 0, 0) => 1.0,
        _ => unreachable!(),
    };
    sign * mag
}

fn modulate_1024qam(bits: &[u8]) -> Vec<Complex32> {
    assert!(bits.len() % 10 == 0, "1024QAM needs bits divisible by 10");
    let d = 682.0f32.sqrt();
    bits.chunks_exact(10)
        .map(|c| {
            let i = qam1024_amplitude(c[0], c[1], c[2], c[3], c[4]);
            let q = qam1024_amplitude(c[5], c[6], c[7], c[8], c[9]);
            Complex32::new(i / d, q / d)
        })
        .collect()
}

fn soft_demod_1024qam(symbols: &[Complex32]) -> Vec<f32> {
    let d = 682.0f32.sqrt();
    let scale = 4.0 / d;
    let t2 = 2.0 / d;
    let t4 = 4.0 / d;
    let t8 = 8.0 / d;
    let t16 = 16.0 / d;
    let mut soft = Vec::with_capacity(symbols.len() * 10);
    for s in symbols {
        // I-axis: b0 (sign), b1 (half), b2 (quarter), b3 (eighth), b4 (sixteenth)
        soft.push(scale * s.re);
        soft.push(scale * (s.re.abs() - t16));
        soft.push(scale * ((s.re.abs() - t16).abs() - t8));
        soft.push(scale * (((s.re.abs() - t16).abs() - t8).abs() - t4));
        soft.push(scale * ((((s.re.abs() - t16).abs() - t8).abs() - t4).abs() - t2));
        // Q-axis: b5 (sign), b6 (half), b7 (quarter), b8 (eighth), b9 (sixteenth)
        soft.push(scale * s.im);
        soft.push(scale * (s.im.abs() - t16));
        soft.push(scale * ((s.im.abs() - t16).abs() - t8));
        soft.push(scale * (((s.im.abs() - t16).abs() - t8).abs() - t4));
        soft.push(scale * ((((s.im.abs() - t16).abs() - t8).abs() - t4).abs() - t2));
    }
    soft
}

/// Apply differential encoding: each output symbol is input[i] * prev_output.
pub fn differential_encode(symbols: &[Complex32]) -> Vec<Complex32> {
    let mut out = Vec::with_capacity(symbols.len());
    let mut prev = Complex32::new(1.0, 0.0);
    for &s in symbols {
        let encoded = s * prev;
        // Normalize to keep on unit circle
        let norm = encoded.norm();
        let encoded = if norm > 0.0 { encoded / norm } else { encoded };
        out.push(encoded);
        prev = encoded;
    }
    out
}

/// Apply differential decoding: output[i] = input[i] * conj(input[i-1]).
pub fn differential_decode(symbols: &[Complex32]) -> Vec<Complex32> {
    let mut out = Vec::with_capacity(symbols.len());
    let mut prev = Complex32::new(1.0, 0.0);
    for &s in symbols {
        let decoded = s * prev.conj();
        out.push(decoded);
        prev = s;
    }
    out
}

/// Decision-directed differential decoding: uses hard decisions on the
/// previous symbol to reduce noise by ~3 dB compared to standard differential.
///
/// `n_phases`: number of PSK phases for the differentially encoded constellation.
/// `phase_offset`: rotation in radians for the constellation (e.g. π/4 for QPSK).
pub fn differential_decode_dd(symbols: &[Complex32], n_phases: usize, phase_offset: f32) -> Vec<Complex32> {
    let mut out = Vec::with_capacity(symbols.len());
    let mut prev = Complex32::new(1.0, 0.0);
    let mut first = true;
    for &s in symbols {
        let prev_hard = if first {
            prev
        } else {
            snap_to_psk(prev, n_phases, phase_offset)
        };
        let decoded = s * prev_hard.conj();
        out.push(decoded);
        prev = s;
        first = false;
    }
    out
}

/// Snap a complex value to the nearest M-PSK constellation point on the unit circle.
fn snap_to_psk(s: Complex32, m: usize, phase_offset: f32) -> Complex32 {
    let angle = s.im.atan2(s.re);
    let step = 2.0 * PI / m as f32;
    let nearest = ((angle - phase_offset) / step).round() * step + phase_offset;
    Complex32::new(nearest.cos(), nearest.sin())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpsk_roundtrip() {
        let bits = vec![0, 1, 0, 0, 1, 1];
        let symbols = modulate_bpsk(&bits);
        let soft = soft_demod_bpsk(&symbols);
        let hard: Vec<u8> = soft.iter().map(|&s| if s > 0.0 { 0 } else { 1 }).collect();
        assert_eq!(hard, bits);
    }

    #[test]
    fn test_qpsk_roundtrip() {
        let bits = vec![0, 0, 0, 1, 1, 0, 1, 1];
        let symbols = modulate_qpsk(&bits);
        let soft = soft_demod_qpsk(&symbols);
        let hard: Vec<u8> = soft.iter().map(|&s| if s > 0.0 { 0 } else { 1 }).collect();
        assert_eq!(hard, bits);
    }

    #[test]
    fn test_8psk_roundtrip() {
        // Phase 1: 8PSK stub operates as QPSK (2 bits per symbol).
        let bits = vec![0, 0, 0, 1, 1, 0, 1, 1];
        let symbols = modulate_8psk(&bits);
        let soft = soft_demod_8psk(&symbols);
        let hard: Vec<u8> = soft.iter().map(|&s| if s > 0.0 { 0 } else { 1 }).collect();
        assert_eq!(hard, bits);
    }

    #[test]
    fn test_64qam_roundtrip() {
        // Test all 64 constellation points
        let mut bits = Vec::new();
        for i in 0..64u8 {
            for b in (0..6).rev() {
                bits.push((i >> b) & 1);
            }
        }
        let symbols = modulate_64qam(&bits);
        assert_eq!(symbols.len(), 64);

        // Verify unit average power
        let avg_power: f32 = symbols.iter().map(|s| s.norm_sqr()).sum::<f32>() / symbols.len() as f32;
        assert!((avg_power - 1.0).abs() < 0.02, "Average power = {}, expected ~1.0", avg_power);

        let soft = soft_demod_64qam(&symbols);
        let hard: Vec<u8> = soft.iter().map(|&s| if s > 0.0 { 0 } else { 1 }).collect();
        assert_eq!(hard, bits);
    }

    #[test]
    fn test_256qam_roundtrip() {
        // Test all 256 constellation points
        let mut bits = Vec::new();
        for i in 0..=255u8 {
            for b in (0..8).rev() {
                bits.push((i >> b) & 1);
            }
        }
        let symbols = modulate_256qam(&bits);
        assert_eq!(symbols.len(), 256);

        // Verify unit average power
        let avg_power: f32 = symbols.iter().map(|s| s.norm_sqr()).sum::<f32>() / symbols.len() as f32;
        assert!((avg_power - 1.0).abs() < 0.02, "Average power = {}, expected ~1.0", avg_power);

        let soft = soft_demod_256qam(&symbols);
        let hard: Vec<u8> = soft.iter().map(|&s| if s > 0.0 { 0 } else { 1 }).collect();
        assert_eq!(hard, bits);
    }

    #[test]
    fn test_1024qam_roundtrip() {
        // Test all 1024 constellation points
        let mut bits = Vec::new();
        for i in 0..1024u16 {
            for b in (0..10).rev() {
                bits.push(((i >> b) & 1) as u8);
            }
        }
        let symbols = modulate_1024qam(&bits);
        assert_eq!(symbols.len(), 1024);

        // Verify unit average power
        let avg_power: f32 = symbols.iter().map(|s| s.norm_sqr()).sum::<f32>() / symbols.len() as f32;
        assert!((avg_power - 1.0).abs() < 0.02, "Average power = {}, expected ~1.0", avg_power);

        let soft = soft_demod_1024qam(&symbols);
        let hard: Vec<u8> = soft.iter().map(|&s| if s > 0.0 { 0 } else { 1 }).collect();
        assert_eq!(hard, bits);
    }

    #[test]
    fn test_differential_roundtrip() {
        let bits = vec![0, 1, 1, 0];
        let symbols = modulate_bpsk(&bits);
        let encoded = differential_encode(&symbols);
        let decoded = differential_decode(&encoded);
        let soft = soft_demod_bpsk(&decoded);
        let hard: Vec<u8> = soft.iter().map(|&s| if s > 0.0 { 0 } else { 1 }).collect();
        assert_eq!(hard, bits);
    }
}
