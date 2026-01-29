/// Channel impairment simulation for testing.
///
/// Provides functions to apply realistic impairments to OFDM waveforms:
/// CFO (carrier frequency offset), AWGN (noise), gain scaling.

use std::f32::consts::PI;
use crate::params::SAMPLE_RATE;

/// Apply a carrier frequency offset (CFO) to a real-valued waveform.
///
/// Uses single-sideband (SSB) frequency shifting via the analytic signal:
/// 1. Form analytic signal z[n] = x[n] + j·H{x}[n] (Hilbert transform)
/// 2. Multiply by exp(j·2π·Δf·n/fs)
/// 3. Take real part
///
/// This shifts all frequencies by exactly +Δf Hz (not ±Δf like cos mixing).
/// Positive Δf shifts the spectrum up, negative shifts it down.
pub fn apply_cfo(samples: &[f32], delta_f_hz: f32) -> Vec<f32> {
    if delta_f_hz.abs() < 1e-10 {
        return samples.to_vec();
    }
    let n = samples.len();
    if n == 0 {
        return Vec::new();
    }
    let fs = SAMPLE_RATE as f32;

    // Form analytic signal via FFT
    let mut spectrum: Vec<num_complex::Complex<f32>> = samples
        .iter()
        .map(|&s| num_complex::Complex::new(s, 0.0))
        .collect();

    let mut planner = rustfft::FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut spectrum);

    // One-sided spectrum: zero negative frequencies, double positive
    // DC (k=0) stays as-is
    for k in 1..((n + 1) / 2) {
        spectrum[k] *= 2.0;
    }
    // Nyquist (k=n/2) stays as-is for even n
    for k in ((n / 2) + 1)..n {
        spectrum[k] = num_complex::Complex::new(0.0, 0.0);
    }

    let ifft = planner.plan_fft_inverse(n);
    ifft.process(&mut spectrum);
    let scale = 1.0 / n as f32;

    // Multiply analytic signal by exp(j·2π·Δf·n/fs) and take real part
    spectrum
        .iter()
        .enumerate()
        .map(|(i, z)| {
            let z_scaled = num_complex::Complex::new(z.re * scale, z.im * scale);
            let phase = 2.0 * PI * delta_f_hz * i as f32 / fs;
            let rot = num_complex::Complex::new(phase.cos(), phase.sin());
            (z_scaled * rot).re
        })
        .collect()
}

/// Apply a carrier frequency offset using complex mixing (analytic signal approach).
///
/// Returns a complex-valued result: x[n] * exp(j·2π·Δf·n/fs).
/// This is the true single-sideband CFO that a baseband receiver would see.
/// Used for computing the Schmidl-Cox autocorrelation on a real preamble:
/// form the analytic signal first, then correlate.
pub fn apply_cfo_complex(samples: &[f32], delta_f_hz: f32) -> Vec<(f32, f32)> {
    let fs = SAMPLE_RATE as f32;
    samples.iter().enumerate().map(|(n, &s)| {
        let phase = 2.0 * PI * delta_f_hz * n as f32 / fs;
        (s * phase.cos(), s * phase.sin())
    }).collect()
}

/// Apply additive white Gaussian noise at a given SNR (dB).
///
/// SNR is defined relative to signal RMS power.
/// Uses a simple LCG-based Box-Muller transform for reproducibility (no rand crate).
pub fn apply_awgn(samples: &[f32], snr_db: f32, seed: u64) -> Vec<f32> {
    let signal_power: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
    let noise_power = signal_power / (10.0f32).powf(snr_db / 10.0);
    let noise_std = noise_power.sqrt();

    let mut rng = LcgRng::new(seed);
    samples.iter().map(|&s| {
        s + noise_std * rng.next_gaussian()
    }).collect()
}

/// Scale waveform amplitude by a constant factor.
pub fn apply_gain(samples: &[f32], gain: f32) -> Vec<f32> {
    samples.iter().map(|&s| s * gain).collect()
}

/// Apply a per-symbol common phase error (CPE) that drifts linearly.
///
/// Simulates residual CFO that wasn't fully corrected, causing each OFDM symbol
/// to have a different phase rotation. `phase_per_symbol` is radians per symbol.
pub fn apply_phase_drift(samples: &[f32], symbol_len: usize, phase_per_symbol: f32) -> Vec<f32> {
    let mut out = samples.to_vec();
    for (sym_idx, chunk) in out.chunks_mut(symbol_len).enumerate() {
        let phase = phase_per_symbol * sym_idx as f32;
        let cos_p = phase.cos();
        let sin_p = phase.sin();
        // For a real signal, rotation by θ is just scaling by cos(θ)
        // plus a Hilbert-transformed component scaled by sin(θ).
        // Since we don't have the Hilbert transform here, we apply the
        // real part of the rotation: x[n] * cos(θ).
        // This is an approximation; for small θ it's ~1.
        // For proper testing, the test should apply this in frequency domain.
        for s in chunk.iter_mut() {
            *s *= cos_p;
        }
        // Note: this loses energy for large angles. The proper approach
        // is to apply phase rotation in the frequency domain after FFT,
        // which the pilot tracking tests will do directly.
        let _ = sin_p; // suppress warning
    }
    out
}

/// Simple LCG-based PRNG for reproducible noise generation.
pub(crate) struct LcgRng {
    state: u64,
}

impl LcgRng {
    pub(crate) fn new(seed: u64) -> Self {
        LcgRng { state: seed | 1 }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    pub(crate) fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Box-Muller transform for Gaussian samples.
    pub(crate) fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// FFT-based convolution and acoustic impairments
// ---------------------------------------------------------------------------

use num_complex::Complex;
use rustfft::FftPlanner;

/// FFT-based linear convolution. Output length = signal.len() + kernel.len() - 1.
fn convolve_fft(signal: &[f32], kernel: &[f32]) -> Vec<f32> {
    if signal.is_empty() || kernel.is_empty() {
        return Vec::new();
    }
    let out_len = signal.len() + kernel.len() - 1;
    let fft_len = out_len.next_power_of_two();

    let mut sig_buf: Vec<Complex<f32>> = signal
        .iter()
        .map(|&s| Complex::new(s, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
        .take(fft_len)
        .collect();

    let mut ker_buf: Vec<Complex<f32>> = kernel
        .iter()
        .map(|&s| Complex::new(s, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
        .take(fft_len)
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_len);
    fft.process(&mut sig_buf);
    fft.process(&mut ker_buf);

    // Pointwise multiply
    for (s, k) in sig_buf.iter_mut().zip(ker_buf.iter()) {
        *s *= k;
    }

    let ifft = planner.plan_fft_inverse(fft_len);
    ifft.process(&mut sig_buf);

    let scale = 1.0 / fft_len as f32;
    sig_buf[..out_len].iter().map(|c| c.re * scale).collect()
}

/// Apply sparse discrete multipath.
///
/// Each tap is `(delay_samples, gain)`. The direct path (delay=0, gain=1.0)
/// should be included explicitly if desired.
/// Output length = `samples.len() + max_delay`.
pub fn apply_multipath(samples: &[f32], taps: &[(usize, f32)]) -> Vec<f32> {
    if samples.is_empty() || taps.is_empty() {
        return samples.to_vec();
    }
    let max_delay = taps.iter().map(|&(d, _)| d).max().unwrap_or(0);
    let out_len = samples.len() + max_delay;
    let mut out = vec![0.0f32; out_len];

    for &(delay, gain) in taps {
        for (i, &s) in samples.iter().enumerate() {
            out[i + delay] += s * gain;
        }
    }
    out
}

/// Apply synthetic room reverb using an exponentially decaying impulse response.
///
/// Generates a deterministic RIR with:
/// - Direct path at sample 0 (gain 1.0)
/// - 6 early reflections at random delays in 1–5 ms, gains 0.05–0.25, random sign
///   (kept within the OFDM cyclic prefix window to avoid severe ISI)
/// - Late reverb: exponentially decaying Gaussian noise from 5 ms to RT60
///   (low-level diffuse tail)
///
/// `rt60_ms`: Time for 60 dB decay (e.g. 100 for a small room, 300 for a hall).
/// `seed`: RNG seed for deterministic generation.
pub fn apply_reverb(samples: &[f32], rt60_ms: f32, seed: u64) -> Vec<f32> {
    if samples.is_empty() || rt60_ms <= 0.0 {
        return samples.to_vec();
    }
    let fs = SAMPLE_RATE as f32;
    let rt60_samples = (rt60_ms * 0.001 * fs) as usize;
    let rir_len = rt60_samples.max(1);

    let mut rir = vec![0.0f32; rir_len];
    let mut rng = LcgRng::new(seed);

    // Direct path
    rir[0] = 1.0;

    // 6 early reflections in 1–5 ms (within CP guard interval)
    // Gains 0.05–0.25 (-12 to -6 dB), realistic for wall/floor reflections
    let early_max = ((5.0 * 0.001 * fs) as usize).min(rir_len - 1);
    for _ in 0..6 {
        let delay = 1 + (rng.next_f32() * (early_max as f32)) as usize;
        let delay = delay.min(rir_len - 1);
        let gain = 0.05 + 0.20 * rng.next_f32(); // 0.05–0.25
        let sign = if rng.next_f32() > 0.5 { 1.0 } else { -1.0 };
        rir[delay] += gain * sign;
    }

    // Late reverb: exponentially decaying Gaussian noise from 5 ms onward
    let late_start = ((5.0 * 0.001 * fs) as usize).min(rir_len);
    let decay_constant = 6.9 / rt60_samples as f32; // ln(10^3) ≈ 6.9 for 60 dB
    for i in late_start..rir_len {
        let envelope = (-decay_constant * i as f32).exp();
        rir[i] += 0.02 * envelope * rng.next_gaussian();
    }

    convolve_fft(samples, &rir)
}

/// Apply a frequency response simulating speaker + microphone roll-off.
///
/// Uses a second-order high-pass and low-pass model:
/// - High-pass: `gain = f² / (f² + f_low²)` — rolls off below `low_rolloff_hz`
/// - Low-pass: `gain = f_high² / (f² + f_high²)` — rolls off above `high_rolloff_hz`
///
/// Typical laptop values: `low=200`, `high=18000`.
/// Same-length output (filtering, not convolution).
pub fn apply_frequency_response(
    samples: &[f32],
    low_rolloff_hz: f32,
    high_rolloff_hz: f32,
) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let n = samples.len();
    let fs = SAMPLE_RATE as f32;
    let fft_len = n.next_power_of_two();

    let mut buf: Vec<Complex<f32>> = samples
        .iter()
        .map(|&s| Complex::new(s, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
        .take(fft_len)
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_len);
    fft.process(&mut buf);

    let f_low_sq = low_rolloff_hz * low_rolloff_hz;
    let f_high_sq = high_rolloff_hz * high_rolloff_hz;

    for k in 0..fft_len {
        let freq = if k <= fft_len / 2 {
            k as f32 * fs / fft_len as f32
        } else {
            (fft_len - k) as f32 * fs / fft_len as f32
        };
        let f_sq = freq * freq;

        // High-pass: attenuate below low_rolloff_hz
        let hp_gain = f_sq / (f_sq + f_low_sq);
        // Low-pass: attenuate above high_rolloff_hz
        let lp_gain = f_high_sq / (f_sq + f_high_sq);

        buf[k] *= hp_gain * lp_gain;
    }

    let ifft = planner.plan_fft_inverse(fft_len);
    ifft.process(&mut buf);

    let scale = 1.0 / fft_len as f32;
    buf[..n].iter().map(|c| c.re * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cfo_zero_is_identity() {
        let signal: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1).sin()).collect();
        let result = apply_cfo(&signal, 0.0);
        for (a, b) in signal.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_awgn_increases_power() {
        let _signal = vec![0.0f32; 1000]; // silence
        let noisy = apply_awgn(&vec![1.0f32; 1000], 10.0, 42);
        let noise_power: f32 = noisy.iter().map(|s| (s - 1.0) * (s - 1.0)).sum::<f32>() / 1000.0;
        // At 10dB SNR with unit signal power, noise power ≈ 0.1
        assert!(noise_power > 0.01 && noise_power < 0.5, "noise_power={}", noise_power);
    }

    #[test]
    fn test_gain_scales_amplitude() {
        let signal = vec![1.0f32, -1.0, 0.5];
        let scaled = apply_gain(&signal, 2.0);
        assert_eq!(scaled, vec![2.0, -2.0, 1.0]);
    }

    #[test]
    fn test_lcg_gaussian_distribution() {
        let mut rng = LcgRng::new(12345);
        let n = 10000;
        let samples: Vec<f32> = (0..n).map(|_| rng.next_gaussian()).collect();
        let mean: f32 = samples.iter().sum::<f32>() / n as f32;
        let variance: f32 = samples.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.1, "mean={}", mean);
        assert!((variance - 1.0).abs() < 0.2, "variance={}", variance);
    }

    // -----------------------------------------------------------------------
    // convolve_fft tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_convolve_fft_impulse() {
        // Convolving with a unit impulse should return the original signal
        let signal: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let kernel = vec![1.0];
        let result = convolve_fft(&signal, &kernel);
        assert_eq!(result.len(), signal.len());
        for (a, b) in signal.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-5, "a={}, b={}", a, b);
        }
    }

    #[test]
    fn test_convolve_fft_output_length() {
        let signal = vec![1.0; 50];
        let kernel = vec![1.0; 10];
        let result = convolve_fft(&signal, &kernel);
        assert_eq!(result.len(), 50 + 10 - 1);
    }

    #[test]
    fn test_convolve_fft_known_result() {
        // [1, 2, 3] * [1, 1] = [1, 3, 5, 3]
        let signal = vec![1.0, 2.0, 3.0];
        let kernel = vec![1.0, 1.0];
        let result = convolve_fft(&signal, &kernel);
        let expected = vec![1.0, 3.0, 5.0, 3.0];
        assert_eq!(result.len(), expected.len());
        for (a, b) in result.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-4, "a={}, b={}", a, b);
        }
    }

    // -----------------------------------------------------------------------
    // apply_multipath tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_multipath_single_tap_no_delay() {
        let signal = vec![1.0, 2.0, 3.0];
        let result = apply_multipath(&signal, &[(0, 1.0)]);
        assert_eq!(result.len(), 3);
        for (a, b) in signal.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_multipath_output_length() {
        let signal = vec![1.0; 100];
        let taps = vec![(0, 1.0), (50, 0.5)];
        let result = apply_multipath(&signal, &taps);
        assert_eq!(result.len(), 150); // 100 + 50
    }

    #[test]
    fn test_multipath_two_taps() {
        let signal = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let taps = vec![(0, 1.0), (2, 0.5)];
        let result = apply_multipath(&signal, &taps);
        assert_eq!(result.len(), 7); // 5 + 2
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
        assert!((result[2] - 0.5).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // apply_reverb tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reverb_preserves_direct_path() {
        // The RIR starts with 1.0, so the output should approximate the input
        // at the beginning (before reflections arrive)
        let signal: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.05).sin()).collect();
        let result = apply_reverb(&signal, 100.0, 42);
        assert!(result.len() > signal.len()); // Longer due to tail
        // First sample should match closely (only direct path contributes)
        assert!(
            (result[0] - signal[0]).abs() < 0.01,
            "result[0]={}, signal[0]={}", result[0], signal[0]
        );
    }

    #[test]
    fn test_reverb_deterministic() {
        let signal: Vec<f32> = (0..500).map(|i| (i as f32 * 0.1).sin()).collect();
        let r1 = apply_reverb(&signal, 100.0, 42);
        let r2 = apply_reverb(&signal, 100.0, 42);
        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_reverb_different_seeds_differ() {
        let signal: Vec<f32> = (0..500).map(|i| (i as f32 * 0.1).sin()).collect();
        let r1 = apply_reverb(&signal, 100.0, 1);
        let r2 = apply_reverb(&signal, 100.0, 2);
        // Should produce different results (check tail region)
        let mut differ = false;
        for i in 200..r1.len().min(r2.len()) {
            if (r1[i] - r2[i]).abs() > 1e-4 {
                differ = true;
                break;
            }
        }
        assert!(differ, "Different seeds should produce different reverb");
    }

    // -----------------------------------------------------------------------
    // apply_frequency_response tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_frequency_response_preserves_length() {
        let signal = vec![1.0; 1000];
        let result = apply_frequency_response(&signal, 200.0, 18000.0);
        assert_eq!(result.len(), 1000);
    }

    #[test]
    fn test_frequency_response_attenuates_dc() {
        // A DC signal (constant) should be strongly attenuated by high-pass
        let signal = vec![1.0; 4800]; // 100ms of DC
        let result = apply_frequency_response(&signal, 200.0, 18000.0);
        let out_power: f32 = result.iter().map(|s| s * s).sum::<f32>() / result.len() as f32;
        let in_power: f32 = signal.iter().map(|s| s * s).sum::<f32>() / signal.len() as f32;
        assert!(
            out_power < in_power * 0.1,
            "DC should be attenuated: in_power={}, out_power={}", in_power, out_power
        );
    }

    #[test]
    fn test_frequency_response_passes_midband() {
        // A 2 kHz tone should pass through with minimal attenuation
        // when low=200Hz, high=18000Hz
        let fs = SAMPLE_RATE as f32;
        let signal: Vec<f32> = (0..4800)
            .map(|i| (2.0 * PI * 2000.0 * i as f32 / fs).sin())
            .collect();
        let result = apply_frequency_response(&signal, 200.0, 18000.0);
        let in_power: f32 = signal.iter().map(|s| s * s).sum::<f32>() / signal.len() as f32;
        let out_power: f32 = result.iter().map(|s| s * s).sum::<f32>() / result.len() as f32;
        let ratio = out_power / in_power;
        assert!(
            ratio > 0.8 && ratio < 1.2,
            "2 kHz should pass through: ratio={}", ratio
        );
    }
}
