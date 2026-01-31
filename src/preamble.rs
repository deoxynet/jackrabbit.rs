/// Preamble generation (PN sequence) and detection.

use num_complex::Complex32;
use crate::params::{FFT_SIZE, CP_LEN, PREAMBLE_SYMBOLS, PREAMBLE_SYMBOLS_SHORT, PREAMBLE_SYMBOLS_MINIMAL};
use crate::ofdm;

/// Generate a PN (pseudo-noise) sequence of length `n` using an LFSR.
fn generate_pn_sequence(n: usize, seed: u32) -> Vec<f32> {
    let mut lfsr = seed | 1;
    let mut seq = Vec::with_capacity(n);
    for _ in 0..n {
        let bit = lfsr & 1;
        let feedback = ((lfsr >> 0) ^ (lfsr >> 2) ^ (lfsr >> 3) ^ (lfsr >> 5)) & 1;
        lfsr = (lfsr >> 1) | (feedback << 15);
        seq.push(if bit == 1 { 1.0 } else { -1.0 });
    }
    seq
}

/// Generate the Schmidl-Cox preamble waveform.
///
/// Uses a frequency-domain design where only even-indexed subcarriers carry
/// PN values. This creates a time-domain signal with two identical halves
/// per OFDM symbol (the Schmidl-Cox property).
///
/// Returns preamble_symbols × (FFT_SIZE + cp_len) samples.
pub fn generate_preamble_symbols(preamble_symbols: usize) -> Vec<f32> {
    generate_preamble_symbols_with_cp(preamble_symbols, CP_LEN)
}

/// Generate preamble with a custom cyclic prefix length.
pub fn generate_preamble_symbols_with_cp(preamble_symbols: usize, cp_len: usize) -> Vec<f32> {
    assert!(preamble_symbols > 0, "Preamble must have at least 1 symbol");
    let pn = generate_pn_sequence(FFT_SIZE / 2, 0xCAFE);

    let mut spectrum = vec![Complex32::new(0.0, 0.0); FFT_SIZE];
    for i in 1..FFT_SIZE / 2 {
        let even_bin = i * 2;
        if even_bin < FFT_SIZE / 2 {
            spectrum[even_bin] = Complex32::new(pn[i], 0.0);
            spectrum[FFT_SIZE - even_bin] = spectrum[even_bin].conj();
        }
    }

    let base_symbol = ofdm::ifft_real(&spectrum);
    let base_with_cp = ofdm::add_cyclic_prefix_n(&base_symbol, cp_len);

    let sym_total = FFT_SIZE + cp_len;
    let mut preamble = Vec::with_capacity(preamble_symbols * sym_total);
    for _ in 0..preamble_symbols {
        preamble.extend_from_slice(&base_with_cp);
    }
    preamble
}

/// Generate the default (long) preamble waveform.
pub fn generate_preamble() -> Vec<f32> {
    generate_preamble_symbols(PREAMBLE_SYMBOLS)
}

/// Generate training symbols (known BPSK data on all active bins).
/// Returns (waveform of 2 × SYMBOL_TOTAL samples, known symbols).
pub fn generate_training(bin_start: usize, bin_end: usize) -> (Vec<f32>, Vec<Complex32>) {
    generate_training_with_cp(bin_start, bin_end, CP_LEN)
}

/// Generate training symbols with a custom cyclic prefix length.
pub fn generate_training_with_cp(bin_start: usize, bin_end: usize, cp_len: usize) -> (Vec<f32>, Vec<Complex32>) {
    generate_training_full(bin_start, bin_end, cp_len, 2)
}

/// Generate training symbols with custom CP and symbol count (1 or 2).
pub fn generate_training_full(bin_start: usize, bin_end: usize, cp_len: usize, n_symbols: usize) -> (Vec<f32>, Vec<Complex32>) {
    assert!(n_symbols >= 1 && n_symbols <= 2, "Training needs 1 or 2 symbols");
    let n_carriers = bin_end - bin_start;
    let pn = generate_pn_sequence(n_carriers, 0xBEEF);

    let known_symbols: Vec<Complex32> = pn.iter()
        .map(|&p| Complex32::new(p, 0.0))
        .collect();

    let sym_total = FFT_SIZE + cp_len;
    let mut waveform = Vec::with_capacity(n_symbols * sym_total);
    for _ in 0..n_symbols {
        let sym = ofdm::synthesize_symbol_cp(&known_symbols, bin_start, bin_end, cp_len);
        waveform.extend_from_slice(&sym);
    }

    (waveform, known_symbols)
}

/// Get the known training symbols for a given band (for channel estimation at RX).
pub fn training_reference(bin_start: usize, bin_end: usize) -> Vec<Complex32> {
    let n_carriers = bin_end - bin_start;
    let pn = generate_pn_sequence(n_carriers, 0xBEEF);
    pn.iter().map(|&p| Complex32::new(p, 0.0)).collect()
}

/// Estimate fractional carrier frequency offset (CFO) from preamble samples.
///
/// Uses Schmidl-Cox autocorrelation at lag N/2 on the analytic signal.
/// The preamble has period-N/2 structure (even-bin PN), so the complex
/// autocorrelation P = Σ z*(n) · z(n+N/2) has phase proportional to CFO.
///
/// Returns estimated CFO in Hz.
pub fn estimate_cfo(preamble_samples: &[f32]) -> f32 {
    estimate_cfo_with_cp(preamble_samples, CP_LEN)
}

/// Estimate CFO with a custom cyclic prefix length.
pub fn estimate_cfo_with_cp(preamble_samples: &[f32], cp_len: usize) -> f32 {
    use rustfft::FftPlanner;
    use num_complex::Complex32;
    use std::f32::consts::PI;

    let lag = FFT_SIZE / 2; // 480
    let symbol_len = FFT_SIZE + cp_len;
    if preamble_samples.len() < symbol_len {
        return 0.0;
    }

    let n_symbols = preamble_samples.len() / symbol_len;
    if n_symbols == 0 {
        return 0.0;
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let ifft = planner.plan_fft_inverse(FFT_SIZE);

    let mut p = Complex32::new(0.0, 0.0);

    for sym_idx in 0..n_symbols {
        let start = sym_idx * symbol_len + cp_len;
        if start + FFT_SIZE > preamble_samples.len() {
            break;
        }
        let segment = &preamble_samples[start..start + FFT_SIZE];

        // Form analytic signal via FFT
        let mut spectrum: Vec<Complex32> = segment
            .iter()
            .map(|&s| Complex32::new(s, 0.0))
            .collect();

        fft.process(&mut spectrum);

        // One-sided spectrum: zero negative frequencies, double positive
        for k in 1..((FFT_SIZE + 1) / 2) {
            spectrum[k] *= 2.0;
        }
        for k in ((FFT_SIZE / 2) + 1)..FFT_SIZE {
            spectrum[k] = Complex32::new(0.0, 0.0);
        }

        ifft.process(&mut spectrum);
        let scale = 1.0 / FFT_SIZE as f32;

        // spectrum now contains the (unnormalized) analytic signal
        // Compute complex autocorrelation at the given lag for this symbol
        for i in 0..lag {
            let z0 = Complex32::new(spectrum[i].re * scale, spectrum[i].im * scale);
            let z1 = Complex32::new(spectrum[i + lag].re * scale, spectrum[i + lag].im * scale);
            p += z0.conj() * z1;
        }
    }

    // Δf = angle(P) × fs / (2π × lag)
    let angle = p.im.atan2(p.re);
    let fs = crate::params::SAMPLE_RATE as f32;
    angle * fs / (2.0 * PI * lag as f32)
}

/// Apply CFO correction to a waveform by SSB frequency-shifting by -Δf Hz,
/// with a phase accumulator for streaming continuity.
///
/// Returns (corrected_samples, end_phase_radians).
pub fn correct_cfo_with_phase(samples: &[f32], delta_f_hz: f32, start_phase: f32) -> (Vec<f32>, f32) {
    use rustfft::FftPlanner;
    use num_complex::Complex32;
    use std::f32::consts::PI;

    if delta_f_hz.abs() < 1e-10 || samples.is_empty() {
        return (samples.to_vec(), start_phase);
    }

    let n = samples.len();
    let fs = crate::params::SAMPLE_RATE as f32;

    // Form analytic signal via FFT
    let mut spectrum: Vec<Complex32> = samples
        .iter()
        .map(|&s| Complex32::new(s, 0.0))
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut spectrum);

    // One-sided spectrum: zero negative frequencies, double positive
    for k in 1..((n + 1) / 2) {
        spectrum[k] *= 2.0;
    }
    for k in ((n / 2) + 1)..n {
        spectrum[k] = Complex32::new(0.0, 0.0);
    }

    let ifft = planner.plan_fft_inverse(n);
    ifft.process(&mut spectrum);
    let scale = 1.0 / n as f32;

    let dphi = -2.0 * PI * delta_f_hz / fs;
    let mut phase = start_phase;
    let mut out = Vec::with_capacity(n);
    for z in &spectrum {
        let z_scaled = Complex32::new(z.re * scale, z.im * scale);
        let rot = Complex32::new(phase.cos(), phase.sin());
        out.push((z_scaled * rot).re);
        phase += dphi;
    }
    let end_phase = phase.rem_euclid(2.0 * PI);
    (out, end_phase)
}

/// Apply CFO correction to a waveform by SSB frequency-shifting by -Δf Hz.
///
/// Forms the analytic signal, multiplies by exp(-j·2π·Δf·n/fs), and takes
/// the real part. This perfectly inverts apply_cfo(samples, +Δf).
pub fn correct_cfo(samples: &[f32], delta_f_hz: f32) -> Vec<f32> {
    correct_cfo_with_phase(samples, delta_f_hz, 0.0).0
}

/// Preamble detector using energy detection + autocorrelation.
///
/// Two-phase detection:
/// 1. Energy detection: find where signal power rises above noise floor
/// 2. Autocorrelation at lag=SYMBOL_TOTAL over a short window confirms
///    the repeating structure and refines timing.
///
/// Returns the sample index where the preamble ends (start of training).
pub struct PreambleDetector {
    detected: bool,
    preamble_symbols: usize,
    overhead_cp: usize,
}

impl PreambleDetector {
    pub fn new() -> Self {
        PreambleDetector {
            detected: false,
            preamble_symbols: 0,
            overhead_cp: CP_LEN,
        }
    }

    pub fn set_overhead_cp(&mut self, cp_len: usize) {
        self.overhead_cp = cp_len;
    }

    pub fn reset(&mut self) {
        self.detected = false;
        self.preamble_symbols = 0;
    }

    /// Scan samples for preamble. Returns Some(preamble_end_offset) if found.
    ///
    /// Uses Schmidl-Cox autocorrelation at lag FFT_SIZE/2. The preamble uses
    /// only even-frequency bins, so each half-symbol is identical. The
    /// normalized autocorrelation M(d) = P(d)² / (R1(d) · R2(d)) approaches
    /// 1.0 during the preamble and 0.0 during noise, regardless of amplitude.
    /// This makes detection robust at very low SNR (e.g. phone→laptop OTA).
    pub fn feed(&mut self, samples: &[f32]) -> Option<usize> {
        if self.detected {
            return None;
        }

        let half = FFT_SIZE / 2; // 480 — the half-symbol lag
        let sym_total = FFT_SIZE + self.overhead_cp;
        let preamble_len_long = PREAMBLE_SYMBOLS * sym_total;
        let preamble_len_short = PREAMBLE_SYMBOLS_SHORT * sym_total;
        let preamble_len_minimal = PREAMBLE_SYMBOLS_MINIMAL * sym_total;
        // Need at least: smallest preamble + half-window for SC metric check after it
        let min_len = preamble_len_minimal + 2 * half;
        if samples.len() < min_len || samples.len() < 2 * half {
            return None;
        }

        let max_d = samples.len() - 2 * half;

        // Initialize running sums for Schmidl-Cox metric at d=0.
        //   P  = Σ x(n) · x(n+half)    (cross-correlation)
        //   R1 = Σ x(n)²               (energy of first half-window)
        //   R2 = Σ x(n+half)²          (energy of second half-window)
        let mut p: f64 = 0.0;
        let mut r1: f64 = 0.0;
        let mut r2: f64 = 0.0;
        for n in 0..half {
            let x0 = samples[n] as f64;
            let x1 = samples[n + half] as f64;
            p += x0 * x1;
            r1 += x0 * x0;
            r2 += x1 * x1;
        }

        let sc_threshold = 0.3;
        let min_plateau = half / 2; // 240 samples — well within the ~420-480 per-symbol plateau

        let mut plateau_start: Option<usize> = None;
        let mut plateau_count: usize = 0;
        let mut best_start: Option<usize> = None;

        for d in 0..max_d {
            let denom = r1 * r2;
            let m = if denom > 1e-20 { (p * p) / denom } else { 0.0 };

            if m > sc_threshold as f64 {
                if plateau_start.is_none() {
                    plateau_start = Some(d);
                    plateau_count = 0;
                }
                plateau_count += 1;
            } else {
                if plateau_count >= min_plateau {
                    best_start = plateau_start;
                    break;
                }
                plateau_start = None;
                plateau_count = 0;
            }

            // Incremental update: shift window by 1 sample
            if d + 2 * half < samples.len() {
                let x_old = samples[d] as f64;
                let x_mid = samples[d + half] as f64;
                let x_new = samples[d + 2 * half] as f64;

                p = p - x_old * x_mid + x_mid * x_new;
                r1 = r1 - x_old * x_old + x_mid * x_mid;
                r2 = r2 - x_mid * x_mid + x_new * x_new;
            }
        }

        // Check if plateau extends to end of scan range
        if best_start.is_none() && plateau_count >= min_plateau {
            best_start = plateau_start;
        }

        let coarse_start = match best_start {
            Some(s) => s,
            None => return None,
        };

        // Fine timing: find the rising edge of the M metric near the coarse start.
        //
        // Strategy: scan a search window to find peak M, then find the FIRST
        // position where M exceeds 90% of the peak. This gives timing that is:
        // - At or slightly before the true preamble start (never late)
        // - CFO-tolerant (P² is sign-insensitive)
        // - Robust to reverb (rising edge tracks the direct path arrival)
        //
        // Being slightly early is safe (OFDM cyclic prefix absorbs it).
        // Being late causes ISI (FFT window captures next-symbol samples).
        let search_end = (coarse_start + half + half / 2).min(max_d);

        // Pass 1: find peak M in the search window (recomputed from scratch)
        let mut fine_peak_m = 0.0f64;
        for d in coarse_start..search_end {
            if d + 2 * half > samples.len() { break; }
            let mut pf: f64 = 0.0;
            let mut r1f: f64 = 0.0;
            let mut r2f: f64 = 0.0;
            for n in 0..half {
                let x0 = samples[d + n] as f64;
                let x1 = samples[d + n + half] as f64;
                pf += x0 * x1;
                r1f += x0 * x0;
                r2f += x1 * x1;
            }
            let denom = r1f * r2f;
            let mf = if denom > 1e-20 { (pf * pf) / denom } else { 0.0 };
            if mf > fine_peak_m {
                fine_peak_m = mf;
            }
        }

        // Pass 2: find first position where M reaches 90% of peak.
        // Energy gate: require window-1 energy ≈ window-2 energy, which
        // rejects positions where window-1 straddles a zero-gap / signal
        // boundary (timing error).  Threshold adapts to CP size: tight
        // for small CP (wired, CP=30) to keep error well within the CP,
        // loose for large CP (OTA, CP=240) so reverb-induced energy
        // imbalance doesn't prevent detection.
        let rising_threshold = fine_peak_m * 0.9;
        let max_err = self.overhead_cp.saturating_sub(10);
        let energy_gate = (1.0 - max_err as f64 / half as f64).max(0.5);
        let mut fine_start = coarse_start;
        for d in coarse_start..search_end {
            if d + 2 * half > samples.len() { break; }
            let mut pf: f64 = 0.0;
            let mut r1f: f64 = 0.0;
            let mut r2f: f64 = 0.0;
            for n in 0..half {
                let x0 = samples[d + n] as f64;
                let x1 = samples[d + n + half] as f64;
                pf += x0 * x1;
                r1f += x0 * x0;
                r2f += x1 * x1;
            }
            let denom = r1f * r2f;
            let mf = if denom > 1e-20 { (pf * pf) / denom } else { 0.0 };
            if mf >= rising_threshold && r1f > energy_gate * r2f {
                fine_start = d;
                break;
            }
        }

        let start = fine_start;

        // Determine preamble length (2, 4, or 8 symbols) by checking if the
        // signal at each boundary still has the preamble's half-symbol repetition.
        // Uses the same Schmidl-Cox metric as detection — works at any
        // alignment within the preamble (robust to timing offsets < CP_LEN).
        //
        // Check boundaries from shortest to longest: if the metric drops below
        // threshold at a boundary, the preamble ended before that point.
        let sc_check = |boundary: usize| -> Option<f64> {
            if boundary + 2 * half > samples.len() {
                return None;
            }
            let mut p_c: f64 = 0.0;
            let mut r1_c: f64 = 0.0;
            let mut r2_c: f64 = 0.0;
            for n in 0..half {
                let x0 = samples[boundary + n] as f64;
                let x1 = samples[boundary + n + half] as f64;
                p_c += x0 * x1;
                r1_c += x0 * x0;
                r2_c += x1 * x1;
            }
            Some(if r1_c * r2_c > 1e-20 { (p_c * p_c) / (r1_c * r2_c) } else { 0.0 })
        };

        // Check at 2-symbol boundary
        let minimal_end = start + preamble_len_minimal;
        let m_minimal = match sc_check(minimal_end) {
            Some(m) => m,
            None => return None,
        };

        let preamble_len = if m_minimal <= 0.3 {
            // No repetition at 2-symbol boundary → minimal preamble (2 symbols)
            preamble_len_minimal
        } else {
            // Still repeating at 2 symbols — check at 4-symbol boundary
            let short_end = start + preamble_len_short;
            let m_short = match sc_check(short_end) {
                Some(m) => m,
                None => return None,
            };
            if m_short <= 0.3 {
                // Stopped at 4 symbols → short preamble
                preamble_len_short
            } else {
                // Still repeating at 4 symbols → long preamble (8 symbols)
                if start + preamble_len_long > samples.len() {
                    return None;
                }
                preamble_len_long
            }
        };
        let preamble_end = start + preamble_len;
        if preamble_end > samples.len() {
            return None;
        }

        self.detected = true;
        self.preamble_symbols = preamble_len / sym_total;
        Some(preamble_end)
    }

    pub fn is_detected(&self) -> bool {
        self.detected
    }

    pub fn preamble_symbols(&self) -> usize {
        self.preamble_symbols
    }

    pub fn preamble_len_samples(&self) -> usize {
        self.preamble_symbols * (FFT_SIZE + self.overhead_cp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::SYMBOL_TOTAL;

    #[test]
    fn test_preamble_generation() {
        let preamble = generate_preamble();
        assert_eq!(preamble.len(), PREAMBLE_SYMBOLS * SYMBOL_TOTAL);

        for &s in &preamble {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn test_preamble_has_repeated_halves() {
        let preamble = generate_preamble();

        let sym_start = crate::params::CP_LEN;
        let first_half = &preamble[sym_start..sym_start + FFT_SIZE / 2];
        let second_half = &preamble[sym_start + FFT_SIZE / 2..sym_start + FFT_SIZE];

        let mut max_diff = 0.0f32;
        for i in 0..FFT_SIZE / 2 {
            max_diff = max_diff.max((first_half[i] - second_half[i]).abs());
        }
        assert!(max_diff < 1e-5, "Halves should match, max_diff={}", max_diff);
    }

    #[test]
    fn test_training_generation() {
        let (waveform, known) = generate_training(20, 80);
        assert_eq!(waveform.len(), 2 * SYMBOL_TOTAL);
        assert_eq!(known.len(), 60);
    }

    #[test]
    fn test_preamble_detection_no_silence() {
        let payload = b"Hello";
        let waveform = crate::encoder::encode(payload, 0, 1.0);

        let mut detector = PreambleDetector::new();
        let result = detector.feed(&waveform);
        assert!(result.is_some(), "Should detect preamble");
        assert_eq!(result.unwrap(), PREAMBLE_SYMBOLS * SYMBOL_TOTAL);
        assert_eq!(detector.preamble_symbols(), PREAMBLE_SYMBOLS);
    }

    #[test]
    fn test_preamble_detection_with_silence() {
        let payload = b"Hello";
        let waveform = crate::encoder::encode(payload, 0, 1.0);

        let mut signal = vec![0.0f32; 4000];
        signal.extend_from_slice(&waveform);
        signal.extend_from_slice(&vec![0.0f32; 4000]);

        let mut detector = PreambleDetector::new();
        let result = detector.feed(&signal);
        assert!(result.is_some(), "Should detect preamble with silence");

        let offset = result.unwrap();
        let expected = 4000 + PREAMBLE_SYMBOLS * SYMBOL_TOTAL;
        // Timing within CP_LEN is valid (Schmidl-Cox rising edge is slightly early)
        let cp = crate::params::CP_LEN;
        assert!(
            offset <= expected && offset + cp >= expected,
            "Preamble end {} should be within CP_LEN ({}) of expected {}",
            offset, cp, expected,
        );
        assert_eq!(detector.preamble_symbols(), PREAMBLE_SYMBOLS);
    }

    #[test]
    fn test_preamble_generation_short() {
        let preamble = generate_preamble_symbols(PREAMBLE_SYMBOLS_SHORT);
        assert_eq!(preamble.len(), PREAMBLE_SYMBOLS_SHORT * SYMBOL_TOTAL);
    }

    #[test]
    fn test_preamble_detection_short() {
        let preamble = generate_preamble_symbols(PREAMBLE_SYMBOLS_SHORT);
        let (training, _) = generate_training(20, 80);
        let mut waveform = Vec::new();
        waveform.extend_from_slice(&preamble);
        waveform.extend_from_slice(&training);

        let mut detector = PreambleDetector::new();
        let result = detector.feed(&waveform);
        assert!(result.is_some(), "Should detect short preamble");
        assert_eq!(result.unwrap(), PREAMBLE_SYMBOLS_SHORT * SYMBOL_TOTAL);
        assert_eq!(detector.preamble_symbols(), PREAMBLE_SYMBOLS_SHORT);
    }

    #[test]
    fn test_preamble_generation_minimal() {
        let preamble = generate_preamble_symbols(PREAMBLE_SYMBOLS_MINIMAL);
        assert_eq!(preamble.len(), PREAMBLE_SYMBOLS_MINIMAL * SYMBOL_TOTAL);
    }

    #[test]
    fn test_preamble_detection_minimal() {
        let preamble = generate_preamble_symbols(PREAMBLE_SYMBOLS_MINIMAL);
        let (training, _) = generate_training(20, 80);
        let mut waveform = Vec::new();
        waveform.extend_from_slice(&preamble);
        waveform.extend_from_slice(&training);

        let mut detector = PreambleDetector::new();
        let result = detector.feed(&waveform);
        assert!(result.is_some(), "Should detect minimal (2-symbol) preamble");
        assert_eq!(result.unwrap(), PREAMBLE_SYMBOLS_MINIMAL * SYMBOL_TOTAL);
        assert_eq!(detector.preamble_symbols(), PREAMBLE_SYMBOLS_MINIMAL);
    }
}
