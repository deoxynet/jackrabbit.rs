/// OFDM modulation: IFFT synthesis, FFT analysis, cyclic prefix, Hermitian symmetry.

use num_complex::Complex32;
use rustfft::FftPlanner;
use crate::params::{FFT_SIZE, CP_LEN};

/// Place data symbols onto subcarrier bins with Hermitian symmetry for real-valued output.
/// `data_bins`: complex symbols for bins [bin_start..bin_end).
/// Returns FFT_SIZE complex frequency-domain vector ready for IFFT.
pub fn build_hermitian_spectrum(data_bins: &[Complex32], bin_start: usize, bin_end: usize) -> Vec<Complex32> {
    let mut spectrum = vec![Complex32::new(0.0, 0.0); FFT_SIZE];

    // Place data in positive frequencies
    for (i, &sym) in data_bins.iter().enumerate() {
        let bin = bin_start + i;
        if bin < bin_end && bin < FFT_SIZE / 2 {
            spectrum[bin] = sym;
        }
    }

    // DC and Nyquist are real
    spectrum[0] = Complex32::new(spectrum[0].re, 0.0);
    if FFT_SIZE % 2 == 0 {
        spectrum[FFT_SIZE / 2] = Complex32::new(spectrum[FFT_SIZE / 2].re, 0.0);
    }

    // Hermitian symmetry: spectrum[N-k] = conj(spectrum[k])
    for k in 1..FFT_SIZE / 2 {
        spectrum[FFT_SIZE - k] = spectrum[k].conj();
    }

    spectrum
}

/// IFFT: frequency domain → time domain. Returns real-valued samples.
pub fn ifft_real(spectrum: &[Complex32]) -> Vec<f32> {
    let n = spectrum.len();
    let mut buffer = spectrum.to_vec();

    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n);
    ifft.process(&mut buffer);

    // IFFT normalization: divide by N
    let scale = 1.0 / n as f32;
    buffer.iter().map(|c| c.re * scale).collect()
}

/// FFT: time domain → frequency domain.
pub fn fft(samples: &[f32]) -> Vec<Complex32> {
    let n = samples.len();
    let mut buffer: Vec<Complex32> = samples.iter().map(|&s| Complex32::new(s, 0.0)).collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    buffer
}

/// Prepend cyclic prefix to a time-domain OFDM symbol.
pub fn add_cyclic_prefix(symbol: &[f32]) -> Vec<f32> {
    assert_eq!(symbol.len(), FFT_SIZE);
    let mut out = Vec::with_capacity(FFT_SIZE + CP_LEN);
    // CP = last CP_LEN samples of the symbol
    out.extend_from_slice(&symbol[FFT_SIZE - CP_LEN..]);
    out.extend_from_slice(symbol);
    out
}

/// Remove cyclic prefix from a received symbol (CP_LEN + FFT_SIZE samples → FFT_SIZE).
pub fn remove_cyclic_prefix(symbol_with_cp: &[f32]) -> Vec<f32> {
    assert!(symbol_with_cp.len() >= CP_LEN + FFT_SIZE);
    symbol_with_cp[CP_LEN..CP_LEN + FFT_SIZE].to_vec()
}

/// Prepend a cyclic prefix of arbitrary length.
pub fn add_cyclic_prefix_n(symbol: &[f32], cp_len: usize) -> Vec<f32> {
    assert_eq!(symbol.len(), FFT_SIZE);
    let mut out = Vec::with_capacity(FFT_SIZE + cp_len);
    out.extend_from_slice(&symbol[FFT_SIZE - cp_len..]);
    out.extend_from_slice(symbol);
    out
}

/// Synthesize one OFDM symbol: data bins → IFFT → add CP → time-domain samples.
pub fn synthesize_symbol(data_bins: &[Complex32], bin_start: usize, bin_end: usize) -> Vec<f32> {
    let spectrum = build_hermitian_spectrum(data_bins, bin_start, bin_end);
    let time_domain = ifft_real(&spectrum);
    add_cyclic_prefix(&time_domain)
}

/// Synthesize with a custom CP length.
pub fn synthesize_symbol_cp(data_bins: &[Complex32], bin_start: usize, bin_end: usize, cp_len: usize) -> Vec<f32> {
    let spectrum = build_hermitian_spectrum(data_bins, bin_start, bin_end);
    let time_domain = ifft_real(&spectrum);
    add_cyclic_prefix_n(&time_domain, cp_len)
}

/// Analyze one OFDM symbol: remove CP → FFT → extract data bins.
pub fn analyze_symbol(symbol_with_cp: &[f32], bin_start: usize, bin_end: usize) -> Vec<Complex32> {
    let samples = remove_cyclic_prefix(symbol_with_cp);
    let spectrum = fft(&samples);
    spectrum[bin_start..bin_end].to_vec()
}

/// Analyze with a custom CP length.
pub fn analyze_symbol_cp(symbol_with_cp: &[f32], bin_start: usize, bin_end: usize, cp_len: usize) -> Vec<Complex32> {
    assert!(symbol_with_cp.len() >= cp_len + FFT_SIZE);
    let samples = &symbol_with_cp[cp_len..cp_len + FFT_SIZE];
    let spectrum = fft(samples);
    spectrum[bin_start..bin_end].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ifft_fft_roundtrip() {
        // Create a simple spectrum with Hermitian symmetry
        let mut spectrum = vec![Complex32::new(0.0, 0.0); FFT_SIZE];
        spectrum[10] = Complex32::new(1.0, 0.5);
        spectrum[FFT_SIZE - 10] = spectrum[10].conj();
        spectrum[20] = Complex32::new(0.3, -0.7);
        spectrum[FFT_SIZE - 20] = spectrum[20].conj();

        let time = ifft_real(&spectrum);
        let recovered = fft(&time);

        for k in 0..FFT_SIZE / 2 + 1 {
            let diff = (recovered[k] - spectrum[k]).norm();
            assert!(diff < 1e-4, "Mismatch at bin {}: diff={}", k, diff);
        }
    }

    #[test]
    fn test_cyclic_prefix_roundtrip() {
        let symbol: Vec<f32> = (0..FFT_SIZE).map(|i| (i as f32 * 0.01).sin()).collect();
        let with_cp = add_cyclic_prefix(&symbol);
        assert_eq!(with_cp.len(), FFT_SIZE + CP_LEN);

        let recovered = remove_cyclic_prefix(&with_cp);
        assert_eq!(recovered, symbol);

        // Verify CP is copy of end
        for i in 0..CP_LEN {
            assert_eq!(with_cp[i], symbol[FFT_SIZE - CP_LEN + i]);
        }
    }

    #[test]
    fn test_hermitian_real_output() {
        let data_bins = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 1.0),
            Complex32::new(-1.0, 0.0),
        ];
        let time = ifft_real(&build_hermitian_spectrum(&data_bins, 20, 23));
        // Output should be real (imaginary part ~0 from Hermitian symmetry)
        for &s in &time {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn test_synthesize_analyze_roundtrip() {
        let bin_start = 20;
        let bin_end = 80;
        let n_carriers = bin_end - bin_start;
        let data: Vec<Complex32> = (0..n_carriers)
            .map(|i| {
                let phase = i as f32 * 0.5;
                Complex32::new(phase.cos(), phase.sin())
            })
            .collect();

        let symbol = synthesize_symbol(&data, bin_start, bin_end);
        assert_eq!(symbol.len(), FFT_SIZE + CP_LEN);

        let recovered = analyze_symbol(&symbol, bin_start, bin_end);
        assert_eq!(recovered.len(), n_carriers);

        for i in 0..n_carriers {
            let diff = (recovered[i] - data[i]).norm();
            assert!(diff < 1e-3, "Mismatch at carrier {}: diff={}", i, diff);
        }
    }
}
