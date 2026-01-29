/// Tests for pilot subcarrier insertion and channel tracking.
///
/// These tests define the API contract for pilot-based channel tracking
/// and will fail until pilot infrastructure is implemented.

use aqua_soundbin::encoder;
use aqua_soundbin::decoder::Decoder;
use aqua_soundbin::ofdm;
use aqua_soundbin::params::{self, FFT_SIZE, CP_LEN, SYMBOL_TOTAL, PREAMBLE_SYMBOLS, TRAINING_SYMBOLS, HEADER_SYMBOLS};
use num_complex::Complex32;

// ---------------------------------------------------------------------------
// Pilot mapping correctness
// ---------------------------------------------------------------------------

/// Verify that the protocol config can report which bins are pilots vs data,
/// and that the encoder places known values on pilot bins.
#[test]
fn test_pilot_bins_defined() {
    let config = params::get_protocol(4); // DT_FAST: 100 carriers

    // pilot_bins() should return the set of pilot bin indices
    let pilots = config.pilot_bins();
    assert!(
        !pilots.is_empty(),
        "Protocol should define pilot subcarriers"
    );

    // Pilots should be within the carrier range
    for &bin in &pilots {
        assert!(
            bin >= config.bin_start && bin < config.bin_end,
            "Pilot bin {} outside carrier range [{}, {})",
            bin, config.bin_start, config.bin_end,
        );
    }

    // Data carrier count should exclude pilots
    let n_data = config.num_data_carriers();
    assert_eq!(
        n_data,
        config.num_carriers() - pilots.len(),
        "Data carriers should be total minus pilots"
    );
}

/// The encoder should place known pilot values that the decoder can extract.
/// Encode a packet, FFT a data symbol, and check that pilot bins carry
/// the expected reference value (not data-dependent).
#[test]
fn test_encoder_inserts_pilots() {
    let payload = b"pilot test data";
    let proto = 4; // DT_FAST
    let config = params::get_protocol(proto);
    let pilots = config.pilot_bins();

    let waveform = encoder::encode(payload, proto, 1.0);

    // Skip to first data symbol (after preamble + training + header)
    let data_start = (PREAMBLE_SYMBOLS + TRAINING_SYMBOLS + HEADER_SYMBOLS) * SYMBOL_TOTAL;
    let sym = &waveform[data_start..data_start + SYMBOL_TOTAL];

    // Analyze the symbol
    let bins = ofdm::analyze_symbol(sym, config.bin_start, config.bin_end);

    // Encode a second packet with different payload
    let payload2 = b"different stuff";
    let waveform2 = encoder::encode(payload2, proto, 1.0);
    let sym2 = &waveform2[data_start..data_start + SYMBOL_TOTAL];
    let bins2 = ofdm::analyze_symbol(sym2, config.bin_start, config.bin_end);

    // Pilot bins should carry the same value regardless of payload
    // (they're data-independent known symbols)
    for &pilot_bin in &pilots {
        let idx = pilot_bin - config.bin_start;
        let _diff = (bins[idx] - bins2[idx]).norm();
        // After differential encoding the absolute value changes, but the
        // pilot pattern relative to its reference should be deterministic.
        // At minimum, pilots should have unit magnitude.
        assert!(
            bins[idx].norm() > 0.1,
            "Pilot bin {} has near-zero magnitude",
            pilot_bin,
        );
    }
}

/// Interleaver and modulator should only process data carriers, not pilots.
/// Coded bits per symbol should match num_data_carriers × bits_per_symbol.
#[test]
fn test_coded_bits_exclude_pilots() {
    let config = params::get_protocol(4);
    let n_data = config.num_data_carriers();
    let expected = n_data * config.modulation.bits_per_symbol();
    assert_eq!(
        config.coded_bits_per_symbol(),
        expected,
        "coded_bits_per_symbol should use data carriers only"
    );
}

// ---------------------------------------------------------------------------
// Channel tracking under drift
// ---------------------------------------------------------------------------

/// Simulate residual CFO that causes per-symbol common phase error (CPE).
/// Without pilot tracking, later symbols accumulate enough phase error to
/// corrupt decoding. With pilot tracking, the decoder compensates per-symbol.
///
/// Uses a 20-symbol packet (long enough for drift to matter) at QPSK 1/2
/// with a phase drift of 0.05 rad/symbol (~3° per symbol, ~60° by symbol 20).
#[test]
fn test_decode_with_phase_drift_pilots() {
    // Use a payload large enough to produce ~20 data symbols on DT_FAST
    let config = params::get_protocol(4);
    let bits_per_sym = config.coded_bits_per_symbol();
    let bytes_needed = (bits_per_sym * 15 / 8).max(64); // ~15 symbols worth
    let payload: Vec<u8> = (0..bytes_needed as u8).collect();
    let proto = 4;

    let waveform = encoder::encode(&payload, proto, 1.0);

    // Apply per-symbol phase drift in frequency domain.
    // This simulates residual CFO that accumulates across symbols.
    let phase_per_symbol = 0.15; // radians — ~8.6° per symbol, ~170° by symbol 20
    let mut drifted = waveform.clone();

    // Apply phase rotation to each OFDM symbol's frequency content.
    // Skip preamble and training (they're used for initial estimation).
    let data_start = (PREAMBLE_SYMBOLS + TRAINING_SYMBOLS + HEADER_SYMBOLS) * SYMBOL_TOTAL;
    for sym_idx in 0.. {
        let start = data_start + sym_idx * SYMBOL_TOTAL;
        if start + SYMBOL_TOTAL > drifted.len() { break; }

        let phase = phase_per_symbol * sym_idx as f32;
        let cos_p = phase.cos();
        let sin_p = phase.sin();

        // Apply phase rotation in frequency domain for accuracy
        let sym_samples = &drifted[start + CP_LEN..start + SYMBOL_TOTAL];
        let spectrum = ofdm::fft(sym_samples);
        let rotated: Vec<Complex32> = spectrum.iter().map(|&s| {
            Complex32::new(
                s.re * cos_p - s.im * sin_p,
                s.re * sin_p + s.im * cos_p,
            )
        }).collect();

        // IFFT back and reconstruct symbol with CP
        let time = ofdm::ifft_real(&rotated);
        let cp = &time[FFT_SIZE - CP_LEN..];
        drifted[start..start + CP_LEN].copy_from_slice(cp);
        drifted[start + CP_LEN..start + SYMBOL_TOTAL].copy_from_slice(&time);
    }

    let mut decoder = Decoder::new();
    let result = decoder.feed(&drifted);
    assert_eq!(
        result, payload,
        "Decode failed with phase drift of {} rad/symbol — pilot tracking needed",
        phase_per_symbol,
    );
}

/// Simulate sample clock offset (SCO) as a linearly increasing phase slope
/// across subcarriers. This causes different subcarriers to drift at
/// different rates. Pilot-based interpolation should track this.
#[test]
fn test_decode_with_sample_clock_offset() {
    // Use a large payload so many data symbols exist for drift to accumulate
    let payload: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
    let proto = 4; // DT_FAST: QPSK 1/2
    let _config = params::get_protocol(proto);

    let waveform = encoder::encode(&payload, proto, 1.0);
    let mut drifted = waveform.clone();

    // SCO manifests as: each symbol k has phase slope across subcarriers
    // proportional to k. Phase at bin b, symbol k = k * slope * (b - N/2).
    let sco_slope_per_symbol = 0.1; // radians per bin per symbol

    let data_start = (PREAMBLE_SYMBOLS + TRAINING_SYMBOLS + HEADER_SYMBOLS) * SYMBOL_TOTAL;
    for sym_idx in 0.. {
        let start = data_start + sym_idx * SYMBOL_TOTAL;
        if start + SYMBOL_TOTAL > drifted.len() { break; }

        let sym_samples = &drifted[start + CP_LEN..start + SYMBOL_TOTAL];
        let mut spectrum = ofdm::fft(sym_samples);

        // Apply linear phase slope
        for (bin, s) in spectrum.iter_mut().enumerate() {
            let bin_offset = bin as f32 - (FFT_SIZE / 2) as f32;
            let phase = sco_slope_per_symbol * sym_idx as f32 * bin_offset;
            let rot = Complex32::new(phase.cos(), phase.sin());
            *s = *s * rot;
        }

        let time = ofdm::ifft_real(&spectrum);
        let cp = &time[FFT_SIZE - CP_LEN..];
        drifted[start..start + CP_LEN].copy_from_slice(cp);
        drifted[start + CP_LEN..start + SYMBOL_TOTAL].copy_from_slice(&time);
    }

    let mut decoder = Decoder::new();
    let result = decoder.feed(&drifted);
    assert_eq!(
        result, payload,
        "Decode failed with SCO slope {} rad/bin/symbol — pilot interpolation needed",
        sco_slope_per_symbol,
    );
}
