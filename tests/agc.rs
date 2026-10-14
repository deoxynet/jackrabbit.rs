/// Tests for automatic gain control (AGC) and noise floor estimation.
///
/// These tests define the API contract for AGC and noise-aware LLR
/// scaling and will fail until the features are implemented.

use jackrabbit::encoder;
use jackrabbit::decoder::Decoder;
use jackrabbit::impairments;
use jackrabbit::ofdm;
use jackrabbit::params::{self, GUARD_BINS, SYMBOL_TOTAL, PREAMBLE_SYMBOLS, TRAINING_SYMBOLS, HEADER_SYMBOLS};

// ---------------------------------------------------------------------------
// AGC behavior
// ---------------------------------------------------------------------------

/// AGC should normalize signal to a target RMS level.
/// The decoder should handle signals that arrive much louder or quieter
/// than the training symbols — e.g. if the mic gain changes mid-packet,
/// or the signal is just very quiet overall.
#[test]
fn test_decode_with_low_volume() {
    let payload = b"quiet signal test";
    let proto = 0; // BPSK 1/2 — most robust

    let waveform = encoder::encode(payload, proto, 1.0);
    // Scale entire waveform to 1% of original amplitude
    let quiet = impairments::apply_gain(&waveform, 0.01);

    let mut decoder = Decoder::new();
    let result = decoder.feed(&quiet);
    assert_eq!(
        result, payload,
        "Decode failed with very low volume (gain=0.01)"
    );
}

/// Signal that's much louder than expected (clipping not simulated,
/// but AGC should prevent overflow in soft-demod scaling).
#[test]
fn test_decode_with_high_volume() {
    let payload = b"loud signal test";
    let proto = 0;

    let waveform = encoder::encode(payload, proto, 1.0);
    let loud = impairments::apply_gain(&waveform, 10.0);

    let mut decoder = Decoder::new();
    let result = decoder.feed(&loud);
    assert_eq!(
        result, payload,
        "Decode failed with high volume (gain=10.0)"
    );
}

/// Data symbols arrive at different amplitude than training.
/// This simulates automatic gain control changing between training and data,
/// which happens on real hardware (e.g., phone mic AGC kicks in late).
///
/// Training at 1.0, data at 0.1. Without AGC renormalization, the
/// channel estimate is 10x too large and equalization destroys the signal.
#[test]
fn test_decode_gain_step_between_training_and_data() {
    let payload = b"gain step test!!";
    let proto = 0; // BPSK 1/2
    let _config = params::get_protocol(proto);

    let waveform = encoder::encode(payload, proto, 1.0);

    let mut modified = waveform.clone();
    let data_start = (PREAMBLE_SYMBOLS + TRAINING_SYMBOLS + HEADER_SYMBOLS) * SYMBOL_TOTAL;

    // Scale data portion by 0.1 (10x quieter than training)
    for s in modified[data_start..].iter_mut() {
        *s *= 0.1;
    }

    let mut decoder = Decoder::new();
    let result = decoder.feed(&modified);
    assert_eq!(
        result, payload,
        "Decode failed with 10x gain step between training and data"
    );
}

// ---------------------------------------------------------------------------
// Noise floor estimation from guard band
// ---------------------------------------------------------------------------

/// The guard band (bins 0–19) carries no signal. During data reception,
/// the energy in these bins is pure noise. The decoder should use this to
/// estimate the noise variance for better LLR scaling.
///
/// Test: estimate noise floor from a noisy data symbol's guard-band bins.
/// The estimate should be close to the known noise variance.
#[test]
fn test_noise_floor_from_guard_band() {
    // Create a noisy data symbol
    let config = params::get_protocol(0);
    let payload = b"noise floor test";
    let waveform = encoder::encode(payload, 0, 1.0);

    // Add noise at known SNR
    let snr_db = 15.0;
    let noisy = impairments::apply_awgn(&waveform, snr_db, 54321);

    // Take first data symbol
    let data_start = (PREAMBLE_SYMBOLS + TRAINING_SYMBOLS + HEADER_SYMBOLS) * SYMBOL_TOTAL;
    let sym = &noisy[data_start..data_start + SYMBOL_TOTAL];
    let spectrum = ofdm::fft(&sym[params::CP_LEN..]);

    // Guard band energy (bins 0 to GUARD_BINS-1 should carry only noise)
    let guard_energy: f32 = spectrum[..GUARD_BINS]
        .iter()
        .map(|s| s.norm_sqr())
        .sum::<f32>() / GUARD_BINS as f32;

    // Signal energy in data bins
    let data_energy: f32 = spectrum[config.bin_start..config.bin_end]
        .iter()
        .map(|s| s.norm_sqr())
        .sum::<f32>() / config.num_carriers() as f32;

    // The guard-band estimate should be much lower than data energy
    // (at 15 dB SNR, noise is ~3% of signal power)
    assert!(
        guard_energy < data_energy * 0.5,
        "Guard band energy ({}) should be much less than data energy ({})",
        guard_energy, data_energy,
    );

    // The estimate should be nonzero (there IS noise)
    assert!(
        guard_energy > 0.0,
        "Guard band energy should be nonzero in noisy signal"
    );
}

/// Noise-aware LLR scaling: decoder should use estimated noise variance
/// to scale soft bits properly. At low SNR, proper scaling improves
/// Viterbi performance by ~1–2 dB compared to fixed scaling.
#[test]
fn test_decode_with_noise_aware_llr() {
    let payload = b"noise-aware LLR";
    let proto = 7; // MT_FAST: QPSK 2/3 — needs good LLR at low SNR

    let waveform = encoder::encode(payload, proto, 1.0);

    // 6 dB SNR — very challenging for QPSK 2/3. Proper LLR normalization
    // (dividing by σ²) gives Viterbi ~1-2 dB effective gain, making the
    // difference between decode success and failure at this operating point.
    let noisy = impairments::apply_awgn(&waveform, 6.0, 99999);

    let mut decoder = Decoder::new();
    let result = decoder.feed(&noisy);
    assert_eq!(
        result, payload,
        "Decode failed at 6 dB SNR — noise-aware LLR scaling needed"
    );
}

/// AGC must handle silence without producing NaN or infinite values.
/// Feed pure silence to the decoder — it should return empty, not panic.
#[test]
fn test_agc_handles_silence() {
    let silence = vec![0.0f32; 48000]; // 1 second of silence

    let mut decoder = Decoder::new();
    let result = decoder.feed(&silence);
    assert!(result.is_empty(), "Silence should not decode to anything");
}

/// AGC must handle near-zero signals (floating-point denormals).
#[test]
fn test_agc_handles_near_zero() {
    let tiny: Vec<f32> = (0..48000).map(|i| (i as f32 * 0.01).sin() * 1e-20).collect();

    let mut decoder = Decoder::new();
    let result = decoder.feed(&tiny);
    // Should not panic, should not decode
    assert!(result.is_empty());
    // All internal state should be finite (no NaN from 0/0)
}
