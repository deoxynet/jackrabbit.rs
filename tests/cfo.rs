/// Tests for carrier frequency offset (CFO) estimation and correction.
///
/// These tests define the API contract for CFO handling and will fail
/// until the estimator and corrector are implemented.

use jackrabbit::encoder;
use jackrabbit::decoder::Decoder;
use jackrabbit::impairments;
use jackrabbit::preamble;

// ---------------------------------------------------------------------------
// Fractional CFO estimation
// ---------------------------------------------------------------------------

/// The preamble has period-N/2 structure (even-bin PN). A Schmidl-Cox
/// estimator computes complex autocorrelation at lag N/2 over the FFT
/// portion (excluding CP) and extracts Δf = angle(P) / (2π · (N/2) / fs).
///
/// Since the waveform is real-valued, we need to form the analytic signal
/// (via Hilbert transform or frequency-domain zeroing of negative bins)
/// before computing the complex autocorrelation.
#[test]
fn test_fractional_cfo_estimate_accuracy() {
    let preamble = preamble::generate_preamble();

    // Apply known CFO values and check that the estimator recovers them
    for &true_cfo_hz in &[0.0, 5.0, -5.0, 12.5, -18.0, 24.0] {
        let impaired = impairments::apply_cfo(&preamble, true_cfo_hz);

        let estimated = preamble::estimate_cfo(&impaired);

        let error = (estimated - true_cfo_hz).abs();
        assert!(
            error < 1.0,
            "CFO estimate too far off: true={} Hz, estimated={} Hz, error={} Hz",
            true_cfo_hz, estimated, error,
        );
    }
}

/// At the Schmidl-Cox detection point, the estimator should resolve CFO
/// within ±(subcarrier_spacing / 2) = ±25 Hz. Values near ±25 Hz are
/// at the ambiguity boundary and may wrap — test that interior values work.
#[test]
fn test_fractional_cfo_near_boundary() {
    let preamble = preamble::generate_preamble();

    for &true_cfo_hz in &[20.0, -20.0, 23.0, -23.0] {
        let impaired = impairments::apply_cfo(&preamble, true_cfo_hz);
        let estimated = preamble::estimate_cfo(&impaired);

        let error = (estimated - true_cfo_hz).abs();
        assert!(
            error < 2.0,
            "Near-boundary CFO estimate failed: true={} Hz, estimated={} Hz",
            true_cfo_hz, estimated,
        );
    }
}

// ---------------------------------------------------------------------------
// End-to-end decode with fractional CFO
// ---------------------------------------------------------------------------

/// Encode with 8PSK rate-2/3 (most sensitive to phase errors), apply a small
/// CFO, and verify the decoder still recovers the payload. The decoder must
/// estimate and correct CFO internally.
#[test]
fn test_decode_with_fractional_cfo() {
    let payload = b"CFO robustness test!";
    let proto = 8; // MT_FASTEST: 8PSK 2/3

    let waveform = encoder::encode(payload, proto, 1.0);

    // 7 Hz offset — about 14% of subcarrier spacing.
    // Without correction, this causes ~50° phase rotation over 10 symbols,
    // which destroys 8PSK.
    let impaired = impairments::apply_cfo(&waveform, 7.0);

    let mut decoder = Decoder::new();
    let result = decoder.feed(&impaired);
    assert_eq!(
        result, payload,
        "Decode failed with 7 Hz CFO on 8PSK 2/3"
    );
}

/// Same as test_decode_with_fractional_cfo, but fed in chunks to ensure
/// streaming CFO correction keeps phase continuity.
#[test]
fn test_decode_with_fractional_cfo_chunked() {
    let payload = b"CFO chunked test!";
    let proto = 8; // MT_FASTEST (stubbed as QPSK)

    let waveform = encoder::encode(payload, proto, 1.0);
    let impaired = impairments::apply_cfo(&waveform, 7.0);

    let mut decoder = Decoder::new();
    let mut result = Vec::new();
    for chunk in impaired.chunks(1024) {
        let r = decoder.feed(chunk);
        if !r.is_empty() {
            result = r;
            break;
        }
    }
    assert_eq!(result, payload, "Chunked decode failed with fractional CFO");
}

/// BPSK at rate-1/2 is more robust — it should tolerate larger CFO.
#[test]
fn test_decode_bpsk_with_moderate_cfo() {
    let payload = b"BPSK handles more CFO";
    let proto = 0; // AUDIBLE_NORMAL: BPSK 1/2

    let waveform = encoder::encode(payload, proto, 1.0);
    let impaired = impairments::apply_cfo(&waveform, 15.0);

    let mut decoder = Decoder::new();
    let result = decoder.feed(&impaired);
    assert_eq!(
        result, payload,
        "Decode failed with 15 Hz CFO on BPSK 1/2"
    );
}

// ---------------------------------------------------------------------------
// Integer CFO detection
// ---------------------------------------------------------------------------

/// Shift the entire waveform's spectrum by exactly +1 subcarrier (50 Hz)
/// by applying a 50 Hz CFO. The integer CFO estimator (using training
/// correlation across candidate offsets) should detect the shift and
/// compensate by adjusting bin indices for header + data extraction.
///
/// This requires both fractional CFO correction (to remove the residual)
/// and integer CFO detection (to find the +1 bin shift).
#[test]
fn test_decode_with_integer_cfo_plus_one() {
    let payload = b"integer CFO +1";
    let proto = 3; // DT_NORMAL: BPSK 1/2 — robust to residual errors

    let waveform = encoder::encode(payload, proto, 1.0);
    // Exactly +1 subcarrier = 50 Hz
    let impaired = impairments::apply_cfo(&waveform, 50.0);

    let mut decoder = Decoder::new();
    let result = decoder.feed(&impaired);
    assert_eq!(
        result, payload,
        "Decode failed with +1 bin (50 Hz) integer CFO"
    );
}

/// Same but -1 subcarrier.
#[test]
fn test_decode_with_integer_cfo_minus_one() {
    let payload = b"integer CFO -1";
    let proto = 3;

    let waveform = encoder::encode(payload, proto, 1.0);
    let impaired = impairments::apply_cfo(&waveform, -50.0);

    let mut decoder = Decoder::new();
    let result = decoder.feed(&impaired);
    assert_eq!(
        result, payload,
        "Decode failed with -1 bin (-50 Hz) integer CFO"
    );
}

// ---------------------------------------------------------------------------
// CFO + noise combined
// ---------------------------------------------------------------------------

/// Real-world scenario: moderate CFO + noise. Should decode at 20 dB SNR
/// with 10 Hz CFO on QPSK 1/2.
#[test]
fn test_decode_cfo_plus_noise() {
    let payload = b"noisy CFO test";
    let proto = 4; // DT_FAST: QPSK 1/2

    let waveform = encoder::encode(payload, proto, 1.0);
    let with_cfo = impairments::apply_cfo(&waveform, 10.0);
    let noisy = impairments::apply_awgn(&with_cfo, 20.0, 7777);

    let mut decoder = Decoder::new();
    let result = decoder.feed(&noisy);
    assert_eq!(
        result, payload,
        "Decode failed with 10 Hz CFO + 20 dB SNR on QPSK 1/2"
    );
}
