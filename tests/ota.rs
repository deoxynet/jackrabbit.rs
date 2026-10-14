/// Simulated over-the-air (OTA) integration tests.
///
/// Chain realistic channel impairments — reverb, frequency response,
/// CFO, AWGN, gain — and verify the decoder still recovers the payload.
///
/// The tests are ordered by increasing difficulty:
/// 1-3: Narrowband protocols through full OTA channel
/// 4: Wideband 16QAM near-field (minimal impairments)
/// 5: Protocol sweep with reverb + CFO + noise
/// 6: Noisy silence padding

use jackrabbit::decoder::Decoder;
use jackrabbit::encoder;
use jackrabbit::impairments;

/// Helper: pad with silence before and after.
fn pad_silence(signal: &[f32], pre_samples: usize, post_samples: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; pre_samples];
    out.extend_from_slice(signal);
    out.extend(vec![0.0f32; post_samples]);
    out
}

// ---------------------------------------------------------------------------
// Test 1: Small room, BPSK (audible band)
//
// Full OTA chain: reverb + freq response + CFO + AWGN + gain.
// BPSK 1/2 is the most robust protocol and handles all impairments.
// ---------------------------------------------------------------------------

#[test]
fn test_ota_small_room_bpsk() {
    let payload = b"small room BPSK test";
    let proto = 0; // AUDIBLE_NORMAL: BPSK 1/2, bins 20-80 (1-4 kHz)

    let waveform = encoder::encode(payload, proto, 1.0);
    let mut signal = impairments::apply_reverb(&waveform, 100.0, 42);
    signal.truncate(waveform.len());
    signal = impairments::apply_frequency_response(&signal, 200.0, 18000.0);
    signal = impairments::apply_cfo(&signal, 5.0);
    signal = impairments::apply_awgn(&signal, 15.0, 7777);
    signal = impairments::apply_gain(&signal, 0.3);

    let padded = pad_silence(&signal, 2400, 2400);
    let mut decoder = Decoder::new();
    let result = decoder.feed(&padded);
    assert_eq!(result, payload, "Small room BPSK decode failed");
}

// ---------------------------------------------------------------------------
// Test 2: Small room, QPSK (data-transfer band)
//
// Reverb + CFO + noise + gain, no frequency response filter.
// DT band (1-6 kHz) with QPSK is moderately robust.
// ---------------------------------------------------------------------------

#[test]
fn test_ota_small_room_qpsk() {
    let payload = b"small room QPSK test";
    let proto = 4; // DT_FAST: QPSK 1/2, bins 20-120 (1-6 kHz)

    let waveform = encoder::encode(payload, proto, 1.0);
    let mut signal = impairments::apply_reverb(&waveform, 50.0, 99);
    signal.truncate(waveform.len());
    signal = impairments::apply_cfo(&signal, 3.0);
    signal = impairments::apply_awgn(&signal, 20.0, 8888);
    signal = impairments::apply_gain(&signal, 0.5);

    let padded = pad_silence(&signal, 2400, 2400);
    let mut decoder = Decoder::new();
    let result = decoder.feed(&padded);
    assert_eq!(result, payload, "Small room QPSK decode failed");
}

// ---------------------------------------------------------------------------
// Test 3: Medium room, BPSK (audible band)
//
// Full OTA chain with longer reverb. Audible-band BPSK is robust enough
// to handle combined reverb + freq response even in medium rooms.
// ---------------------------------------------------------------------------

#[test]
fn test_ota_medium_room_bpsk() {
    let payload = b"medium room test";
    let proto = 0; // AUDIBLE_NORMAL: BPSK 1/2

    let waveform = encoder::encode(payload, proto, 1.0);
    let mut signal = impairments::apply_reverb(&waveform, 200.0, 123);
    signal.truncate(waveform.len());
    signal = impairments::apply_frequency_response(&signal, 200.0, 18000.0);
    signal = impairments::apply_cfo(&signal, 4.0);
    signal = impairments::apply_awgn(&signal, 18.0, 5555);
    signal = impairments::apply_gain(&signal, 0.4);

    let padded = pad_silence(&signal, 4800, 4800);
    let mut decoder = Decoder::new();
    let result = decoder.feed(&padded);
    assert_eq!(result, payload, "Medium room BPSK decode failed");
}

// ---------------------------------------------------------------------------
// Test 4: Near-field, 16QAM (wide band, short CP)
//
// 16QAM with short CP needs near-ideal conditions. This tests the
// near-field scenario: CFO + noise + mild gain, no reverb/freq response.
// Verifies the full encode→impair→decode chain works for the highest
// throughput protocol.
// ---------------------------------------------------------------------------

#[test]
fn test_ota_near_field_16qam() {
    let payload = b"near-field 16QAM";
    let proto = 11; // UT_FASTEST: 16QAM 3/4, short CP, bins 20-320

    let waveform = encoder::encode(payload, proto, 1.0);
    let signal = impairments::apply_cfo(&waveform, 2.0);
    let signal = impairments::apply_awgn(&signal, 28.0, 3333);
    let signal = impairments::apply_gain(&signal, 0.8);

    let padded = pad_silence(&signal, 2400, 2400);
    let mut decoder = Decoder::new();
    let result = decoder.feed(&padded);
    assert_eq!(result, payload, "Near-field 16QAM decode failed");
}

// ---------------------------------------------------------------------------
// Test 5: Protocol sweep through OTA channel
//
// Tests multiple protocols through reverb + CFO + noise + gain.
// Exercises audible, DT, and MT bandwidth classes.
// ---------------------------------------------------------------------------

#[test]
fn test_ota_protocol_sweep() {
    let payload = b"protocol sweep";

    // Proto 0: Audible BPSK 1/2 (bins 20-80)
    // Proto 3: DT BPSK 1/2 (bins 20-120)
    // Proto 6: MT BPSK 1/2 (bins 20-160)
    for &proto in &[0u8, 3, 6] {
        let waveform = encoder::encode(payload, proto, 1.0);
        let mut signal = impairments::apply_reverb(&waveform, 80.0, 300 + proto as u64);
        signal.truncate(waveform.len());
        signal = impairments::apply_cfo(&signal, 3.0);
        signal = impairments::apply_awgn(&signal, 18.0, 4000 + proto as u64);
        signal = impairments::apply_gain(&signal, 0.5);

        let padded = pad_silence(&signal, 2400, 4800);
        let mut decoder = Decoder::new();
        let result = decoder.feed(&padded);
        assert_eq!(
            result, payload,
            "Protocol sweep failed for proto {}",
            proto
        );
    }
}

// ---------------------------------------------------------------------------
// Test 6: OTA-robust protocol (protocol 15)
//
// Maximally robust config: BPSK 1/2 LDPC, audible band, full CP, dense pilots.
// Harder conditions than test 1 (longer reverb, freq response, same CFO/noise/gain).
// ---------------------------------------------------------------------------

#[test]
fn test_ota_robust_protocol() {
    let payload = b"OTA robust protocol test";
    let proto = 15; // OTA_ROBUST: BPSK 1/2 LDPC, bins 20-80, dense pilots

    let waveform = encoder::encode(payload, proto, 1.0);
    let mut signal = impairments::apply_reverb(&waveform, 150.0, 55);
    signal.truncate(waveform.len());
    signal = impairments::apply_frequency_response(&signal, 200.0, 18000.0);
    signal = impairments::apply_cfo(&signal, 4.0);
    signal = impairments::apply_awgn(&signal, 18.0, 6666);
    signal = impairments::apply_gain(&signal, 0.4);

    let padded = pad_silence(&signal, 4800, 4800);
    let mut decoder = Decoder::new();
    let result = decoder.feed(&padded);
    assert_eq!(result, payload, "OTA robust protocol decode failed");
}

// ---------------------------------------------------------------------------
// Test 7: Noisy silence padding
//
// Tests that the decoder handles background noise in the silence regions
// before and after the signal (as would occur with a real microphone).
// ---------------------------------------------------------------------------

#[test]
fn test_ota_noisy_silence_padding() {
    let payload = b"noisy silence test";
    let proto = 0; // BPSK 1/2

    let waveform = encoder::encode(payload, proto, 1.0);
    let mut signal = impairments::apply_reverb(&waveform, 100.0, 42);
    signal.truncate(waveform.len());
    signal = impairments::apply_frequency_response(&signal, 200.0, 18000.0);
    signal = impairments::apply_cfo(&signal, 3.0);
    signal = impairments::apply_awgn(&signal, 15.0, 1234);
    signal = impairments::apply_gain(&signal, 0.3);

    // Add noisy silence before and after (0.5s = 24000 samples)
    let noise_std = 0.01;
    let pre_len = 24000;
    let post_len = 24000;

    let mut padded = Vec::with_capacity(pre_len + signal.len() + post_len);

    // Generate noisy silence using simple LCG (same approach as the lib)
    let mut state: u64 = 9999 | 1;
    let mut lcg_f32 = || -> f32 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    let mut lcg_gaussian = || -> f32 {
        let u1 = lcg_f32().max(1e-10);
        let u2 = lcg_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    };

    for _ in 0..pre_len {
        padded.push(noise_std * lcg_gaussian());
    }
    padded.extend_from_slice(&signal);
    for _ in 0..post_len {
        padded.push(noise_std * lcg_gaussian());
    }

    let mut decoder = Decoder::new();
    let result = decoder.feed(&padded);
    assert_eq!(result, payload, "Noisy silence padding decode failed");
}
