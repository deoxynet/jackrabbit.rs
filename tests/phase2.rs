/// Phase 2 coverage: UT band, 16QAM, rate 3/4, and short preamble/CP options.
///
/// These tests are expected to fail until Phase 2 is implemented.

use jackrabbit::decoder::Decoder;
use jackrabbit::encoder;
use jackrabbit::params::{self, ProtocolConfig, SYMBOL_TOTAL, PREAMBLE_SYMBOLS, TRAINING_SYMBOLS, HEADER_SYMBOLS};

fn find_ut_protocol() -> Option<&'static ProtocolConfig> {
    params::PROTOCOLS
        .iter()
        .find(|p| p.bin_start == 20 && p.bin_end == 320)
}

#[test]
fn test_ut_protocol_exists() {
    assert!(
        find_ut_protocol().is_some(),
        "Expected a UT protocol with bins 20..320 (300 carriers)"
    );
}

#[test]
fn test_ut_loopback() {
    let config = find_ut_protocol().expect("UT protocol not defined");
    let payload = b"ut loopback";

    let waveform = encoder::encode(payload, config.id, 1.0);
    assert!(
        !waveform.is_empty(),
        "UT encode produced empty waveform"
    );

    let mut decoder = Decoder::new();
    let result = decoder.feed(&waveform);
    assert_eq!(result, payload, "UT loopback failed");
}

#[test]
fn test_16qam_protocol_exists() {
    let has_16qam = params::PROTOCOLS
        .iter()
        .any(|p| p.modulation.bits_per_symbol() == 4);
    assert!(has_16qam, "Expected at least one 16QAM protocol (4 bits/symbol)");
}

#[test]
fn test_rate_three_fourths_exists() {
    let has_three_fourths = params::PROTOCOLS
        .iter()
        .any(|p| p.code_rate.ratio() == (3, 4));
    assert!(has_three_fourths, "Expected at least one rate-3/4 protocol");
}

/// Ensure at least one protocol uses a shorter preamble and/or shorter CP
/// than the global defaults. This enforces having a short-preamble/short-CP
/// option alongside the long default.
#[test]
fn test_short_packet_option_exists() {
    let payload = b"short option";
    let mut has_shorter = false;

    for config in params::PROTOCOLS.iter() {
        let n_data = config.data_symbols_for(payload.len());
        let baseline_symbols = PREAMBLE_SYMBOLS + TRAINING_SYMBOLS + HEADER_SYMBOLS + n_data;
        let baseline_len = baseline_symbols * SYMBOL_TOTAL;

        let actual_len = encoder::encode(payload, config.id, 1.0).len();
        if actual_len < baseline_len {
            has_shorter = true;
            break;
        }
    }

    assert!(
        has_shorter,
        "Expected at least one protocol with shorter preamble and/or CP than the global defaults"
    );
}
