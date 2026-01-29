/// Phase 3 coverage: LDPC codes and adaptive protocol selection.

use aqua_soundbin::decoder::Decoder;
use aqua_soundbin::encoder;
use aqua_soundbin::params::{self, FecType};

fn find_ldpc_protocols() -> Vec<&'static params::ProtocolConfig> {
    params::PROTOCOLS
        .iter()
        .filter(|p| p.fec_type == FecType::Ldpc)
        .collect()
}

#[test]
fn test_ldpc_protocol_exists() {
    let ldpc_protos = find_ldpc_protocols();
    assert!(
        !ldpc_protos.is_empty(),
        "Expected at least one protocol with FecType::Ldpc"
    );
}

#[test]
fn test_ldpc_loopback() {
    for config in find_ldpc_protocols() {
        let payload = b"ldpc loopback test";
        let waveform = encoder::encode(payload, config.id, 1.0);
        assert!(
            !waveform.is_empty(),
            "LDPC protocol {} produced empty waveform",
            config.name
        );

        let mut decoder = Decoder::new();
        let result = decoder.feed(&waveform);
        assert_eq!(
            result, payload,
            "LDPC loopback failed for protocol {} (id={})",
            config.name, config.id
        );
    }
}

#[test]
fn test_ldpc_various_sizes() {
    // Use the first LDPC protocol for this test
    let config = find_ldpc_protocols()[0];

    for &size in &[1usize, 40, 100, 500, 1023] {
        let payload: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let waveform = encoder::encode(&payload, config.id, 1.0);
        assert!(
            !waveform.is_empty(),
            "Protocol {} produced empty waveform for size {}",
            config.name, size
        );

        let mut decoder = Decoder::new();
        let result = decoder.feed(&waveform);
        assert_eq!(
            result, payload,
            "LDPC decode failed for protocol {} with payload size {}",
            config.name, size
        );
    }
}

#[test]
fn test_channel_report_after_decode() {
    let payload = b"channel report test";
    let proto = 0; // Use a convolutional protocol to test report generation
    let waveform = encoder::encode(payload, proto, 1.0);

    let mut decoder = Decoder::new();
    let result = decoder.feed(&waveform);
    assert_eq!(result, payload);

    let report = decoder.channel_report();
    assert!(
        report.is_some(),
        "Expected channel report after successful decode"
    );
    let report = report.unwrap();
    assert!(!report.bands.is_empty(), "Expected at least one band in report");
    assert!(
        !report.viable_protocols.is_empty(),
        "Expected at least one viable protocol"
    );
}

#[test]
fn test_channel_report_snr_reasonable() {
    // Clean loopback should show high SNR
    let payload = b"snr check";
    let proto = 4; // DT_FAST: QPSK, rate 1/2, bins 20-120
    let waveform = encoder::encode(payload, proto, 1.0);

    let mut decoder = Decoder::new();
    let result = decoder.feed(&waveform);
    assert_eq!(result, payload);

    let report = decoder.channel_report().expect("No channel report");
    // The DT band (bins 20-120) is band index 1
    let dt_band = &report.bands[1];
    assert!(
        dt_band.snr_db > 10.0,
        "Expected clean channel SNR > 10 dB, got {} dB",
        dt_band.snr_db
    );
}

#[test]
fn test_recommended_protocol_decodes() {
    // Get a channel report from a clean decode
    let payload = b"recommend test";
    let proto = 4;
    let waveform = encoder::encode(payload, proto, 1.0);

    let mut decoder = Decoder::new();
    let result = decoder.feed(&waveform);
    assert_eq!(result, payload);

    let report = decoder.channel_report().expect("No channel report");
    let rec = report.recommended_protocol;

    // Encode with the recommended protocol and verify it decodes
    let waveform2 = encoder::encode(payload, rec, 1.0);
    let mut decoder2 = Decoder::new();
    let result2 = decoder2.feed(&waveform2);
    assert_eq!(
        result2, payload,
        "Recommended protocol {} failed to decode",
        rec
    );
}

#[test]
fn test_existing_protocols_unchanged() {
    // Verify original convolutional protocols still work.
    // Protocol 8 was repurposed from MT_FASTEST (Conv) to WIRED_FAST (LDPC).
    let payload = b"protocol test";
    for proto_id in 0..12u8 {
        let config = params::get_protocol(proto_id);
        if proto_id != 8 {
            assert_eq!(
                config.fec_type,
                FecType::Convolutional,
                "Protocol {} should be Convolutional",
                proto_id
            );
        }

        let waveform = encoder::encode(payload, proto_id, 1.0);
        let mut decoder = Decoder::new();
        let result = decoder.feed(&waveform);
        assert_eq!(
            result, payload,
            "Regression: protocol {} (id={}) loopback failed",
            config.name, proto_id
        );
    }
}

#[test]
fn test_ldpc_channel_report_after_decode() {
    // Verify channel report works for LDPC protocols too
    let config = find_ldpc_protocols()[0];
    let payload = b"ldpc channel report";
    let waveform = encoder::encode(payload, config.id, 1.0);

    let mut decoder = Decoder::new();
    let result = decoder.feed(&waveform);
    assert_eq!(result, payload);

    let report = decoder.channel_report();
    assert!(
        report.is_some(),
        "Expected channel report after LDPC decode"
    );
}
