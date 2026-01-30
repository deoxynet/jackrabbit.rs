/// Header encode/decode.
///
/// Header: 1 OFDM symbol using BPSK rate-1/2 on bins 20–79 (60 carriers).
/// Carries 18 info bits: 5-bit protocol_id + 12-bit payload_length + 1-bit parity.

use num_complex::Complex32;
use crate::params::{HEADER_BINS_START, HEADER_BINS_END, HEADER_INFO_BITS};
use crate::fec;
use crate::modulation;
use crate::ofdm;

/// Encode header into 18 info bits → FEC → BPSK symbols for 60 carriers.
pub fn encode_header(protocol_id: u8, payload_len: usize) -> Vec<Complex32> {
    assert!(protocol_id < 32, "Protocol ID must fit in 5 bits");
    assert!(payload_len <= 4095, "Payload length must fit in 12 bits");

    // Pack 18 bits: [proto_id(5)] [payload_len(12)] [parity(1)]
    let mut info_bits = [0u8; HEADER_INFO_BITS];

    // Protocol ID: bits 0–4
    for i in 0..5 {
        info_bits[i] = ((protocol_id >> (4 - i)) & 1) as u8;
    }

    // Payload length: bits 5–16
    for i in 0..12 {
        info_bits[5 + i] = ((payload_len >> (11 - i)) & 1) as u8;
    }

    // Parity bit: XOR of first 17 bits
    let parity: u8 = info_bits[..17].iter().fold(0u8, |acc, &b| acc ^ b);
    info_bits[17] = parity;

    // FEC encode at rate 1/2
    let coded = fec::conv_encode(&info_bits);

    // We have (18 + 6) * 2 = 48 coded bits, but need 60 for 60 BPSK carriers.
    // Pad with zeros to fill all carriers.
    let n_carriers = HEADER_BINS_END - HEADER_BINS_START;
    let mut padded = coded;
    padded.resize(n_carriers, 0);

    // BPSK modulate
    modulation::modulate(&padded, crate::params::Modulation::Bpsk)
}

/// Encode header into an OFDM symbol waveform.
pub fn encode_header_symbol(protocol_id: u8, payload_len: usize) -> Vec<f32> {
    let symbols = encode_header(protocol_id, payload_len);
    ofdm::synthesize_symbol(&symbols, HEADER_BINS_START, HEADER_BINS_END)
}

/// Encode header into an OFDM symbol waveform with custom cyclic prefix.
pub fn encode_header_symbol_with_cp(protocol_id: u8, payload_len: usize, cp_len: usize) -> Vec<f32> {
    let symbols = encode_header(protocol_id, payload_len);
    ofdm::synthesize_symbol_cp(&symbols, HEADER_BINS_START, HEADER_BINS_END, cp_len)
}

/// Decode header from received frequency-domain bins.
/// Returns (protocol_id, payload_len) or None if parity check fails.
pub fn decode_header(received_bins: &[Complex32]) -> Option<(u8, usize)> {
    let n_carriers = HEADER_BINS_END - HEADER_BINS_START;
    assert_eq!(received_bins.len(), n_carriers);

    // Soft-demodulate BPSK
    let soft = modulation::soft_demodulate(received_bins, crate::params::Modulation::Bpsk);

    // Take only the coded bits (not the padding)
    let coded_len = (HEADER_INFO_BITS + 6) * 2; // (18 + K-1) * 2 = 48
    let soft_coded = &soft[..coded_len.min(soft.len())];

    // Viterbi decode
    let decoded = fec::viterbi_decode(soft_coded, HEADER_INFO_BITS);
    if decoded.len() < HEADER_INFO_BITS {
        return None;
    }

    // Check parity
    let parity: u8 = decoded[..17].iter().fold(0u8, |acc, &b| acc ^ b);
    if parity != decoded[17] {
        return None;
    }

    // Extract protocol ID (5 bits)
    let mut proto_id: u8 = 0;
    for i in 0..5 {
        proto_id |= decoded[i] << (4 - i);
    }

    // Extract payload length (12 bits)
    let mut payload_len: usize = 0;
    for i in 0..12 {
        payload_len |= (decoded[5 + i] as usize) << (11 - i);
    }

    if proto_id as usize >= crate::params::PROTOCOLS.len() || payload_len > crate::params::MAX_PAYLOAD {
        return None;
    }

    Some((proto_id, payload_len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        for proto_id in 0..crate::params::PROTOCOLS.len() as u8 {
            for &payload_len in &[0, 1, 64, 128, 512, 1023, 2048, 4095] {
                let symbols = encode_header(proto_id, payload_len);
                let decoded = decode_header(&symbols);
                assert_eq!(
                    decoded,
                    Some((proto_id, payload_len)),
                    "Failed for proto={}, len={}",
                    proto_id,
                    payload_len
                );
            }
        }
    }

    #[test]
    fn test_header_via_ofdm() {
        let proto_id = 4;
        let payload_len = 256;

        let waveform = encode_header_symbol(proto_id, payload_len);
        let received_bins = ofdm::analyze_symbol(&waveform, HEADER_BINS_START, HEADER_BINS_END);
        let decoded = decode_header(&received_bins);
        assert_eq!(decoded, Some((proto_id, payload_len)));
    }
}
