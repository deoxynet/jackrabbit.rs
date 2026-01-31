/// Header encode/decode.
///
/// Header: 1 OFDM symbol using BPSK rate-1/2 on bins 20–79 (60 carriers).
/// Carries 24 info bits: 2-bit frame_type + 5-bit seq_num + 5-bit protocol_id + 12-bit payload_length.
/// Conv encode: (24 + 6) * 2 = 60 coded bits = exactly 60 BPSK carriers (no padding).

use num_complex::Complex32;
use crate::params::{HEADER_BINS_START, HEADER_BINS_END, HEADER_INFO_BITS};
use crate::fec;
use crate::modulation;
use crate::ofdm;

/// Frame type carried in the header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameType {
    Data = 0,
    Ack = 1,
    Reset = 2,
}

impl FrameType {
    /// Decode from a 2-bit value. Returns None for reserved values.
    pub fn from_bits(val: u8) -> Option<Self> {
        match val {
            0 => Some(FrameType::Data),
            1 => Some(FrameType::Ack),
            2 => Some(FrameType::Reset),
            _ => None,
        }
    }
}

/// Structured header metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeaderInfo {
    pub frame_type: FrameType,   // 2 bits (0-3)
    pub seq_num: u8,             // 5 bits (0-31)
    pub protocol_id: u8,         // 5 bits (0-31)
    pub payload_len: usize,      // 12 bits (0-4095)
}

/// Encode header into 24 info bits → FEC → BPSK symbols for 60 carriers.
pub fn encode_header(info: &HeaderInfo) -> Vec<Complex32> {
    assert!(info.protocol_id < 32, "Protocol ID must fit in 5 bits");
    assert!(info.payload_len <= 4095, "Payload length must fit in 12 bits");
    assert!(info.seq_num < 32, "Sequence number must fit in 5 bits");

    // Pack 24 bits: [frame_type(2)] [seq_num(5)] [protocol_id(5)] [payload_len(12)]
    let mut info_bits = [0u8; HEADER_INFO_BITS];

    // Frame type: bits 0–1
    let ft = info.frame_type as u8;
    for i in 0..2 {
        info_bits[i] = (ft >> (1 - i)) & 1;
    }

    // Sequence number: bits 2–6
    for i in 0..5 {
        info_bits[2 + i] = (info.seq_num >> (4 - i)) & 1;
    }

    // Protocol ID: bits 7–11
    for i in 0..5 {
        info_bits[7 + i] = (info.protocol_id >> (4 - i)) & 1;
    }

    // Payload length: bits 12–23
    for i in 0..12 {
        info_bits[12 + i] = ((info.payload_len >> (11 - i)) & 1) as u8;
    }

    // FEC encode at rate 1/2: (24 + 6) * 2 = 60 coded bits = 60 carriers exactly
    let coded = fec::conv_encode(&info_bits);
    debug_assert_eq!(coded.len(), 60, "24 info bits should produce exactly 60 coded bits");

    // BPSK modulate
    modulation::modulate(&coded, crate::params::Modulation::Bpsk)
}

/// Encode header into an OFDM symbol waveform.
pub fn encode_header_symbol(info: &HeaderInfo) -> Vec<f32> {
    let symbols = encode_header(info);
    ofdm::synthesize_symbol(&symbols, HEADER_BINS_START, HEADER_BINS_END)
}

/// Encode header into an OFDM symbol waveform with custom cyclic prefix.
pub fn encode_header_symbol_with_cp(info: &HeaderInfo, cp_len: usize) -> Vec<f32> {
    let symbols = encode_header(info);
    ofdm::synthesize_symbol_cp(&symbols, HEADER_BINS_START, HEADER_BINS_END, cp_len)
}

/// Decode header from received frequency-domain bins.
/// Returns HeaderInfo or None if decode fails validation.
pub fn decode_header(received_bins: &[Complex32]) -> Option<HeaderInfo> {
    let n_carriers = HEADER_BINS_END - HEADER_BINS_START;
    assert_eq!(received_bins.len(), n_carriers);

    // Soft-demodulate BPSK
    let soft = modulation::soft_demodulate(received_bins, crate::params::Modulation::Bpsk);

    // All 60 soft bits are coded (no padding to skip)
    let coded_len = (HEADER_INFO_BITS + 6) * 2; // (24 + K-1) * 2 = 60
    debug_assert_eq!(coded_len, n_carriers);
    let soft_coded = &soft[..coded_len.min(soft.len())];

    // Viterbi decode
    let decoded = fec::viterbi_decode(soft_coded, HEADER_INFO_BITS);
    if decoded.len() < HEADER_INFO_BITS {
        return None;
    }

    // Extract frame_type (2 bits)
    let mut ft_val: u8 = 0;
    for i in 0..2 {
        ft_val |= decoded[i] << (1 - i);
    }
    let frame_type = FrameType::from_bits(ft_val)?;

    // Extract seq_num (5 bits)
    let mut seq_num: u8 = 0;
    for i in 0..5 {
        seq_num |= decoded[2 + i] << (4 - i);
    }

    // Extract protocol ID (5 bits)
    let mut proto_id: u8 = 0;
    for i in 0..5 {
        proto_id |= decoded[7 + i] << (4 - i);
    }

    // Extract payload length (12 bits)
    let mut payload_len: usize = 0;
    for i in 0..12 {
        payload_len |= (decoded[12 + i] as usize) << (11 - i);
    }

    if proto_id as usize >= crate::params::PROTOCOLS.len() || payload_len > crate::params::MAX_PAYLOAD {
        return None;
    }

    Some(HeaderInfo {
        frame_type,
        seq_num,
        protocol_id: proto_id,
        payload_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        for proto_id in 0..crate::params::PROTOCOLS.len() as u8 {
            for &payload_len in &[0, 1, 64, 128, 512, 1023, 2048, 4095] {
                let info = HeaderInfo {
                    frame_type: FrameType::Data,
                    seq_num: 0,
                    protocol_id: proto_id,
                    payload_len,
                };
                let symbols = encode_header(&info);
                let decoded = decode_header(&symbols);
                assert_eq!(
                    decoded,
                    Some(info),
                    "Failed for proto={}, len={}",
                    proto_id,
                    payload_len
                );
            }
        }
    }

    #[test]
    fn test_header_all_seq_nums() {
        for seq in 0..32u8 {
            let info = HeaderInfo {
                frame_type: FrameType::Data,
                seq_num: seq,
                protocol_id: 0,
                payload_len: 100,
            };
            let symbols = encode_header(&info);
            let decoded = decode_header(&symbols).expect("decode failed");
            assert_eq!(decoded.seq_num, seq, "seq_num mismatch for seq={}", seq);
        }
    }

    #[test]
    fn test_header_frame_types() {
        for &ft in &[FrameType::Data, FrameType::Ack, FrameType::Reset] {
            let info = HeaderInfo {
                frame_type: ft,
                seq_num: 5,
                protocol_id: 3,
                payload_len: 256,
            };
            let symbols = encode_header(&info);
            let decoded = decode_header(&symbols).expect("decode failed");
            assert_eq!(decoded.frame_type, ft);
        }
    }

    #[test]
    fn test_header_24_bits_produce_60_coded() {
        let info = HeaderInfo {
            frame_type: FrameType::Data,
            seq_num: 0,
            protocol_id: 0,
            payload_len: 0,
        };
        let symbols = encode_header(&info);
        assert_eq!(symbols.len(), 60, "24 info bits should produce exactly 60 BPSK symbols");
    }

    #[test]
    fn test_header_via_ofdm() {
        let info = HeaderInfo {
            frame_type: FrameType::Data,
            seq_num: 0,
            protocol_id: 4,
            payload_len: 256,
        };

        let waveform = encode_header_symbol(&info);
        let received_bins = ofdm::analyze_symbol(&waveform, HEADER_BINS_START, HEADER_BINS_END);
        let decoded = decode_header(&received_bins);
        assert_eq!(decoded, Some(info));
    }
}
