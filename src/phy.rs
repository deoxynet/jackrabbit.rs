/// PHY layer: frame abstraction and link management.

use crate::decoder::Decoder;
use crate::encoder;
use crate::header::{FrameType, HeaderInfo};
use crate::params::{self, BitloadProfile};
use crate::adaptive::ChannelReport;

/// A PHY-layer frame with metadata and payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhyFrame {
    pub frame_type: FrameType,
    pub seq_num: u8,
    pub protocol_id: u8,
    pub payload: Vec<u8>,
}

impl PhyFrame {
    /// Create a data frame.
    pub fn data(seq_num: u8, protocol_id: u8, payload: Vec<u8>) -> Self {
        PhyFrame {
            frame_type: FrameType::Data,
            seq_num,
            protocol_id,
            payload,
        }
    }

    /// Create an ACK frame. The payload carries the acked sequence number.
    pub fn ack(seq_num: u8, protocol_id: u8, acked_seq: u8) -> Self {
        PhyFrame {
            frame_type: FrameType::Ack,
            seq_num,
            protocol_id,
            payload: vec![acked_seq],
        }
    }

    /// Create a RESET frame. The payload carries the next expected sequence number.
    pub fn reset(seq_num: u8, protocol_id: u8, next_seq: u8) -> Self {
        PhyFrame {
            frame_type: FrameType::Reset,
            seq_num,
            protocol_id,
            payload: vec![next_seq],
        }
    }

    fn header_info(&self) -> HeaderInfo {
        HeaderInfo {
            frame_type: self.frame_type,
            seq_num: self.seq_num,
            protocol_id: self.protocol_id,
            payload_len: self.payload.len(),
        }
    }

    /// Encode this frame into audio samples.
    pub fn encode(&self, volume: f32) -> Vec<f32> {
        encoder::encode_with_header(
            &self.payload,
            &self.header_info(),
            volume,
            params::PREAMBLE_SYMBOLS,
            params::CP_LEN,
            params::TRAINING_SYMBOLS,
        )
    }

    /// Encode with all configurable parameters.
    pub fn encode_full(
        &self,
        volume: f32,
        preamble_symbols: usize,
        overhead_cp_len: usize,
        training_symbols: usize,
    ) -> Vec<f32> {
        encoder::encode_with_header(
            &self.payload,
            &self.header_info(),
            volume,
            preamble_symbols,
            overhead_cp_len,
            training_symbols,
        )
    }

    /// Encode with all configurable parameters and an optional data CP override.
    pub fn encode_full_with_data_cp(
        &self,
        volume: f32,
        preamble_symbols: usize,
        overhead_cp_len: usize,
        training_symbols: usize,
        data_cp_len: Option<usize>,
    ) -> Vec<f32> {
        encoder::encode_with_header_data_cp(
            &self.payload,
            &self.header_info(),
            volume,
            preamble_symbols,
            overhead_cp_len,
            training_symbols,
            data_cp_len,
        )
    }

    /// Encode with a bitloading profile for per-band modulation.
    pub fn encode_bitloaded(
        &self,
        volume: f32,
        preamble_symbols: usize,
        overhead_cp_len: usize,
        training_symbols: usize,
        profile: &BitloadProfile,
    ) -> Vec<f32> {
        encoder::encode_bitloaded(
            &self.payload,
            &self.header_info(),
            volume,
            preamble_symbols,
            overhead_cp_len,
            training_symbols,
            profile,
        )
    }

    /// Encode with a bitloading profile and an optional data CP override.
    pub fn encode_bitloaded_with_data_cp(
        &self,
        volume: f32,
        preamble_symbols: usize,
        overhead_cp_len: usize,
        training_symbols: usize,
        data_cp_len: Option<usize>,
        profile: &BitloadProfile,
    ) -> Vec<f32> {
        encoder::encode_bitloaded_with_data_cp(
            &self.payload,
            &self.header_info(),
            volume,
            preamble_symbols,
            overhead_cp_len,
            training_symbols,
            data_cp_len,
            profile,
        )
    }
}

/// A stateful PHY link that manages TX sequencing and RX duplicate detection.
pub struct PhyLink {
    decoder: Decoder,
    tx_seq: u8,
    rx_last: Option<(FrameType, u8, u8)>,
    default_protocol: u8,
    volume: f32,
}

impl PhyLink {
    /// Create a new PHY link.
    pub fn new(protocol_id: u8, volume: f32) -> Self {
        PhyLink {
            decoder: Decoder::new(),
            tx_seq: 0,
            rx_last: None,
            default_protocol: protocol_id,
            volume,
        }
    }

    /// Encode a data frame, auto-incrementing the sequence number.
    /// Returns (audio samples, sequence number used).
    pub fn send_data(&mut self, payload: &[u8]) -> (Vec<f32>, u8) {
        let seq = self.tx_seq;
        let frame = PhyFrame::data(seq, self.default_protocol, payload.to_vec());
        let samples = frame.encode(self.volume);
        self.tx_seq = (self.tx_seq + 1) & 0x1F;
        (samples, seq)
    }

    /// Encode an ACK frame for the given sequence number.
    pub fn send_ack(&mut self, acked_seq: u8) -> Vec<f32> {
        let seq = self.tx_seq;
        let frame = PhyFrame::ack(seq, self.default_protocol, acked_seq);
        let samples = frame.encode(self.volume);
        self.tx_seq = (self.tx_seq + 1) & 0x1F;
        samples
    }

    /// Feed samples to the decoder. Returns a PhyFrame on successful decode.
    pub fn receive(&mut self, samples: &[f32]) -> Option<PhyFrame> {
        let (info, payload) = self.decoder.feed_frame(samples)?;
        Some(PhyFrame {
            frame_type: info.frame_type,
            seq_num: info.seq_num,
            protocol_id: info.protocol_id,
            payload,
        })
    }

    /// Feed samples and filter duplicate sequence numbers.
    /// Returns None if the frame's seq_num matches the last received.
    pub fn receive_dedup(&mut self, samples: &[f32]) -> Option<PhyFrame> {
        let frame = self.receive(samples)?;
        let key = (frame.frame_type, frame.seq_num, frame.protocol_id);
        if self.rx_last == Some(key) {
            return None;
        }
        self.rx_last = Some(key);
        Some(frame)
    }

    /// Set the cyclic prefix length for overhead symbols on the decoder.
    pub fn set_overhead_cp(&mut self, cp_len: usize) {
        self.decoder.set_overhead_cp(cp_len);
    }

    /// Set the number of training symbols on the decoder.
    pub fn set_training_symbols(&mut self, n: usize) {
        self.decoder.set_training_symbols(n);
    }

    /// Access the underlying decoder (e.g. for channel reports).
    pub fn decoder(&self) -> &Decoder {
        &self.decoder
    }

    /// Get the channel quality report from the last successful decode.
    pub fn channel_report(&self) -> Option<&ChannelReport> {
        self.decoder.channel_report()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phy_frame_data_roundtrip() {
        let payload = b"PHY data test";
        let frame = PhyFrame::data(3, 0, payload.to_vec());
        let samples = frame.encode(1.0);

        let mut decoder = Decoder::new();
        let (info, decoded) = decoder.feed_frame(&samples).expect("decode failed");
        assert_eq!(decoded, payload);
        assert_eq!(info.frame_type, FrameType::Data);
        assert_eq!(info.seq_num, 3);
        assert_eq!(info.protocol_id, 0);
    }

    #[test]
    fn test_phy_frame_ack_roundtrip() {
        let frame = PhyFrame::ack(7, 0, 5);
        let samples = frame.encode(1.0);

        let mut decoder = Decoder::new();
        let (info, decoded) = decoder.feed_frame(&samples).expect("decode failed");
        assert_eq!(info.frame_type, FrameType::Ack);
        assert_eq!(info.seq_num, 7);
        assert_eq!(decoded, vec![5u8]);
    }

    #[test]
    fn test_phy_link_seq_auto_increment() {
        let mut link = PhyLink::new(0, 1.0);

        let payload = b"seq test";
        let (_, seq0) = link.send_data(payload);
        let (_, seq1) = link.send_data(payload);
        let (_, seq2) = link.send_data(payload);

        assert_eq!(seq0, 0);
        assert_eq!(seq1, 1);
        assert_eq!(seq2, 2);
    }

    #[test]
    fn test_phy_link_seq_wraps_at_32() {
        let mut link = PhyLink::new(0, 1.0);
        let payload = b"wrap";

        for i in 0..32u8 {
            let (_, seq) = link.send_data(payload);
            assert_eq!(seq, i, "seq mismatch at iteration {}", i);
        }
        // Should wrap to 0
        let (_, seq) = link.send_data(payload);
        assert_eq!(seq, 0, "seq should wrap to 0");
    }

    #[test]
    fn test_phy_link_receive() {
        let frame = PhyFrame::data(5, 0, b"link rx".to_vec());
        let samples = frame.encode(1.0);

        let mut link = PhyLink::new(0, 1.0);
        let received = link.receive(&samples).expect("receive failed");
        assert_eq!(received.frame_type, FrameType::Data);
        assert_eq!(received.seq_num, 5);
        assert_eq!(received.payload, b"link rx");
    }

    #[test]
    fn test_phy_link_dedup() {
        let frame = PhyFrame::data(3, 0, b"dedup".to_vec());
        let samples = frame.encode(1.0);

        let mut link = PhyLink::new(0, 1.0);

        // First receive should work
        let r1 = link.receive_dedup(&samples);
        assert!(r1.is_some());

        // Same seq again should be filtered
        let r2 = link.receive_dedup(&samples);
        assert!(r2.is_none());
    }

    #[test]
    fn test_phy_link_dedup_allows_different_frame_types() {
        let data = PhyFrame::data(2, 0, b"data".to_vec());
        let ack = PhyFrame::ack(2, 0, 2);
        let data_samples = data.encode(1.0);
        let ack_samples = ack.encode(1.0);

        let mut link = PhyLink::new(0, 1.0);

        let r1 = link.receive_dedup(&data_samples);
        assert!(matches!(r1.as_ref().map(|f| f.frame_type), Some(FrameType::Data)));

        let r2 = link.receive_dedup(&ack_samples);
        assert!(matches!(r2.as_ref().map(|f| f.frame_type), Some(FrameType::Ack)));
    }

    #[test]
    fn test_backward_compat_old_encode_feed() {
        // Old API: encode() + feed() should still work
        let payload = b"backward compat";
        let waveform = encoder::encode(payload, 0, 1.0);
        let mut decoder = Decoder::new();
        let result = decoder.feed(&waveform);
        assert_eq!(result, payload);
    }
}
