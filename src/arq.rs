/// Selective Repeat ARQ: reliable delivery with multiple frames in flight.
///
/// Window of 16 frames (half the 32-entry sequence space), per-frame ACKs,
/// receiver reordering buffer, only retransmit lost frames.

use crate::decoder::Decoder;
use crate::header::FrameType;
use crate::params;
use crate::phy::PhyFrame;

const SEQ_BITS: u8 = 5;
const SEQ_SPACE: u8 = 1 << SEQ_BITS; // 32
const SEQ_MASK: u8 = SEQ_SPACE - 1;  // 0x1F
const WINDOW_SIZE: u8 = SEQ_SPACE / 2; // 16

/// Is `seq` in [base, base+size) mod SEQ_SPACE?
fn seq_in_range(seq: u8, base: u8, size: u8) -> bool {
    (seq.wrapping_sub(base) & SEQ_MASK) < size
}

/// Events returned by [`SelectiveRepeat::receive`].
#[derive(Debug)]
pub enum ArqEvent {
    /// In-order data delivered. Caller must transmit `ack` samples.
    /// `payloads` contains 1+ consecutive payloads (gap-fill delivers buffered frames).
    /// Empty `payloads` means out-of-order frame was buffered but not yet deliverable.
    ReceivedData { payloads: Vec<Vec<u8>>, ack: Vec<f32> },
    /// One or more pending sends were acknowledged (window advanced).
    Acked,
    /// Duplicate / below-window data. Caller must transmit `ack` (re-ack).
    DuplicateAck(Vec<f32>),
    /// Sender reset: drop buffered RX window and resync to next_seq.
    Reset { next_seq: u8 },
}

struct TxSlot {
    frame: PhyFrame,
    retries: u32,
}

pub struct SelectiveRepeat {
    decoder: Decoder,
    // TX
    tx_base: u8,                               // oldest unacked seq
    tx_next: u8,                               // next seq to assign
    tx_window: [Option<TxSlot>; 16],           // indexed by seq & 0x0F
    // RX
    rx_base: u8,                               // next expected in-order seq
    rx_window: [Option<Vec<u8>>; 16],          // buffered OOO payloads
    // Config
    protocol_id: u8,
    volume: f32,
    preamble_symbols: usize,
    overhead_cp: usize,
    training_symbols: usize,
    max_retries: u32,
}

const NONE_TX: Option<TxSlot> = None;
const NONE_RX: Option<Vec<u8>> = None;

impl SelectiveRepeat {
    /// Default config (OTA settings).
    pub fn new(protocol_id: u8, volume: f32) -> Self {
        SelectiveRepeat {
            decoder: Decoder::new(),
            tx_base: 0,
            tx_next: 0,
            tx_window: [NONE_TX; 16],
            rx_base: 0,
            rx_window: [NONE_RX; 16],
            protocol_id,
            volume,
            preamble_symbols: params::PREAMBLE_SYMBOLS,
            overhead_cp: params::CP_LEN,
            training_symbols: params::TRAINING_SYMBOLS,
            max_retries: 5,
        }
    }

    /// Cable-optimized config (short preamble, protocol CP, 1 training symbol).
    pub fn new_wired(protocol_id: u8, volume: f32, cp_len: usize) -> Self {
        let mut decoder = Decoder::new();
        decoder.set_overhead_cp(cp_len);
        decoder.set_training_symbols(1);
        SelectiveRepeat {
            decoder,
            tx_base: 0,
            tx_next: 0,
            tx_window: [NONE_TX; 16],
            rx_base: 0,
            rx_window: [NONE_RX; 16],
            protocol_id,
            volume,
            preamble_symbols: params::PREAMBLE_SYMBOLS_MINIMAL,
            overhead_cp: cp_len,
            training_symbols: 1,
            max_retries: 5,
        }
    }

    /// Queue data to send. Returns encoded samples.
    /// Returns None if the window is full.
    pub fn send(&mut self, payload: &[u8]) -> Option<Vec<f32>> {
        if !self.can_send() {
            return None;
        }
        if payload.is_empty() || payload.len() > params::MAX_PAYLOAD {
            return None;
        }
        let seq = self.tx_next;
        let frame = PhyFrame::data(seq, self.protocol_id, payload.to_vec());
        let samples = frame.encode_full(
            self.volume,
            self.preamble_symbols,
            self.overhead_cp,
            self.training_symbols,
        );
        if samples.is_empty() {
            return None;
        }
        self.tx_window[(seq & 0x0F) as usize] = Some(TxSlot {
            frame,
            retries: 0,
        });
        self.tx_next = (self.tx_next + 1) & SEQ_MASK;
        Some(samples)
    }

    /// Feed received audio. Returns event if a frame decoded.
    pub fn receive(&mut self, samples: &[f32]) -> Option<ArqEvent> {
        let (info, payload) = self.decoder.feed_frame(samples)?;
        if info.protocol_id != self.protocol_id {
            return None;
        }
        match info.frame_type {
            FrameType::Data => {
                let s = info.seq_num;
                if seq_in_range(s, self.rx_base, WINDOW_SIZE) {
                    // In current window — buffer
                    self.rx_window[(s & 0x0F) as usize] = Some(payload);
                    // Drain consecutive filled slots from rx_base
                    let mut payloads = Vec::new();
                    while let Some(data) = self.rx_window[(self.rx_base & 0x0F) as usize].take() {
                        payloads.push(data);
                        self.rx_base = (self.rx_base + 1) & SEQ_MASK;
                    }
                    // Cumulative ACK: only when rx_base advanced (consecutive delivery)
                    let ack = if !payloads.is_empty() {
                        let ack_seq = (self.rx_base.wrapping_sub(1)) & SEQ_MASK;
                        self.build_ack(ack_seq)
                    } else {
                        Vec::new() // OOO frame buffered, no ACK yet
                    };
                    Some(ArqEvent::ReceivedData { payloads, ack })
                } else if seq_in_range(s, self.rx_base.wrapping_sub(WINDOW_SIZE) & SEQ_MASK, WINDOW_SIZE) {
                    // In previous window — already delivered, re-ACK with cumulative seq
                    let ack_seq = (self.rx_base.wrapping_sub(1)) & SEQ_MASK;
                    let ack_samples = self.build_ack(ack_seq);
                    Some(ArqEvent::DuplicateAck(ack_samples))
                } else {
                    None
                }
            }
            FrameType::Ack => {
                let acked_seq = payload.first().copied().unwrap_or(0);
                let inflight = (self.tx_next.wrapping_sub(self.tx_base) & SEQ_MASK) as u8;
                if inflight > 0 && seq_in_range(acked_seq, self.tx_base, inflight) {
                    // Cumulative ACK: clear all slots from tx_base through acked_seq
                    let count = ((acked_seq.wrapping_sub(self.tx_base)) & SEQ_MASK) + 1;
                    let mut cleared = false;
                    for _ in 0..count {
                        if self.tx_window[(self.tx_base & 0x0F) as usize].is_some() {
                            self.tx_window[(self.tx_base & 0x0F) as usize] = None;
                            cleared = true;
                        }
                        self.tx_base = (self.tx_base + 1) & SEQ_MASK;
                    }
                    // Continue past any pre-cleared slots
                    while self.tx_base != self.tx_next
                        && self.tx_window[(self.tx_base & 0x0F) as usize].is_none()
                    {
                        self.tx_base = (self.tx_base + 1) & SEQ_MASK;
                    }
                    if cleared { return Some(ArqEvent::Acked); }
                }
                None
            }
            FrameType::Reset => {
                let next_seq = payload.first().copied().unwrap_or(0) & SEQ_MASK;
                // Drop any buffered out-of-order payloads and resync.
                for slot in self.rx_window.iter_mut() {
                    *slot = None;
                }
                self.rx_base = next_seq;
                Some(ArqEvent::Reset { next_seq })
            }
        }
    }

    /// Called when the caller's timer fires. Returns retransmit samples
    /// for the oldest unacked frame, or None if idle or max_retries exceeded.
    pub fn timeout(&mut self) -> Option<Vec<f32>> {
        if self.is_idle() {
            return None;
        }
        let inflight = (self.tx_next.wrapping_sub(self.tx_base) & SEQ_MASK) as u8;
        for i in 0..inflight {
            let seq = (self.tx_base + i) & SEQ_MASK;
            let slot = &mut self.tx_window[(seq & 0x0F) as usize];
            if let Some(ref mut tx) = slot {
                tx.retries += 1;
                if tx.retries > self.max_retries {
                    // Give up on the entire window to avoid RX deadlock; reset both ends.
                    return Some(self.reset_tx_window());
                }
                return Some(tx.frame.encode_full(
                    self.volume,
                    self.preamble_symbols,
                    self.overhead_cp,
                    self.training_symbols,
                ));
            }
        }
        None
    }

    /// Retransmit ALL unacked frames in one burst. Returns concatenated samples
    /// for every in-flight frame that hasn't exceeded max_retries.
    /// Frames exceeding max_retries are dropped.
    pub fn timeout_burst(&mut self) -> Vec<Vec<f32>> {
        if self.is_idle() {
            return Vec::new();
        }
        let mut retransmits = Vec::new();
        let inflight = (self.tx_next.wrapping_sub(self.tx_base) & SEQ_MASK) as u8;
        for i in 0..inflight {
            let seq = (self.tx_base + i) & SEQ_MASK;
            let slot = &mut self.tx_window[(seq & 0x0F) as usize];
            if let Some(ref mut tx) = slot {
                tx.retries += 1;
                if tx.retries > self.max_retries {
                    // Any drop would deadlock RX; reset instead.
                    return vec![self.reset_tx_window()];
                } else {
                    retransmits.push(tx.frame.encode_full(
                        self.volume,
                        self.preamble_symbols,
                        self.overhead_cp,
                        self.training_symbols,
                    ));
                }
            }
        }
        // Advance tx_base past any dropped frames
        while self.tx_base != self.tx_next
            && self.tx_window[(self.tx_base & 0x0F) as usize].is_none()
        {
            self.tx_base = (self.tx_base + 1) & SEQ_MASK;
        }
        retransmits
    }

    /// Window not full — ready to accept new send().
    pub fn can_send(&self) -> bool {
        ((self.tx_next.wrapping_sub(self.tx_base)) & SEQ_MASK) < WINDOW_SIZE
    }

    /// No frames in flight.
    pub fn is_idle(&self) -> bool {
        self.tx_base == self.tx_next
    }

    /// Access decoder for channel reports.
    pub fn decoder(&self) -> &Decoder {
        &self.decoder
    }

    /// Reset the decoder's buffer. Called when the audio stream is known to be
    /// discontinuous (e.g. after the half-duplex gate blocked input).
    pub fn reset_decoder(&mut self) {
        self.decoder.reset();
    }

    fn build_ack(&mut self, acked_seq: u8) -> Vec<f32> {
        let frame = PhyFrame::ack(acked_seq, self.protocol_id, acked_seq);
        frame.encode_full(
            self.volume,
            self.preamble_symbols,
            self.overhead_cp,
            self.training_symbols,
        )
    }

    fn reset_tx_window(&mut self) -> Vec<f32> {
        let next_seq = self.tx_next & SEQ_MASK;
        for slot in self.tx_window.iter_mut() {
            *slot = None;
        }
        self.tx_base = self.tx_next;
        let frame = PhyFrame::reset(next_seq, self.protocol_id, next_seq);
        frame.encode_full(
            self.volume,
            self.preamble_symbols,
            self.overhead_cp,
            self.training_symbols,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::FrameType;

    const PROTO: u8 = 0;
    const VOL: f32 = 1.0;

    // 1. send returns samples, can_send/is_idle
    #[test]
    fn test_send_returns_samples_can_send_is_idle() {
        let mut arq = SelectiveRepeat::new(PROTO, VOL);
        assert!(arq.is_idle());
        assert!(arq.can_send());
        let samples = arq.send(b"hello");
        assert!(samples.is_some());
        assert!(!samples.unwrap().is_empty());
        assert!(!arq.is_idle());
        assert!(arq.can_send()); // window size 8, only 1 used
    }

    // 2. window full (16 sends) → 17th returns None
    #[test]
    fn test_window_full() {
        let mut arq = SelectiveRepeat::new(PROTO, VOL);
        for i in 0..16 {
            let r = arq.send(format!("frame{}", i).as_bytes());
            assert!(r.is_some(), "send {} should succeed", i);
        }
        assert!(!arq.can_send());
        let r = arq.send(b"overflow");
        assert!(r.is_none());
    }

    // 3. receive data → ReceivedData with payload + ack
    #[test]
    fn test_receive_data_returns_received_data() {
        let mut sender = SelectiveRepeat::new(PROTO, VOL);
        let mut receiver = SelectiveRepeat::new(PROTO, VOL);

        let samples = sender.send(b"payload").unwrap();
        let event = receiver.receive(&samples);
        assert!(event.is_some());
        match event.unwrap() {
            ArqEvent::ReceivedData { payloads, ack } => {
                assert_eq!(payloads.len(), 1);
                assert_eq!(payloads[0], b"payload");
                assert!(!ack.is_empty());
            }
            other => panic!("expected ReceivedData, got {:?}", other),
        }
    }

    // 4. receive ack → Acked, is_idle
    #[test]
    fn test_receive_ack_returns_acked() {
        let mut arq = SelectiveRepeat::new(PROTO, VOL);
        let _samples = arq.send(b"data");
        assert!(!arq.is_idle());

        // Build an ack for seq 0
        let ack_frame = PhyFrame::ack(0, PROTO, 0);
        let ack_samples = ack_frame.encode(VOL);

        let event = arq.receive(&ack_samples);
        assert!(event.is_some());
        match event.unwrap() {
            ArqEvent::Acked => {}
            other => panic!("expected Acked, got {:?}", other),
        }
        assert!(arq.is_idle());
    }

    // 5. out-of-order delivery: send 0,1,2 → rx gets 0,2 then 1 → delivers [0], then [1,2]
    #[test]
    fn test_out_of_order_delivery() {
        let mut sender = SelectiveRepeat::new(PROTO, VOL);
        let mut receiver = SelectiveRepeat::new(PROTO, VOL);

        let s0 = sender.send(b"frame0").unwrap();
        let s1 = sender.send(b"frame1").unwrap();
        let s2 = sender.send(b"frame2").unwrap();

        // Receive frame 0 — in-order, delivers immediately
        match receiver.receive(&s0).unwrap() {
            ArqEvent::ReceivedData { payloads, .. } => {
                assert_eq!(payloads.len(), 1);
                assert_eq!(payloads[0], b"frame0");
            }
            other => panic!("expected ReceivedData, got {:?}", other),
        }

        // Receive frame 2 (skipping 1) — buffered, no in-order delivery
        match receiver.receive(&s2).unwrap() {
            ArqEvent::ReceivedData { payloads, .. } => {
                assert!(payloads.is_empty(), "OOO frame should not deliver");
            }
            other => panic!("expected ReceivedData, got {:?}", other),
        }

        // Receive frame 1 — gap fill, delivers [1, 2]
        match receiver.receive(&s1).unwrap() {
            ArqEvent::ReceivedData { payloads, .. } => {
                assert_eq!(payloads.len(), 2);
                assert_eq!(payloads[0], b"frame1");
                assert_eq!(payloads[1], b"frame2");
            }
            other => panic!("expected ReceivedData, got {:?}", other),
        }
    }

    // 6. duplicate data → DuplicateAck
    #[test]
    fn test_receive_duplicate_data() {
        let mut sender = SelectiveRepeat::new(PROTO, VOL);
        let mut receiver = SelectiveRepeat::new(PROTO, VOL);

        let samples = sender.send(b"dup test").unwrap();

        // First receive → ReceivedData
        let event1 = receiver.receive(&samples);
        assert!(matches!(event1, Some(ArqEvent::ReceivedData { .. })));

        // Second receive of same data → DuplicateAck (now in previous window)
        let event2 = receiver.receive(&samples);
        assert!(event2.is_some());
        match event2.unwrap() {
            ArqEvent::DuplicateAck(ack) => {
                assert!(!ack.is_empty());
            }
            other => panic!("expected DuplicateAck, got {:?}", other),
        }
    }

    // 7. timeout retransmits oldest unacked
    #[test]
    fn test_timeout_retransmit() {
        let mut arq = SelectiveRepeat::new(PROTO, VOL);
        let _original = arq.send(b"retry me").unwrap();
        assert!(!arq.is_idle());

        let retransmit = arq.timeout();
        assert!(retransmit.is_some());
        assert!(!retransmit.unwrap().is_empty());
        assert!(!arq.is_idle());

        // Verify retransmit decodes to the same payload
        let mut decoder = Decoder::new();
        let rt_samples = arq.timeout().unwrap();
        let (info, payload) = decoder.feed_frame(&rt_samples).unwrap();
        assert_eq!(payload, b"retry me");
        assert_eq!(info.seq_num, 0);
    }

    // 8. timeout max_retries → reset, returns RESET frame, is_idle
    #[test]
    fn test_timeout_max_retries_drops() {
        let mut arq = SelectiveRepeat::new(PROTO, VOL);
        arq.send(b"doomed");
        assert!(!arq.is_idle());

        // Exhaust retries (max_retries = 5)
        for i in 0..5 {
            let r = arq.timeout();
            assert!(r.is_some(), "retry {} should succeed", i);
        }

        // Next timeout exceeds max → RESET frame emitted
        let r = arq.timeout();
        assert!(r.is_some());
        let reset_samples = r.unwrap();
        let mut decoder = Decoder::new();
        let (info, payload) = decoder.feed_frame(&reset_samples).unwrap();
        assert_eq!(info.frame_type, FrameType::Reset);
        assert_eq!(payload.len(), 1);
        assert!(arq.is_idle());
    }

    // 9. full roundtrip: 3 frames through encode/decode, all acked
    #[test]
    fn test_full_roundtrip() {
        let mut sender = SelectiveRepeat::new(PROTO, VOL);
        let mut receiver = SelectiveRepeat::new(PROTO, VOL);

        let mut all_acks = Vec::new();

        // Send 3 frames
        for msg in &[b"msg0" as &[u8], b"msg1", b"msg2"] {
            let data_samples = sender.send(msg).unwrap();
            let event = receiver.receive(&data_samples).unwrap();
            match event {
                ArqEvent::ReceivedData { payloads, ack } => {
                    assert_eq!(payloads.len(), 1);
                    all_acks.push(ack);
                }
                other => panic!("expected ReceivedData, got {:?}", other),
            }
        }
        assert!(!sender.is_idle());

        // Feed all acks back to sender
        for ack_samples in &all_acks {
            let event = sender.receive(ack_samples).unwrap();
            assert!(matches!(event, ArqEvent::Acked));
        }
        assert!(sender.is_idle());
    }

    // 10. out-of-order roundtrip via PHY: encode/decode with reordered delivery
    #[test]
    fn test_out_of_order_roundtrip_via_phy() {
        let mut sender = SelectiveRepeat::new(PROTO, VOL);
        let mut receiver = SelectiveRepeat::new(PROTO, VOL);

        let s0 = sender.send(b"aaa").unwrap();
        let s1 = sender.send(b"bbb").unwrap();
        let s2 = sender.send(b"ccc").unwrap();

        // Deliver out of order: 2, 0, 1
        let mut all_payloads: Vec<Vec<u8>> = Vec::new();
        let mut all_acks: Vec<Vec<f32>> = Vec::new();

        // Frame 2 first — OOO, buffered
        match receiver.receive(&s2).unwrap() {
            ArqEvent::ReceivedData { payloads, ack } => {
                assert!(payloads.is_empty());
                all_acks.push(ack);
            }
            other => panic!("expected ReceivedData, got {:?}", other),
        }

        // Frame 0 — in-order, delivers [aaa]
        match receiver.receive(&s0).unwrap() {
            ArqEvent::ReceivedData { payloads, ack } => {
                all_payloads.extend(payloads);
                all_acks.push(ack);
            }
            other => panic!("expected ReceivedData, got {:?}", other),
        }

        // Frame 1 — gap fill, delivers [bbb, ccc]
        match receiver.receive(&s1).unwrap() {
            ArqEvent::ReceivedData { payloads, ack } => {
                all_payloads.extend(payloads);
                all_acks.push(ack);
            }
            other => panic!("expected ReceivedData, got {:?}", other),
        }

        assert_eq!(all_payloads.len(), 3);
        assert_eq!(all_payloads[0], b"aaa");
        assert_eq!(all_payloads[1], b"bbb");
        assert_eq!(all_payloads[2], b"ccc");

        // Feed acks back — sender should become idle
        for ack in &all_acks {
            let _ = sender.receive(ack);
        }
        assert!(sender.is_idle());
    }

    // 11. reset frame clears RX window and advances base
    #[test]
    fn test_reset_resyncs_receiver() {
        let mut sender = SelectiveRepeat::new(PROTO, VOL);
        let mut receiver = SelectiveRepeat::new(PROTO, VOL);

        // Send seq 0 and 1, deliver only seq 1 (OOO) so it gets buffered.
        let _s0 = sender.send(b"frame0").unwrap();
        let s1 = sender.send(b"frame1").unwrap();
        let event = receiver.receive(&s1).unwrap();
        assert!(matches!(event, ArqEvent::ReceivedData { payloads, .. } if payloads.is_empty()));

        // Now send a reset to seq 5.
        let reset = PhyFrame::reset(5, PROTO, 5).encode(VOL);
        let event = receiver.receive(&reset).unwrap();
        match event {
            ArqEvent::Reset { next_seq } => assert_eq!(next_seq, 5),
            other => panic!("expected Reset, got {:?}", other),
        }

        // Receiver should accept seq 5 immediately (in-order).
        let s5 = PhyFrame::data(5, PROTO, b"fresh".to_vec()).encode(VOL);
        match receiver.receive(&s5).unwrap() {
            ArqEvent::ReceivedData { payloads, .. } => {
                assert_eq!(payloads.len(), 1);
                assert_eq!(payloads[0], b"fresh");
            }
            other => panic!("expected ReceivedData, got {:?}", other),
        }
    }
}
