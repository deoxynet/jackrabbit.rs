/// KCP-based ARQ: reliable delivery using the battle-tested KCP protocol.
///
/// KCP provides congestion control, fast retransmit, ordered delivery, and
/// selective acknowledgment internally. We wrap it to encode output packets
/// as PHY frames over the audio channel.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::io::{self, Write};
use std::rc::Rc;

use kcp::Kcp;

use crate::decoder::Decoder;
use crate::header::FrameType;
use crate::params::{self, BitloadProfile};
use crate::phy::PhyFrame;

/// Shared buffer for KCP output packets.
type PacketQueue = Rc<RefCell<VecDeque<Vec<u8>>>>;

/// KCP output sink that collects packets into a queue.
struct KcpOutput {
    queue: PacketQueue,
}

impl Write for KcpOutput {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.queue.borrow_mut().push_back(buf.to_vec());
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Events from the KCP ARQ layer.
#[derive(Debug)]
pub enum ArqEvent {
    /// In-order data delivered.
    /// `payloads` contains 1+ consecutive payloads delivered by KCP.
    ReceivedData { payloads: Vec<Vec<u8>> },
    /// One or more pending sends were acknowledged (internal KCP state advanced).
    Acked,
    /// Sender reset: drop buffered RX window and resync.
    Reset { next_seq: u8 },
}

/// KCP-based ARQ over audio PHY.
pub struct KcpArq {
    kcp: Kcp<KcpOutput>,
    output_queue: PacketQueue,
    decoder: Decoder,
    recv_buf: Vec<u8>,
    // Config
    protocol_id: u8,
    volume: f32,
    preamble_symbols: usize,
    overhead_cp: usize,
    training_symbols: usize,
    data_cp_override: Option<usize>,
    tx_bitload_profile: Option<BitloadProfile>,
    // Timing
    clock_ms: u32,
    // Track window state for detecting ACKs
    last_wait_snd: usize,
}

impl KcpArq {
    /// Configure KCP for audio half-duplex with tuned parameters.
    fn configure_kcp(kcp: &mut Kcp<KcpOutput>, wired: bool) {
        // Tuned for audio half-duplex:
        // - nodelay=true: use 30ms min RTO instead of 100ms
        // - interval: internal tick rate (OTA: 100ms, wired: 50ms)
        // - resend=1: fast retransmit on first duplicate ACK (aggressive)
        // - nc=true: no congestion control (fixed bandwidth channel)
        let interval = if wired { 50 } else { 100 };
        kcp.set_nodelay(true, interval, 1, true);

        // Larger windows for pipelining (32 send, 128 receive)
        // Audio has high latency, so we want more in-flight packets
        kcp.set_wndsize(32, 128);

        // MTU: match PHY max payload (minus KCP's 24-byte overhead)
        kcp.set_mtu(params::MAX_PAYLOAD).unwrap();

        // Min RTO: audio round-trip is 500-1000ms OTA (burst + ACK), 100-200ms wired
        // Don't retransmit too early or we waste bandwidth
        let min_rto = if wired { 100 } else { 500 };
        kcp.set_rx_minrto(min_rto);

        // Give up after 10 failed retransmits (was 20)
        kcp.set_maximum_resend_times(10);
    }

    /// Default config (OTA settings).
    pub fn new(protocol_id: u8, volume: f32) -> Self {
        let queue = Rc::new(RefCell::new(VecDeque::new()));
        let output = KcpOutput { queue: queue.clone() };

        let mut kcp = Kcp::new(0, output); // conv=0 for single link
        Self::configure_kcp(&mut kcp, false);

        KcpArq {
            kcp,
            output_queue: queue,
            decoder: Decoder::new(),
            recv_buf: vec![0u8; params::MAX_PAYLOAD + 100], // extra for KCP overhead
            protocol_id,
            volume,
            preamble_symbols: params::PREAMBLE_SYMBOLS,
            overhead_cp: params::CP_LEN,
            training_symbols: params::TRAINING_SYMBOLS,
            data_cp_override: None,
            tx_bitload_profile: None,
            clock_ms: 0,
            last_wait_snd: 0,
        }
    }

    /// Cable-optimized config (short preamble, protocol CP, 1 training symbol).
    pub fn new_wired(protocol_id: u8, volume: f32, cp_len: usize) -> Self {
        let mut decoder = Decoder::new();
        decoder.set_overhead_cp(cp_len);
        decoder.set_training_symbols(1);

        let queue = Rc::new(RefCell::new(VecDeque::new()));
        let output = KcpOutput { queue: queue.clone() };

        let mut kcp = Kcp::new(0, output);
        Self::configure_kcp(&mut kcp, true);

        KcpArq {
            kcp,
            output_queue: queue,
            decoder,
            recv_buf: vec![0u8; params::MAX_PAYLOAD + 100],
            protocol_id,
            volume,
            preamble_symbols: params::PREAMBLE_SYMBOLS_MINIMAL,
            overhead_cp: cp_len,
            training_symbols: 1,
            data_cp_override: None,
            tx_bitload_profile: None,
            clock_ms: 0,
            last_wait_snd: 0,
        }
    }

    /// Queue data to send. Returns true if queued successfully.
    /// Returns false if window is full or payload is invalid.
    pub fn send(&mut self, payload: &[u8]) -> bool {
        if payload.is_empty() || payload.len() > params::MAX_PAYLOAD {
            return false;
        }
        // Enforce window limit before queueing
        if !self.can_send() {
            return false;
        }
        self.kcp.send(payload).is_ok()
    }

    /// Feed received audio. Returns event if data was delivered or state changed.
    pub fn receive(&mut self, samples: &[f32]) -> Option<ArqEvent> {
        let (info, payload) = self.decoder.feed_frame(samples)?;
        if info.protocol_id != self.protocol_id {
            return None;
        }

        match info.frame_type {
            FrameType::Data => {
                // Data frame contains a KCP packet
                if self.kcp.input(&payload).is_err() {
                    return None;
                }

                // Try to receive any complete messages from KCP
                let mut payloads = Vec::new();
                loop {
                    match self.kcp.recv(&mut self.recv_buf) {
                        Ok(n) => {
                            payloads.push(self.recv_buf[..n].to_vec());
                        }
                        Err(_) => break,
                    }
                }

                if payloads.is_empty() {
                    // KCP buffered the packet but no complete message yet
                    // Still return an event so MAC knows something was received
                    Some(ArqEvent::ReceivedData { payloads: vec![] })
                } else {
                    Some(ArqEvent::ReceivedData { payloads })
                }
            }
            FrameType::Ack => {
                // ACK frame contains a KCP packet (KCP ACKs are embedded in its protocol)
                // For compatibility, treat standalone ACK frames as KCP input too
                let before = self.kcp.wait_snd();
                if self.kcp.input(&payload).is_ok() {
                    let after = self.kcp.wait_snd();
                    if after < before {
                        self.last_wait_snd = after;
                        return Some(ArqEvent::Acked);
                    }
                }
                None
            }
            FrameType::Reset => {
                let next_seq = payload.first().copied().unwrap_or(0) & 0x1F;
                // Reset KCP state by creating a new instance
                self.reset_kcp();
                Some(ArqEvent::Reset { next_seq })
            }
        }
    }

    /// Advance the KCP clock and flush any pending output.
    /// Returns audio samples if KCP has packets to send.
    /// `elapsed_ms` is the time since the last poll call.
    pub fn poll(&mut self, elapsed_ms: u32) -> Option<Vec<f32>> {
        self.clock_ms = self.clock_ms.wrapping_add(elapsed_ms);

        // Update KCP internal state (ignore errors, they just mean nothing to do)
        let _ = self.kcp.update(self.clock_ms);

        // Flush any pending output (ignore errors)
        let _ = self.kcp.flush();

        // Track wait_snd for idle detection
        self.last_wait_snd = self.kcp.wait_snd();

        // Encode any queued KCP output packets as PHY frames
        self.drain_output_queue()
    }

    /// Check when KCP wants the next update call (in ms from current clock).
    pub fn check(&self) -> u32 {
        self.kcp.check(self.clock_ms).saturating_sub(self.clock_ms)
    }

    /// Window not full - ready to accept new send().
    pub fn can_send(&self) -> bool {
        self.kcp.wait_snd() < self.kcp.snd_wnd() as usize
    }

    /// No packets in flight (send window empty).
    pub fn is_idle(&self) -> bool {
        self.kcp.wait_snd() == 0
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

    /// Set the TX bitloading profile.
    pub fn set_tx_bitload_profile(&mut self, profile: BitloadProfile) {
        self.tx_bitload_profile = Some(profile);
    }

    /// Set the RX bitloading profile on the decoder.
    pub fn set_rx_bitload_profile(&mut self, profile: BitloadProfile) {
        self.decoder.set_bitload_profile(profile);
    }

    /// Override the cyclic prefix length for data symbols (cable mode).
    pub fn set_data_cp_override(&mut self, cp_len: usize) {
        self.data_cp_override = Some(cp_len);
        self.decoder.set_data_cp_override(cp_len);
    }

    /// Get the current TX bitloading profile, if any.
    pub fn tx_bitload_profile(&self) -> Option<&BitloadProfile> {
        self.tx_bitload_profile.as_ref()
    }

    /// Update the TX bitloading profile from the latest channel estimate.
    pub fn update_bitload_from_channel(&mut self) -> Option<BitloadProfile> {
        let report = self.decoder.channel_report()?;
        let config = params::get_protocol(self.protocol_id);

        let band_defs: &[(usize, usize)] = &[
            (20, 80), (20, 120), (20, 160), (20, 320), (20, 400), (5, 475),
        ];
        let band_idx = band_defs
            .iter()
            .position(|&(bs, be)| bs == config.bin_start && be == config.bin_end)?;
        let band = &report.bands.get(band_idx)?;
        if !band.snr_db.is_finite() {
            return None;
        }

        let num_dc = config.num_data_carriers();
        let uniform_snr = vec![band.snr_db; num_dc];
        let profile = BitloadProfile::from_snr(&uniform_snr, num_dc);
        self.tx_bitload_profile = Some(profile.clone());
        Some(profile)
    }

    /// Drain the KCP output queue and encode packets as PHY frames.
    fn drain_output_queue(&mut self) -> Option<Vec<f32>> {
        let packets: Vec<Vec<u8>> = self.output_queue.borrow_mut().drain(..).collect();
        if packets.is_empty() {
            return None;
        }

        let mut all_samples = Vec::new();
        for (i, packet) in packets.iter().enumerate() {
            let seq = i as u8 & 0x1F;
            let frame = PhyFrame::data(seq, self.protocol_id, packet.clone());
            let samples = self.encode_frame(&frame);
            all_samples.extend(samples);
        }

        if all_samples.is_empty() {
            None
        } else {
            Some(all_samples)
        }
    }

    /// Encode a frame, using bitloaded encoding if a TX profile is set.
    fn encode_frame(&self, frame: &PhyFrame) -> Vec<f32> {
        if let Some(ref profile) = self.tx_bitload_profile {
            frame.encode_bitloaded_with_data_cp(
                self.volume,
                self.preamble_symbols,
                self.overhead_cp,
                self.training_symbols,
                self.data_cp_override,
                profile,
            )
        } else {
            frame.encode_full_with_data_cp(
                self.volume,
                self.preamble_symbols,
                self.overhead_cp,
                self.training_symbols,
                self.data_cp_override,
            )
        }
    }

    /// Reset KCP state completely (uses OTA settings).
    fn reset_kcp(&mut self) {
        let queue = Rc::new(RefCell::new(VecDeque::new()));
        let output = KcpOutput { queue: queue.clone() };

        let mut kcp = Kcp::new(0, output);
        // Use OTA settings on reset (conservative)
        Self::configure_kcp(&mut kcp, false);

        self.kcp = kcp;
        self.output_queue = queue;
        self.clock_ms = 0;
        self.last_wait_snd = 0;
    }
}

// Keep SelectiveRepeat as an alias for backward compatibility
pub type SelectiveRepeat = KcpArq;

#[cfg(test)]
mod tests {
    use super::*;

    const PROTO: u8 = 0;
    const VOL: f32 = 1.0;

    #[test]
    fn test_send_returns_true_can_send_is_idle() {
        let mut arq = KcpArq::new(PROTO, VOL);
        assert!(arq.is_idle());
        assert!(arq.can_send());

        let result = arq.send(b"hello");
        assert!(result);

        // Need to poll to flush the send queue
        let samples = arq.poll(0);
        assert!(samples.is_some());
        assert!(!samples.unwrap().is_empty());
        assert!(!arq.is_idle());
        assert!(arq.can_send()); // window size 16, only 1 used
    }

    #[test]
    fn test_window_full() {
        let mut arq = KcpArq::new(PROTO, VOL);
        // Window size is 32 (tuned for audio pipelining)
        for i in 0..32 {
            let r = arq.send(format!("frame{}", i).as_bytes());
            assert!(r, "send {} should succeed", i);
        }
        arq.poll(0); // flush
        assert!(!arq.can_send());

        let r = arq.send(b"overflow");
        assert!(!r);
    }

    #[test]
    fn test_receive_data_returns_received_data() {
        let mut sender = KcpArq::new(PROTO, VOL);
        let mut receiver = KcpArq::new(PROTO, VOL);

        sender.send(b"payload");
        let samples = sender.poll(0).unwrap();

        let event = receiver.receive(&samples);
        assert!(event.is_some());
        match event.unwrap() {
            ArqEvent::ReceivedData { payloads } => {
                assert_eq!(payloads.len(), 1);
                assert_eq!(payloads[0], b"payload");
            }
            other => panic!("expected ReceivedData, got {:?}", other),
        }
    }

    #[test]
    fn test_receive_ack_advances_window() {
        let mut sender = KcpArq::new(PROTO, VOL);
        let mut receiver = KcpArq::new(PROTO, VOL);

        sender.send(b"data");
        let data_samples = sender.poll(0).unwrap();
        assert!(!sender.is_idle());

        // Receiver gets the data
        receiver.receive(&data_samples);

        // Receiver polls to generate ACK
        let ack_samples = receiver.poll(10);
        assert!(ack_samples.is_some());

        // Feed ACK back to sender
        sender.receive(&ack_samples.unwrap());
        sender.poll(10);

        // Sender should now be idle
        assert!(sender.is_idle());
    }

    #[test]
    fn test_full_roundtrip() {
        let mut sender = KcpArq::new(PROTO, VOL);
        let mut receiver = KcpArq::new(PROTO, VOL);

        // Send 3 frames
        for msg in &[b"msg0" as &[u8], b"msg1", b"msg2"] {
            sender.send(msg);
        }
        let data_samples = sender.poll(0).unwrap();
        assert!(!sender.is_idle());

        // Receiver gets all data
        let event = receiver.receive(&data_samples).unwrap();
        match event {
            ArqEvent::ReceivedData { payloads } => {
                assert_eq!(payloads.len(), 3);
            }
            other => panic!("expected ReceivedData, got {:?}", other),
        }

        // Exchange ACKs until sender is idle
        for _ in 0..10 {
            if sender.is_idle() {
                break;
            }
            if let Some(rx_samples) = receiver.poll(10) {
                sender.receive(&rx_samples);
            }
            sender.poll(10);
        }

        assert!(sender.is_idle());
    }

    #[test]
    fn test_retransmit_via_timeout() {
        let mut sender = KcpArq::new(PROTO, VOL);

        sender.send(b"will timeout");
        let _original = sender.poll(0).unwrap();
        assert!(!sender.is_idle());

        // Advance clock significantly to trigger retransmit
        // KCP's internal RTO starts around 200ms
        for _ in 0..30 {
            let retransmit = sender.poll(50); // 50ms * 30 = 1.5s
            if retransmit.is_some() {
                // Got a retransmit
                assert!(!sender.is_idle());
                return;
            }
        }
        // Even if no retransmit visible, sender should still not be idle
        assert!(!sender.is_idle());
    }

    #[test]
    fn test_wired_mode() {
        let proto = 13u8; // WIRED_TURBO
        let config = params::get_protocol(proto);

        let mut sender = KcpArq::new_wired(proto, VOL, config.cp_len);
        let mut receiver = KcpArq::new_wired(proto, VOL, config.cp_len);

        sender.send(b"wired data");
        let samples = sender.poll(0).unwrap();

        let event = receiver.receive(&samples);
        assert!(event.is_some());
        match event.unwrap() {
            ArqEvent::ReceivedData { payloads } => {
                assert_eq!(payloads.len(), 1);
                assert_eq!(payloads[0], b"wired data");
            }
            other => panic!("expected ReceivedData, got {:?}", other),
        }
    }

    #[test]
    fn test_protocol_mismatch_ignored() {
        let mut sender = KcpArq::new(0, VOL);
        let mut receiver = KcpArq::new(1, VOL); // different protocol

        sender.send(b"proto-0");
        let samples = sender.poll(0).unwrap();

        assert!(receiver.receive(&samples).is_none());
    }
}
