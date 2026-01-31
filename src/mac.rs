/// Half-Duplex MAC layer: protocol engine between ARQ and audio/TAP I/O.
///
/// No I/O ownership — caller pushes/pulls samples and payloads.
/// Implicit state machine:
///   - Transmitting: `!output_buf.is_empty()`
///   - Waiting for ACK: `output_buf.is_empty() && !arq.is_idle()`
///   - Ready to send: `output_buf.is_empty() && arq.can_send() && !tx_queue.is_empty()`
///   - Idle: `output_buf.is_empty() && arq.is_idle() && tx_queue.is_empty()`

use std::collections::VecDeque;
use crate::arq::{ArqEvent, SelectiveRepeat};
use crate::params;

pub struct HalfDuplexMac {
    arq: SelectiveRepeat,
    tx_queue: VecDeque<Vec<u8>>,
    output_buf: VecDeque<f32>,
    rx_queue: VecDeque<Vec<u8>>,
    clock: u64,
    last_tx_end: u64,
    timeout_samples: u64,
    gap_samples: usize,
    burst_limit: usize,       // max frames sent per burst
    burst_sent: usize,        // frames sent in current burst
    was_transmitting: bool,   // decoder reset on TX→RX transition
    stop_and_wait: bool,      // only send new data when no inflight frames
    // Deferred ACK: buffer ACKs during a received burst, flush after silence
    pending_acks: Vec<Vec<f32>>,
    quiet_samples: usize,        // consecutive low-energy samples (actual silence)
    ack_quiet_threshold: usize,  // quiet samples needed before flushing ACKs
}

impl HalfDuplexMac {
    /// OTA configuration.
    pub fn new(protocol_id: u8, volume: f32) -> Self {
        HalfDuplexMac {
            arq: SelectiveRepeat::new(protocol_id, volume),
            tx_queue: VecDeque::new(),
            output_buf: VecDeque::new(),
            rx_queue: VecDeque::new(),
            clock: 0,
            last_tx_end: 0,
            timeout_samples: 48_000, // 1s at 48kHz
            gap_samples: params::SYMBOL_TOTAL * 2, // 2,400 = 50ms inter-frame silence
            burst_limit: 1, // robust half-duplex: one frame per burst
            burst_sent: 0,
            was_transmitting: false,
            stop_and_wait: true,
            pending_acks: Vec::new(),
            quiet_samples: 0,
            ack_quiet_threshold: params::SYMBOL_TOTAL * 4, // 4800 = 100ms of quiet
        }
    }

    /// Cable-optimized configuration (short preamble, protocol CP, 1 training symbol).
    pub fn new_wired(protocol_id: u8, volume: f32, cp_len: usize) -> Self {
        let sym = params::FFT_SIZE + cp_len;
        HalfDuplexMac {
            arq: SelectiveRepeat::new_wired(protocol_id, volume, cp_len),
            tx_queue: VecDeque::new(),
            output_buf: VecDeque::new(),
            rx_queue: VecDeque::new(),
            clock: 0,
            last_tx_end: 0,
            timeout_samples: 24_000, // 500ms at 48kHz
            gap_samples: sym * 2, // 2 symbols inter-frame silence
            burst_limit: 1, // robust half-duplex: one frame per burst
            burst_sent: 0,
            was_transmitting: false,
            stop_and_wait: true,
            pending_acks: Vec::new(),
            quiet_samples: 0,
            ack_quiet_threshold: sym * 4, // ~4 symbols of quiet
        }
    }

    /// Queue a payload for transmission.
    pub fn enqueue(&mut self, payload: Vec<u8>) {
        self.tx_queue.push_back(payload);
    }

    /// Feed received audio samples. Half-duplex gate: ignored while transmitting.
    pub fn feed(&mut self, samples: &[f32]) {
        if self.is_transmitting() {
            return;
        }
        // If we just stopped transmitting, reset decoder to discard
        // stale samples from before the half-duplex gate blocked input.
        // The audio stream is discontinuous across TX→RX transitions.
        if self.was_transmitting {
            self.was_transmitting = false;
            self.arq.reset_decoder();
        }
        // Track energy-based silence: only count truly quiet samples.
        // This avoids triggering ACK flush during another frame's preamble/data.
        let energy: f32 = samples.iter().map(|s| s * s).sum::<f32>();
        if energy < 1e-6 * samples.len() as f32 {
            self.quiet_samples += samples.len();
        } else {
            self.quiet_samples = 0;
        }

        if let Some(event) = self.arq.receive(samples) {
            match event {
                ArqEvent::ReceivedData { payloads, ack } => {
                    for p in payloads {
                        self.rx_queue.push_back(p);
                    }
                    if !ack.is_empty() {
                        // Always defer ACKs until silence is detected.
                        // Sending immediately risks arriving while the
                        // peer's output ring still has samples (echo),
                        // causing the peer's decoder to false-lock on its
                        // own echo and miss our ACK entirely.
                        self.pending_acks.push(ack);
                    }
                }
                ArqEvent::Acked => {}
                ArqEvent::DuplicateAck(ack) => {
                    if !ack.is_empty() {
                        self.pending_acks.push(ack);
                    }
                }
                ArqEvent::Reset { .. } => {
                    // Drop any queued ACKs for the old window.
                    self.pending_acks.clear();
                }
            }
        }
    }

    /// Advance the clock and perform timeout/burst-send logic.
    pub fn poll(&mut self, elapsed_samples: usize) {
        self.clock += elapsed_samples as u64;
        if self.is_transmitting() {
            return;
        }
        // Flush deferred ACKs after sufficient quiet (sender's burst ended).
        // Uses energy-based silence so we don't flush mid-burst during
        // preamble/training/data processing of the next frame.
        if !self.pending_acks.is_empty()
            && self.quiet_samples >= self.ack_quiet_threshold
        {
            let ack = self.pending_acks.pop().unwrap();
            self.pending_acks.clear();
            self.append_with_gap(&ack);
            return; // send ACK before any new data
        }
        // Timeout: retransmit oldest unacked frame (single-frame to allow
        // ACK flow in half-duplex — large bursts block ACK reception)
        if !self.arq.is_idle()
            && self.clock.saturating_sub(self.last_tx_end) > self.timeout_samples
        {
            if let Some(retransmit) = self.arq.timeout() {
                self.append_with_gap(&retransmit);
            }
            self.last_tx_end = self.clock;
        }
        // Burst-send queued payloads (respecting burst_limit).
        // Only send when no pending ACKs — if we have ACKs buffered,
        // we're still receiving a burst and must not interrupt it by
        // transmitting (half-duplex gate would block further reception).
        if self.pending_acks.is_empty() {
            if self.stop_and_wait && !self.arq.is_idle() {
                return;
            }
            self.try_send_burst();
        }
    }

    /// Drain samples into the output buffer. Fills remainder with 0.0.
    /// Returns the number of samples drained from the output buffer.
    pub fn fill_output(&mut self, out: &mut [f32]) -> usize {
        let n = out.len().min(self.output_buf.len());
        for sample in out[..n].iter_mut() {
            *sample = self.output_buf.pop_front().unwrap();
        }
        for sample in out[n..].iter_mut() {
            *sample = 0.0;
        }
        if n > 0 && self.output_buf.is_empty() {
            self.last_tx_end = self.clock;
            self.was_transmitting = true;
            self.burst_sent = 0; // burst fully transmitted; allow next burst
        }
        n
    }

    /// Dequeue the next received payload, if any.
    pub fn recv(&mut self) -> Option<Vec<u8>> {
        self.rx_queue.pop_front()
    }

    /// True if the output buffer has samples (including inter-frame gaps).
    pub fn is_transmitting(&self) -> bool {
        !self.output_buf.is_empty()
    }

    /// True if no pending TX, no in-flight ARQ frames, and no queued payloads.
    pub fn is_idle(&self) -> bool {
        self.output_buf.is_empty() && self.arq.is_idle() && self.tx_queue.is_empty()
    }

    /// Number of payloads waiting in the TX queue (not yet handed to ARQ).
    pub fn tx_pending(&self) -> usize {
        self.tx_queue.len()
    }

    /// Number of audio samples buffered for output.
    pub fn output_available(&self) -> usize {
        self.output_buf.len()
    }

    /// Set max frames per TX burst. Higher = more throughput on clean links.
    pub fn set_burst_limit(&mut self, limit: usize) {
        self.burst_limit = limit.max(1);
    }

    /// Enable/disable stop-and-wait. When true, no new data is sent while
    /// any frame is in-flight. When false, the ARQ window fills freely.
    pub fn set_stop_and_wait(&mut self, enabled: bool) {
        self.stop_and_wait = enabled;
    }

    /// Set the retransmit timeout in samples.
    pub fn set_timeout(&mut self, samples: u64) {
        self.timeout_samples = samples;
    }

    /// Set the inter-frame gap in samples.
    pub fn set_gap(&mut self, samples: usize) {
        self.gap_samples = samples;
    }

    /// Set silence duration (in samples) before deferred ACKs are flushed.
    pub fn set_ack_quiet_threshold(&mut self, samples: usize) {
        self.ack_quiet_threshold = samples;
    }

    /// Inform the MAC about the total queued output backlog (in samples).
    /// This is used to align timeouts with actual playback completion.
    pub fn note_output_backlog(&mut self, backlog_samples: usize) {
        self.last_tx_end = self
            .clock
            .saturating_add(backlog_samples as u64);
    }

    /// Access the underlying ARQ state.
    pub fn arq(&self) -> &SelectiveRepeat {
        &self.arq
    }

    fn append_with_gap(&mut self, samples: &[f32]) {
        self.output_buf.extend(samples.iter());
        for _ in 0..self.gap_samples {
            self.output_buf.push_back(0.0);
        }
    }

    fn try_send_burst(&mut self) {
        while self.burst_sent < self.burst_limit
            && self.arq.can_send()
            && !self.tx_queue.is_empty()
        {
            let payload = self.tx_queue.pop_front().unwrap();
            if let Some(samples) = self.arq.send(&payload) {
                self.append_with_gap(&samples);
                self.burst_sent += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phy::PhyFrame;

    const PROTO: u8 = 0;
    const VOL: f32 = 1.0;

    // 1. enqueue + poll → has output
    #[test]
    fn test_enqueue_poll_has_output() {
        let mut mac = HalfDuplexMac::new(PROTO, VOL);
        assert!(mac.is_idle());
        mac.enqueue(b"hello".to_vec());
        assert!(!mac.is_idle()); // queued payload
        mac.poll(0);
        assert!(mac.is_transmitting());
        let mut buf = vec![0.0f32; 1024];
        let n = mac.fill_output(&mut buf);
        assert!(n > 0);
    }

    // 2. feed data → recv payload, ACK always deferred until silence
    #[test]
    fn test_feed_data_recv_payload() {
        let mut sender_arq = SelectiveRepeat::new(PROTO, VOL);
        let data_samples = sender_arq.send(b"test payload").unwrap();

        let mut mac = HalfDuplexMac::new(PROTO, VOL);
        mac.feed(&data_samples);
        let payload = mac.recv();
        assert!(payload.is_some());
        assert_eq!(payload.unwrap(), b"test payload");
        // ACK is deferred until silence detected (prevents echo-gate deadlock)
        assert!(!mac.is_transmitting(), "ACK should be deferred");
        assert!(!mac.pending_acks.is_empty(), "pending ACK should exist");

        // Feed enough silence to trigger ack quiet threshold, then poll to flush
        let silence = vec![0.0f32; mac.ack_quiet_threshold + 1];
        mac.feed(&silence);
        mac.poll(0);
        assert!(mac.is_transmitting(), "ACK should now be in output_buf");
    }

    // 2b. deferred ACK works the same regardless of stop-and-wait setting
    #[test]
    fn test_feed_data_recv_payload_pipelined() {
        let mut sender_arq = SelectiveRepeat::new(PROTO, VOL);
        let data_samples = sender_arq.send(b"test payload").unwrap();

        let mut mac = HalfDuplexMac::new(PROTO, VOL);
        mac.stop_and_wait = false;
        mac.feed(&data_samples);
        let payload = mac.recv();
        assert!(payload.is_some());
        assert_eq!(payload.unwrap(), b"test payload");
        assert!(!mac.is_transmitting(), "ACK should be deferred");
        assert!(!mac.pending_acks.is_empty(), "pending ACK should exist");

        // Feed enough silence to trigger ack quiet threshold, then poll to flush
        let silence = vec![0.0f32; mac.ack_quiet_threshold + 1];
        mac.feed(&silence);
        mac.poll(0);
        assert!(mac.is_transmitting(), "ACK should now be in output_buf");
    }

    // 3. half-duplex gate: while output_buf has data, feed() is a no-op
    #[test]
    fn test_half_duplex_gate() {
        let mut mac = HalfDuplexMac::new(PROTO, VOL);
        mac.enqueue(b"tx data".to_vec());
        mac.poll(0);
        assert!(mac.is_transmitting());

        // Feed data while transmitting — should be ignored
        let mut sender_arq = SelectiveRepeat::new(PROTO, VOL);
        let data_samples = sender_arq.send(b"incoming").unwrap();
        mac.feed(&data_samples);
        assert!(mac.recv().is_none(), "half-duplex gate should block feed");
    }

    // 4. ACK priority: pending ACKs block data send, ACK flushes first
    #[test]
    fn test_ack_priority() {
        let mut mac = HalfDuplexMac::new(PROTO, VOL);
        mac.enqueue(b"queued data".to_vec());

        // Feed incoming data — ACK is deferred (pending_acks)
        let mut sender_arq = SelectiveRepeat::new(PROTO, VOL);
        let data_samples = sender_arq.send(b"incoming data").unwrap();
        mac.feed(&data_samples);
        assert!(!mac.is_transmitting(), "ACK should be deferred");

        // poll without sufficient silence → data NOT sent (pending ACKs block burst)
        mac.poll(0);
        assert!(!mac.is_transmitting(), "should not send data while pending ACKs exist");
        assert_eq!(mac.tx_pending(), 1, "queued data should still be pending");

        // Feed silence to meet quiet threshold, then poll → ACK flushes first
        let silence = vec![0.0f32; mac.ack_quiet_threshold + 1];
        mac.feed(&silence);
        mac.poll(0);
        assert!(mac.is_transmitting(), "deferred ACK should now be in output_buf");
        assert_eq!(mac.tx_pending(), 1, "queued data still waiting behind ACK");

        // Drain ACK output, then poll → queued data sent
        let total = mac.output_buf.len();
        let mut buf = vec![0.0f32; total];
        mac.fill_output(&mut buf);
        mac.poll(0);
        assert!(mac.is_transmitting(), "queued data should now be sent");
        assert_eq!(mac.tx_pending(), 0);
    }

    // 5. burst fill: enqueue 4 payloads, poll once, verify only one sent
    #[test]
    fn test_burst_fill() {
        let mut mac = HalfDuplexMac::new(PROTO, VOL);
        for i in 0..4 {
            mac.enqueue(format!("burst-{}", i).into_bytes());
        }
        mac.poll(0);
        assert!(mac.is_transmitting());
        assert_eq!(mac.tx_pending(), 3, "one payload should be sent per burst");
        assert!(!mac.arq().is_idle(), "ARQ should have in-flight frames");
    }

    // 6. timeout retransmit
    #[test]
    fn test_timeout_retransmit() {
        let mut mac = HalfDuplexMac::new(PROTO, VOL);
        mac.enqueue(b"timeout test".to_vec());
        mac.poll(0);
        assert!(mac.is_transmitting());

        // Drain all output (simulates audio playback)
        let total = mac.output_buf.len();
        let mut buf = vec![0.0f32; total];
        mac.fill_output(&mut buf);
        assert!(!mac.is_transmitting());
        assert!(!mac.arq().is_idle(), "frame should be in-flight");

        // Advance clock past timeout
        mac.poll(50_000);
        assert!(
            mac.is_transmitting(),
            "timeout should have triggered retransmit"
        );
    }

    // 8. ACK reception should not reset timeout for other in-flight frames
    #[test]
    fn test_ack_does_not_reset_timeout() {
        let mut mac = HalfDuplexMac::new(PROTO, VOL);
        mac.enqueue(b"frame0".to_vec());
        mac.enqueue(b"frame1".to_vec());
        mac.poll(0);

        // Drain all output (simulates audio playback)
        let total = mac.output_buf.len();
        let mut buf = vec![0.0f32; total];
        mac.fill_output(&mut buf);
        assert!(!mac.arq().is_idle(), "frames should be in-flight");

        // Advance clock but stay below timeout
        mac.poll(40_000);
        assert!(!mac.is_transmitting());

        // Deliver ACK for seq 0 (leaves seq 1 in-flight)
        let ack_samples = PhyFrame::ack(0, PROTO, 0).encode(VOL);
        mac.feed(&ack_samples);
        assert!(!mac.is_transmitting(), "ACK should not enqueue TX");

        // Advance clock past timeout since last TX end
        mac.poll(20_000);
        assert!(
            mac.is_transmitting(),
            "timeout should trigger retransmit despite ACK reception"
        );
    }

    // 7. state queries: fresh idle, after enqueue+poll transmitting, after round-trip idle
    #[test]
    fn test_state_queries() {
        let mut mac_a = HalfDuplexMac::new(PROTO, VOL);
        let mut mac_b = HalfDuplexMac::new(PROTO, VOL);

        // Fresh → idle
        assert!(mac_a.is_idle());
        assert!(!mac_a.is_transmitting());

        // Enqueue + poll → transmitting
        mac_a.enqueue(b"state test".to_vec());
        mac_a.poll(0);
        assert!(mac_a.is_transmitting());
        assert!(!mac_a.is_idle());

        // Full round-trip using step loop (handles deferred ACKs)
        let chunk = 4800;
        for _ in 0..500 {
            if mac_a.is_idle() && mac_b.is_idle() && mac_b.pending_acks.is_empty() {
                break;
            }
            let mut buf = vec![0.0f32; chunk];
            mac_a.fill_output(&mut buf);
            mac_b.feed(&buf);
            buf.fill(0.0);
            mac_b.fill_output(&mut buf);
            mac_a.feed(&buf);
            mac_a.poll(chunk);
            mac_b.poll(chunk);
        }

        assert_eq!(mac_b.recv().unwrap(), b"state test");
        assert!(mac_a.is_idle(), "a should be idle after round-trip");
    }
}
