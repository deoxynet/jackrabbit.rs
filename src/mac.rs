/// Half-Duplex MAC layer: protocol engine between ARQ and audio/TAP I/O.
///
/// No I/O ownership — caller pushes/pulls samples and payloads.
/// Implicit state machine:
///   - Transmitting: `!output_buf.is_empty()`
///   - Waiting for ACK: `output_buf.is_empty() && !arq.is_idle()`
///   - Ready to send: `output_buf.is_empty() && arq.can_send() && !tx_queue.is_empty()`
///   - Idle: `output_buf.is_empty() && arq.is_idle() && tx_queue.is_empty()`

use std::collections::VecDeque;
use crate::arq::{ArqEvent, KcpArq};
use crate::params;

pub struct HalfDuplexMac {
    arq: KcpArq,
    tx_queue: VecDeque<Vec<u8>>,
    output_buf: VecDeque<f32>,
    rx_queue: VecDeque<Vec<u8>>,
    clock: u64,
    clock_ms: u32,
    last_tx_end: u64,
    timeout_samples: u64,
    gap_samples: usize,
    burst_limit: usize,       // max frames sent per burst
    burst_sent: usize,        // frames sent in current burst
    was_transmitting: bool,   // decoder reset on TX→RX transition
    stop_and_wait: bool,      // only send new data when no inflight frames
    // Deferred poll: wait for silence before flushing KCP output
    needs_poll: bool,
    quiet_samples: usize,        // consecutive low-energy samples (actual silence)
    poll_quiet_threshold: usize, // quiet samples needed before polling KCP
}

impl HalfDuplexMac {
    /// OTA configuration.
    pub fn new(protocol_id: u8, volume: f32) -> Self {
        HalfDuplexMac {
            arq: KcpArq::new(protocol_id, volume),
            tx_queue: VecDeque::new(),
            output_buf: VecDeque::new(),
            rx_queue: VecDeque::new(),
            clock: 0,
            clock_ms: 0,
            last_tx_end: 0,
            timeout_samples: 48_000, // 1s at 48kHz
            gap_samples: params::SYMBOL_TOTAL * 2, // 2,400 = 50ms inter-frame silence
            burst_limit: 1, // robust half-duplex: one frame per burst
            burst_sent: 0,
            was_transmitting: false,
            stop_and_wait: true,
            needs_poll: false,
            quiet_samples: 48_000, // presume clear at startup (1s)
            poll_quiet_threshold: params::SYMBOL_TOTAL * 4, // 4800 = 100ms of quiet
        }
    }

    /// Cable-optimized configuration (short preamble, protocol CP, 1 training symbol).
    pub fn new_wired(protocol_id: u8, volume: f32, cp_len: usize) -> Self {
        let sym = params::FFT_SIZE + cp_len;
        HalfDuplexMac {
            arq: KcpArq::new_wired(protocol_id, volume, cp_len),
            tx_queue: VecDeque::new(),
            output_buf: VecDeque::new(),
            rx_queue: VecDeque::new(),
            clock: 0,
            clock_ms: 0,
            last_tx_end: 0,
            timeout_samples: 24_000, // 500ms at 48kHz
            gap_samples: sym * 2, // 2 symbols inter-frame silence
            burst_limit: 1, // robust half-duplex: one frame per burst
            burst_sent: 0,
            was_transmitting: false,
            stop_and_wait: true,
            needs_poll: false,
            quiet_samples: 48_000, // presume clear at startup (1s)
            poll_quiet_threshold: sym * 4, // ~4 symbols of quiet
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
        if self.was_transmitting {
            self.was_transmitting = false;
            self.arq.reset_decoder();
        }
        // Track energy-based silence: only count truly quiet samples.
        let energy: f32 = samples.iter().map(|s| s * s).sum::<f32>();
        if energy < 1e-6 * samples.len() as f32 {
            self.quiet_samples += samples.len();
        } else {
            self.quiet_samples = 0;
        }

        if let Some(event) = self.arq.receive(samples) {
            match event {
                ArqEvent::ReceivedData { payloads } => {
                    for p in payloads {
                        self.rx_queue.push_back(p);
                    }
                    // KCP generates ACKs internally; mark that we need to poll
                    self.needs_poll = true;
                }
                ArqEvent::Acked => {}
                ArqEvent::Reset { .. } => {
                    // Drop any pending state
                    self.needs_poll = false;
                }
            }
        }
    }

    /// Advance the clock and perform timeout/burst-send logic.
    pub fn poll(&mut self, elapsed_samples: usize) {
        self.clock += elapsed_samples as u64;
        // Convert samples to ms (48000 samples/sec = 48 samples/ms)
        let elapsed_ms = (elapsed_samples / 48).max(1) as u32;
        self.clock_ms = self.clock_ms.wrapping_add(elapsed_ms);

        if self.is_transmitting() {
            return;
        }

        // Flush deferred KCP output after sufficient quiet (sender's burst ended).
        if self.needs_poll && self.quiet_samples >= self.poll_quiet_threshold {
            if let Some(samples) = self.arq.poll(elapsed_ms) {
                self.quiet_samples = 0;
                self.needs_poll = false;
                self.output_buf.extend(samples.iter());
                return; // send response before any new data
            }
            self.needs_poll = false;
        }

        // Regular KCP update for retransmission timing
        if !self.arq.is_idle() {
            let since_tx = self.clock.saturating_sub(self.last_tx_end);
            // Poll KCP periodically for retransmits
            if since_tx > self.timeout_samples / 4 {
                if let Some(samples) = self.arq.poll(elapsed_ms) {
                    self.append_with_gap(&samples);
                    self.last_tx_end = self.clock;
                    return;
                }
            }
        }

        // Burst-send queued payloads (respecting burst_limit).
        // Carrier sense: require gap_samples of silence before sending data.
        if !self.needs_poll && self.quiet_samples >= self.gap_samples {
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

    /// Set silence duration (in samples) before deferred responses are flushed.
    pub fn set_ack_quiet_threshold(&mut self, samples: usize) {
        self.poll_quiet_threshold = samples;
    }

    /// Override the cyclic prefix length for data symbols (cable mode).
    pub fn set_data_cp_override(&mut self, cp_len: usize) {
        self.arq.set_data_cp_override(cp_len);
    }

    /// Inform the MAC about the total queued output backlog (in samples).
    /// This is used to align timeouts with actual playback completion.
    pub fn note_output_backlog(&mut self, backlog_samples: usize) {
        self.last_tx_end = self
            .clock
            .saturating_add(backlog_samples as u64);
    }

    /// Access the underlying ARQ state.
    pub fn arq(&self) -> &KcpArq {
        &self.arq
    }

    fn append_with_gap(&mut self, samples: &[f32]) {
        self.quiet_samples = 0; // can't be quiet while transmitting
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
            if self.arq.send(&payload) {
                // Poll immediately to get the encoded samples
                if let Some(samples) = self.arq.poll(0) {
                    self.append_with_gap(&samples);
                    self.burst_sent += 1;
                }
            }
        }
    }
}

// Backward compatibility alias
pub type SelectiveRepeat = KcpArq;

#[cfg(test)]
mod tests {
    use super::*;

    const PROTO: u8 = 0;
    const VOL: f32 = 1.0;

    // Helper to step both MACs
    fn step(a: &mut HalfDuplexMac, b: &mut HalfDuplexMac, chunk: usize) {
        let mut buf = vec![0.0f32; chunk];
        a.fill_output(&mut buf);
        b.feed(&buf);
        buf.fill(0.0);
        b.fill_output(&mut buf);
        a.feed(&buf);
        a.poll(chunk);
        b.poll(chunk);
    }

    fn run_until_idle(a: &mut HalfDuplexMac, b: &mut HalfDuplexMac, chunk: usize, max_steps: usize) -> usize {
        for i in 0..max_steps {
            if a.is_idle() && b.is_idle() {
                return i;
            }
            step(a, b, chunk);
        }
        max_steps
    }

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

    // 2. feed data → recv payload
    #[test]
    fn test_feed_data_recv_payload() {
        let mut sender = HalfDuplexMac::new(PROTO, VOL);
        let mut receiver = HalfDuplexMac::new(PROTO, VOL);

        sender.enqueue(b"test payload".to_vec());
        sender.poll(0);

        let chunk = 4800;
        let steps = run_until_idle(&mut sender, &mut receiver, chunk, 500);
        assert!(steps < 500, "transfer did not complete");

        let payload = receiver.recv();
        assert!(payload.is_some());
        assert_eq!(payload.unwrap(), b"test payload");
    }

    // 3. half-duplex gate: while output_buf has data, feed() is a no-op
    #[test]
    fn test_half_duplex_gate() {
        let mut mac = HalfDuplexMac::new(PROTO, VOL);
        mac.enqueue(b"tx data".to_vec());
        mac.poll(0);
        assert!(mac.is_transmitting());

        // Create some data samples to feed
        let mut sender = KcpArq::new(PROTO, VOL);
        sender.send(b"incoming");
        let data_samples = sender.poll(0).unwrap();

        mac.feed(&data_samples);
        assert!(mac.recv().is_none(), "half-duplex gate should block feed");
    }

    // 4. burst fill: enqueue 4 payloads, poll once, verify only one sent
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

    // 5. timeout retransmit
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

    // 6. state queries: fresh idle, after enqueue+poll transmitting
    #[test]
    fn test_state_queries() {
        let mut mac = HalfDuplexMac::new(PROTO, VOL);

        // Fresh → idle
        assert!(mac.is_idle());
        assert!(!mac.is_transmitting());

        // Enqueue + poll → transmitting
        mac.enqueue(b"state test".to_vec());
        mac.poll(0);
        assert!(mac.is_transmitting());
        assert!(!mac.is_idle());
    }

    // 7. Full round-trip transfer
    #[test]
    fn test_full_roundtrip() {
        let mut mac_a = HalfDuplexMac::new(PROTO, VOL);
        let mut mac_b = HalfDuplexMac::new(PROTO, VOL);

        mac_a.enqueue(b"roundtrip test".to_vec());
        mac_a.poll(0);

        let chunk = 4800;
        let steps = run_until_idle(&mut mac_a, &mut mac_b, chunk, 500);
        assert!(steps < 500, "transfer did not complete");

        let payload = mac_b.recv().expect("B should have received data");
        assert_eq!(payload, b"roundtrip test");
        assert!(mac_a.is_idle());
    }
}
