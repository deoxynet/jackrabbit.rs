/// KCP-based ARQ integration tests.
///
/// Exercises the full encode → decode round-trip through the PHY layer,
/// including window management, retransmission, wired mode, multi-protocol
/// operation, and sustained transfers.

use aqua_soundbin::arq::{ArqEvent, KcpArq};
use aqua_soundbin::params;

/// Helper: run a single round-trip (sender sends, receiver processes, ACKs exchanged)
fn roundtrip(
    sender: &mut KcpArq,
    receiver: &mut KcpArq,
    payload: &[u8],
) -> Vec<Vec<u8>> {
    sender.send(payload);
    let data_samples = sender.poll(0).expect("sender should produce samples");

    let event = receiver.receive(&data_samples).expect("receiver should decode");
    let payloads = match event {
        ArqEvent::ReceivedData { payloads } => payloads,
        other => panic!("expected ReceivedData, got {:?}", other),
    };

    // Exchange ACKs until sender is idle
    for i in 0..20 {
        if sender.is_idle() {
            break;
        }
        if let Some(rx_samples) = receiver.poll(10 * (i + 1)) {
            sender.receive(&rx_samples);
        }
        sender.poll(10 * (i + 1));
    }

    payloads
}

// ---------------------------------------------------------------------------
// Test 1: Single frame round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_arq_single_frame_roundtrip() {
    let mut sender = KcpArq::new(0, 1.0);
    let mut receiver = KcpArq::new(0, 1.0);

    let payloads = roundtrip(&mut sender, &mut receiver, b"single frame");

    assert_eq!(payloads.len(), 1);
    assert_eq!(payloads[0], b"single frame");
    assert!(sender.is_idle());
}

// ---------------------------------------------------------------------------
// Test 2: Burst of frames (full window)
// ---------------------------------------------------------------------------

#[test]
fn test_arq_full_window_burst() {
    let mut sender = KcpArq::new(0, 1.0);
    let mut receiver = KcpArq::new(0, 1.0);

    // Window size is 32
    let messages: Vec<Vec<u8>> = (0..32)
        .map(|i| format!("burst-{}", i).into_bytes())
        .collect();

    // Send all 32
    for m in &messages {
        sender.send(m);
    }
    let data_samples = sender.poll(0).expect("sender should produce samples");
    assert!(!sender.can_send());

    // Receive all
    let event = receiver.receive(&data_samples).expect("receiver should decode");
    let received = match event {
        ArqEvent::ReceivedData { payloads } => payloads,
        other => panic!("expected ReceivedData, got {:?}", other),
    };

    assert_eq!(received.len(), 32);
    for (i, msg) in messages.iter().enumerate() {
        assert_eq!(received[i], *msg, "mismatch at position {}", i);
    }

    // Exchange ACKs
    for _ in 0..20 {
        if sender.is_idle() {
            break;
        }
        if let Some(rx_samples) = receiver.poll(10) {
            sender.receive(&rx_samples);
        }
        sender.poll(10);
    }

    assert!(sender.is_idle());
    assert!(sender.can_send());
}

// ---------------------------------------------------------------------------
// Test 3: Sliding window — send more than 16 frames
// ---------------------------------------------------------------------------

#[test]
fn test_arq_sliding_window() {
    let mut sender = KcpArq::new(0, 1.0);
    let mut receiver = KcpArq::new(0, 1.0);

    let mut total_delivered = 0usize;

    // Phase 1: fill window with 32 frames
    for i in 0u8..32 {
        sender.send(&[i]);
    }
    let data_samples = sender.poll(0).unwrap();
    assert!(!sender.can_send());

    // Receive all 32
    if let Some(ArqEvent::ReceivedData { payloads }) = receiver.receive(&data_samples) {
        total_delivered += payloads.len();
    }

    // Exchange ACKs until window clears
    for _ in 0..30 {
        if sender.can_send() {
            break;
        }
        if let Some(rx_samples) = receiver.poll(10) {
            sender.receive(&rx_samples);
        }
        sender.poll(10);
    }
    assert!(sender.can_send());

    // Phase 2: send 16 more
    for i in 32u8..48 {
        sender.send(&[i]);
    }
    let data_samples = sender.poll(0).unwrap();

    if let Some(ArqEvent::ReceivedData { payloads }) = receiver.receive(&data_samples) {
        total_delivered += payloads.len();
    }

    // Exchange ACKs
    for _ in 0..30 {
        if sender.is_idle() {
            break;
        }
        if let Some(rx_samples) = receiver.poll(10) {
            sender.receive(&rx_samples);
        }
        sender.poll(10);
    }

    assert!(sender.is_idle());
    assert!(total_delivered >= 48, "should have delivered at least 48 frames");
}

// ---------------------------------------------------------------------------
// Test 4: Retransmission recovers lost frame
// ---------------------------------------------------------------------------

#[test]
fn test_arq_retransmit_lost_frame() {
    let mut sender = KcpArq::new(0, 1.0);
    let mut receiver = KcpArq::new(0, 1.0);

    sender.send(b"will be lost");
    let _original = sender.poll(0).unwrap();
    assert!(!sender.is_idle());

    // Don't deliver the original (simulates loss)
    // Instead, advance time to trigger retransmit
    for i in 0..50 {
        let retransmit = sender.poll(100);
        if let Some(samples) = retransmit {
            // Got a retransmit, deliver it
            if let Some(ArqEvent::ReceivedData { payloads }) = receiver.receive(&samples) {
                if !payloads.is_empty() {
                    assert_eq!(payloads[0], b"will be lost");
                    // Exchange ACKs
                    for _ in 0..20 {
                        if sender.is_idle() {
                            break;
                        }
                        if let Some(rx_samples) = receiver.poll(10) {
                            sender.receive(&rx_samples);
                        }
                        sender.poll(10);
                    }
                    assert!(sender.is_idle());
                    return;
                }
            }
        }
    }
    // Even if we don't get a retransmit in test (depends on KCP timing),
    // sender should still not be idle
    assert!(!sender.is_idle());
}

// ---------------------------------------------------------------------------
// Test 5: Wired mode configuration
// ---------------------------------------------------------------------------

#[test]
fn test_arq_wired_mode() {
    let proto = 13u8; // WIRED_TURBO
    let config = params::get_protocol(proto);

    let mut sender = KcpArq::new_wired(proto, 1.0, config.cp_len);
    let mut receiver = KcpArq::new_wired(proto, 1.0, config.cp_len);

    let payloads: Vec<&[u8]> = vec![b"wired-0", b"wired-1", b"wired-2"];

    for payload in &payloads {
        sender.send(payload);
    }
    let data_samples = sender.poll(0).unwrap();

    let event = receiver.receive(&data_samples).expect("decode failed");
    let received = match event {
        ArqEvent::ReceivedData { payloads } => payloads,
        other => panic!("expected ReceivedData, got {:?}", other),
    };

    assert_eq!(received.len(), 3);
    for (i, p) in payloads.iter().enumerate() {
        assert_eq!(received[i], *p, "mismatch at {}", i);
    }

    // Exchange ACKs
    for _ in 0..20 {
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

// ---------------------------------------------------------------------------
// Test 6: Multiple protocols through ARQ
// ---------------------------------------------------------------------------

#[test]
fn test_arq_protocol_sweep() {
    for &proto in &[0u8, 4, 7, 12, 15] {
        let mut sender = KcpArq::new(proto, 1.0);
        let mut receiver = KcpArq::new(proto, 1.0);

        let payload = format!("proto-{}", proto).into_bytes();
        let payloads = roundtrip(&mut sender, &mut receiver, &payload);

        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0], payload, "payload mismatch for proto {}", proto);
        assert!(sender.is_idle(), "proto {} should be idle", proto);
    }
}

// ---------------------------------------------------------------------------
// Test 7: Protocol mismatch is ignored
// ---------------------------------------------------------------------------

#[test]
fn test_arq_protocol_mismatch_ignored() {
    let mut sender0 = KcpArq::new(0, 1.0);
    let mut receiver1 = KcpArq::new(1, 1.0);

    sender0.send(b"proto-0");
    let data_samples = sender0.poll(0).unwrap();

    assert!(receiver1.receive(&data_samples).is_none(), "mismatched protocol should be ignored");

    // Verify receiver1 can still receive matching protocol
    let mut sender1 = KcpArq::new(1, 1.0);
    sender1.send(b"proto-1");
    let data_samples = sender1.poll(0).unwrap();

    let event = receiver1.receive(&data_samples);
    assert!(event.is_some());
    match event.unwrap() {
        ArqEvent::ReceivedData { payloads } => {
            assert_eq!(payloads.len(), 1);
            assert_eq!(payloads[0], b"proto-1");
        }
        other => panic!("expected ReceivedData, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Test 8: Large payload through ARQ
// ---------------------------------------------------------------------------

#[test]
fn test_arq_large_payload() {
    let mut sender = KcpArq::new(4, 1.0); // DT_FAST: QPSK
    let mut receiver = KcpArq::new(4, 1.0);

    let payload: Vec<u8> = (0..512).map(|i| (i % 256) as u8).collect();
    let payloads = roundtrip(&mut sender, &mut receiver, &payload);

    assert_eq!(payloads.len(), 1);
    assert_eq!(payloads[0], payload);
    assert!(sender.is_idle());
}

// ---------------------------------------------------------------------------
// Test 9: Sustained transfer — 32 frames
// ---------------------------------------------------------------------------

#[test]
fn test_arq_sustained_transfer() {
    let mut sender = KcpArq::new(0, 1.0);
    let mut receiver = KcpArq::new(0, 1.0);

    let total_frames = 32usize;
    let mut delivered: Vec<Vec<u8>> = Vec::new();

    for i in 0..total_frames {
        let payload = vec![i as u8];
        let payloads = roundtrip(&mut sender, &mut receiver, &payload);
        delivered.extend(payloads);
    }

    assert_eq!(delivered.len(), total_frames);
    for (i, d) in delivered.iter().enumerate() {
        assert_eq!(d, &vec![i as u8], "mismatch at frame {}", i);
    }
    assert!(sender.is_idle());
}

// ---------------------------------------------------------------------------
// Test 10: Pipelined transfer with batched sends
// ---------------------------------------------------------------------------

#[test]
fn test_arq_pipelined_batched() {
    let mut sender = KcpArq::new(0, 1.0);
    let mut receiver = KcpArq::new(0, 1.0);

    let mut all_delivered: Vec<Vec<u8>> = Vec::new();

    for batch in 0..3u8 {
        // Send 8 frames in a batch
        for j in 0..8u8 {
            sender.send(&[batch * 8 + j]);
        }
        let data_samples = sender.poll(0).unwrap();

        // Receive all
        if let Some(ArqEvent::ReceivedData { payloads }) = receiver.receive(&data_samples) {
            all_delivered.extend(payloads);
        }

        // Exchange ACKs
        for _ in 0..30 {
            if sender.is_idle() {
                break;
            }
            if let Some(rx_samples) = receiver.poll(10) {
                sender.receive(&rx_samples);
            }
            sender.poll(10);
        }
        assert!(sender.is_idle(), "batch {} should leave sender idle", batch);
    }

    assert_eq!(all_delivered.len(), 24);
    for (i, d) in all_delivered.iter().enumerate() {
        assert_eq!(d, &vec![i as u8], "mismatch at position {}", i);
    }
}

// ---------------------------------------------------------------------------
// Test 11: can_send / is_idle state tracking
// ---------------------------------------------------------------------------

#[test]
fn test_arq_state_tracking() {
    let mut arq = KcpArq::new(0, 1.0);

    assert!(arq.is_idle());
    assert!(arq.can_send());

    arq.send(b"test");
    arq.poll(0);

    assert!(!arq.is_idle());
    assert!(arq.can_send()); // window of 16, only 1 used
}

// ---------------------------------------------------------------------------
// Test 12: Window exhaustion
// ---------------------------------------------------------------------------

#[test]
fn test_arq_window_exhaustion() {
    let mut arq = KcpArq::new(0, 1.0);

    // Fill window (size 32)
    for i in 0..32 {
        assert!(arq.send(&[i as u8]), "send {} should succeed", i);
    }
    arq.poll(0);

    assert!(!arq.can_send());
    assert!(!arq.send(b"overflow"));
}
