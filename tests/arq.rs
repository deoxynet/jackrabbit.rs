/// Selective Repeat ARQ integration tests.
///
/// Exercises the full encode → decode → ACK round-trip through the PHY layer,
/// including out-of-order delivery, window management, retransmission, wired
/// mode, multi-protocol operation, and impaired channels.

use aqua_soundbin::arq::{ArqEvent, SelectiveRepeat};
use aqua_soundbin::decoder::Decoder;
use aqua_soundbin::header::FrameType;
use aqua_soundbin::params;

// ---------------------------------------------------------------------------
// Test 1: Single frame round-trip
//
// One frame through the full ARQ + PHY stack: send → receive → ack → idle.
// ---------------------------------------------------------------------------

#[test]
fn test_arq_single_frame_roundtrip() {
    let mut sender = SelectiveRepeat::new(0, 1.0);
    let mut receiver = SelectiveRepeat::new(0, 1.0);

    let data_samples = sender.send(b"single frame").unwrap();
    assert!(!sender.is_idle());

    let ack_samples = match receiver.receive(&data_samples).unwrap() {
        ArqEvent::ReceivedData { payloads, ack } => {
            assert_eq!(payloads.len(), 1);
            assert_eq!(payloads[0], b"single frame");
            ack
        }
        other => panic!("expected ReceivedData, got {:?}", other),
    };

    match sender.receive(&ack_samples).unwrap() {
        ArqEvent::Acked => {}
        other => panic!("expected Acked, got {:?}", other),
    }
    assert!(sender.is_idle());
}

// ---------------------------------------------------------------------------
// Test 2: Burst of 16 frames (full window)
//
// Fill the window, deliver all in order, ACK all, verify sender goes idle.
// ---------------------------------------------------------------------------

#[test]
fn test_arq_full_window_burst() {
    let mut sender = SelectiveRepeat::new(0, 1.0);
    let mut receiver = SelectiveRepeat::new(0, 1.0);

    let messages: Vec<Vec<u8>> = (0..16)
        .map(|i| format!("burst-{}", i).into_bytes())
        .collect();

    // Send all 16
    let data_samples: Vec<Vec<f32>> = messages
        .iter()
        .map(|m| sender.send(m).unwrap())
        .collect();
    assert!(!sender.can_send());

    // Receive all 8 in order, collect ACKs
    let mut acks = Vec::new();
    for (i, samples) in data_samples.iter().enumerate() {
        match receiver.receive(samples).unwrap() {
            ArqEvent::ReceivedData { payloads, ack } => {
                assert_eq!(payloads.len(), 1);
                assert_eq!(payloads[0], messages[i]);
                acks.push(ack);
            }
            other => panic!("frame {}: expected ReceivedData, got {:?}", i, other),
        }
    }

    // ACK all — sender goes idle
    for ack in &acks {
        match sender.receive(ack).unwrap() {
            ArqEvent::Acked => {}
            other => panic!("expected Acked, got {:?}", other),
        }
    }
    assert!(sender.is_idle());
    assert!(sender.can_send());
}

// ---------------------------------------------------------------------------
// Test 3: Out-of-order delivery with reordering
//
// Send frames 0–4, deliver to receiver in scrambled order.
// Verify the reordering buffer produces correct in-order output.
// ---------------------------------------------------------------------------

#[test]
fn test_arq_out_of_order_reordering() {
    let mut sender = SelectiveRepeat::new(0, 1.0);
    let mut receiver = SelectiveRepeat::new(0, 1.0);

    let msgs: Vec<&[u8]> = vec![b"A", b"BB", b"CCC", b"DDDD", b"EEEEE"];
    let samples: Vec<Vec<f32>> = msgs.iter().map(|m| sender.send(m).unwrap()).collect();

    // Deliver in order: 0, 3, 4, 1, 2
    let delivery_order = [0usize, 3, 4, 1, 2];
    let mut delivered: Vec<Vec<u8>> = Vec::new();
    let mut acks: Vec<Vec<f32>> = Vec::new();

    for &idx in &delivery_order {
        match receiver.receive(&samples[idx]).unwrap() {
            ArqEvent::ReceivedData { payloads, ack } => {
                delivered.extend(payloads);
                acks.push(ack);
            }
            other => panic!("frame {}: expected ReceivedData, got {:?}", idx, other),
        }
    }

    // All 5 should be delivered in original order
    assert_eq!(delivered.len(), 5);
    for (i, msg) in msgs.iter().enumerate() {
        assert_eq!(delivered[i], *msg, "mismatch at position {}", i);
    }

    // ACK all — sender should go idle
    for ack in &acks {
        let _ = sender.receive(ack);
    }
    assert!(sender.is_idle());
}

// ---------------------------------------------------------------------------
// Test 4: Sliding window — send more than 16 frames
//
// Send 16, ACK first 8 (window slides), send 8 more, ACK remaining.
// Verifies window advancement and seq wrap behavior.
// ---------------------------------------------------------------------------

#[test]
fn test_arq_sliding_window() {
    let mut sender = SelectiveRepeat::new(0, 1.0);
    let mut receiver = SelectiveRepeat::new(0, 1.0);

    // Phase 1: fill window with frames 0–15
    let mut all_data = Vec::new();
    for i in 0u8..16 {
        all_data.push(sender.send(&[i]).unwrap());
    }
    assert!(!sender.can_send());

    // Receive all 16, collect ACKs
    let mut acks = Vec::new();
    for samples in &all_data {
        match receiver.receive(samples).unwrap() {
            ArqEvent::ReceivedData { ack, .. } => acks.push(ack),
            other => panic!("expected ReceivedData, got {:?}", other),
        }
    }

    // ACK first 8 — window slides, can send again
    for ack in &acks[..8] {
        match sender.receive(ack).unwrap() {
            ArqEvent::Acked => {}
            other => panic!("expected Acked, got {:?}", other),
        }
    }
    assert!(sender.can_send());

    // Phase 2: send 8 more (seqs 16–23)
    let mut more_data = Vec::new();
    for i in 16u8..24 {
        more_data.push(sender.send(&[i]).unwrap());
    }

    // Receive new frames
    let mut more_acks = Vec::new();
    for samples in &more_data {
        match receiver.receive(samples).unwrap() {
            ArqEvent::ReceivedData { payloads, ack } => {
                assert_eq!(payloads.len(), 1);
                more_acks.push(ack);
            }
            other => panic!("expected ReceivedData, got {:?}", other),
        }
    }

    // ACK remaining phase 1 (8–15) + phase 2 (16–23)
    for ack in acks[8..].iter().chain(more_acks.iter()) {
        let _ = sender.receive(ack);
    }
    assert!(sender.is_idle());
}

// ---------------------------------------------------------------------------
// Test 5: Sequence number wraps around 32
//
// Push 40 frames through the ARQ, 8 at a time, verifying seq wrap.
// ---------------------------------------------------------------------------

#[test]
fn test_arq_seq_wraparound() {
    let mut sender = SelectiveRepeat::new(0, 1.0);
    let mut receiver = SelectiveRepeat::new(0, 1.0);

    let mut total_delivered = 0usize;

    for batch in 0..5u8 {
        // Send 8 frames
        let mut batch_data = Vec::new();
        for j in 0..8u8 {
            let payload = vec![batch * 8 + j];
            batch_data.push(sender.send(&payload).unwrap());
        }

        // Receive and ACK all 8
        for samples in &batch_data {
            match receiver.receive(samples).unwrap() {
                ArqEvent::ReceivedData { payloads, ack } => {
                    total_delivered += payloads.len();
                    match sender.receive(&ack).unwrap() {
                        ArqEvent::Acked => {}
                        other => panic!("expected Acked, got {:?}", other),
                    }
                }
                other => panic!("expected ReceivedData, got {:?}", other),
            }
        }
        assert!(sender.is_idle());
    }

    assert_eq!(total_delivered, 40);
}

// ---------------------------------------------------------------------------
// Test 6: Retransmission recovers lost frame
//
// Send 3 frames. "Lose" frame 1 (don't deliver it). Timeout retransmits it.
// Deliver the retransmit → gap-fill delivers frames 1 and 2.
// ---------------------------------------------------------------------------

#[test]
fn test_arq_retransmit_lost_frame() {
    let mut sender = SelectiveRepeat::new(0, 1.0);
    let mut receiver = SelectiveRepeat::new(0, 1.0);

    let s0 = sender.send(b"zero").unwrap();
    let _s1 = sender.send(b"one").unwrap(); // "lost" — not delivered
    let s2 = sender.send(b"two").unwrap();

    // Deliver frame 0 — in-order
    let ack0 = match receiver.receive(&s0).unwrap() {
        ArqEvent::ReceivedData { payloads, ack } => {
            assert_eq!(payloads, vec![b"zero".to_vec()]);
            ack
        }
        other => panic!("expected ReceivedData, got {:?}", other),
    };

    // Deliver frame 2 — OOO, buffered, no cumulative ACK (rx_base didn't advance)
    match receiver.receive(&s2).unwrap() {
        ArqEvent::ReceivedData { payloads, ack } => {
            assert!(payloads.is_empty(), "OOO should buffer, not deliver");
            assert!(ack.is_empty(), "OOO should not generate cumulative ACK");
        }
        other => panic!("expected ReceivedData, got {:?}", other),
    }

    // ACK frame 0 at sender (cumulative ACK covers only seq 0)
    sender.receive(&ack0).unwrap();
    // tx_base should have advanced to 1 (frames 1,2 still unacked)
    assert!(!sender.is_idle());

    // Timeout retransmits frame 1
    let retransmit = sender.timeout().unwrap();

    // Deliver the retransmit — gap-fill delivers frames 1 and 2
    match receiver.receive(&retransmit).unwrap() {
        ArqEvent::ReceivedData { payloads, ack } => {
            assert_eq!(payloads.len(), 2);
            assert_eq!(payloads[0], b"one");
            assert_eq!(payloads[1], b"two");
            // ACK frame 1 at sender
            sender.receive(&ack).unwrap();
        }
        other => panic!("expected ReceivedData, got {:?}", other),
    }

    assert!(sender.is_idle());
}

// ---------------------------------------------------------------------------
// Test 7: Duplicate ACK from receiver (retransmit of already-delivered frame)
//
// Receiver gets frame 0, ACKs it. Frame 0 retransmitted (simulating late
// timeout). Receiver sees it as previous-window → DuplicateAck.
// ---------------------------------------------------------------------------

#[test]
fn test_arq_duplicate_triggers_reack() {
    let mut sender = SelectiveRepeat::new(0, 1.0);
    let mut receiver = SelectiveRepeat::new(0, 1.0);

    let data = sender.send(b"dup-check").unwrap();

    // Receiver gets and delivers the frame
    let ack = match receiver.receive(&data).unwrap() {
        ArqEvent::ReceivedData { payloads, ack } => {
            assert_eq!(payloads[0], b"dup-check");
            ack
        }
        other => panic!("expected ReceivedData, got {:?}", other),
    };

    // Sender gets ACK — idle
    sender.receive(&ack).unwrap();
    assert!(sender.is_idle());

    // Retransmit arrives late at receiver → DuplicateAck
    match receiver.receive(&data).unwrap() {
        ArqEvent::DuplicateAck(reack) => {
            assert!(!reack.is_empty());
        }
        other => panic!("expected DuplicateAck, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Test 8: Wired mode configuration
//
// Uses new_wired() with a cable protocol, verifying the reduced-overhead
// path (minimal preamble, short CP, 1 training symbol) works end-to-end.
// ---------------------------------------------------------------------------

#[test]
fn test_arq_wired_mode() {
    let proto = 13u8; // WIRED_TURBO: QAM256, 3/4 LDPC, cp=30
    let config = params::get_protocol(proto);

    let mut sender = SelectiveRepeat::new_wired(proto, 1.0, config.cp_len);
    let mut receiver = SelectiveRepeat::new_wired(proto, 1.0, config.cp_len);

    // Send 3 frames through wired ARQ
    let payloads: Vec<&[u8]> = vec![b"wired-0", b"wired-1", b"wired-2"];
    let mut acks = Vec::new();

    for payload in &payloads {
        let data = sender.send(payload).unwrap();
        match receiver.receive(&data).unwrap() {
            ArqEvent::ReceivedData {
                payloads: delivered,
                ack,
            } => {
                assert_eq!(delivered.len(), 1);
                assert_eq!(delivered[0], *payload);
                acks.push(ack);
            }
            other => panic!("expected ReceivedData, got {:?}", other),
        }
    }

    for ack in &acks {
        match sender.receive(ack).unwrap() {
            ArqEvent::Acked => {}
            other => panic!("expected Acked, got {:?}", other),
        }
    }
    assert!(sender.is_idle());
}

// ---------------------------------------------------------------------------
// Test 9: Multiple protocols through ARQ
//
// Verifies that different protocol configs all work through the ARQ layer.
// ---------------------------------------------------------------------------

#[test]
fn test_arq_protocol_sweep() {
    for &proto in &[0u8, 4, 7, 12, 15] {
        let mut sender = SelectiveRepeat::new(proto, 1.0);
        let mut receiver = SelectiveRepeat::new(proto, 1.0);

        let payload = format!("proto-{}", proto).into_bytes();
        let data = sender.send(&payload).unwrap();

        let ack = match receiver.receive(&data).unwrap() {
            ArqEvent::ReceivedData {
                payloads: delivered,
                ack,
            } => {
                assert_eq!(
                    delivered[0], payload,
                    "payload mismatch for proto {}",
                    proto
                );
                ack
            }
            other => panic!("proto {}: expected ReceivedData, got {:?}", proto, other),
        };

        match sender.receive(&ack).unwrap() {
            ArqEvent::Acked => {}
            other => panic!("proto {}: expected Acked, got {:?}", proto, other),
        }
        assert!(sender.is_idle(), "proto {} should be idle", proto);
    }
}

// ---------------------------------------------------------------------------
// Test 10: Protocol mismatch is ignored
//
// Frames with a different protocol_id should be dropped without affecting
// subsequent valid decodes.
// ---------------------------------------------------------------------------

#[test]
fn test_arq_protocol_mismatch_ignored() {
    let mut sender0 = SelectiveRepeat::new(0, 1.0);
    let mut receiver1 = SelectiveRepeat::new(1, 1.0);

    let data0 = sender0.send(b"proto-0").unwrap();
    assert!(receiver1.receive(&data0).is_none(), "mismatched protocol should be ignored");

    let mut sender1 = SelectiveRepeat::new(1, 1.0);
    let data1 = sender1.send(b"proto-1").unwrap();
    match receiver1.receive(&data1).unwrap() {
        ArqEvent::ReceivedData { payloads, .. } => {
            assert_eq!(payloads.len(), 1);
            assert_eq!(payloads[0], b"proto-1");
        }
        other => panic!("expected ReceivedData, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Test 11: Large payload through ARQ
//
// Sends a multi-symbol payload to verify the ARQ doesn't interfere with
// large frame encoding/decoding.
// ---------------------------------------------------------------------------

#[test]
fn test_arq_large_payload() {
    let mut sender = SelectiveRepeat::new(4, 1.0); // DT_FAST: QPSK
    let mut receiver = SelectiveRepeat::new(4, 1.0);

    let payload: Vec<u8> = (0..512).map(|i| (i % 256) as u8).collect();
    let data = sender.send(&payload).unwrap();

    let ack = match receiver.receive(&data).unwrap() {
        ArqEvent::ReceivedData {
            payloads: delivered,
            ack,
        } => {
            assert_eq!(delivered.len(), 1);
            assert_eq!(delivered[0], payload);
            ack
        }
        other => panic!("expected ReceivedData, got {:?}", other),
    };

    sender.receive(&ack).unwrap();
    assert!(sender.is_idle());
}

// ---------------------------------------------------------------------------
// Test 12: Sustained transfer — 32 frames with sliding window
//
// Simulates a realistic bulk transfer: continuously send while ACKing,
// keeping the pipeline full. Verifies no state corruption over many
// window rotations.
// ---------------------------------------------------------------------------

#[test]
fn test_arq_sustained_transfer() {
    let mut sender = SelectiveRepeat::new(0, 1.0);
    let mut receiver = SelectiveRepeat::new(0, 1.0);

    let total_frames = 32usize;
    let mut delivered: Vec<Vec<u8>> = Vec::new();

    // Send one, receive, ACK, repeat. Exercises window sliding across
    // many seq rotations (32 frames = 2 full seq-space wraps).
    for i in 0..total_frames {
        let payload = vec![i as u8];
        let data = sender.send(&payload).unwrap();

        match receiver.receive(&data).unwrap() {
            ArqEvent::ReceivedData {
                payloads,
                ack,
            } => {
                delivered.extend(payloads);
                sender.receive(&ack).unwrap();
            }
            other => panic!("frame {}: expected ReceivedData, got {:?}", i, other),
        }
    }

    assert_eq!(delivered.len(), total_frames);
    for (i, d) in delivered.iter().enumerate() {
        assert_eq!(d, &vec![i as u8], "mismatch at frame {}", i);
    }
    assert!(sender.is_idle());
}

// ---------------------------------------------------------------------------
// Test 13: Pipelined transfer with batched ACKs
//
// Send 8 frames, then ACK them all at once. Repeat for 3 batches (24 total).
// Verifies the window slides correctly with deferred ACKs.
// ---------------------------------------------------------------------------

#[test]
fn test_arq_pipelined_batched_acks() {
    let mut sender = SelectiveRepeat::new(0, 1.0);
    let mut receiver = SelectiveRepeat::new(0, 1.0);

    let mut all_delivered: Vec<Vec<u8>> = Vec::new();

    for batch in 0..3u8 {
        let mut batch_samples = Vec::new();
        // Send 8 frames
        for j in 0..8u8 {
            let payload = vec![batch * 8 + j];
            batch_samples.push(sender.send(&payload).unwrap());
        }

        // Receive all 8, collect ACKs
        let mut batch_acks = Vec::new();
        for samples in &batch_samples {
            match receiver.receive(samples).unwrap() {
                ArqEvent::ReceivedData { payloads, ack } => {
                    all_delivered.extend(payloads);
                    batch_acks.push(ack);
                }
                other => panic!("expected ReceivedData, got {:?}", other),
            }
        }

        // ACK all 8 at once
        for ack in &batch_acks {
            sender.receive(ack).unwrap();
        }
        assert!(sender.is_idle(), "batch {} should leave sender idle", batch);
    }

    assert_eq!(all_delivered.len(), 24);
    for (i, d) in all_delivered.iter().enumerate() {
        assert_eq!(d, &vec![i as u8], "mismatch at position {}", i);
    }
}

// ---------------------------------------------------------------------------
// Test 14: Cumulative ACK clears entire window
//
// Send 4 frames in order, deliver all. Cumulative ACKs cover [0], [0..1],
// [0..2], [0..3]. Sending the last ACK first makes sender idle immediately.
// ---------------------------------------------------------------------------

#[test]
fn test_arq_cumulative_ack() {
    let mut sender = SelectiveRepeat::new(0, 1.0);
    let mut receiver = SelectiveRepeat::new(0, 1.0);

    // Send 4
    let mut data_samples = Vec::new();
    for i in 0..4u8 {
        data_samples.push(sender.send(&[i]).unwrap());
    }

    // Receive in order, collect cumulative ACKs
    let mut acks = Vec::new();
    for samples in &data_samples {
        match receiver.receive(samples).unwrap() {
            ArqEvent::ReceivedData { ack, .. } => acks.push(ack),
            other => panic!("expected ReceivedData, got {:?}", other),
        }
    }

    // ACK(3) is cumulative: clears slots 0,1,2,3 → sender idle
    sender.receive(&acks[3]).unwrap();
    assert!(sender.is_idle(), "cumulative ACK(3) should clear all 4 frames");
}

// ---------------------------------------------------------------------------
// Test 15: Cumulative ACK header seq = rx_base - 1
//
// Ensures cumulative ACK frames carry the highest contiguous delivered seq.
// ---------------------------------------------------------------------------

#[test]
fn test_arq_cumulative_ack_header_seq() {
    let mut sender = SelectiveRepeat::new(0, 1.0);
    let mut receiver = SelectiveRepeat::new(0, 1.0);

    // Send 3 frames, deliver all in order
    let s0 = sender.send(b"zero").unwrap();
    let s1 = sender.send(b"one").unwrap();
    let s2 = sender.send(b"two").unwrap();

    // Receive frame 0 → cumulative ACK(0)
    let ack0 = match receiver.receive(&s0).unwrap() {
        ArqEvent::ReceivedData { payloads, ack } => {
            assert_eq!(payloads.len(), 1);
            ack
        }
        other => panic!("expected ReceivedData, got {:?}", other),
    };

    // Receive frame 1 → cumulative ACK(1)
    let _ack1 = match receiver.receive(&s1).unwrap() {
        ArqEvent::ReceivedData { payloads, ack } => {
            assert_eq!(payloads.len(), 1);
            ack
        }
        other => panic!("expected ReceivedData, got {:?}", other),
    };

    // Receive frame 2 → cumulative ACK(2)
    let ack2 = match receiver.receive(&s2).unwrap() {
        ArqEvent::ReceivedData { payloads, ack } => {
            assert_eq!(payloads.len(), 1);
            ack
        }
        other => panic!("expected ReceivedData, got {:?}", other),
    };

    // Verify ACK(0) header
    let mut decoder = Decoder::new();
    let (info, payload) = decoder.feed_frame(&ack0).expect("ack0 decode failed");
    assert_eq!(info.frame_type, FrameType::Ack);
    assert_eq!(info.seq_num, 0, "first cumulative ACK should be for seq 0");
    assert_eq!(payload[0], 0);

    // Verify ACK(2) header — cumulative covers 0,1,2
    let mut decoder = Decoder::new();
    let (info, payload) = decoder.feed_frame(&ack2).expect("ack2 decode failed");
    assert_eq!(info.frame_type, FrameType::Ack);
    assert_eq!(info.seq_num, 2, "last cumulative ACK should be for seq 2");
    assert_eq!(payload[0], 2);

    // Sending ACK(2) to sender should clear all 3 frames
    sender.receive(&ack2).unwrap();
    assert!(sender.is_idle());
}
