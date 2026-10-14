/// Half-Duplex MAC integration tests.
///
/// Exercises the MAC layer end-to-end: unidirectional and bidirectional
/// transfers, burst delivery, lost-frame recovery, wired mode, and
/// sustained throughput.

use aqua_soundbin::mac::HalfDuplexMac;
use aqua_soundbin::params;

/// Simulate one half-duplex audio exchange step between two MACs.
///
/// a fills output → b feeds it; b fills output → a feeds it; both poll.
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

/// Run steps until both MACs are idle or max iterations reached.
fn run_until_idle(a: &mut HalfDuplexMac, b: &mut HalfDuplexMac, chunk: usize, max_steps: usize) -> usize {
    for i in 0..max_steps {
        if a.is_idle() && b.is_idle() {
            return i;
        }
        step(a, b, chunk);
    }
    max_steps
}

// 1. Unidirectional transfer: A sends 1 payload to B
#[test]
fn test_mac_unidirectional_transfer() {
    let mut a = HalfDuplexMac::new(0, 1.0);
    let mut b = HalfDuplexMac::new(0, 1.0);

    a.enqueue(b"hello from A".to_vec());
    a.poll(0);

    let chunk = 4800; // 100ms at 48kHz
    let steps = run_until_idle(&mut a, &mut b, chunk, 500);
    assert!(steps < 500, "transfer did not complete in time");

    let payload = b.recv().expect("B should have received data");
    assert_eq!(payload, b"hello from A");
    assert!(b.recv().is_none());
}

// 2. Bidirectional exchange: A sends first, then B sends back
#[test]
fn test_mac_bidirectional_exchange() {
    let mut a = HalfDuplexMac::new(0, 1.0);
    let mut b = HalfDuplexMac::new(0, 1.0);

    let chunk = 4800;

    // Phase 1: A sends to B
    a.enqueue(b"from A".to_vec());
    a.poll(0);
    let steps = run_until_idle(&mut a, &mut b, chunk, 500);
    assert!(steps < 500, "A→B transfer did not complete");

    let payload = b.recv().expect("B should have received A's payload");
    assert_eq!(payload, b"from A");

    // Phase 2: B sends to A
    b.enqueue(b"from B".to_vec());
    b.poll(0);
    let steps = run_until_idle(&mut a, &mut b, chunk, 500);
    assert!(steps < 500, "B→A transfer did not complete");

    let payload = a.recv().expect("A should have received B's payload");
    assert_eq!(payload, b"from B");
}

// 3. Burst transfer: 16 payloads through one direction (full window)
#[test]
fn test_mac_burst_transfer() {
    let mut a = HalfDuplexMac::new(0, 1.0);
    let mut b = HalfDuplexMac::new(0, 1.0);

    let messages: Vec<Vec<u8>> = (0..16)
        .map(|i| format!("burst-{:02}", i).into_bytes())
        .collect();
    for msg in &messages {
        a.enqueue(msg.clone());
    }
    a.poll(0);

    let chunk = 4800;
    let steps = run_until_idle(&mut a, &mut b, chunk, 2000);
    assert!(steps < 2000, "burst transfer did not complete");

    let mut received = Vec::new();
    while let Some(p) = b.recv() {
        received.push(p);
    }
    assert_eq!(received.len(), 16, "should receive all 16 payloads");
    for (i, msg) in messages.iter().enumerate() {
        assert_eq!(received[i], *msg, "mismatch at position {}", i);
    }
}

// 4. Lost frame recovery: zero out one frame's samples, verify timeout + retransmit
#[test]
fn test_mac_lost_frame_recovery() {
    let mut a = HalfDuplexMac::new(0, 1.0);
    let mut b = HalfDuplexMac::new(0, 1.0);

    a.enqueue(b"recover me".to_vec());
    a.poll(0);

    let chunk = 4800;

    // First step: corrupt A's output (simulate lost frame)
    let mut buf = vec![0.0f32; chunk];
    a.fill_output(&mut buf);
    // Zero out the buffer to simulate total loss
    buf.fill(0.0);
    b.feed(&buf);
    buf.fill(0.0);
    b.fill_output(&mut buf);
    a.feed(&buf);
    a.poll(chunk);
    b.poll(chunk);

    // B should not have received anything yet
    assert!(b.recv().is_none(), "corrupted frame should not decode");

    // Continue stepping — timeout should trigger retransmit
    let steps = run_until_idle(&mut a, &mut b, chunk, 500);
    assert!(steps < 500, "recovery did not complete");

    let payload = b.recv().expect("B should have received retransmitted data");
    assert_eq!(payload, b"recover me");
}

// 5. Wired mode round-trip
#[test]
fn test_mac_wired_mode() {
    let proto = 13u8; // WIRED_TURBO
    let config = params::get_protocol(proto);

    let mut a = HalfDuplexMac::new_wired(proto, 1.0, config.cp_len);
    let mut b = HalfDuplexMac::new_wired(proto, 1.0, config.cp_len);

    a.enqueue(b"wired transfer".to_vec());
    a.poll(0);

    let chunk = 2400; // 50ms
    let steps = run_until_idle(&mut a, &mut b, chunk, 500);
    assert!(steps < 500, "wired transfer did not complete");

    let payload = b.recv().expect("B should have received wired data");
    assert_eq!(payload, b"wired transfer");
}

// 6. Sustained transfer: 48 payloads, all delivered in order
#[test]
fn test_mac_sustained_transfer() {
    let mut a = HalfDuplexMac::new(0, 1.0);
    let mut b = HalfDuplexMac::new(0, 1.0);

    let total = 48;
    let messages: Vec<Vec<u8>> = (0..total)
        .map(|i| format!("sustained-{:02}", i).into_bytes())
        .collect();

    // Enqueue all up front
    for msg in &messages {
        a.enqueue(msg.clone());
    }
    a.poll(0);

    let chunk = 4800;
    let max_steps = 5000;
    for _ in 0..max_steps {
        if a.is_idle() {
            break;
        }
        step(&mut a, &mut b, chunk);
    }
    assert!(a.is_idle(), "sender should be idle after sustained transfer");

    let mut received = Vec::new();
    while let Some(p) = b.recv() {
        received.push(p);
    }
    assert_eq!(
        received.len(),
        total,
        "should receive all {} payloads, got {}",
        total,
        received.len()
    );
    for (i, msg) in messages.iter().enumerate() {
        assert_eq!(received[i], *msg, "mismatch at position {}", i);
    }
}

// 7. Throughput benchmark: measure bytes/sec through MAC loopback for each protocol class
#[test]
fn test_mac_throughput_benchmark() {
    let sample_rate = 48_000usize;
    let chunk = 4800; // 100ms

    struct Case {
        name: &'static str,
        proto: u8,
        wired: bool,
        payload_size: usize,
        num_payloads: usize,
    }

    let cases = [
        Case { name: "AUDIBLE_NORMAL (BPSK 1/2)", proto: 0, wired: false, payload_size: 50, num_payloads: 16 },
        Case { name: "DT_FAST (QPSK 1/2)", proto: 4, wired: false, payload_size: 200, num_payloads: 16 },
        Case { name: "UT_FASTEST (QAM16 3/4)", proto: 11, wired: false, payload_size: 2000, num_payloads: 16 },
        Case { name: "WIRED_TURBO (QAM256 3/4 LDPC)", proto: 13, wired: true, payload_size: 1000, num_payloads: 32 },
    ];

    eprintln!("\n--- MAC Throughput Benchmark ---");
    for case in &cases {
        let config = params::get_protocol(case.proto);
        let (mut a, mut b) = if case.wired {
            (
                HalfDuplexMac::new_wired(case.proto, 1.0, config.cp_len),
                HalfDuplexMac::new_wired(case.proto, 1.0, config.cp_len),
            )
        } else {
            (
                HalfDuplexMac::new(case.proto, 1.0),
                HalfDuplexMac::new(case.proto, 1.0),
            )
        };

        let payload: Vec<u8> = (0..case.payload_size).map(|i| (i % 256) as u8).collect();
        for _ in 0..case.num_payloads {
            a.enqueue(payload.clone());
        }
        a.poll(0);

        let mut total_samples = 0usize;
        let max_steps = 20_000;
        for _ in 0..max_steps {
            if a.is_idle() {
                break;
            }
            step(&mut a, &mut b, chunk);
            total_samples += chunk;
        }
        assert!(a.is_idle(), "{}: transfer did not complete", case.name);

        let mut received = Vec::new();
        while let Some(p) = b.recv() {
            received.push(p);
        }
        assert_eq!(received.len(), case.num_payloads, "{}: payload count mismatch", case.name);

        let total_bytes = case.num_payloads * case.payload_size;
        let duration_secs = total_samples as f64 / sample_rate as f64;
        let throughput_bps = (total_bytes as f64 * 8.0) / duration_secs;

        eprintln!(
            "  {:40} {:5} B x {:3} = {:6} B in {:6.2}s = {:8.0} bps ({:.1} kbps)",
            case.name,
            case.payload_size,
            case.num_payloads,
            total_bytes,
            duration_secs,
            throughput_bps,
            throughput_bps / 1000.0,
        );
    }

    // Pipelined mode (stop_and_wait = false)
    eprintln!("\n--- Pipelined Mode (window=32) ---");
    for case in &cases {
        let config = params::get_protocol(case.proto);
        let (mut a, mut b) = if case.wired {
            (
                HalfDuplexMac::new_wired(case.proto, 1.0, config.cp_len),
                HalfDuplexMac::new_wired(case.proto, 1.0, config.cp_len),
            )
        } else {
            (
                HalfDuplexMac::new(case.proto, 1.0),
                HalfDuplexMac::new(case.proto, 1.0),
            )
        };

        // Pipelining only helps wired (OTA half-duplex needs ACK time between bursts)
        if case.wired {
            a.set_stop_and_wait(false);
            b.set_stop_and_wait(false);
            a.set_burst_limit(8);
            b.set_burst_limit(8);
        }

        let payload: Vec<u8> = (0..case.payload_size).map(|i| (i % 256) as u8).collect();
        for _ in 0..case.num_payloads {
            a.enqueue(payload.clone());
        }
        a.poll(0);

        let mut total_samples = 0usize;
        let max_steps = 20_000;
        for _ in 0..max_steps {
            if a.is_idle() {
                break;
            }
            step(&mut a, &mut b, chunk);
            total_samples += chunk;
        }
        assert!(a.is_idle(), "{} (pipelined): transfer did not complete", case.name);

        let mut received = Vec::new();
        while let Some(p) = b.recv() {
            received.push(p);
        }
        assert_eq!(received.len(), case.num_payloads, "{} (pipelined): payload count mismatch", case.name);

        let total_bytes = case.num_payloads * case.payload_size;
        let duration_secs = total_samples as f64 / sample_rate as f64;
        let throughput_bps = (total_bytes as f64 * 8.0) / duration_secs;

        eprintln!(
            "  {:40} {:5} B x {:3} = {:6} B in {:6.2}s = {:8.0} bps ({:.1} kbps)",
            case.name,
            case.payload_size,
            case.num_payloads,
            total_bytes,
            duration_secs,
            throughput_bps,
            throughput_bps / 1000.0,
        );
    }
    eprintln!("--- End Benchmark ---\n");
}
