use aqua_soundbin::arq::SelectiveRepeat;
use aqua_soundbin::params;
use aqua_soundbin::decoder::Decoder;

fn main() {
    let proto = 13u8;
    let config = params::get_protocol(proto);

    // Build 2 wired frames with gap
    let mut sender = SelectiveRepeat::new_wired(proto, 1.0, config.cp_len);
    let f0 = sender.send(b"w0").unwrap();
    let f1 = sender.send(b"w1").unwrap();
    let gap = vec![0.0f32; 9600];

    let mut stream: Vec<f32> = Vec::new();
    stream.extend_from_slice(&f0);
    stream.extend(&gap);
    stream.extend_from_slice(&f1);
    stream.extend(&gap);
    eprintln!("Stream: {} samples (f0={} + gap={} + f1={} + gap={})",
        stream.len(), f0.len(), gap.len(), f1.len(), gap.len());

    // Test 1: Feed entire stream to decoder at once
    let mut dec = Decoder::new();
    dec.set_overhead_cp(config.cp_len);
    dec.set_training_symbols(1);

    let r1 = dec.feed_frame(&stream);
    eprintln!("Feed all at once: {:?}", r1.as_ref().map(|(h,p)| (&h.seq_num, p.len())));

    // After first decode, try feeding more (empty) to trigger second decode
    for i in 0..5 {
        let r = dec.feed_frame(&[0.0; 4800]);
        if let Some((h, p)) = &r {
            eprintln!("Extra feed {}: decoded seq={} payload={}", i, h.seq_num, p.len());
        } else {
            let di = dec.debug_info();
            eprintln!("Extra feed {}: None (state={} buf={} event={})", i, di.state, di.buffer_len, di.last_event);
        }
    }

    // Test 2: Feed in chunks of 4800
    eprintln!("\n--- Chunked feed ---");
    let mut dec2 = Decoder::new();
    dec2.set_overhead_cp(config.cp_len);
    dec2.set_training_symbols(1);
    let mut decoded = 0;
    for (ci, chunk) in stream.chunks(4800).enumerate() {
        let r = dec2.feed_frame(chunk);
        if let Some((h, _p)) = &r {
            decoded += 1;
            eprintln!("Chunk {}: DECODED seq={}", ci, h.seq_num);
        }
        if ci < 15 {
            let di = dec2.debug_info();
            if di.last_event != "" || ci < 8 {
                eprintln!("  Chunk {} state: {} buf={} event={} preambles={}",
                    ci, di.state, di.buffer_len, di.last_event, di.preamble_count);
            }
        }
    }
    eprintln!("Chunked: decoded {}/2", decoded);
}
