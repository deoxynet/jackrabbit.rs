use jackrabbit::encoder;
use jackrabbit::decoder::Decoder;
use jackrabbit::impairments;
use jackrabbit::fec;
use jackrabbit::ofdm;
use jackrabbit::crc;
use jackrabbit::params::{self, SYMBOL_TOTAL, PREAMBLE_SYMBOLS, TRAINING_SYMBOLS, HEADER_SYMBOLS};

fn main() {
    let payload = b"noise-aware LLR";
    let proto = 8; // MT_FASTEST: 8PSK 2/3

    let waveform = encoder::encode(payload, proto, 1.0);
    let config = params::get_protocol(proto);

    // Compute expected coded bits for comparison
    let with_crc = crc::append_crc(payload);
    let info_bits: Vec<u8> = with_crc.iter().flat_map(|&b| (0..8).rev().map(move |i| (b >> i) & 1)).collect();
    let expected_coded = fec::encode(&info_bits, config.code_rate);
    eprintln!("Expected coded bits: {}", expected_coded.len());
    eprintln!("Coded bits per symbol: {}", config.coded_bits_per_symbol());
    eprintln!("Info bits per symbol: {}", config.info_bits_per_symbol());
    eprintln!("Data symbols needed: {}", config.data_symbols_for(payload.len()));

    // Try different SNR values
    for snr in &[20.0, 12.0, 10.0, 8.0, 7.5, 7.0, 6.5, 6.0, 5.0] {
        let noisy = impairments::apply_awgn(&waveform, *snr, 99999);
        let mut decoder = Decoder::new();
        let result = decoder.feed(&noisy);
        let ok = result == payload;
        eprintln!("SNR={:.1} dB: decode {}", snr, if ok { "OK" } else { "FAIL" });
    }

    // Test with no noise to verify loopback
    eprintln!("\nLoopback (no noise):");
    let mut decoder = Decoder::new();
    let result = decoder.feed(&waveform);
    eprintln!("  result = {:?}", result == payload.to_vec());

    // Check coded bit errors at 6 dB
    eprintln!("\nCoded bit error analysis at 6 dB:");
    let noisy = impairments::apply_awgn(&waveform, 6.0, 99999);

    // Extract data symbol and process it manually
    let data_start = (PREAMBLE_SYMBOLS + TRAINING_SYMBOLS + HEADER_SYMBOLS) * SYMBOL_TOTAL;
    let sym = &noisy[data_start..data_start + SYMBOL_TOTAL];
    let spectrum = ofdm::fft(&sym[params::CP_LEN..]);

    // Get clean symbol for comparison
    let clean_sym = &waveform[data_start..data_start + SYMBOL_TOTAL];
    let clean_spectrum = ofdm::fft(&clean_sym[params::CP_LEN..]);

    // Check a few bins
    eprintln!("Bin 20 clean: {:?}, noisy: {:?}", clean_spectrum[20], spectrum[20]);
    eprintln!("Bin 100 clean: {:?}, noisy: {:?}", clean_spectrum[100], spectrum[100]);

    // Noise check
    let half_fft = params::FFT_SIZE / 2;
    let mut noise_sum = 0.0f32;
    let mut noise_count = 0;
    for k in 0..config.bin_start {
        noise_sum += spectrum[k].norm_sqr();
        noise_count += 1;
    }
    for k in config.bin_end..half_fft {
        noise_sum += spectrum[k].norm_sqr();
        noise_count += 1;
    }
    let noise_per_bin = noise_sum / noise_count as f32;
    eprintln!("Noise per bin: {:.2}, bins used: {}", noise_per_bin, noise_count);

    let signal_per_bin: f32 = spectrum[config.bin_start..config.bin_end]
        .iter().map(|s| s.norm_sqr()).sum::<f32>() / config.num_carriers() as f32;
    eprintln!("Signal per bin: {:.2}", signal_per_bin);
    eprintln!("Estimated SNR per bin: {:.1} dB", 10.0 * ((signal_per_bin - noise_per_bin) / noise_per_bin).log10());

    // 6 dB with many seeds to find pass rate
    eprintln!("\n6 dB pass rate with 100 seeds:");
    let mut pass = 0;
    let total = 100;
    for seed in 0..total {
        let noisy = impairments::apply_awgn(&waveform, 6.0, seed * 1000 + 42);
        let mut decoder = Decoder::new();
        let result = decoder.feed(&noisy);
        if result == payload { pass += 1; }
    }
    eprintln!("6 dB: {}/{} passed ({:.0}%)", pass, total, pass as f32 / total as f32 * 100.0);

    // 7 dB with many seeds
    eprintln!("\n7 dB pass rate with 100 seeds:");
    let mut pass7 = 0;
    for seed in 0..total {
        let noisy = impairments::apply_awgn(&waveform, 7.0, seed * 1000 + 42);
        let mut decoder = Decoder::new();
        let result = decoder.feed(&noisy);
        if result == payload { pass7 += 1; }
    }
    eprintln!("7 dB: {}/{} passed ({:.0}%)", pass7, total, pass7 as f32 / total as f32 * 100.0);

    // 8 dB
    eprintln!("\n8 dB pass rate with 100 seeds:");
    let mut pass8 = 0;
    for seed in 0..total {
        let noisy = impairments::apply_awgn(&waveform, 8.0, seed * 1000 + 42);
        let mut decoder = Decoder::new();
        let result = decoder.feed(&noisy);
        if result == payload { pass8 += 1; }
    }
    eprintln!("8 dB: {}/{} passed ({:.0}%)", pass8, total, pass8 as f32 / total as f32 * 100.0);
}
