use aqua_soundbin::params;

fn main() {
    for &(proto_id, payload, preamble, cp, training) in &[
        (0u8, 50usize, params::PREAMBLE_SYMBOLS, params::CP_LEN, params::TRAINING_SYMBOLS),
        (4, 200, params::PREAMBLE_SYMBOLS, params::CP_LEN, params::TRAINING_SYMBOLS),
        (11, 500, params::PREAMBLE_SYMBOLS, params::CP_LEN, params::TRAINING_SYMBOLS),
        (11, 1200, params::PREAMBLE_SYMBOLS, params::CP_LEN, params::TRAINING_SYMBOLS),
        (13, 1000, params::PREAMBLE_SYMBOLS_MINIMAL, params::CP_LEN_ULTRA, 1),
        (13, 2000, params::PREAMBLE_SYMBOLS_MINIMAL, params::CP_LEN_ULTRA, 1),
        (16, 2000, params::PREAMBLE_SYMBOLS_MINIMAL, params::CP_LEN_CABLE, 1),
    ] {
        let config = params::get_protocol(proto_id);
        let total = config.total_samples_for_max(payload, preamble, cp, training);
        let data_syms = config.data_symbols_for(payload);
        eprintln!("{:20} {:5}B payload: {} samples ({:.0}ms), {} data syms, sym_total={}",
            config.name, payload, total, total as f64 / 48.0, data_syms, config.symbol_total());
    }
}
