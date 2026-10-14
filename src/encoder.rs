/// Full TX chain: data → FEC → interleave → modulate → IFFT → CP → assemble.

use crate::crc;
use crate::fec;
use crate::header::{self, HeaderInfo, FrameType};
use crate::interleaver;
use crate::ldpc;
use crate::modulation;
use crate::ofdm;
use crate::params;
use crate::preamble;

/// Encode a payload into an OFDM waveform.
///
/// Returns f32 samples at 48kHz sample rate.
pub fn encode(payload: &[u8], protocol_id: u8, volume: f32) -> Vec<f32> {
    encode_full(payload, protocol_id, volume, params::PREAMBLE_SYMBOLS, params::CP_LEN)
}

/// Encode with a configurable preamble length (in OFDM symbols).
pub fn encode_with_preamble_symbols(
    payload: &[u8],
    protocol_id: u8,
    volume: f32,
    preamble_symbols: usize,
) -> Vec<f32> {
    encode_full(payload, protocol_id, volume, preamble_symbols, params::CP_LEN)
}

/// Encode with configurable preamble length and overhead CP length.
pub fn encode_full(
    payload: &[u8],
    protocol_id: u8,
    volume: f32,
    preamble_symbols: usize,
    overhead_cp_len: usize,
) -> Vec<f32> {
    encode_max(payload, protocol_id, volume, preamble_symbols, overhead_cp_len, params::TRAINING_SYMBOLS)
}

/// Encode with all configurable parameters: preamble count, overhead CP, training symbols.
///
/// `overhead_cp_len` controls the cyclic prefix for preamble, training, and
/// header symbols. Data symbols always use the protocol's own CP.
/// `training_symbols`: 1 or 2 training symbols (1 for cable, 2 for OTA/default).
pub fn encode_max(
    payload: &[u8],
    protocol_id: u8,
    volume: f32,
    preamble_symbols: usize,
    overhead_cp_len: usize,
    training_symbols: usize,
) -> Vec<f32> {
    let header_info = HeaderInfo {
        frame_type: FrameType::Data,
        seq_num: 0,
        protocol_id,
        payload_len: payload.len(),
    };
    encode_inner(payload, &header_info, volume, preamble_symbols, overhead_cp_len, training_symbols)
}

/// Encode with explicit header metadata (frame type, sequence number, etc).
pub fn encode_with_header(
    payload: &[u8],
    header_info: &HeaderInfo,
    volume: f32,
    preamble_symbols: usize,
    overhead_cp_len: usize,
    training_symbols: usize,
) -> Vec<f32> {
    encode_inner(payload, header_info, volume, preamble_symbols, overhead_cp_len, training_symbols)
}

/// Encode with explicit header metadata and an optional data CP override.
pub fn encode_with_header_data_cp(
    payload: &[u8],
    header_info: &HeaderInfo,
    volume: f32,
    preamble_symbols: usize,
    overhead_cp_len: usize,
    training_symbols: usize,
    data_cp_len: Option<usize>,
) -> Vec<f32> {
    encode_inner_bitloaded(
        payload,
        header_info,
        volume,
        preamble_symbols,
        overhead_cp_len,
        training_symbols,
        data_cp_len,
        None,
    )
}

/// Encode with a bitloading profile for per-band modulation.
///
/// When `profile` is None or uniform, behaves identically to `encode_max`.
/// When `profile` specifies varying modulation per band, each band's data
/// carriers use that band's QAM order.
pub fn encode_bitloaded(
    payload: &[u8],
    header_info: &HeaderInfo,
    volume: f32,
    preamble_symbols: usize,
    overhead_cp_len: usize,
    training_symbols: usize,
    profile: &params::BitloadProfile,
) -> Vec<f32> {
    encode_inner_bitloaded(
        payload, header_info, volume, preamble_symbols,
        overhead_cp_len, training_symbols, None, Some(profile),
    )
}

fn encode_inner(
    payload: &[u8],
    header_info: &HeaderInfo,
    volume: f32,
    preamble_symbols: usize,
    overhead_cp_len: usize,
    training_symbols: usize,
) -> Vec<f32> {
    encode_inner_bitloaded(
        payload, header_info, volume, preamble_symbols,
        overhead_cp_len, training_symbols, None, None,
    )
}

/// Encode with a bitloading profile and an optional data CP override.
pub fn encode_bitloaded_with_data_cp(
    payload: &[u8],
    header_info: &HeaderInfo,
    volume: f32,
    preamble_symbols: usize,
    overhead_cp_len: usize,
    training_symbols: usize,
    data_cp_len: Option<usize>,
    profile: &params::BitloadProfile,
) -> Vec<f32> {
    encode_inner_bitloaded(
        payload,
        header_info,
        volume,
        preamble_symbols,
        overhead_cp_len,
        training_symbols,
        data_cp_len,
        Some(profile),
    )
}

fn encode_inner_bitloaded(
    payload: &[u8],
    header_info: &HeaderInfo,
    volume: f32,
    preamble_symbols: usize,
    overhead_cp_len: usize,
    training_symbols: usize,
    data_cp_len: Option<usize>,
    profile: Option<&params::BitloadProfile>,
) -> Vec<f32> {
    if payload.is_empty() || payload.len() > params::MAX_PAYLOAD {
        return Vec::new();
    }
    let protocol_id = header_info.protocol_id;
    if protocol_id as usize >= params::PROTOCOLS.len() {
        return Vec::new();
    }
    assert!(preamble_symbols > 0, "Preamble must have at least 1 symbol");
    if preamble_symbols != params::PREAMBLE_SYMBOLS
        && preamble_symbols != params::PREAMBLE_SYMBOLS_SHORT
        && preamble_symbols != params::PREAMBLE_SYMBOLS_MINIMAL
    {
        return Vec::new();
    }

    let config = params::get_protocol(protocol_id);

    // 1. CRC-16 append
    let with_crc = crc::append_crc(payload);

    // 2. Bit-pack payload → bit stream
    let bit_stream = bytes_to_bits(&with_crc);

    // 3. Append tail bits (K-1 = 6 zeros to flush encoder)
    // (conv_encode handles this internally, but we need to account for them in symbol packing)

    // 4. FEC encode at protocol's rate
    let coded_bits = match config.fec_type {
        params::FecType::Convolutional => fec::encode(&bit_stream, config.code_rate),
        params::FecType::Ldpc => {
            let code = ldpc::get_ldpc_code(config.code_rate);
            ldpc::ldpc_encode_payload(&bit_stream, code)
        }
    };

    // 5. Partition coded bits into N symbols
    let num_data_carriers = config.num_data_carriers();
    let coded_bits_per_sym = match profile {
        Some(p) => p.coded_bits_per_symbol(num_data_carriers),
        None => config.coded_bits_per_symbol(),
    };
    if coded_bits_per_sym == 0 {
        return Vec::new();
    }
    let n_data_symbols = (coded_bits.len() + coded_bits_per_sym - 1) / coded_bits_per_sym;

    // Pad coded bits to fill complete symbols
    let mut padded_coded = coded_bits;
    padded_coded.resize(n_data_symbols * coded_bits_per_sym, 0);

    // 6. Generate preamble (using overhead CP)
    let preamble_waveform = preamble::generate_preamble_symbols_with_cp(preamble_symbols, overhead_cp_len);

    // 7. Generate training symbols (using overhead CP, configurable count)
    assert!(training_symbols >= 1 && training_symbols <= 2, "Training symbols must be 1 or 2");
    let (training_waveform, _) = preamble::generate_training_full(
        params::TRAINING_BIN_START,
        params::TRAINING_BIN_END,
        overhead_cp_len,
        training_symbols,
    );

    // 8. Generate header symbol (using overhead CP)
    let header_waveform = header::encode_header_symbol_with_cp(header_info, overhead_cp_len);

    // 9. Generate data symbols (using per-protocol CP length)
    let data_cp_len = data_cp_len.unwrap_or(config.cp_len);
    let data_sym_total = params::FFT_SIZE + data_cp_len;
    let mut data_waveform = Vec::with_capacity(n_data_symbols * data_sym_total);

    let pilot_bins = config.pilot_bins();
    let pilot_syms = params::pilot_symbols(pilot_bins.len());
    // Differential encoding is used for PSK modulations (BPSK, QPSK, PSK8)
    // where only phase carries information. QAM modulations need coherent
    // detection (amplitude carries info). When bitloading is active, we must
    // skip differential encoding because:
    //   1. QAM bands would lose amplitude info (diff encode normalizes to unit circle)
    //   2. Null bands would poison subsequent carriers (zero propagates through prev)
    let use_differential = profile.is_none()
        && config.modulation != params::Modulation::Qam16
        && config.modulation != params::Modulation::Qam64
        && config.modulation != params::Modulation::Qam256;

    for sym_idx in 0..n_data_symbols {
        let start = sym_idx * coded_bits_per_sym;
        let end = start + coded_bits_per_sym;
        let sym_coded_bits = &padded_coded[start..end];

        // Per-symbol processing: interleave → modulate → insert pilots → [differential encode] → OFDM
        let interleaved = interleaver::interleave(sym_coded_bits, interleaver::symbol_seed(sym_idx));
        let modulated = match profile {
            Some(p) => modulation::modulate_bitloaded(&interleaved, p, num_data_carriers),
            None => modulation::modulate(&interleaved, config.modulation),
        };
        let full_carriers = params::merge_data_pilots(
            &modulated, &pilot_syms, &pilot_bins,
            config.bin_start, config.num_carriers(),
        );
        let tx_carriers = if use_differential {
            modulation::differential_encode(&full_carriers)
        } else {
            full_carriers
        };
        let symbol_samples = ofdm::synthesize_symbol_cp(
            &tx_carriers,
            config.bin_start,
            config.bin_end,
            data_cp_len,
        );
        data_waveform.extend_from_slice(&symbol_samples);
    }

    // 10. Assemble: preamble ++ training ++ header ++ data
    let total_len = preamble_waveform.len() + training_waveform.len()
        + header_waveform.len() + data_waveform.len();
    let mut waveform = Vec::with_capacity(total_len);
    waveform.extend_from_slice(&preamble_waveform);
    waveform.extend_from_slice(&training_waveform);
    waveform.extend_from_slice(&header_waveform);
    waveform.extend_from_slice(&data_waveform);

    // 11. Scale by volume and normalize peak to 0.95
    let peak = waveform.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 {
        let scale = volume * 0.95 / peak;
        for s in &mut waveform {
            *s *= scale;
        }
    }

    waveform
}

/// Convert bytes to bits (MSB first).
fn bytes_to_bits(data: &[u8]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(data.len() * 8);
    for &byte in data {
        for i in (0..8).rev() {
            bits.push((byte >> i) & 1);
        }
    }
    bits
}

/// Convert bits to bytes (MSB first).
pub fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity((bits.len() + 7) / 8);
    for chunk in bits.chunks(8) {
        let mut byte = 0u8;
        for (i, &bit) in chunk.iter().enumerate() {
            byte |= bit << (7 - i);
        }
        bytes.push(byte);
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_bits_roundtrip() {
        let data = vec![0xA5, 0x3C, 0xFF, 0x00];
        let bits = bytes_to_bits(&data);
        assert_eq!(bits.len(), 32);
        let recovered = bits_to_bytes(&bits);
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_encode_produces_waveform() {
        let payload = b"Hello";
        let waveform = encode(payload, 0, 1.0);

        // Should have preamble + training + header + data symbols
        let config = params::get_protocol(0);
        let expected_samples = config.total_samples_for(payload.len());
        assert_eq!(waveform.len(), expected_samples);

        // Peak should be ≤ 0.95
        let peak = waveform.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(peak <= 0.96, "Peak too high: {}", peak);
        assert!(peak > 0.5, "Peak too low: {}", peak);
    }

    #[test]
    fn test_encode_all_protocols() {
        let payload = b"test";
        for proto in 0..params::PROTOCOLS.len() as u8 {
            let waveform = encode(payload, proto, 1.0);
            assert!(!waveform.is_empty(), "Protocol {} produced empty waveform", proto);
            // All samples should be finite
            for &s in &waveform {
                assert!(s.is_finite(), "Protocol {} produced non-finite sample", proto);
            }
        }
    }

    #[test]
    fn test_encode_short_preamble_length() {
        let payload = b"short preamble";
        let proto = 0;
        let config = params::get_protocol(proto);
        let preamble_symbols = params::PREAMBLE_SYMBOLS_SHORT;
        let waveform = encode_with_preamble_symbols(payload, proto, 1.0, preamble_symbols);

        let fixed_symbols = preamble_symbols + params::TRAINING_SYMBOLS + params::HEADER_SYMBOLS;
        let n_data = config.data_symbols_for(payload.len());
        let expected = fixed_symbols * params::SYMBOL_TOTAL + n_data * config.symbol_total();
        assert_eq!(waveform.len(), expected);
    }

    #[test]
    fn test_encode_single_training_symbol() {
        let payload = b"single training";
        let proto = 13u8; // WIRED_TURBO
        let config = params::get_protocol(proto);
        let overhead_cp = config.cp_len;
        let preamble_symbols = params::PREAMBLE_SYMBOLS_MINIMAL;

        let waveform_2ts = encode_max(payload, proto, 1.0, preamble_symbols, overhead_cp, 2);
        let waveform_1ts = encode_max(payload, proto, 1.0, preamble_symbols, overhead_cp, 1);

        // Single training should be one overhead symbol shorter
        let overhead_sym = params::FFT_SIZE + overhead_cp;
        assert_eq!(
            waveform_2ts.len() - waveform_1ts.len(), overhead_sym,
            "Single training should save exactly one overhead symbol"
        );
    }

    #[test]
    fn test_encode_with_overhead_cp() {
        let payload = b"overhead cp test";
        let proto = 13; // WIRED_TURBO, cp_len=30
        let config = params::get_protocol(proto);
        let overhead_cp = config.cp_len; // Use protocol's CP for overhead too
        let preamble_symbols = params::PREAMBLE_SYMBOLS_SHORT;
        let waveform = encode_full(payload, proto, 1.0, preamble_symbols, overhead_cp);

        let overhead_sym_total = params::FFT_SIZE + overhead_cp;
        let fixed_symbols = preamble_symbols + params::TRAINING_SYMBOLS + params::HEADER_SYMBOLS;
        let n_data = config.data_symbols_for(payload.len());
        let expected = fixed_symbols * overhead_sym_total + n_data * config.symbol_total();
        assert_eq!(waveform.len(), expected, "Waveform length mismatch with overhead CP={}", overhead_cp);

        // Should be shorter than using default CP
        let default_waveform = encode_with_preamble_symbols(payload, proto, 1.0, preamble_symbols);
        assert!(waveform.len() < default_waveform.len(), "overhead CP should produce shorter waveform");
    }
}
