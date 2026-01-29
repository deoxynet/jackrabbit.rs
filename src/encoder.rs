/// Full TX chain: data → FEC → interleave → modulate → IFFT → CP → assemble.

use crate::crc;
use crate::fec;
use crate::header;
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
    encode_with_preamble_symbols(payload, protocol_id, volume, params::PREAMBLE_SYMBOLS)
}

/// Encode with a configurable preamble length (in OFDM symbols).
pub fn encode_with_preamble_symbols(
    payload: &[u8],
    protocol_id: u8,
    volume: f32,
    preamble_symbols: usize,
) -> Vec<f32> {
    assert!(!payload.is_empty(), "Payload must not be empty");
    assert!(payload.len() <= params::MAX_PAYLOAD, "Payload too large");
    if protocol_id as usize >= params::PROTOCOLS.len() {
        return Vec::new();
    }
    assert!(preamble_symbols > 0, "Preamble must have at least 1 symbol");
    if preamble_symbols != params::PREAMBLE_SYMBOLS
        && preamble_symbols != params::PREAMBLE_SYMBOLS_SHORT
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
    let coded_bits_per_sym = config.coded_bits_per_symbol();
    let n_data_symbols = (coded_bits.len() + coded_bits_per_sym - 1) / coded_bits_per_sym;

    // Pad coded bits to fill complete symbols
    let mut padded_coded = coded_bits;
    padded_coded.resize(n_data_symbols * coded_bits_per_sym, 0);

    // 6. Generate preamble
    let preamble_waveform = preamble::generate_preamble_symbols(preamble_symbols);

    // 7. Generate training symbols
    let (training_waveform, _) = preamble::generate_training(config.bin_start, config.bin_end);

    // 8. Generate header symbol
    let header_waveform = header::encode_header_symbol(protocol_id, payload.len());

    // 9. Generate data symbols (using per-protocol CP length)
    let data_sym_total = config.symbol_total();
    let mut data_waveform = Vec::with_capacity(n_data_symbols * data_sym_total);

    let pilot_bins = config.pilot_bins();
    let pilot_syms = params::pilot_symbols(pilot_bins.len());
    let use_differential = config.modulation != params::Modulation::Qam16
        && config.modulation != params::Modulation::Qam64
        && config.modulation != params::Modulation::Qam256;

    for sym_idx in 0..n_data_symbols {
        let start = sym_idx * coded_bits_per_sym;
        let end = start + coded_bits_per_sym;
        let sym_coded_bits = &padded_coded[start..end];

        // Per-symbol processing: interleave → modulate → insert pilots → [differential encode] → OFDM
        let interleaved = interleaver::interleave(sym_coded_bits, interleaver::symbol_seed(sym_idx));
        let modulated = modulation::modulate(&interleaved, config.modulation);
        let full_carriers = params::merge_data_pilots(
            &modulated, &pilot_syms, &pilot_bins,
            config.bin_start, config.num_carriers(),
        );
        // 16QAM uses coherent detection (amplitude carries info), skip differential
        let tx_carriers = if use_differential {
            modulation::differential_encode(&full_carriers)
        } else {
            full_carriers
        };
        let symbol_samples = ofdm::synthesize_symbol_cp(&tx_carriers, config.bin_start, config.bin_end, config.cp_len);
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
}
