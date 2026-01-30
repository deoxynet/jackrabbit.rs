/// OFDM system parameters and protocol configurations.

use num_complex::Complex32;

pub const SAMPLE_RATE: u32 = 48_000;
pub const FFT_SIZE: usize = 960;
pub const CP_LEN: usize = 240;
pub const CP_LEN_SHORT: usize = 120; // 1/8 of FFT_SIZE, for near-field / low-reverb
pub const CP_LEN_MINIMAL: usize = 60; // 1.25ms, zero multipath on wired connection
pub const CP_LEN_ULTRA: usize = 30;  // 0.625ms, minimal guard for clean wired links
pub const CP_LEN_CABLE: usize = 15; // 0.3125ms, direct cable (zero multipath, handles small clock offset)
pub const SYMBOL_TOTAL: usize = FFT_SIZE + CP_LEN; // 1200 samples (25ms)
pub const GUARD_BINS: usize = 20; // 0–1000 Hz unused
/// Training band covers the widest possible protocol range.
/// All protocols generate training on this fixed range so the decoder
/// can estimate the channel before knowing which protocol is in use.
pub const TRAINING_BIN_START: usize = 5;
pub const TRAINING_BIN_END: usize = 475;
pub const PREAMBLE_SYMBOLS: usize = 8;
pub const PREAMBLE_SYMBOLS_SHORT: usize = 4;
pub const PREAMBLE_SYMBOLS_MINIMAL: usize = 2; // Cable-only: high-SNR, no multipath
pub const PREAMBLE_SAMPLES: usize = PREAMBLE_SYMBOLS * SYMBOL_TOTAL; // 9600 (200ms)
pub const TRAINING_SYMBOLS: usize = 2;
pub const TRAINING_SYMBOLS_SHORT: usize = 1; // Cable-only: high-SNR, single training symbol
pub const TRAINING_SAMPLES: usize = TRAINING_SYMBOLS * SYMBOL_TOTAL; // 2400 (50ms)
pub const HEADER_SYMBOLS: usize = 1;
pub const HEADER_BINS_START: usize = 20;
pub const HEADER_BINS_END: usize = 80; // bins 20–79 (60 carriers)
pub const HEADER_INFO_BITS: usize = 18; // 5-bit proto + 12-bit length + 1-bit parity
pub const MAX_PAYLOAD: usize = 4095;
pub const RING_BUFFER_SIZE: usize = 131072; // 2^17, ~2.7s at 48kHz

/// Spacing between pilot subcarriers (in bins).
pub const PILOT_SPACING: usize = 10;
/// Offset of first pilot from bin_start.
pub const PILOT_OFFSET: usize = 5;

/// Modulation scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modulation {
    Bpsk,  // 1 bit/symbol
    Qpsk,  // 2 bits/symbol
    Psk8,  // Phase 1: stubbed as QPSK (2 bits/symbol)
    Qam16, // 4 bits/symbol — Gray-coded 4×4 constellation
    Qam64,  // 6 bits/symbol — Gray-coded 8×8 constellation
    Qam256,  // 8 bits/symbol — Gray-coded 16×16 constellation
    Qam1024, // 10 bits/symbol — Gray-coded 32×32 constellation
}

impl Modulation {
    pub fn bits_per_symbol(self) -> usize {
        match self {
            Modulation::Bpsk => 1,
            Modulation::Qpsk => 2,
            Modulation::Psk8 => 2, // Phase 1 stub: operates as QPSK
            Modulation::Qam16 => 4,
            Modulation::Qam64 => 6,
            Modulation::Qam256 => 8,
            Modulation::Qam1024 => 10,
        }
    }
}

/// FEC implementation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FecType {
    Convolutional,
    Ldpc,
}

/// FEC code rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeRate {
    Half,        // rate 1/2
    TwoThird,    // rate 2/3 (punctured from 1/2)
    ThreeFourth, // rate 3/4 (punctured from 1/2)
    FiveSixth,   // rate 5/6 (punctured from 1/2, or native LDPC)
}

impl CodeRate {
    /// Output bits per input bit.
    pub fn ratio(self) -> (usize, usize) {
        match self {
            CodeRate::Half => (1, 2),
            CodeRate::TwoThird => (2, 3),
            CodeRate::ThreeFourth => (3, 4),
            CodeRate::FiveSixth => (5, 6),
        }
    }
}

/// Protocol configuration.
#[derive(Debug, Clone)]
pub struct ProtocolConfig {
    pub id: u8,
    pub name: &'static str,
    pub bin_start: usize,  // first used subcarrier bin
    pub bin_end: usize,    // one past last used subcarrier bin
    pub modulation: Modulation,
    pub code_rate: CodeRate,
    pub fec_type: FecType,
    pub cp_len: usize,     // cyclic prefix length (default CP_LEN=240)
    pub pilot_spacing: usize, // spacing between pilot subcarriers (default PILOT_SPACING=10)
}

impl ProtocolConfig {
    pub fn num_carriers(&self) -> usize {
        self.bin_end - self.bin_start
    }

    /// Total samples per OFDM symbol for this protocol (FFT + CP).
    pub fn symbol_total(&self) -> usize {
        FFT_SIZE + self.cp_len
    }

    /// Coded bits per OFDM data symbol for this protocol.
    /// Uses data carriers only (excludes pilot subcarriers).
    pub fn coded_bits_per_symbol(&self) -> usize {
        self.num_data_carriers() * self.modulation.bits_per_symbol()
    }

    /// Info bits per OFDM data symbol (after FEC).
    pub fn info_bits_per_symbol(&self) -> usize {
        let coded = self.coded_bits_per_symbol();
        let (k, n) = self.code_rate.ratio();
        coded * k / n
    }

    /// Total coded bits produced for a given payload length (bytes), including CRC-16.
    pub fn coded_bits_for(&self, payload_len: usize) -> usize {
        let info_bits = (payload_len + 2) * 8; // +2 for CRC-16
        match self.fec_type {
            FecType::Convolutional => {
                let tail_bits = 6; // K-1 for conv encoder flush
                // conv_encode produces (info_bits + tail_bits) * 2 rate-1/2 coded bits
                let rate_half_coded = (info_bits + tail_bits) * 2;
                match self.code_rate {
                    CodeRate::Half => rate_half_coded,
                    // puncture keeps 3 out of every 4 bits
                    CodeRate::TwoThird => {
                        let full_groups = rate_half_coded / 4;
                        let remainder = rate_half_coded % 4;
                        full_groups * 3 + remainder.min(3)
                    }
                    // puncture pattern [1,1,0,1,1,0]: keeps 4 out of every 6 bits
                    CodeRate::ThreeFourth => {
                        let full_groups = rate_half_coded / 6;
                        let remainder = rate_half_coded % 6;
                        let pattern = [true, true, false, true, true, false];
                        let kept_in_rem = pattern[..remainder].iter().filter(|&&x| x).count();
                        full_groups * 4 + kept_in_rem
                    }
                    // puncture pattern [1,1,1,0,0,1,0,0,1,1]: keeps 6 out of every 10 bits
                    CodeRate::FiveSixth => {
                        let pattern = [true, true, true, false, false, true, false, false, true, true];
                        let full_groups = rate_half_coded / 10;
                        let remainder = rate_half_coded % 10;
                        let kept_in_rem = pattern[..remainder].iter().filter(|&&x| x).count();
                        full_groups * 6 + kept_in_rem
                    }
                }
            }
            FecType::Ldpc => {
                // LDPC block coding: ceil(info_bits / k) * n
                let (k, n) = crate::ldpc::ldpc_block_params(self.code_rate);
                let num_blocks = (info_bits + k - 1) / k;
                num_blocks * n
            }
        }
    }

    /// Number of data symbols needed for a given payload length (bytes), including CRC-16.
    /// Accounts for FEC tail bits (K-1 = 6 zeros to flush the convolutional encoder).
    pub fn data_symbols_for(&self, payload_len: usize) -> usize {
        let coded_bits = self.coded_bits_for(payload_len);
        let coded_per_sym = self.coded_bits_per_symbol();
        (coded_bits + coded_per_sym - 1) / coded_per_sym
    }

    /// Pilot subcarrier bin indices for this protocol.
    ///
    /// Pilots are evenly spaced across the carrier range, carrying known
    /// BPSK values that enable per-symbol channel tracking.
    pub fn pilot_bins(&self) -> Vec<usize> {
        let mut pilots = Vec::new();
        let mut bin = self.bin_start + PILOT_OFFSET;
        while bin < self.bin_end {
            pilots.push(bin);
            bin += self.pilot_spacing;
        }
        pilots
    }

    /// Number of data-carrying subcarriers (total minus pilots).
    pub fn num_data_carriers(&self) -> usize {
        self.num_carriers() - self.pilot_bins().len()
    }

    /// Total samples for a packet with the given payload length.
    /// Preamble, training, and header always use the standard CP (SYMBOL_TOTAL).
    /// Data symbols use this protocol's cp_len.
    pub fn total_samples_for(&self, payload_len: usize) -> usize {
        self.total_samples_for_full(payload_len, PREAMBLE_SYMBOLS, CP_LEN)
    }

    /// Total samples with configurable preamble count and overhead CP length.
    pub fn total_samples_for_full(
        &self,
        payload_len: usize,
        preamble_symbols: usize,
        overhead_cp_len: usize,
    ) -> usize {
        self.total_samples_for_max(payload_len, preamble_symbols, overhead_cp_len, TRAINING_SYMBOLS)
    }

    /// Total samples with all configurable parameters including training symbols.
    pub fn total_samples_for_max(
        &self,
        payload_len: usize,
        preamble_symbols: usize,
        overhead_cp_len: usize,
        training_symbols: usize,
    ) -> usize {
        let n_data = self.data_symbols_for(payload_len);
        let overhead_sym_total = FFT_SIZE + overhead_cp_len;
        let fixed_symbols = preamble_symbols + training_symbols + HEADER_SYMBOLS;
        fixed_symbols * overhead_sym_total + n_data * self.symbol_total()
    }
}

/// All protocol configurations.
pub static PROTOCOLS: [ProtocolConfig; 17] = [
    // Audible: 1–4 kHz → bins 20–80 (60 carriers)
    ProtocolConfig { id: 0, name: "AUDIBLE_NORMAL",  bin_start: 20, bin_end: 80,  modulation: Modulation::Bpsk, code_rate: CodeRate::Half,        fec_type: FecType::Convolutional, cp_len: CP_LEN, pilot_spacing: PILOT_SPACING },
    ProtocolConfig { id: 1, name: "AUDIBLE_FAST",    bin_start: 20, bin_end: 80,  modulation: Modulation::Qpsk, code_rate: CodeRate::Half,        fec_type: FecType::Convolutional, cp_len: CP_LEN, pilot_spacing: PILOT_SPACING },
    ProtocolConfig { id: 2, name: "AUDIBLE_FASTEST", bin_start: 20, bin_end: 80,  modulation: Modulation::Qpsk, code_rate: CodeRate::TwoThird,    fec_type: FecType::Convolutional, cp_len: CP_LEN, pilot_spacing: PILOT_SPACING },
    // Data Transfer: 1–6 kHz → bins 20–120 (100 carriers)
    ProtocolConfig { id: 3, name: "DT_NORMAL",       bin_start: 20, bin_end: 120, modulation: Modulation::Bpsk, code_rate: CodeRate::Half,        fec_type: FecType::Convolutional, cp_len: CP_LEN, pilot_spacing: PILOT_SPACING },
    ProtocolConfig { id: 4, name: "DT_FAST",         bin_start: 20, bin_end: 120, modulation: Modulation::Qpsk, code_rate: CodeRate::Half,        fec_type: FecType::Convolutional, cp_len: CP_LEN, pilot_spacing: PILOT_SPACING },
    ProtocolConfig { id: 5, name: "DT_FASTEST",      bin_start: 20, bin_end: 120, modulation: Modulation::Qpsk, code_rate: CodeRate::TwoThird,    fec_type: FecType::Convolutional, cp_len: CP_LEN, pilot_spacing: PILOT_SPACING },
    // Multi-Tone: 1–8 kHz → bins 20–160 (140 carriers)
    ProtocolConfig { id: 6, name: "MT_NORMAL",        bin_start: 20, bin_end: 160, modulation: Modulation::Bpsk, code_rate: CodeRate::Half,        fec_type: FecType::Convolutional, cp_len: CP_LEN, pilot_spacing: PILOT_SPACING },
    ProtocolConfig { id: 7, name: "MT_FAST",          bin_start: 20, bin_end: 160, modulation: Modulation::Qpsk, code_rate: CodeRate::TwoThird,    fec_type: FecType::Convolutional, cp_len: CP_LEN, pilot_spacing: PILOT_SPACING },
    ProtocolConfig { id: 8, name: "WIRED_FAST",        bin_start: 20, bin_end: 320, modulation: Modulation::Qam64, code_rate: CodeRate::ThreeFourth, fec_type: FecType::Ldpc, cp_len: CP_LEN_MINIMAL, pilot_spacing: 15 },
    // Ultra-Tone: 1–16 kHz → bins 20–320 (300 carriers)
    ProtocolConfig { id: 9,  name: "UT_NORMAL",       bin_start: 20, bin_end: 320, modulation: Modulation::Bpsk,  code_rate: CodeRate::Half,        fec_type: FecType::Convolutional, cp_len: CP_LEN, pilot_spacing: PILOT_SPACING },
    ProtocolConfig { id: 10, name: "UT_FAST",         bin_start: 20, bin_end: 320, modulation: Modulation::Qpsk,  code_rate: CodeRate::TwoThird,    fec_type: FecType::Convolutional, cp_len: CP_LEN, pilot_spacing: PILOT_SPACING },
    ProtocolConfig { id: 11, name: "UT_FASTEST",      bin_start: 20, bin_end: 320, modulation: Modulation::Qam16, code_rate: CodeRate::ThreeFourth, fec_type: FecType::Convolutional, cp_len: CP_LEN_SHORT, pilot_spacing: PILOT_SPACING },
    // LDPC protocols
    ProtocolConfig { id: 12, name: "DT_LDPC",         bin_start: 20, bin_end: 120, modulation: Modulation::Qpsk,  code_rate: CodeRate::Half,        fec_type: FecType::Ldpc, cp_len: CP_LEN, pilot_spacing: PILOT_SPACING },
    ProtocolConfig { id: 13, name: "WIRED_TURBO",      bin_start: 20, bin_end: 400, modulation: Modulation::Qam256, code_rate: CodeRate::ThreeFourth, fec_type: FecType::Ldpc, cp_len: CP_LEN_ULTRA, pilot_spacing: 30 },
    ProtocolConfig { id: 14, name: "WIRED_MAX",        bin_start: 20, bin_end: 400, modulation: Modulation::Qam64, code_rate: CodeRate::ThreeFourth, fec_type: FecType::Ldpc, cp_len: CP_LEN_MINIMAL, pilot_spacing: 15 },
    // OTA-robust: audible band, BPSK 1/2 LDPC, full CP, dense pilots
    ProtocolConfig { id: 15, name: "OTA_ROBUST",      bin_start: 20, bin_end: 80,  modulation: Modulation::Bpsk,  code_rate: CodeRate::Half,        fec_type: FecType::Ldpc, cp_len: CP_LEN, pilot_spacing: 5 },
    // Wired extreme: 3.5mm cable, max throughput. 1024QAM, rate 5/6 LDPC, near-full bandwidth, minimal CP/pilots.
    ProtocolConfig { id: 16, name: "WIRED_EXTREME",   bin_start: 5,  bin_end: 475, modulation: Modulation::Qam1024, code_rate: CodeRate::FiveSixth,  fec_type: FecType::Ldpc, cp_len: CP_LEN_CABLE, pilot_spacing: 60 },
];

pub fn get_protocol(id: u8) -> &'static ProtocolConfig {
    &PROTOCOLS[id as usize]
}

/// Generate the known pilot symbol values for a given number of pilots.
/// Uses BPSK +1 for all pilots (known reference).
pub fn pilot_symbols(n: usize) -> Vec<Complex32> {
    vec![Complex32::new(1.0, 0.0); n]
}

/// Merge data symbols and pilot symbols into a full carrier vector.
///
/// `data` has `num_data_carriers` entries; `pilots` has `pilot_bins.len()` entries.
/// Returns a vector of `num_carriers` entries with pilots inserted at their positions.
pub fn merge_data_pilots(
    data: &[Complex32],
    pilots: &[Complex32],
    pilot_bins: &[usize],
    bin_start: usize,
    num_carriers: usize,
) -> Vec<Complex32> {
    let mut result = Vec::with_capacity(num_carriers);
    let mut d_idx = 0;
    let mut p_idx = 0;
    for bin in bin_start..bin_start + num_carriers {
        if p_idx < pilot_bins.len() && bin == pilot_bins[p_idx] {
            result.push(pilots[p_idx]);
            p_idx += 1;
        } else {
            result.push(data[d_idx]);
            d_idx += 1;
        }
    }
    result
}

/// Extract data symbols from a full carrier vector, excluding pilots.
pub fn extract_data_bins(
    all_bins: &[Complex32],
    pilot_bins: &[usize],
    bin_start: usize,
    num_carriers: usize,
) -> Vec<Complex32> {
    let mut data = Vec::with_capacity(num_carriers - pilot_bins.len());
    let mut p_idx = 0;
    for (i, &val) in all_bins.iter().enumerate() {
        let bin = bin_start + i;
        if p_idx < pilot_bins.len() && bin == pilot_bins[p_idx] {
            p_idx += 1;
        } else {
            data.push(val);
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_third_padding_case() {
        let config = &PROTOCOLS[2]; // QPSK 2/3, 60 carriers
        let payload_len = 1;
        let coded_bits = config.coded_bits_for(payload_len);
        let coded_per_sym = config.coded_bits_per_symbol();
        let total_bits = config.data_symbols_for(payload_len) * coded_per_sym;

        assert!(coded_bits < coded_per_sym);
        assert!(total_bits > coded_bits);
    }
}
