/// Stateful streaming RX decoder.
///
/// State machine:
/// IDLE → preamble detected → COLLECT_TRAINING → COLLECT_HEADER → COLLECT_DATA → done.
///
/// All incoming samples are buffered. The Schmidl-Cox detector runs over the
/// buffer to find the preamble. Once found, we know where training starts and
/// process the remaining symbols sequentially.

use num_complex::Complex32;

use crate::adaptive::{self, ChannelReport};
use crate::channel;
use crate::crc;
use crate::encoder;
use crate::fec;
use crate::header;
use crate::interleaver;
use crate::ldpc;
use crate::modulation;
use crate::ofdm;
use crate::params::{self, ProtocolConfig, SYMBOL_TOTAL, TRAINING_SYMBOLS};
use crate::preamble;

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Idle,
    CollectTraining,
    CollectHeader,
    CollectData,
}

/// Diagnostic info for debugging real-time decode issues.
#[derive(Debug, Clone)]
pub struct DecoderDebugInfo {
    pub state: &'static str,
    pub buffer_len: usize,
    pub total_samples_fed: u64,
    pub max_amplitude: f32,
    pub rms_amplitude: f32,
    /// Last significant event: "preamble", "header_ok", "header_fail",
    /// "crc_ok", "crc_fail", or "" if none yet.
    pub last_event: &'static str,
    /// Counter of preamble detections (including false ones).
    pub preamble_count: u32,
}

pub struct Decoder {
    state: State,
    /// All samples received, indexed by position
    buffer: Vec<f32>,
    /// Current read position in buffer
    read_pos: usize,
    detector: preamble::PreambleDetector,

    // After CFO estimation
    cfo_hz: f32,
    fractional_cfo_hz: f32,
    cfo_phase: f32,
    /// Integer CFO in bins (detected from training correlation)
    integer_cfo_bins: i32,

    // After training
    channel_estimate: Vec<Complex32>,
    protocol_config: Option<&'static ProtocolConfig>,

    // After header
    payload_len: usize,
    n_data_symbols: usize,

    // Per-symbol phase advance from residual CFO (estimated from training pair)
    phase_delta: f32,

    // During data collection
    data_soft_bits: Vec<f32>,
    data_symbols_collected: usize,

    // Noise variance accumulation for channel quality reporting
    noise_var_sum: f32,
    noise_var_count: usize,

    // Channel quality report (populated after successful decode)
    last_channel_report: Option<ChannelReport>,

    // Diagnostics
    total_samples_fed: u64,
    recent_max_amplitude: f32,
    recent_rms_sum: f64,
    recent_rms_count: u64,
    last_event: &'static str,
    preamble_count: u32,
}

impl Decoder {
    pub fn new() -> Self {
        Decoder {
            state: State::Idle,
            buffer: Vec::new(),
            read_pos: 0,
            detector: preamble::PreambleDetector::new(),
            cfo_hz: 0.0,
            fractional_cfo_hz: 0.0,
            cfo_phase: 0.0,
            integer_cfo_bins: 0,
            channel_estimate: Vec::new(),
            protocol_config: None,
            payload_len: 0,
            n_data_symbols: 0,
            phase_delta: 0.0,
            data_soft_bits: Vec::new(),
            data_symbols_collected: 0,
            noise_var_sum: 0.0,
            noise_var_count: 0,
            last_channel_report: None,
            total_samples_fed: 0,
            recent_max_amplitude: 0.0,
            recent_rms_sum: 0.0,
            recent_rms_count: 0,
            last_event: "",
            preamble_count: 0,
        }
    }

    fn data_symbol_phase(phase_delta: f32, data_symbol_index: usize, data_symbol_len: usize) -> f32 {
        let phase_per_sample = phase_delta / SYMBOL_TOTAL as f32;
        let base_samples = (TRAINING_SYMBOLS + params::HEADER_SYMBOLS) as f32 * SYMBOL_TOTAL as f32;
        let data_samples = data_symbol_index as f32 * data_symbol_len as f32;
        (base_samples + data_samples) * phase_per_sample
    }

    pub fn reset(&mut self) {
        self.state = State::Idle;
        self.buffer.clear();
        self.read_pos = 0;
        self.detector.reset();
        self.reset_decode_state();
    }

    /// Reset after a failed decode attempt, preserving unprocessed samples.
    ///
    /// Instead of clearing the entire buffer (which discards any real signal
    /// that arrived during a false detection), this drains only the samples
    /// that were already consumed and lets the preamble detector re-scan
    /// the remainder. This is critical for OTA reliability: if a false
    /// preamble triggers header_fail, the real signal's samples survive.
    fn reset_for_retry(&mut self) {
        let skip = self.read_pos;
        self.state = State::Idle;
        if skip > 0 && skip <= self.buffer.len() {
            self.buffer.drain(..skip);
        }
        self.read_pos = 0;
        self.detector.reset();
        self.reset_decode_state();
    }

    fn reset_decode_state(&mut self) {
        self.cfo_hz = 0.0;
        self.fractional_cfo_hz = 0.0;
        self.cfo_phase = 0.0;
        self.integer_cfo_bins = 0;
        self.channel_estimate.clear();
        self.protocol_config = None;
        self.payload_len = 0;
        self.n_data_symbols = 0;
        self.phase_delta = 0.0;
        self.data_soft_bits.clear();
        self.data_symbols_collected = 0;
        self.noise_var_sum = 0.0;
        self.noise_var_count = 0;
        // Note: last_channel_report and diagnostic counters are NOT cleared
        // on reset, so callers can read them after decode completes.
    }

    /// Get the channel quality report from the last successful decode.
    /// Returns `None` if no successful decode has occurred yet.
    pub fn channel_report(&self) -> Option<&ChannelReport> {
        self.last_channel_report.as_ref()
    }

    /// How many unread samples are available.
    fn available(&self) -> usize {
        self.buffer.len() - self.read_pos
    }

    /// Read n samples from current position.
    fn read_samples(&mut self, n: usize) -> &[f32] {
        let start = self.read_pos;
        self.read_pos += n;
        &self.buffer[start..start + n]
    }

    /// Diagnostic snapshot for debugging real-time issues.
    pub fn debug_info(&self) -> DecoderDebugInfo {
        let rms = if self.recent_rms_count > 0 {
            (self.recent_rms_sum / self.recent_rms_count as f64).sqrt() as f32
        } else {
            0.0
        };
        DecoderDebugInfo {
            state: match self.state {
                State::Idle => "idle",
                State::CollectTraining => "training",
                State::CollectHeader => "header",
                State::CollectData => "data",
            },
            buffer_len: self.buffer.len(),
            total_samples_fed: self.total_samples_fed,
            max_amplitude: self.recent_max_amplitude,
            rms_amplitude: rms,
            last_event: self.last_event,
            preamble_count: self.preamble_count,
        }
    }

    /// Feed samples to the decoder. Returns decoded payload bytes if a complete
    /// packet is successfully decoded, or empty vec if still waiting for more data.
    pub fn feed(&mut self, samples: &[f32]) -> Vec<u8> {
        // Track diagnostics
        self.total_samples_fed += samples.len() as u64;
        for &s in samples {
            let abs = s.abs();
            if abs > self.recent_max_amplitude {
                self.recent_max_amplitude = abs;
            }
            self.recent_rms_sum += (s as f64) * (s as f64);
        }
        self.recent_rms_count += samples.len() as u64;

        // In idle state, trim buffer BEFORE appending to prevent unbounded growth.
        // This preserves the full contents of the current feed call while discarding
        // old samples that the detector already scanned without finding a preamble.
        if self.state == State::Idle && self.buffer.len() > params::RING_BUFFER_SIZE {
            let trim = self.buffer.len() - params::RING_BUFFER_SIZE;
            self.buffer.drain(..trim);
            self.read_pos = self.read_pos.saturating_sub(trim);
        }

        if self.fractional_cfo_hz.abs() > 0.1 && self.state != State::Idle {
            let (corrected, end_phase) = preamble::correct_cfo_with_phase(
                samples,
                self.fractional_cfo_hz,
                self.cfo_phase,
            );
            self.cfo_phase = end_phase;
            self.buffer.extend_from_slice(&corrected);
        } else {
            self.buffer.extend_from_slice(samples);
        }

        self.process()
    }

    fn process(&mut self) -> Vec<u8> {
        // Timeout: if the buffer has grown far beyond what any valid signal
        // needs while in a non-Idle state, we're stuck (corrupted header
        // claiming too many data symbols, or other pathological case).
        // Max valid signal ≈ 8 symbols preamble + 2 training + 1 header
        // + ~300 data symbols = ~375K samples. Use 480K as a generous limit.
        const STUCK_LIMIT: usize = 480_000;
        if self.state != State::Idle && self.buffer.len() > STUCK_LIMIT {
            self.last_event = "timeout";
            self.reset_for_retry();
            // Fall through to Idle scanning below
        }

        // Retry limit: on header_fail, we re-scan for another preamble.
        // Limit retries to prevent spinning if ambient noise (e.g. a fan
        // tone) produces sustained Schmidl-Cox plateaus.
        let mut retries = 0u32;
        const MAX_RETRIES: u32 = 3;

        loop {
            match self.state {
                State::Idle => {
                    if let Some(offset) = self.try_detect() {
                        // offset = sample index where preamble ends / training starts
                        self.last_event = "preamble";
                        self.preamble_count += 1;
                        // Estimate CFO from preamble samples
                        let preamble_len = self.detector.preamble_len_samples();
                        let preamble_start = offset.saturating_sub(preamble_len);
                        let preamble_slice = &self.buffer[preamble_start..offset];
                        self.cfo_hz = preamble::estimate_cfo(preamble_slice);

                        // Decompose CFO into integer bins and fractional Hz.
                        // Only correct the fractional part in time domain;
                        // the integer part is handled by bin offset detection
                        // from training correlation.
                        let subcarrier_spacing = params::SAMPLE_RATE as f32 / params::FFT_SIZE as f32;
                        let fractional_cfo = self.cfo_hz
                            - (self.cfo_hz / subcarrier_spacing).round() * subcarrier_spacing;
                        self.fractional_cfo_hz = fractional_cfo;

                        if fractional_cfo.abs() > 0.1 {
                            let (corrected, end_phase) = preamble::correct_cfo_with_phase(
                                &self.buffer[offset..],
                                fractional_cfo,
                                0.0,
                            );
                            self.cfo_phase = end_phase;
                            self.buffer.truncate(offset);
                            self.buffer.extend_from_slice(&corrected);
                        }

                        self.read_pos = offset;
                        self.state = State::CollectTraining;
                    } else {
                        return Vec::new();
                    }
                }
                State::CollectTraining => {
                    if self.available() < TRAINING_SYMBOLS * SYMBOL_TOTAL {
                        return Vec::new();
                    }
                    self.process_training();
                    self.state = State::CollectHeader;
                }
                State::CollectHeader => {
                    if self.available() < SYMBOL_TOTAL {
                        return Vec::new();
                    }
                    if !self.process_header() {
                        self.last_event = "header_fail";
                        retries += 1;
                        if retries >= MAX_RETRIES {
                            // Too many false detections — give up and wait
                            // for the next feed() call with fresh samples.
                            return Vec::new();
                        }
                        // Preserve unprocessed samples so the real preamble
                        // (if present later in the buffer) can still be found.
                        self.reset_for_retry();
                        continue; // Re-enter loop to scan for another preamble
                    }
                    self.last_event = "header_ok";
                    self.state = State::CollectData;
                }
                State::CollectData => {
                    return self.process_data();
                }
            }
        }
    }

    fn try_detect(&mut self) -> Option<usize> {
        self.detector.feed(&self.buffer)
    }

    fn process_training(&mut self) {
        // Use wide band for channel estimation so it works for any protocol.
        // Must cover the widest protocol (WIRED_TURBO: bins 20–460).
        let bin_start = params::GUARD_BINS;
        let bin_end = 460;

        let sym1 = self.read_samples(SYMBOL_TOTAL).to_vec();
        let sym2 = self.read_samples(SYMBOL_TOTAL).to_vec();

        // Detect integer CFO by trying bin offsets [-2, -1, 0, +1, +2]
        // and finding the one that maximizes correlation with the known reference.
        //
        // Uses block-coherent correlation: divide per-bin correlations into
        // small blocks (4 bins), sum coherently within each block (timing phase
        // is ~constant over 4 bins), then sum magnitudes across blocks. This is
        // robust to both timing-induced phase ramps (from early preamble
        // detection within CP) and channel phase variations (from reverb).
        let reference = preamble::training_reference(bin_start, bin_end);
        let mut best_offset: i32 = 0;
        let mut best_corr = 0.0f32;
        const BLOCK_SIZE: usize = 4;

        for offset in -2i32..=2 {
            let shifted_start = (bin_start as i32 + offset).max(1) as usize;
            let shifted_end = (bin_end as i32 + offset).min(params::FFT_SIZE as i32 / 2) as usize;
            if shifted_end <= shifted_start {
                continue;
            }
            let n = shifted_end - shifted_start;
            if n != reference.len() {
                continue;
            }

            let bins1 = ofdm::analyze_symbol(&sym1, shifted_start, shifted_end);
            let bins2 = ofdm::analyze_symbol(&sym2, shifted_start, shifted_end);

            // Per-bin correlation with reference for both training symbols
            let corr1: Vec<Complex32> = bins1.iter().zip(reference.iter())
                .map(|(y, r)| *y * r.conj())
                .collect();
            let corr2: Vec<Complex32> = bins2.iter().zip(reference.iter())
                .map(|(y, r)| *y * r.conj())
                .collect();

            // Block-coherent sum: coherent within blocks, non-coherent across.
            // For a 50-sample timing offset, phase rotates 0.327 rad/bin, so a
            // 4-bin block spans ~1.3 rad (93% coherence). Channel phase from
            // reverb is also smooth within 4 bins. Between blocks, magnitudes
            // add regardless of accumulated phase.
            let mut total = 0.0f32;
            for (blk1, blk2) in corr1.chunks(BLOCK_SIZE).zip(corr2.chunks(BLOCK_SIZE)) {
                let sum1: Complex32 = blk1.iter().copied().sum();
                let sum2: Complex32 = blk2.iter().copied().sum();
                // Phase-align sym2 block to sym1 block, then average
                let align = sum2 * sum1.conj();
                let phase = align.im.atan2(align.re);
                let comp = Complex32::new((-phase).cos(), (-phase).sin());
                let avg = sum1 + sum2 * comp;
                total += avg.norm();
            }

            if total > best_corr {
                best_corr = total;
                best_offset = offset;
            }
        }

        self.integer_cfo_bins = best_offset;

        // Re-analyze with the best offset
        let adj_start = (bin_start as i32 + best_offset) as usize;
        let adj_end = (bin_end as i32 + best_offset) as usize;
        let bins1 = ofdm::analyze_symbol(&sym1, adj_start, adj_end);
        let bins2 = ofdm::analyze_symbol(&sym2, adj_start, adj_end);

        // Estimate per-symbol phase delta from the training pair.
        // With integer CFO, each symbol accumulates a phase rotation of
        // 2π·Δf·SYMBOL_TOTAL/fs. The training pair reveals this as
        // arg(Σ Y2[k]·conj(Y1[k])).
        let mut phase_sum = Complex32::new(0.0, 0.0);
        for (b2, b1) in bins2.iter().zip(bins1.iter()) {
            phase_sum += *b2 * b1.conj();
        }
        self.phase_delta = phase_sum.im.atan2(phase_sum.re);

        // Average both training symbols for channel estimation, after
        // compensating the second symbol for the per-symbol phase rotation.
        // This gives a 3 dB noise reduction vs using a single symbol,
        // which is critical at low SNR.
        let phase_comp = Complex32::new((-self.phase_delta).cos(), (-self.phase_delta).sin());
        self.channel_estimate = reference
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if x.norm_sqr() > 1e-10 {
                    let bins2_comp = bins2[i] * phase_comp;
                    (bins1[i] + bins2_comp) / (x * 2.0)
                } else {
                    Complex32::new(1.0, 0.0)
                }
            })
            .collect();
    }

    fn process_header(&mut self) -> bool {
        let sym = self.read_samples(SYMBOL_TOTAL).to_vec();
        let adj_start = (params::HEADER_BINS_START as i32 + self.integer_cfo_bins) as usize;
        let adj_end = (params::HEADER_BINS_END as i32 + self.integer_cfo_bins) as usize;
        let bins = ofdm::analyze_symbol(&sym, adj_start, adj_end);

        let h_offset = adj_start - (params::GUARD_BINS as i32 + self.integer_cfo_bins) as usize;
        let h_end = adj_end - (params::GUARD_BINS as i32 + self.integer_cfo_bins) as usize;
        let h_slice = &self.channel_estimate[h_offset..h_end];
        let equalized = channel::equalize(&bins, h_slice);

        // Compensate per-symbol phase drift on the header.
        // The header is 2 symbols after training sym 1 (training1=0, training2=1, header=2).
        // Data symbols don't need this because differential encoding cancels constant phase.
        let header_phase = 2.0 * self.phase_delta;
        let phase_comp = Complex32::new((-header_phase).cos(), (-header_phase).sin());
        let corrected: Vec<Complex32> = equalized.iter().map(|&e| e * phase_comp).collect();

        if let Some((proto_id, payload_len)) = header::decode_header(&corrected) {
            let config = params::get_protocol(proto_id);
            self.protocol_config = Some(config);
            self.payload_len = payload_len;
            self.n_data_symbols = config.data_symbols_for(payload_len);
            self.data_soft_bits.clear();
            self.data_symbols_collected = 0;
            true
        } else {
            false
        }
    }

    fn process_data(&mut self) -> Vec<u8> {
        let config = self.protocol_config.unwrap();
        let adj_start = (config.bin_start as i32 + self.integer_cfo_bins) as usize;
        let adj_end = (config.bin_end as i32 + self.integer_cfo_bins) as usize;
        let data_sym_total = config.symbol_total();
        let use_differential = config.modulation != params::Modulation::Qam16
            && config.modulation != params::Modulation::Qam64
            && config.modulation != params::Modulation::Qam256;

        let pilot_bins = config.pilot_bins();
        let known_pilots = params::pilot_symbols(pilot_bins.len());

        while self.data_symbols_collected < self.n_data_symbols {
            if self.available() < data_sym_total {
                return Vec::new();
            }

            let sym = self.read_samples(data_sym_total).to_vec();

            // Full FFT for both data extraction and noise estimation
            let full_spectrum = ofdm::fft(&sym[config.cp_len..]);

            // Estimate noise variance from all out-of-band bins (below and above signal).
            let half_fft = params::FFT_SIZE / 2;
            let mut noise_sum = 0.0f32;
            let mut noise_count = 0usize;
            for k in 0..adj_start {
                noise_sum += full_spectrum[k].norm_sqr();
                noise_count += 1;
            }
            for k in adj_end..half_fft {
                noise_sum += full_spectrum[k].norm_sqr();
                noise_count += 1;
            }
            let noise_per_bin = if noise_count > 0 { noise_sum / noise_count as f32 } else { 1e-10 };
            let noise_var = (noise_per_bin / 2.0).max(1e-10);

            // Accumulate noise variance for channel quality reporting
            self.noise_var_sum += noise_var;
            self.noise_var_count += 1;

            // Extract data bins
            let bins: Vec<Complex32> = full_spectrum[adj_start..adj_end].to_vec();

            let h_offset = adj_start - (params::GUARD_BINS as i32 + self.integer_cfo_bins) as usize;
            let h_end = adj_end - (params::GUARD_BINS as i32 + self.integer_cfo_bins) as usize;
            let h_slice = &self.channel_estimate[h_offset..h_end];
            let equalized = channel::equalize_mmse(&bins, h_slice, noise_var);

            // Compensate per-symbol phase rotation from residual CFO.
            let sym_phase = Decoder::data_symbol_phase(
                self.phase_delta,
                self.data_symbols_collected,
                data_sym_total,
            );
            let sym_comp = Complex32::new((-sym_phase).cos(), (-sym_phase).sin());
            let equalized_corrected: Vec<Complex32> = equalized.iter()
                .map(|&e| e * sym_comp).collect();

            // For PSK modulations: decision-directed differential decoding.
            // For QAM: coherent detection (amplitude carries info, no differential).
            let decoded = if use_differential {
                let (n_phases, phase_offset) = match config.modulation {
                    params::Modulation::Bpsk => (2, 0.0),
                    params::Modulation::Qpsk => (8, 0.0),
                    params::Modulation::Psk8 => (8, 0.0),
                    params::Modulation::Qam16 | params::Modulation::Qam64 | params::Modulation::Qam256 => unreachable!(),
                };
                modulation::differential_decode_dd(&equalized_corrected, n_phases, phase_offset)
            } else {
                equalized_corrected.clone()
            };

            // Pilot-based per-symbol channel tracking.
            let corrected = self.pilot_correct(&decoded, config, &pilot_bins, &known_pilots);

            // Extract data carriers only (exclude pilots)
            let data_decoded = params::extract_data_bins(
                &corrected, &pilot_bins, config.bin_start, config.num_carriers(),
            );

            let soft = modulation::soft_demodulate(&data_decoded, config.modulation);

            // Noise-aware LLR scaling: multiply raw LLR by |H|²/(2σ²).
            let avg_h_sq: f32 = h_slice.iter().map(|h| h.norm_sqr()).sum::<f32>()
                / h_slice.len() as f32;
            let llr_scale = (avg_h_sq / (2.0 * noise_var)).min(100.0);

            let scaled_soft: Vec<f32> = soft.iter().map(|&s| s * llr_scale).collect();

            let deinterleaved = interleaver::deinterleave_soft(
                &scaled_soft,
                interleaver::symbol_seed(self.data_symbols_collected),
            );

            self.data_soft_bits.extend_from_slice(&deinterleaved);
            self.data_symbols_collected += 1;
        }

        // All data symbols collected — FEC decode
        let total_info_bits = (self.payload_len + 2) * 8; // +2 for CRC
        let coded_bits_len = config.coded_bits_for(self.payload_len);
        let usable_soft = &self.data_soft_bits[..coded_bits_len.min(self.data_soft_bits.len())];

        let decoded_bits = match config.fec_type {
            params::FecType::Convolutional => fec::decode(
                usable_soft,
                total_info_bits,
                config.code_rate,
            ),
            params::FecType::Ldpc => {
                let code = ldpc::get_ldpc_code(config.code_rate);
                ldpc::ldpc_decode_payload(usable_soft, total_info_bits, code, 30)
            }
        };

        let decoded_bytes = encoder::bits_to_bytes(&decoded_bits);

        let result = match crc::check_crc(&decoded_bytes) {
            Some(payload) => {
                self.last_event = "crc_ok";
                // Compute channel quality report on successful decode
                let avg_noise_var = if self.noise_var_count > 0 {
                    self.noise_var_sum / self.noise_var_count as f32
                } else {
                    1e-10
                };
                let est_bin_start = (params::GUARD_BINS as i32 + self.integer_cfo_bins).max(0) as usize;
                let trained_bin_start = (config.bin_start as i32 + self.integer_cfo_bins).max(0) as usize;
                let trained_bin_end = (config.bin_end as i32 + self.integer_cfo_bins).max(0) as usize;
                self.last_channel_report = Some(adaptive::estimate_channel_quality(
                    &self.channel_estimate,
                    avg_noise_var,
                    est_bin_start,
                    trained_bin_start,
                    trained_bin_end,
                ));
                payload
            }
            None => {
                self.last_event = "crc_fail";
                // Preserve remaining buffer for retry (another preamble
                // might be present if this was a false/corrupted detection).
                self.reset_for_retry();
                return Vec::new();
            }
        };

        // Successful decode — full reset (buffer no longer needed).
        self.reset();
        result
    }

    /// Per-symbol pilot-based channel correction.
    ///
    /// Compares received pilot values (after differential decoding) to known
    /// reference values. Estimates:
    ///   1. Common Phase Error (CPE): uniform gain/phase across all bins.
    ///   2. Sample Clock Offset (SCO): linear phase slope across bins.
    ///
    /// Applies the inverse correction to all decoded bins.
    fn pilot_correct(
        &self,
        decoded: &[Complex32],
        config: &ProtocolConfig,
        pilot_bins: &[usize],
        known_pilots: &[Complex32],
    ) -> Vec<Complex32> {
        if pilot_bins.is_empty() {
            return decoded.to_vec();
        }

        // Extract received pilot values
        let pilot_indices: Vec<usize> = pilot_bins.iter()
            .map(|&b| b - config.bin_start)
            .collect();

        // Compute per-pilot error: received / expected
        // This gives the channel distortion at each pilot position.
        let mut pilot_errors: Vec<Complex32> = Vec::with_capacity(pilot_bins.len());
        for (i, &idx) in pilot_indices.iter().enumerate() {
            let received = decoded[idx];
            let expected = known_pilots[i];
            if expected.norm_sqr() > 1e-10 {
                pilot_errors.push(received / expected);
            } else {
                pilot_errors.push(Complex32::new(1.0, 0.0));
            }
        }

        // Fit a linear model: error[k] = a * bin_position + b
        // where bin_position is relative to the band center.
        let center = (config.bin_start + config.bin_end) as f32 / 2.0;
        let n = pilot_errors.len() as f32;

        // Compute mean (intercept estimate) and slope via least squares
        let mut sum_x = 0.0f32;
        let mut sum_xx = 0.0f32;
        let mut sum_err = Complex32::new(0.0, 0.0);
        let mut sum_x_err = Complex32::new(0.0, 0.0);

        for (i, &err) in pilot_errors.iter().enumerate() {
            let x = pilot_bins[i] as f32 - center;
            sum_x += x;
            sum_xx += x * x;
            sum_err += err;
            sum_x_err += err * x;
        }

        let mean_x = sum_x / n;
        let mean_err = sum_err / n;
        let mean_x_err = sum_x_err / n;
        let var_x = sum_xx / n - mean_x * mean_x;

        // Slope (complex) and intercept
        let slope = if var_x > 1e-10 {
            (mean_x_err - mean_err * mean_x) / var_x
        } else {
            Complex32::new(0.0, 0.0)
        };
        let intercept = mean_err - slope * mean_x;

        // Apply inverse correction to all bins
        decoded
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                let x = (config.bin_start + i) as f32 - center;
                let correction = intercept + slope * x;
                let corr_sq = correction.norm_sqr();
                if corr_sq > 1e-10 {
                    d * correction.conj() / corr_sq
                } else {
                    d
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder;

    #[test]
    fn test_decode_encoded_signal() {
        let payload = b"Hello, OFDM!";
        let waveform = encoder::encode(payload, 0, 1.0);

        let mut decoder = Decoder::new();
        let result = decoder.feed(&waveform);
        assert_eq!(result, payload, "Loopback decode failed");
    }

    #[test]
    fn test_decode_with_silence_padding() {
        let payload = b"test";
        let waveform = encoder::encode(payload, 1, 0.8);

        let mut signal = vec![0.0f32; 4096];
        signal.extend_from_slice(&waveform);
        signal.extend_from_slice(&vec![0.0f32; 4096]);

        let mut decoder = Decoder::new();
        let result = decoder.feed(&signal);
        assert_eq!(result, payload);
    }

    #[test]
    fn test_decode_chunked() {
        let payload = b"chunked decode test";
        let waveform = encoder::encode(payload, 3, 1.0);

        let mut decoder = Decoder::new();
        let chunk_size = 4096;
        let mut result = Vec::new();

        for chunk in waveform.chunks(chunk_size) {
            let r = decoder.feed(chunk);
            if !r.is_empty() {
                result = r;
                break;
            }
        }

        assert_eq!(result, payload, "Chunked decode failed");
    }

    #[test]
    fn test_decode_all_protocols() {
        for proto in 0..params::PROTOCOLS.len() as u8 {
            let payload = b"protocol test";
            let waveform = encoder::encode(payload, proto, 1.0);

            let mut decoder = Decoder::new();
            let result = decoder.feed(&waveform);
            assert_eq!(result, payload, "Protocol {} loopback failed", proto);
        }
    }

    #[test]
    fn test_decode_various_sizes() {
        for &size in &[1, 16, 64, 128, 256, 512, 1023] {
            let payload: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            let waveform = encoder::encode(&payload, 0, 1.0);

            let mut decoder = Decoder::new();
            let result = decoder.feed(&waveform);
            assert_eq!(result, payload, "Size {} loopback failed", size);
        }
    }

    #[test]
    fn test_decode_two_third_with_padding() {
        let payload = b"x";
        let proto = 2; // QPSK 2/3
        let config = params::get_protocol(proto);
        let coded_bits = config.coded_bits_for(payload.len());
        let coded_per_sym = config.coded_bits_per_symbol();
        assert_ne!(coded_bits % coded_per_sym, 0, "Expected padding for this payload");

        let waveform = encoder::encode(payload, proto, 1.0);
        let mut decoder = Decoder::new();
        let result = decoder.feed(&waveform);
        assert_eq!(result, payload);
    }

    #[test]
    fn test_decode_short_preamble() {
        let payload = b"short preamble decode";
        let proto = 0;
        let waveform = encoder::encode_with_preamble_symbols(
            payload,
            proto,
            1.0,
            params::PREAMBLE_SYMBOLS_SHORT,
        );

        let mut decoder = Decoder::new();
        let result = decoder.feed(&waveform);
        assert_eq!(result, payload);
    }

    /// After a failed decode (header_fail or crc_fail), the decoder should
    /// preserve remaining buffer samples so a subsequent real signal can
    /// still be detected. This simulates the OTA scenario where a false
    /// preamble detection clears the buffer, destroying the real signal.
    #[test]
    fn test_retry_after_failed_detection() {
        let payload = b"retry works";
        let waveform = encoder::encode(payload, 0, 1.0);

        // Feed the signal in 4096-sample chunks, as the AudioWorklet does.
        // Prefix with 24000 samples of silence (0.5s).
        let mut signal = vec![0.0f32; 24000];
        signal.extend_from_slice(&waveform);
        signal.extend_from_slice(&vec![0.0f32; 24000]);

        let mut decoder = Decoder::new();
        let chunk_size = 4096;
        let mut result = Vec::new();

        for chunk in signal.chunks(chunk_size) {
            let r = decoder.feed(chunk);
            if !r.is_empty() {
                result = r;
                break;
            }
        }

        assert_eq!(result, payload, "Chunked decode with silence padding failed");
    }

    /// Verify that after a complete signal, the decoder can handle another
    /// signal fed in chunks (tests that reset() properly clears state).
    #[test]
    fn test_sequential_decodes_chunked() {
        let payload1 = b"first message";
        let payload2 = b"second message";
        let waveform1 = encoder::encode(payload1, 0, 1.0);
        let waveform2 = encoder::encode(payload2, 0, 1.0);

        let mut signal = vec![0.0f32; 4800];
        signal.extend_from_slice(&waveform1);
        signal.extend_from_slice(&vec![0.0f32; 9600]); // gap between signals
        signal.extend_from_slice(&waveform2);
        signal.extend_from_slice(&vec![0.0f32; 4800]);

        let mut decoder = Decoder::new();
        let chunk_size = 4096;
        let mut results: Vec<Vec<u8>> = Vec::new();

        for chunk in signal.chunks(chunk_size) {
            let r = decoder.feed(chunk);
            if !r.is_empty() {
                results.push(r);
            }
        }

        assert_eq!(results.len(), 2, "Expected 2 decoded messages, got {}", results.len());
        assert_eq!(results[0], payload1, "First message mismatch");
        assert_eq!(results[1], payload2, "Second message mismatch");
    }

    #[test]
    fn test_phase_scaling_for_short_cp() {
        let phase_delta = 1.0f32;
        let long_len = SYMBOL_TOTAL;
        let short_len = params::FFT_SIZE + params::CP_LEN_SHORT;

        let long0 = Decoder::data_symbol_phase(phase_delta, 0, long_len);
        let long1 = Decoder::data_symbol_phase(phase_delta, 1, long_len);
        assert!((long1 - long0 - phase_delta).abs() < 1e-6);

        let short0 = Decoder::data_symbol_phase(phase_delta, 0, short_len);
        let short1 = Decoder::data_symbol_phase(phase_delta, 1, short_len);
        let expected = phase_delta * (short_len as f32 / SYMBOL_TOTAL as f32);
        assert!((short1 - short0 - expected).abs() < 1e-6);
    }
}
