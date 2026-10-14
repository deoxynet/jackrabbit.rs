/// Adaptive protocol selection based on channel quality estimation.
///
/// Estimates SNR from channel estimate and noise variance, then recommends
/// the highest-throughput protocol whose required SNR is below the measured SNR.

use num_complex::Complex32;
use crate::params::{self, FecType, Modulation, CodeRate, BitloadLevel, BitloadProfile, BITLOAD_BANDS};

/// Per-band SNR report.
#[derive(Debug, Clone)]
pub struct BandReport {
    pub snr_db: f32,
    pub snr_min_db: f32,
    pub num_carriers: usize,
}

/// Channel quality report with protocol recommendation.
#[derive(Debug, Clone)]
pub struct ChannelReport {
    pub bands: Vec<BandReport>,
    pub recommended_protocol: u8,
    pub viable_protocols: Vec<u8>,
}

/// SNR threshold entry for protocol recommendation.
struct SnrThreshold {
    required_snr_db: f32,
    throughput_score: f32, // higher = more data per second
}

/// Get the throughput score for a protocol (bits per OFDM symbol time).
fn throughput_score(config: &params::ProtocolConfig) -> f32 {
    let bits_per_carrier = config.modulation.bits_per_symbol() as f32;
    let (k, n) = config.code_rate.ratio();
    let code_rate = k as f32 / n as f32;
    let carriers = config.num_data_carriers() as f32;
    let symbol_time = config.symbol_total() as f32 / params::SAMPLE_RATE as f32;
    carriers * bits_per_carrier * code_rate / symbol_time
}

/// Required SNR (dB) for a given modulation + FEC combination.
/// Includes ~3 dB implementation margin.
fn required_snr_db(modulation: Modulation, code_rate: CodeRate, fec_type: FecType) -> f32 {
    match (modulation, code_rate, fec_type) {
        (Modulation::Bpsk, CodeRate::Half, FecType::Ldpc) => 1.5,
        (Modulation::Bpsk, CodeRate::Half, FecType::Convolutional) => 3.0,
        (Modulation::Qpsk, CodeRate::Half, FecType::Ldpc) => 4.0,
        (Modulation::Qpsk, CodeRate::Half, FecType::Convolutional) => 6.0,
        (Modulation::Qpsk, CodeRate::TwoThird, FecType::Convolutional) => 9.0,
        (Modulation::Qpsk, CodeRate::TwoThird, FecType::Ldpc) => 7.0,
        (Modulation::Psk8, CodeRate::TwoThird, FecType::Convolutional) => 9.0,
        (Modulation::Qam16, CodeRate::ThreeFourth, FecType::Ldpc) => 16.0,
        (Modulation::Qam16, CodeRate::ThreeFourth, FecType::Convolutional) => 18.0,
        (Modulation::Qam64, CodeRate::ThreeFourth, FecType::Ldpc) => 22.0,
        (Modulation::Qam64, CodeRate::ThreeFourth, FecType::Convolutional) => 24.0,
        (Modulation::Qam256, CodeRate::ThreeFourth, FecType::Ldpc) => 28.0,
        (Modulation::Qam256, CodeRate::ThreeFourth, FecType::Convolutional) => 30.0,
        (Modulation::Qam256, CodeRate::FiveSixth, FecType::Ldpc) => 30.0,
        // Conservative fallback for unlisted combinations
        _ => 20.0,
    }
}

/// Estimate channel quality from a channel estimate and noise variance.
///
/// `channel_est`: Per-subcarrier channel frequency response H[k] covering the
///   widest band (e.g. bins 20-320).
/// `noise_var`: Estimated noise variance per subcarrier.
/// `trained_bin_start`/`trained_bin_end`: Bin range where training symbols were transmitted.
///
/// Returns a `ChannelReport` with per-band SNR and protocol recommendations.
pub fn estimate_channel_quality(
    channel_est: &[Complex32],
    noise_var: f32,
    est_bin_start: usize,
    trained_bin_start: usize,
    trained_bin_end: usize,
) -> ChannelReport {
    let noise_var = noise_var.max(1e-10);

    // Define the band boundaries used by protocols (absolute bin numbers).
    let band_defs: &[(usize, usize, &str)] = &[
        (20, 80, "audible"),        // 60 carriers
        (20, 120, "dt"),            // 100 carriers
        (20, 160, "mt"),            // 140 carriers
        (20, 320, "ut"),            // 300 carriers
        (20, 400, "wired"),         // 380 carriers (20 kHz)
        (5, 475, "wired_extreme"),  // 470 carriers (near-Nyquist, cable only)
    ];

    let mut bands = Vec::new();
    for &(bin_start, bin_end, _name) in band_defs {
        // Check that this band is fully within the trained range
        let fully_trained = bin_start >= trained_bin_start && bin_end <= trained_bin_end;
        if !fully_trained {
            bands.push(BandReport {
                snr_db: f32::NAN,
                snr_min_db: f32::NAN,
                num_carriers: bin_end - bin_start,
            });
            continue;
        }

        // Index into channel_est (which starts at est_bin_start)
        let start_idx = bin_start.saturating_sub(est_bin_start);
        let end_idx = bin_end.saturating_sub(est_bin_start).min(channel_est.len());

        if start_idx >= end_idx {
            bands.push(BandReport {
                snr_db: f32::NAN,
                snr_min_db: f32::NAN,
                num_carriers: bin_end - bin_start,
            });
            continue;
        }

        let slice = &channel_est[start_idx..end_idx];
        let mut snr_sum = 0.0f32;
        let mut snr_min = f32::MAX;

        for h in slice {
            let h_sq = h.norm_sqr();
            let snr_linear = h_sq / noise_var;
            snr_sum += snr_linear;
            if snr_linear < snr_min {
                snr_min = snr_linear;
            }
        }

        let avg_snr = snr_sum / slice.len() as f32;
        let snr_db = 10.0 * avg_snr.max(1e-10).log10();
        let snr_min_db = 10.0 * snr_min.max(1e-10).log10();

        bands.push(BandReport {
            snr_db,
            snr_min_db,
            num_carriers: bin_end - bin_start,
        });
    }

    // Build threshold list for all protocols
    let mut thresholds: Vec<SnrThreshold> = Vec::new();
    for config in params::PROTOCOLS.iter() {
        let req_snr = required_snr_db(config.modulation, config.code_rate, config.fec_type);
        let tp = throughput_score(config);
        thresholds.push(SnrThreshold {
            required_snr_db: req_snr,
            throughput_score: tp,
        });
    }

    // For each protocol, determine if the channel supports it
    let mut viable = Vec::new();
    for (i, config) in params::PROTOCOLS.iter().enumerate() {
        // Find the band that matches this protocol's carrier range
        let band_idx = band_defs
            .iter()
            .position(|&(bs, be, _)| bs == config.bin_start && be == config.bin_end);

        if let Some(bi) = band_idx {
            if bands[bi].snr_db.is_finite()
                && bands[bi].snr_db >= thresholds[i].required_snr_db
            {
                viable.push(config.id);
            }
        }
    }

    // Recommend the highest-throughput viable protocol
    let recommended = viable
        .iter()
        .map(|&id| {
            let idx = id as usize;
            (id, thresholds[idx].throughput_score)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(id, _)| id)
        .unwrap_or(0); // fallback to most robust

    ChannelReport {
        bands,
        recommended_protocol: recommended,
        viable_protocols: viable,
    }
}

/// Convenience: recommend a single protocol from a channel report.
pub fn recommend_protocol(report: &ChannelReport) -> u8 {
    report.recommended_protocol
}

/// SNR thresholds (dB) for each bitloading level, with built-in safety margin.
/// These are conservative: ~3-5 dB above theoretical minimums to ensure
/// both TX and RX agree on the same level despite estimation differences.
const BITLOAD_SNR_THRESHOLDS: [(BitloadLevel, f32); 5] = [
    (BitloadLevel::Bpsk,   5.0),   // BPSK viable above ~2 dB, use 5 dB threshold
    (BitloadLevel::Qpsk,   10.0),  // QPSK viable above ~7 dB
    (BitloadLevel::Qam16,  18.0),  // 16QAM viable above ~15 dB
    (BitloadLevel::Qam64,  25.0),  // 64QAM viable above ~22 dB
    (BitloadLevel::Qam256, 32.0),  // 256QAM viable above ~28 dB
];

/// Select the highest bitloading level whose SNR threshold is below the
/// measured band-average SNR.
fn snr_to_bitload_level(snr_db: f32) -> BitloadLevel {
    let mut best = BitloadLevel::Null;
    for &(level, threshold) in &BITLOAD_SNR_THRESHOLDS {
        if snr_db >= threshold {
            best = level;
        }
    }
    best
}

impl BitloadProfile {
    /// Generate a bitloading profile from per-subcarrier SNR measurements.
    ///
    /// Divides `num_data_carriers` into `BITLOAD_BANDS` bands, computes the
    /// average SNR for each band from the per-carrier SNR values, and selects
    /// the highest modulation level whose threshold is below the band SNR.
    ///
    /// `per_carrier_snr_db`: one entry per data carrier (pilots already excluded).
    pub fn from_snr(per_carrier_snr_db: &[f32], num_data_carriers: usize) -> Self {
        let dummy = Self::uniform(BitloadLevel::Null);
        let boundaries = dummy.band_boundaries(num_data_carriers);
        let mut bands = [BitloadLevel::Null; BITLOAD_BANDS];

        for (i, &(start, end)) in boundaries.iter().enumerate() {
            if start >= end || start >= per_carrier_snr_db.len() {
                bands[i] = BitloadLevel::Null;
                continue;
            }
            let end = end.min(per_carrier_snr_db.len());
            let slice = &per_carrier_snr_db[start..end];
            let avg_snr: f32 = slice.iter().sum::<f32>() / slice.len() as f32;
            bands[i] = snr_to_bitload_level(avg_snr);
        }

        BitloadProfile { bands }
    }
}

/// Hysteresis margins for bitloading level changes:
/// - To increase modulation: band SNR must exceed threshold + HYSTERESIS_UP_DB
/// - To decrease modulation: band SNR must drop below threshold - HYSTERESIS_DOWN_DB
const HYSTERESIS_UP_DB: f32 = 2.0;
const HYSTERESIS_DOWN_DB: f32 = 1.0;

/// Update a bitloading profile with hysteresis to prevent rapid toggling.
///
/// `current`: the currently active profile.
/// `per_carrier_snr_db`: fresh per-carrier SNR measurements.
/// `num_data_carriers`: total data carriers for band boundary computation.
///
/// Only changes a band's level if the SNR is sufficiently above (for upgrade)
/// or below (for downgrade) the threshold, preventing oscillation.
pub fn update_profile_with_hysteresis(
    current: &BitloadProfile,
    per_carrier_snr_db: &[f32],
    num_data_carriers: usize,
) -> BitloadProfile {
    let boundaries = current.band_boundaries(num_data_carriers);
    let mut bands = current.bands;

    for (i, &(start, end)) in boundaries.iter().enumerate() {
        if start >= end || start >= per_carrier_snr_db.len() {
            continue;
        }
        let end = end.min(per_carrier_snr_db.len());
        let slice = &per_carrier_snr_db[start..end];
        let avg_snr: f32 = slice.iter().sum::<f32>() / slice.len() as f32;

        let desired = snr_to_bitload_level(avg_snr);
        let current_bps = current.bands[i].bits_per_symbol();
        let desired_bps = desired.bits_per_symbol();

        if desired_bps > current_bps {
            // Upgrade: require SNR to be above threshold + hysteresis margin
            let upgraded = snr_to_bitload_level(avg_snr - HYSTERESIS_UP_DB);
            if upgraded.bits_per_symbol() > current_bps {
                bands[i] = desired;
            }
        } else if desired_bps < current_bps {
            // Downgrade: change only if SNR dropped below threshold - margin
            let downgraded = snr_to_bitload_level(avg_snr + HYSTERESIS_DOWN_DB);
            if downgraded.bits_per_symbol() < current_bps {
                bands[i] = desired;
            }
        }
        // Equal: no change
    }

    BitloadProfile { bands }
}

/// Compute per-subcarrier SNR in dB from a channel estimate and noise variance.
///
/// Returns a Vec with one entry per subcarrier in the channel estimate,
/// representing `10*log10(|H[k]|² / noise_var)`.
pub fn per_subcarrier_snr_db(
    channel_est: &[Complex32],
    noise_var: f32,
) -> Vec<f32> {
    let nv = noise_var.max(1e-10);
    channel_est
        .iter()
        .map(|h| {
            let snr_linear = h.norm_sqr() / nv;
            10.0 * snr_linear.max(1e-10).log10()
        })
        .collect()
}

/// Compute per-subcarrier LLR scale factors: |H[k]|² / (2 * noise_var).
///
/// Used to weight soft bits from each subcarrier proportionally to its
/// channel quality, giving the FEC decoder better reliability information.
/// Clamped to [0, max_scale] to prevent extreme values on very strong carriers.
pub fn per_subcarrier_llr_scale(
    channel_est: &[Complex32],
    noise_var: f32,
    max_scale: f32,
) -> Vec<f32> {
    let nv = noise_var.max(1e-10);
    let inv_2nv = 1.0 / (2.0 * nv);
    channel_est
        .iter()
        .map(|h| (h.norm_sqr() * inv_2nv).min(max_scale))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_channel(n: usize, gain: f32) -> Vec<Complex32> {
        vec![Complex32::new(gain, 0.0); n]
    }

    #[test]
    fn test_snr_computation() {
        let h = make_channel(300, 1.0); // |H|^2 = 1.0
        let noise_var = 0.01; // SNR = 1/0.01 = 100 → 20 dB
        let report = estimate_channel_quality(&h, noise_var, 20, 20, 320);

        // The UT band (bins 20-320, 300 carriers) should have SNR ~20 dB
        assert!(report.bands.len() >= 4);
        let ut_band = &report.bands[3]; // last band = UT
        assert!(
            (ut_band.snr_db - 20.0).abs() < 1.0,
            "Expected ~20 dB SNR, got {} dB",
            ut_band.snr_db
        );
    }

    #[test]
    fn test_recommendation_high_snr() {
        // Very clean channel: should recommend the fastest protocol
        let h = make_channel(300, 10.0); // |H|^2 = 100 → SNR = 100/0.001 = 50 dB
        let noise_var = 0.001;
        let report = estimate_channel_quality(&h, noise_var, 20, 20, 320);

        // Should recommend one of the fastest (highest throughput) protocols
        let rec = recommend_protocol(&report);
        let config = params::get_protocol(rec);
        // At high SNR, should pick a high-order modulation or wide band
        assert!(
            report.viable_protocols.len() > 8,
            "Expected many viable protocols at high SNR, got {}",
            report.viable_protocols.len()
        );
        // The recommended protocol should have good throughput
        let tp = throughput_score(config);
        assert!(tp > 100.0, "Expected high throughput, got {}", tp);
    }

    #[test]
    fn test_recommendation_low_snr() {
        // Noisy channel: should recommend a robust protocol
        let h = make_channel(300, 0.1); // |H|^2 = 0.01
        let noise_var = 0.01; // SNR = 0.01/0.01 = 1 → 0 dB
        let report = estimate_channel_quality(&h, noise_var, 20, 20, 320);

        // At 0 dB, very few (or no) protocols should be viable
        assert!(
            report.viable_protocols.len() <= 2,
            "Expected few viable protocols at 0 dB SNR, got {}",
            report.viable_protocols.len()
        );
    }

    #[test]
    fn test_per_subcarrier_snr_db() {
        let h = make_channel(10, 1.0);
        let noise_var = 0.01;
        let snr = per_subcarrier_snr_db(&h, noise_var);
        assert_eq!(snr.len(), 10);
        // |H|^2 = 1.0, noise = 0.01 → SNR = 100 → 20 dB
        for &s in &snr {
            assert!((s - 20.0).abs() < 0.1, "Expected ~20 dB, got {}", s);
        }
    }

    #[test]
    fn test_per_subcarrier_llr_scale() {
        let h = vec![
            Complex32::new(1.0, 0.0),  // |H|^2 = 1.0
            Complex32::new(0.1, 0.0),  // |H|^2 = 0.01
            Complex32::new(2.0, 0.0),  // |H|^2 = 4.0
        ];
        let noise_var = 0.5;
        let scales = per_subcarrier_llr_scale(&h, noise_var, 100.0);
        assert_eq!(scales.len(), 3);
        // scale = |H|^2 / (2 * 0.5) = |H|^2
        assert!((scales[0] - 1.0).abs() < 0.01);
        assert!((scales[1] - 0.01).abs() < 0.001);
        assert!((scales[2] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_bitload_from_snr_high() {
        // All carriers at 35 dB → should get QAM256
        let snr = vec![35.0f32; 90];
        let profile = BitloadProfile::from_snr(&snr, 90);
        for &b in &profile.bands {
            assert_eq!(b, BitloadLevel::Qam256);
        }
    }

    #[test]
    fn test_bitload_from_snr_low() {
        // All carriers at 2 dB → below all thresholds → Null
        let snr = vec![2.0f32; 90];
        let profile = BitloadProfile::from_snr(&snr, 90);
        for &b in &profile.bands {
            assert_eq!(b, BitloadLevel::Null);
        }
    }

    #[test]
    fn test_bitload_from_snr_mixed() {
        // Clear separation: first quarter at 8 dB (BPSK), last quarter at 27 dB (QAM64)
        let num_dc = 96; // evenly divisible by 16
        let per_band = num_dc / BITLOAD_BANDS;
        let mut snr = Vec::with_capacity(num_dc);
        for i in 0..BITLOAD_BANDS {
            let level = if i < 4 { 8.0 } else if i < 12 { 12.0 } else { 27.0 };
            snr.extend(std::iter::repeat(level).take(per_band));
        }
        let profile = BitloadProfile::from_snr(&snr, num_dc);
        // First 4 bands at 8 dB → BPSK (threshold 5 dB)
        for i in 0..4 {
            assert_eq!(profile.bands[i], BitloadLevel::Bpsk, "band {} should be BPSK", i);
        }
        // Middle bands at 12 dB → QPSK (threshold 10 dB)
        for i in 4..12 {
            assert_eq!(profile.bands[i], BitloadLevel::Qpsk, "band {} should be QPSK", i);
        }
        // Last 4 bands at 27 dB → QAM64 (threshold 25 dB)
        for i in 12..16 {
            assert_eq!(profile.bands[i], BitloadLevel::Qam64, "band {} should be QAM64", i);
        }
    }

    #[test]
    fn test_hysteresis_prevents_toggle() {
        // Current profile: all QAM16 (threshold = 18 dB)
        let current = BitloadProfile::uniform(BitloadLevel::Qam16);
        let num_dc = 90;

        // SNR slightly above QAM64 threshold (25 dB) but within hysteresis up margin
        // Threshold for QAM64 = 25, need 25+2 = 27 to upgrade
        let snr_borderline = vec![26.0f32; num_dc]; // above 25 but below 27
        let updated = update_profile_with_hysteresis(&current, &snr_borderline, num_dc);
        for &b in &updated.bands {
            assert_eq!(b, BitloadLevel::Qam16, "Should NOT upgrade with borderline SNR");
        }

        // SNR well above threshold + margin → should upgrade
        let snr_high = vec![28.0f32; num_dc];
        let updated = update_profile_with_hysteresis(&current, &snr_high, num_dc);
        for &b in &updated.bands {
            assert_eq!(b, BitloadLevel::Qam64, "Should upgrade with high SNR");
        }
    }

    /// Downgrade hysteresis: must drop below threshold - margin to downgrade.
    #[test]
    fn test_hysteresis_prevents_downgrade() {
        let current = BitloadProfile::uniform(BitloadLevel::Qam64);
        let num_dc = 96;

        // SNR just below QAM64 threshold (25) but within downgrade margin (25-1=24)
        let snr_borderline = vec![24.5f32; num_dc]; // below 25 but above 24
        let updated = update_profile_with_hysteresis(&current, &snr_borderline, num_dc);
        for &b in &updated.bands {
            assert_eq!(b, BitloadLevel::Qam64, "Should NOT downgrade with borderline SNR");
        }

        // SNR drops below threshold - margin → should downgrade
        let snr_low = vec![23.0f32; num_dc]; // below 24, snr_to_bitload_level(23+1=24) → QAM16
        let updated = update_profile_with_hysteresis(&current, &snr_low, num_dc);
        for &b in &updated.bands {
            assert_eq!(b, BitloadLevel::Qam16, "Should downgrade when SNR drops below margin");
        }
    }

    /// SNR exactly at a threshold boundary maps to that level.
    #[test]
    fn test_snr_exact_threshold_boundaries() {
        assert_eq!(snr_to_bitload_level(5.0), BitloadLevel::Bpsk);
        assert_eq!(snr_to_bitload_level(10.0), BitloadLevel::Qpsk);
        assert_eq!(snr_to_bitload_level(18.0), BitloadLevel::Qam16);
        assert_eq!(snr_to_bitload_level(25.0), BitloadLevel::Qam64);
        assert_eq!(snr_to_bitload_level(32.0), BitloadLevel::Qam256);

        // Just below each threshold → previous level
        assert_eq!(snr_to_bitload_level(4.99), BitloadLevel::Null);
        assert_eq!(snr_to_bitload_level(9.99), BitloadLevel::Bpsk);
        assert_eq!(snr_to_bitload_level(17.99), BitloadLevel::Qpsk);
        assert_eq!(snr_to_bitload_level(24.99), BitloadLevel::Qam16);
        assert_eq!(snr_to_bitload_level(31.99), BitloadLevel::Qam64);
    }

    /// Negative SNR (noise > signal) maps to Null.
    #[test]
    fn test_snr_negative_maps_to_null() {
        assert_eq!(snr_to_bitload_level(-5.0), BitloadLevel::Null);
        assert_eq!(snr_to_bitload_level(-100.0), BitloadLevel::Null);
        assert_eq!(snr_to_bitload_level(0.0), BitloadLevel::Null);
    }

    /// Per-subcarrier SNR and LLR scale with zero/tiny noise variance.
    #[test]
    fn test_per_subcarrier_zero_noise_clamped() {
        let h = vec![Complex32::new(1.0, 0.0)];
        // noise_var = 0 gets clamped to 1e-10, should not panic or produce Inf
        let snr = per_subcarrier_snr_db(&h, 0.0);
        assert!(snr[0].is_finite(), "SNR should be finite with zero noise");
        assert!(snr[0] > 90.0, "SNR should be very high, got {}", snr[0]);

        let scale = per_subcarrier_llr_scale(&h, 0.0, 100.0);
        assert_eq!(scale[0], 100.0, "LLR scale should clamp to max_scale");
    }

    /// Per-subcarrier SNR with zero channel gain.
    #[test]
    fn test_per_subcarrier_zero_channel() {
        let h = vec![Complex32::new(0.0, 0.0)];
        let snr = per_subcarrier_snr_db(&h, 0.1);
        assert!(snr[0].is_finite(), "SNR should be finite with zero channel");
        assert!(snr[0] < -50.0, "SNR should be very negative, got {}", snr[0]);

        let scale = per_subcarrier_llr_scale(&h, 0.1, 100.0);
        assert!((scale[0] - 0.0).abs() < 1e-6, "Zero channel should give zero LLR scale");
    }

    /// from_snr with fewer carriers than SNR entries truncates gracefully.
    #[test]
    fn test_from_snr_more_snr_entries_than_carriers() {
        // 96 carrier slots, but only 80 SNR entries provided
        let snr = vec![20.0f32; 80];
        let profile = BitloadProfile::from_snr(&snr, 96);
        // First bands get SNR data, last bands get Null (out of range)
        let boundaries = profile.band_boundaries(96);
        for (i, &(start, _end)) in boundaries.iter().enumerate() {
            if start < 80 {
                assert_ne!(profile.bands[i], BitloadLevel::Null,
                    "Band {} with SNR data should not be Null", i);
            }
        }
    }

    /// from_snr with empty SNR produces all-Null profile.
    #[test]
    fn test_from_snr_empty_input() {
        let snr: Vec<f32> = vec![];
        let profile = BitloadProfile::from_snr(&snr, 90);
        for &b in &profile.bands {
            assert_eq!(b, BitloadLevel::Null);
        }
    }
}
