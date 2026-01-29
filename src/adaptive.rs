/// Adaptive protocol selection based on channel quality estimation.
///
/// Estimates SNR from channel estimate and noise variance, then recommends
/// the highest-throughput protocol whose required SNR is below the measured SNR.

use num_complex::Complex32;
use crate::params::{self, FecType, Modulation, CodeRate};

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
    let bin_offset = est_bin_start as i32 - params::GUARD_BINS as i32;

    // Define the band boundaries used by protocols
    let band_defs: &[(usize, usize, &str)] = &[
        (20, 80, "audible"),   // 60 carriers
        (20, 120, "dt"),       // 100 carriers
        (20, 160, "mt"),       // 140 carriers
        (20, 320, "ut"),       // 300 carriers
        (20, 410, "wired_max"),  // 390 carriers (up to 20.5 kHz)
        (20, 460, "wired_turbo"),// 440 carriers (up to 23 kHz)
    ];

    let mut bands = Vec::new();
    for &(bin_start, bin_end, _name) in band_defs {
        let band_start_i = bin_start as i32 + bin_offset;
        let band_end_i = bin_end as i32 + bin_offset;
        if band_start_i < 0 || band_end_i <= band_start_i {
            bands.push(BandReport {
                snr_db: f32::NAN,
                snr_min_db: f32::NAN,
                num_carriers: bin_end - bin_start,
            });
            continue;
        }

        let band_start = band_start_i as usize;
        let band_end = band_end_i as usize;
        let fully_trained = band_start >= trained_bin_start && band_end <= trained_bin_end;
        if !fully_trained {
            bands.push(BandReport {
                snr_db: f32::NAN,
                snr_min_db: f32::NAN,
                num_carriers: bin_end - bin_start,
            });
            continue;
        }

        let start_idx = band_start.saturating_sub(est_bin_start);
        let end_idx = band_end.saturating_sub(est_bin_start).min(channel_est.len());

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
}
