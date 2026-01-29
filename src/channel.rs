/// Channel estimation from training symbols and zero-forcing equalization.

use num_complex::Complex32;

/// Estimate channel frequency response from training symbols.
/// Uses average of two training symbols: H[k] = avg(Y1[k], Y2[k]) / X[k]
/// where X[k] is the known training reference.
pub fn estimate_channel(
    received_bins_1: &[Complex32],
    received_bins_2: &[Complex32],
    reference: &[Complex32],
) -> Vec<Complex32> {
    assert_eq!(received_bins_1.len(), reference.len());
    assert_eq!(received_bins_2.len(), reference.len());

    reference
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let y_avg = (received_bins_1[i] + received_bins_2[i]) * 0.5;
            if x.norm_sqr() > 1e-10 {
                y_avg / x
            } else {
                Complex32::new(1.0, 0.0) // no data on this carrier, assume unity
            }
        })
        .collect()
}

/// Zero-forcing equalization: Y[k] / H[k].
/// Delegates to MMSE with noise_var=0.0 (backward compatible).
pub fn equalize(received_bins: &[Complex32], channel: &[Complex32]) -> Vec<Complex32> {
    equalize_mmse(received_bins, channel, 0.0)
}

/// MMSE equalization: X_hat[k] = H*[k] · Y[k] / (|H[k]|² + noise_var).
///
/// When noise_var=0.0, degrades to zero-forcing. For positive noise_var,
/// suppresses noise amplification on weak subcarriers instead of zeroing them.
pub fn equalize_mmse(
    received_bins: &[Complex32],
    channel: &[Complex32],
    noise_var: f32,
) -> Vec<Complex32> {
    assert_eq!(received_bins.len(), channel.len());

    received_bins
        .iter()
        .zip(channel.iter())
        .map(|(&y, &h)| {
            let h_sq = h.norm_sqr();
            let denom = h_sq + noise_var;
            if denom > 1e-10 {
                y * h.conj() / denom
            } else {
                Complex32::new(0.0, 0.0)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_channel_estimation_unity() {
        let n = 60;
        let reference: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new(if i % 2 == 0 { 1.0 } else { -1.0 }, 0.0))
            .collect();

        // Unity channel: received = reference
        let h = estimate_channel(&reference, &reference, &reference);
        for &hi in &h {
            assert!((hi - Complex32::new(1.0, 0.0)).norm() < 1e-5);
        }
    }

    #[test]
    fn test_channel_estimation_and_equalization() {
        let n = 60;
        let reference: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new(if i % 2 == 0 { 1.0 } else { -1.0 }, 0.0))
            .collect();

        // Simulated channel: phase rotation + attenuation
        let channel: Vec<Complex32> = (0..n)
            .map(|i| {
                let phase = i as f32 * 0.1;
                let mag = 0.5 + 0.5 * (i as f32 * 0.05).cos();
                Complex32::new(mag * phase.cos(), mag * phase.sin())
            })
            .collect();

        // Received = channel * reference
        let received: Vec<Complex32> = reference
            .iter()
            .zip(channel.iter())
            .map(|(&x, &h)| h * x)
            .collect();

        let h_est = estimate_channel(&received, &received, &reference);

        // Equalize a test signal through the same channel
        let test_data: Vec<Complex32> = (0..n)
            .map(|i| {
                let angle = i as f32 * PI / 4.0;
                Complex32::new(angle.cos(), angle.sin())
            })
            .collect();

        let received_data: Vec<Complex32> = test_data
            .iter()
            .zip(channel.iter())
            .map(|(&x, &h)| h * x)
            .collect();

        let equalized = equalize(&received_data, &h_est);

        for i in 0..n {
            let diff = (equalized[i] - test_data[i]).norm();
            assert!(diff < 0.1, "Equalization error at {}: {}", i, diff);
        }
    }

    #[test]
    fn test_mmse_zero_noise_matches_zf() {
        let n = 16;
        let channel: Vec<Complex32> = (0..n)
            .map(|i| {
                let phase = i as f32 * 0.3;
                Complex32::new(phase.cos(), phase.sin()) * 0.8
            })
            .collect();
        let received: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new(i as f32 * 0.1, -(i as f32) * 0.05))
            .collect();

        let zf = equalize(&received, &channel);
        let mmse = equalize_mmse(&received, &channel, 0.0);

        for i in 0..n {
            let diff = (zf[i] - mmse[i]).norm();
            assert!(diff < 1e-6, "MMSE(noise=0) != ZF at bin {}: diff={}", i, diff);
        }
    }

    #[test]
    fn test_mmse_preserves_weak_subcarriers() {
        // With a weak channel tap and noise, ZF amplifies noise badly.
        // MMSE should keep the output bounded.
        let strong = Complex32::new(1.0, 0.0);
        let weak = Complex32::new(0.01, 0.0); // |H|^2 = 0.0001
        let channel = vec![strong, weak, strong, weak];
        let received = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(0.5, 0.0), // noise-dominated on weak subcarrier
            Complex32::new(1.0, 0.0),
            Complex32::new(0.5, 0.0),
        ];

        let noise_var = 0.1;
        let mmse = equalize_mmse(&received, &channel, noise_var);

        // On weak subcarriers (|H|^2=0.0001), MMSE denom = 0.0001 + 0.1 = 0.1001
        // so output ~ H* * Y / 0.1001 which is small, not amplified.
        for i in [1, 3] {
            assert!(
                mmse[i].norm() < 1.0,
                "MMSE should suppress weak subcarrier {}: got norm={}",
                i,
                mmse[i].norm()
            );
        }

        // On strong subcarriers (|H|^2=1.0), MMSE ~ ZF
        for i in [0, 2] {
            let expected = received[i] * channel[i].conj() / (channel[i].norm_sqr() + noise_var);
            let diff = (mmse[i] - expected).norm();
            assert!(diff < 1e-6, "MMSE strong subcarrier {} error: {}", i, diff);
        }
    }
}
