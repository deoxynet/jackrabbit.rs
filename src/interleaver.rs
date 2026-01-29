/// Seeded pseudo-random bit interleaver/deinterleaver per OFDM symbol.
///
/// Uses a simple LCG PRNG to generate a permutation of bit indices.

/// Generate a permutation table for `n` elements using a seed.
fn make_permutation(n: usize, seed: u32) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    // Fisher-Yates shuffle with LCG PRNG
    let mut rng = seed as u64;
    for i in (1..n).rev() {
        // LCG: x = (a*x + c) mod m
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = ((rng >> 33) as usize) % (i + 1);
        indices.swap(i, j);
    }
    indices
}

/// Interleave bits: output[perm[i]] = input[i].
pub fn interleave(bits: &[u8], seed: u32) -> Vec<u8> {
    let perm = make_permutation(bits.len(), seed);
    let mut out = vec![0u8; bits.len()];
    for (i, &p) in perm.iter().enumerate() {
        out[p] = bits[i];
    }
    out
}

/// Interleave soft bits: output[i] = input[perm[i]].
pub fn deinterleave_soft(soft_bits: &[f32], seed: u32) -> Vec<f32> {
    let perm = make_permutation(soft_bits.len(), seed);
    let mut out = vec![0.0f32; soft_bits.len()];
    for (i, &p) in perm.iter().enumerate() {
        out[i] = soft_bits[p];
    }
    out
}

/// Compute interleaver seed for a given symbol index.
pub fn symbol_seed(symbol_index: usize) -> u32 {
    // Mix symbol index with a constant for better randomness
    (symbol_index as u32).wrapping_mul(2654435761) ^ 0xDEADBEEF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interleave_deinterleave() {
        let bits: Vec<u8> = vec![0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0];
        let seed = symbol_seed(0);

        let interleaved = interleave(&bits, seed);
        // Should be a permutation (same bits, different order)
        assert_ne!(interleaved, bits);

        // Convert to soft: 0→+1.0, 1→-1.0
        let soft: Vec<f32> = interleaved.iter().map(|&b| if b == 0 { 1.0 } else { -1.0 }).collect();
        let deinterleaved = deinterleave_soft(&soft, seed);
        let hard: Vec<u8> = deinterleaved.iter().map(|&s| if s > 0.0 { 0 } else { 1 }).collect();
        assert_eq!(hard, bits);
    }

    #[test]
    fn test_permutation_is_permutation() {
        let perm = make_permutation(100, 42);
        let mut sorted = perm.clone();
        sorted.sort();
        let expected: Vec<usize> = (0..100).collect();
        assert_eq!(sorted, expected);
    }
}
