/// QC-LDPC encoder and min-sum belief propagation decoder.
///
/// Uses IEEE 802.11n base matrices with N=648, Z=27 (lifting factor).
/// Supports rates 1/2 (k=324) and 3/4 (k=486).
///
/// Encoder: Systematic encoding exploiting the dual-diagonal parity structure.
/// Decoder: Normalized min-sum BP, max 30 iterations, early termination on syndrome check.

use crate::params::CodeRate;

/// Lifting factor for all codes.
const Z: usize = 27;

/// LDPC code descriptor.
pub struct LdpcCode {
    /// Codeword length.
    pub n: usize,
    /// Information bits.
    pub k: usize,
    /// Parity bits (n - k).
    pub m: usize,
    /// Lifting factor.
    pub z: usize,
    /// Base matrix rows.
    pub m_base: usize,
    /// Base matrix columns.
    pub n_base: usize,
    /// Base matrix entries (row-major). -1 = zero block, 0..Z-1 = circulant shift value.
    pub base_matrix: &'static [i16],
}

/// Return (k, n) for an LDPC block given the code rate.
/// Used by params.rs for coded_bits_for() calculation.
pub fn ldpc_block_params(rate: CodeRate) -> (usize, usize) {
    match rate {
        CodeRate::Half => (LDPC_648_R12.k, LDPC_648_R12.n),
        CodeRate::ThreeFourth => (LDPC_648_R34.k, LDPC_648_R34.n),
        CodeRate::FiveSixth => (LDPC_648_R56.k, LDPC_648_R56.n),
        // Rate 2/3 not used for LDPC but return something valid
        CodeRate::TwoThird => (LDPC_648_R12.k, LDPC_648_R12.n),
    }
}

/// Get the LDPC code for a given rate.
pub fn get_ldpc_code(rate: CodeRate) -> &'static LdpcCode {
    match rate {
        CodeRate::Half => &LDPC_648_R12,
        CodeRate::ThreeFourth => &LDPC_648_R34,
        CodeRate::FiveSixth => &LDPC_648_R56,
        CodeRate::TwoThird => &LDPC_648_R12, // fallback
    }
}

// IEEE 802.11n base matrix for N=648, rate 1/2.
// 12 rows × 24 columns, Z=27. m=12, k=12 (in base), so m*Z=324, k*Z=324, n=648.
//
// From IEEE 802.11n-2009 Table R.1 (Z=27).
// -1 means zero block (no connection).
#[rustfmt::skip]
static BASE_MATRIX_R12: [i16; 12 * 24] = [
    0,  -1, -1, -1,  0,  0, -1, -1,  0, -1, -1,  0,  1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   22,   0, -1, -1, 17, -1,  0,  0, 12, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    6,  -1,  0, -1, 10, -1, -1, -1, 24, -1,  0, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1,
    2,  -1, -1,  0, 20, -1, -1, -1, 25,  0, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1,
   23,  -1, -1, -1,  3, -1, -1, -1,  0, -1,  9, 11, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1,
   24,  -1, 23,  1, 17, -1,  3, -1, 10, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1,
   25,  -1, -1, -1,  8,  6, -1, -1,  7, 24, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1,
   26,  -1, -1, -1, 12, -1, -1, -1, 15, -1,  0, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1,
    9,  -1, -1, -1,  2, -1, -1, -1, 24, -1, -1,  0, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1,
    2, 24, -1, -1,  5, -1, -1, -1, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1,
   13,  -1, -1, -1,  4, -1, -1, -1,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,
    7,  -1, -1, -1, 14, -1, -1, -1, 12, -1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,
];

// IEEE 802.11n base matrix for N=648, rate 3/4.
// 6 rows × 24 columns, Z=27. m=6, k=18 (in base), so m*Z=162, k*Z=486, n=648.
//
// From IEEE 802.11n-2009 Table R.3 (Z=27).
#[rustfmt::skip]
static BASE_MATRIX_R34: [i16; 6 * 24] = [
   16, 17, 22, 24,  9,  3, 14, -1,  4,  2,  7, -1, 26, -1,  2, -1, 21, -1,  1,  0, -1, -1, -1, -1,
   25, 12, 12,  3,  3, 26,  6, 21, -1, 15, 22, -1, 15, -1,  4, -1, -1, 16, -1,  0,  0, -1, -1, -1,
   25, 18, 26, 16, 22, 23,  9, -1,  0, -1,  4, -1,  4, -1, 14, -1, -1,  0,  0, -1,  0,  0, -1, -1,
    9,  7,  0,  1, 17, -1, -1,  7,  3, -1,  3, 23, -1, 16, -1, -1, 21, -1,  0, -1, -1,  0,  0, -1,
   24,  5, 26,  7,  1, -1, -1, 15, 24, 15, -1,  8, -1, 13, -1, 11, -1, -1, -1, -1, -1, -1,  0,  0,
    2,  2, 19, 14, 24,  1, 15, 19, -1, 21, -1,  2, -1, 24, -1,  3, -1, -1, -1, -1, -1, -1, -1,  0,
];

pub static LDPC_648_R12: LdpcCode = LdpcCode {
    n: 648,
    k: 324,
    m: 324,
    z: Z,
    m_base: 12,
    n_base: 24,
    base_matrix: &BASE_MATRIX_R12,
};

pub static LDPC_648_R34: LdpcCode = LdpcCode {
    n: 648,
    k: 486,
    m: 162,
    z: Z,
    m_base: 6,
    n_base: 24,
    base_matrix: &BASE_MATRIX_R34,
};

// IEEE 802.11n base matrix for N=648, rate 5/6.
// 4 rows × 24 columns, Z=27. m=4, k=20 (in base), so m*Z=108, k*Z=540, n=648.
//
// From IEEE 802.11n-2009 Table R.4 (Z=27).
#[rustfmt::skip]
static BASE_MATRIX_R56: [i16; 4 * 24] = [
   17, 13,  8, 21,  9,  3, 18, 12, 10,  0,  4, 15, 19,  2,  5, 10, 26, 19, 13, 13,  1,  0, -1, -1,
    3, 12, 11, 14, 11, 25,  5, 18,  0,  9,  2, 26, 26, 10, 24,  7, 14, 20,  4,  2, -1,  0,  0, -1,
   22, 16,  4,  3, 10, 21, 12,  5, 21, 14, 19,  5, -1,  8,  5, 18, 11,  5,  5, 15,  0, -1,  0,  0,
    7,  7, 14, 14,  4, 16, 16, 24, 24, 10,  1,  7, 15,  6, 10, 26,  8, 18, 21, 14, -1, -1, -1,  0,
];

pub static LDPC_648_R56: LdpcCode = LdpcCode {
    n: 648,
    k: 540,
    m: 108,
    z: Z,
    m_base: 4,
    n_base: 24,
    base_matrix: &BASE_MATRIX_R56,
};

/// Expand a base matrix entry into a Z×Z circulant permutation matrix.
/// Returns the column index that has a 1 in the given row.
/// For shift value `s`: row `r` has a 1 at column `(r + s) % Z`.
#[inline]
fn circulant_col(row_in_block: usize, shift: usize) -> usize {
    (row_in_block + shift) % Z
}

/// Encode information bits using the QC-LDPC code.
///
/// The code is systematic: output = [info_bits | parity_bits].
/// Uses the efficient dual-diagonal encoding for 802.11n parity structure.
///
/// Input: `info_bits` -- exactly `code.k` bits (0 or 1).
/// Output: `code.n` coded bits.
pub fn ldpc_encode(info_bits: &[u8], code: &LdpcCode) -> Vec<u8> {
    assert_eq!(info_bits.len(), code.k, "Info bits length must be exactly k");

    let mut codeword = vec![0u8; code.n];

    // Systematic part: copy info bits
    codeword[..code.k].copy_from_slice(info_bits);

    // Compute parity bits using H * c = 0.
    // The parity portion of H for 802.11n has a dual-diagonal / staircase structure.
    // We compute parity sequentially: for each parity check row, solve for the
    // corresponding parity bit given all previously computed parity bits.
    //
    // For each check row i, the syndrome contribution from info bits is:
    //   s_i = sum over j in [0, k_base) of P_{i,j} * info_block_j
    // where P_{i,j} is the (i,j) entry of the base matrix for info columns.
    //
    // The parity structure in the base matrix typically has a lower-triangular
    // or dual-diagonal form. We handle the general case by Gaussian elimination
    // on the parity sub-matrix.

    // Step 1: Compute syndrome from information part: s = H_info * info
    let mut syndrome = vec![0u8; code.m];
    for bi in 0..code.m_base {
        for bj in 0..code.n_base {
            if bj * Z >= code.k {
                // This is a parity column, skip for info syndrome
                continue;
            }
            let shift = code.base_matrix[bi * code.n_base + bj];
            if shift < 0 {
                continue;
            }
            let s = shift as usize;
            // For each row in this Z×Z block
            for r in 0..Z {
                let c = circulant_col(r, s);
                let global_col = bj * Z + c;
                if global_col < code.k {
                    syndrome[bi * Z + r] ^= codeword[global_col];
                }
            }
        }
    }

    // Step 2: Solve for parity bits.
    // The parity sub-matrix of H (columns k..n) has a specific structure.
    // For 802.11n, the parity columns form a (approximately) lower-triangular
    // structure with the dual-diagonal. We solve row by row.
    //
    // We use a direct approach: for each parity check row, accumulate the
    // contribution of already-determined parity bits, then solve for the
    // remaining unknown parity bit on the diagonal.
    //
    // Build the parity sub-matrix connectivity for efficient back-substitution.
    // For the 802.11n codes, each check equation involves at most a few parity blocks.

    // Build sparse representation of parity sub-matrix
    // parity_entries[check_row] = list of (parity_bit_col, connected)
    let k_base = code.k / Z;

    // For rate 1/2 (and other 802.11n codes), the parity part of H
    // has a specific dual-diagonal structure that allows sequential encoding.
    // We'll use a general approach: encode using the expanded H matrix on parity columns.

    // Collect parity column connections per check row
    let mut parity_connections: Vec<Vec<(usize, usize)>> = vec![Vec::new(); code.m];
    for bi in 0..code.m_base {
        for bj in k_base..code.n_base {
            let shift = code.base_matrix[bi * code.n_base + bj];
            if shift < 0 {
                continue;
            }
            let s = shift as usize;
            let parity_block = bj - k_base;
            for r in 0..Z {
                let c = circulant_col(r, s);
                let parity_col = parity_block * Z + c;
                let check_row = bi * Z + r;
                parity_connections[check_row].push((parity_col, 1));
            }
        }
    }

    // Solve for parity using Gaussian elimination on the parity sub-matrix.
    // We build a dense m×m binary matrix and the syndrome vector, then solve.
    // For m=324 or m=162, this is manageable.
    let m = code.m;

    // Use bit-packed rows for efficiency
    let words_per_row = (m + 63) / 64;
    let mut matrix: Vec<u64> = vec![0u64; m * words_per_row];
    let mut rhs = syndrome.clone();

    // Fill matrix from parity connections
    for row in 0..m {
        for &(col, _) in &parity_connections[row] {
            if col < m {
                let word = col / 64;
                let bit = col % 64;
                matrix[row * words_per_row + word] |= 1u64 << bit;
            }
        }
    }

    // Gaussian elimination with partial pivoting
    let mut pivot_row = vec![0usize; m];
    for i in 0..m {
        pivot_row[i] = i;
    }

    for col in 0..m {
        let word = col / 64;
        let bit = col % 64;
        let mask = 1u64 << bit;

        // Find pivot
        let mut found = false;
        for row in col..m {
            let pr = pivot_row[row];
            if matrix[pr * words_per_row + word] & mask != 0 {
                if row != col {
                    pivot_row.swap(row, col);
                }
                found = true;
                break;
            }
        }

        if !found {
            continue; // Singular column (shouldn't happen for valid LDPC)
        }

        let pivot = pivot_row[col];

        // Eliminate below
        for row in 0..m {
            if row == col {
                continue;
            }
            let pr = pivot_row[row];
            if matrix[pr * words_per_row + word] & mask != 0 {
                // XOR row
                for w in 0..words_per_row {
                    matrix[pr * words_per_row + w] ^= matrix[pivot * words_per_row + w];
                }
                rhs[pr] ^= rhs[pivot];
            }
        }
    }

    // Read solution
    let mut parity = vec![0u8; m];
    for col in 0..m {
        let pr = pivot_row[col];
        parity[col] = rhs[pr];
    }

    // Place parity bits
    codeword[code.k..].copy_from_slice(&parity);

    codeword
}

/// Decode LLRs using normalized min-sum belief propagation.
///
/// Input: `llr` -- `code.n` log-likelihood ratios (positive = more likely 0).
/// Output: decoded information bits (length `code.k`).
///
/// Uses normalized min-sum with factor 0.75, max 30 iterations,
/// early termination when syndrome check passes.
pub fn ldpc_decode(llr: &[f32], code: &LdpcCode, max_iter: usize) -> Vec<u8> {
    assert_eq!(llr.len(), code.n, "LLR length must be exactly n");

    let norm_factor: f32 = 0.75;

    // Build sparse H matrix: for each check node, list of variable node indices
    // For each variable node, list of (check_node_index, position_in_check_list)
    let mut check_to_var: Vec<Vec<usize>> = vec![Vec::new(); code.m];
    let mut var_to_check: Vec<Vec<usize>> = vec![Vec::new(); code.n];

    for bi in 0..code.m_base {
        for bj in 0..code.n_base {
            let shift = code.base_matrix[bi * code.n_base + bj];
            if shift < 0 {
                continue;
            }
            let s = shift as usize;
            for r in 0..Z {
                let check = bi * Z + r;
                let var = bj * Z + circulant_col(r, s);
                check_to_var[check].push(var);
                var_to_check[var].push(check);
            }
        }
    }

    // Messages: check-to-variable
    // Indexed as c2v[check][position_in_check_neighbor_list]
    let mut c2v: Vec<Vec<f32>> = check_to_var
        .iter()
        .map(|neighbors| vec![0.0f32; neighbors.len()])
        .collect();

    // Variable beliefs (channel + extrinsic)
    let mut belief: Vec<f32> = llr.to_vec();

    for _iter in 0..max_iter {
        // Variable-to-check messages: v2c[check][j] = belief[var] - c2v[check][j]
        // Check node update (min-sum):
        for (ci, neighbors) in check_to_var.iter().enumerate() {
            let degree = neighbors.len();
            if degree == 0 {
                continue;
            }

            // Compute incoming v2c messages
            let v2c: Vec<f32> = neighbors
                .iter()
                .enumerate()
                .map(|(j, &var)| belief[var] - c2v[ci][j])
                .collect();

            // Min-sum: product of signs, minimum of |v2c|
            // For each output j: sign = product of signs excluding j,
            //                     magnitude = min of |v2c| excluding j
            let mut sign_product: i8 = 1;
            let mut min1 = f32::MAX; // smallest
            let mut min2 = f32::MAX; // second smallest
            let mut min1_idx = 0;

            for (j, &msg) in v2c.iter().enumerate() {
                if msg < 0.0 {
                    sign_product = -sign_product;
                }
                let abs_msg = msg.abs();
                if abs_msg < min1 {
                    min2 = min1;
                    min1 = abs_msg;
                    min1_idx = j;
                } else if abs_msg < min2 {
                    min2 = abs_msg;
                }
            }

            for (j, &msg) in v2c.iter().enumerate() {
                let sign_j = if msg < 0.0 { -sign_product } else { sign_product };
                let mag_j = if j == min1_idx { min2 } else { min1 };
                c2v[ci][j] = sign_j as f32 * mag_j * norm_factor;
            }
        }

        // Variable node update: belief = channel_llr + sum of incoming c2v
        for (vi, checks) in var_to_check.iter().enumerate() {
            let mut sum = llr[vi];
            for &ci in checks {
                // Find position of vi in check_to_var[ci]
                if let Some(pos) = check_to_var[ci].iter().position(|&v| v == vi) {
                    sum += c2v[ci][pos];
                }
            }
            belief[vi] = sum;
        }

        // Syndrome check for early termination
        if syndrome_check(&belief, &check_to_var) {
            break;
        }
    }

    // Hard decision on beliefs
    belief[..code.k].iter().map(|&b| if b >= 0.0 { 0 } else { 1 }).collect()
}

/// Check if the hard decisions satisfy all parity checks (H * c = 0).
fn syndrome_check(belief: &[f32], check_to_var: &[Vec<usize>]) -> bool {
    for neighbors in check_to_var {
        let mut parity = 0u8;
        for &var in neighbors {
            if belief[var] < 0.0 {
                parity ^= 1;
            }
        }
        if parity != 0 {
            return false;
        }
    }
    true
}

/// Verify that a codeword satisfies H * c = 0.
/// Used for testing encoder correctness.
pub fn verify_codeword(codeword: &[u8], code: &LdpcCode) -> bool {
    assert_eq!(codeword.len(), code.n);

    for bi in 0..code.m_base {
        for r in 0..Z {
            let mut parity = 0u8;
            for bj in 0..code.n_base {
                let shift = code.base_matrix[bi * code.n_base + bj];
                if shift < 0 {
                    continue;
                }
                let s = shift as usize;
                let col = bj * Z + circulant_col(r, s);
                parity ^= codeword[col];
            }
            if parity != 0 {
                return false;
            }
        }
    }
    true
}

/// Encode a variable-length payload into one or more LDPC blocks.
///
/// Splits info bits into k-sized blocks, pads the last block with zeros,
/// encodes each block, and concatenates the coded output.
pub fn ldpc_encode_payload(info_bits: &[u8], code: &LdpcCode) -> Vec<u8> {
    let mut coded = Vec::new();
    let mut offset = 0;

    while offset < info_bits.len() {
        let remaining = info_bits.len() - offset;
        let block_len = remaining.min(code.k);

        let mut block = vec![0u8; code.k];
        block[..block_len].copy_from_slice(&info_bits[offset..offset + block_len]);

        let codeword = ldpc_encode(&block, code);
        coded.extend_from_slice(&codeword);

        offset += block_len;
    }

    // If info_bits is empty, produce one all-zero block
    if info_bits.is_empty() {
        let block = vec![0u8; code.k];
        let codeword = ldpc_encode(&block, code);
        coded.extend_from_slice(&codeword);
    }

    coded
}

/// Decode one or more LDPC blocks from soft LLR values.
///
/// Splits the LLR stream into n-sized blocks, decodes each, concatenates
/// the info bits, and truncates to the expected number of info bits.
pub fn ldpc_decode_payload(
    soft_bits: &[f32],
    expected_info_bits: usize,
    code: &LdpcCode,
    max_iter: usize,
) -> Vec<u8> {
    let mut decoded = Vec::new();
    let mut offset = 0;

    while offset + code.n <= soft_bits.len() {
        let block = &soft_bits[offset..offset + code.n];
        let info = ldpc_decode(block, code, max_iter);
        decoded.extend_from_slice(&info);
        offset += code.n;
    }

    decoded.truncate(expected_info_bits);
    decoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ldpc_encode_produces_valid_codeword_r12() {
        let mut info = vec![0u8; LDPC_648_R12.k];
        // Set some bits
        for i in 0..info.len() {
            info[i] = ((i * 7 + 3) % 2) as u8;
        }
        let codeword = ldpc_encode(&info, &LDPC_648_R12);
        assert_eq!(codeword.len(), LDPC_648_R12.n);
        assert!(
            verify_codeword(&codeword, &LDPC_648_R12),
            "Syndrome check failed for rate 1/2 codeword"
        );
        // Verify systematic: info bits preserved
        assert_eq!(&codeword[..LDPC_648_R12.k], &info[..]);
    }

    #[test]
    fn test_ldpc_encode_produces_valid_codeword_r34() {
        let mut info = vec![0u8; LDPC_648_R34.k];
        for i in 0..info.len() {
            info[i] = ((i * 13 + 5) % 2) as u8;
        }
        let codeword = ldpc_encode(&info, &LDPC_648_R34);
        assert_eq!(codeword.len(), LDPC_648_R34.n);
        assert!(
            verify_codeword(&codeword, &LDPC_648_R34),
            "Syndrome check failed for rate 3/4 codeword"
        );
        assert_eq!(&codeword[..LDPC_648_R34.k], &info[..]);
    }

    #[test]
    fn test_ldpc_encode_decode_rate_half_clean() {
        let mut info = vec![0u8; LDPC_648_R12.k];
        for i in 0..info.len() {
            info[i] = ((i * 11 + 7) % 2) as u8;
        }
        let codeword = ldpc_encode(&info, &LDPC_648_R12);

        // Perfect channel: LLR = +1 for bit 0, -1 for bit 1
        let llr: Vec<f32> = codeword.iter().map(|&b| if b == 0 { 1.0 } else { -1.0 }).collect();
        let decoded = ldpc_decode(&llr, &LDPC_648_R12, 30);
        assert_eq!(decoded, info, "Rate 1/2 clean decode failed");
    }

    #[test]
    fn test_ldpc_encode_decode_rate_three_fourths_clean() {
        let mut info = vec![0u8; LDPC_648_R34.k];
        for i in 0..info.len() {
            info[i] = ((i * 3 + 1) % 2) as u8;
        }
        let codeword = ldpc_encode(&info, &LDPC_648_R34);

        let llr: Vec<f32> = codeword.iter().map(|&b| if b == 0 { 1.0 } else { -1.0 }).collect();
        let decoded = ldpc_decode(&llr, &LDPC_648_R34, 30);
        assert_eq!(decoded, info, "Rate 3/4 clean decode failed");
    }

    #[test]
    fn test_ldpc_decode_with_soft_errors() {
        let mut info = vec![0u8; LDPC_648_R12.k];
        for i in 0..info.len() {
            info[i] = ((i * 17 + 3) % 2) as u8;
        }
        let codeword = ldpc_encode(&info, &LDPC_648_R12);

        // Create LLRs with some flipped signs (errors)
        let mut llr: Vec<f32> = codeword.iter().map(|&b| if b == 0 { 2.0 } else { -2.0 }).collect();

        // Flip ~5% of bits (with weaker confidence)
        for i in (0..llr.len()).step_by(20) {
            llr[i] = -llr[i] * 0.3;
        }

        let decoded = ldpc_decode(&llr, &LDPC_648_R12, 30);
        assert_eq!(decoded, info, "LDPC should correct sparse errors");
    }

    #[test]
    fn test_ldpc_early_termination() {
        // With clean input, the decoder should converge in 1-2 iterations
        let info = vec![0u8; LDPC_648_R12.k];
        let codeword = ldpc_encode(&info, &LDPC_648_R12);
        let llr: Vec<f32> = codeword.iter().map(|&b| if b == 0 { 5.0 } else { -5.0 }).collect();

        // Even with max_iter=2, clean data should decode correctly
        let decoded = ldpc_decode(&llr, &LDPC_648_R12, 2);
        assert_eq!(decoded, info, "Early termination should handle clean data");
    }

    #[test]
    fn test_ldpc_multi_block() {
        // Test payload spanning multiple LDPC blocks
        let code = &LDPC_648_R12;
        let info_len = code.k * 2 + 100; // 2+ blocks
        let mut info = vec![0u8; info_len];
        for i in 0..info_len {
            info[i] = ((i * 23 + 9) % 2) as u8;
        }

        let coded = ldpc_encode_payload(&info, code);
        let expected_blocks = (info_len + code.k - 1) / code.k;
        assert_eq!(coded.len(), expected_blocks * code.n);

        // Decode with clean LLRs
        let llr: Vec<f32> = coded.iter().map(|&b| if b == 0 { 3.0 } else { -3.0 }).collect();
        let decoded = ldpc_decode_payload(&llr, info_len, code, 30);
        assert_eq!(decoded, info);
    }

    #[test]
    fn test_ldpc_all_zeros_codeword() {
        // All-zeros should be a valid codeword
        let info = vec![0u8; LDPC_648_R12.k];
        let codeword = ldpc_encode(&info, &LDPC_648_R12);
        assert!(codeword.iter().all(|&b| b == 0), "All-zero info should produce all-zero codeword");
        assert!(verify_codeword(&codeword, &LDPC_648_R12));
    }

    #[test]
    fn test_verify_codeword_detects_error() {
        let info = vec![0u8; LDPC_648_R12.k];
        let mut codeword = ldpc_encode(&info, &LDPC_648_R12);
        // Flip one bit
        codeword[0] ^= 1;
        assert!(
            !verify_codeword(&codeword, &LDPC_648_R12),
            "Should detect single bit error"
        );
    }

    #[test]
    fn test_ldpc_encode_produces_valid_codeword_r56() {
        let mut info = vec![0u8; LDPC_648_R56.k];
        for i in 0..info.len() {
            info[i] = ((i * 11 + 3) % 2) as u8;
        }
        let codeword = ldpc_encode(&info, &LDPC_648_R56);
        assert_eq!(codeword.len(), LDPC_648_R56.n);
        assert!(
            verify_codeword(&codeword, &LDPC_648_R56),
            "Syndrome check failed for rate 5/6 codeword"
        );
        assert_eq!(&codeword[..LDPC_648_R56.k], &info[..]);
    }

    #[test]
    fn test_ldpc_encode_decode_rate_five_sixths_clean() {
        let mut info = vec![0u8; LDPC_648_R56.k];
        for i in 0..info.len() {
            info[i] = ((i * 7 + 5) % 2) as u8;
        }
        let codeword = ldpc_encode(&info, &LDPC_648_R56);

        let llr: Vec<f32> = codeword.iter().map(|&b| if b == 0 { 3.0 } else { -3.0 }).collect();
        let decoded = ldpc_decode(&llr, &LDPC_648_R56, 30);
        assert_eq!(decoded, info, "Rate 5/6 clean decode failed");
    }
}
