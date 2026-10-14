use std::fmt::Display;

use num_complex::Complex;
use num_traits::Zero;

#[macro_use]
pub(crate) mod common;

pub(crate) mod array_utils;
pub(crate) mod fft_cache;
pub(crate) mod fft_helper;
pub(crate) mod math_utils;
pub(crate) mod twiddles;

pub mod butterflies;
mod dft;
mod radix4;
pub(crate) mod radixn;
mod planner;

pub use crate::fft::common::FftNum;
pub use crate::fft::planner::FftPlanner;

/// A trait that allows FFT algorithms to report their expected input/output size
pub trait Length {
    /// The FFT size that this algorithm can process
    fn len(&self) -> usize;
}

/// Represents a FFT direction, IE a forward FFT or an inverse FFT
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum FftDirection {
    Forward,
    Inverse,
}
impl FftDirection {
    #[inline]
    pub fn opposite_direction(&self) -> FftDirection {
        match self {
            Self::Forward => Self::Inverse,
            Self::Inverse => Self::Forward,
        }
    }
}
impl Display for FftDirection {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        match self {
            Self::Forward => f.write_str("Forward"),
            Self::Inverse => f.write_str("Inverse"),
        }
    }
}

/// A trait that allows FFT algorithms to report whether they compute forward FFTs or inverse FFTs
pub trait Direction {
    fn fft_direction(&self) -> FftDirection;
}

/// Trait for algorithms that compute FFTs.
pub trait Fft<T: FftNum>: Length + Direction + Sync + Send {
    /// Computes a FFT in-place.
    fn process(&self, buffer: &mut [Complex<T>]) {
        let mut scratch = vec![Complex::zero(); self.get_inplace_scratch_len()];
        self.process_with_scratch(buffer, &mut scratch);
    }

    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]);

    fn process_outofplace_with_scratch(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    );

    fn process_immutable_with_scratch(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    );

    fn get_inplace_scratch_len(&self) -> usize;
    fn get_outofplace_scratch_len(&self) -> usize;
    fn get_immutable_scratch_len(&self) -> usize;
}
