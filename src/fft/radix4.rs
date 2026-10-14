use std::sync::Arc;

use num_complex::Complex;

use crate::fft::butterflies::{
    Butterfly1, Butterfly16, Butterfly2, Butterfly32, Butterfly4, Butterfly8,
};
use crate::fft::radixn::butterfly_4;
use crate::fft::array_utils::bitreversed_transpose;
use crate::fft::{common::FftNum, twiddles, FftDirection};
use crate::fft::{Direction, Fft, Length};

/// FFT algorithm optimized for power-of-two sizes
pub struct Radix4<T> {
    twiddles: Box<[Complex<T>]>,

    base_fft: Arc<dyn Fft<T>>,
    base_len: usize,

    len: usize,
    direction: FftDirection,
    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    immut_scratch_len: usize,
}

impl<T: FftNum> Radix4<T> {
    pub fn new(len: usize, direction: FftDirection) -> Self {
        assert!(
            len.is_power_of_two(),
            "Radix4 algorithm requires a power-of-two input size. Got {}",
            len
        );

        let exponent = len.trailing_zeros();
        let (base_exponent, base_fft) = match exponent {
            0 => (0, Arc::new(Butterfly1::new(direction)) as Arc<dyn Fft<T>>),
            1 => (1, Arc::new(Butterfly2::new(direction)) as Arc<dyn Fft<T>>),
            2 => (2, Arc::new(Butterfly4::new(direction)) as Arc<dyn Fft<T>>),
            3 => (3, Arc::new(Butterfly8::new(direction)) as Arc<dyn Fft<T>>),
            _ => {
                if exponent % 2 == 1 {
                    (5, Arc::new(Butterfly32::new(direction)) as Arc<dyn Fft<T>>)
                } else {
                    (4, Arc::new(Butterfly16::new(direction)) as Arc<dyn Fft<T>>)
                }
            }
        };

        Self::new_with_base((exponent - base_exponent) / 2, base_fft)
    }

    /// Constructs a Radix4 instance which computes FFTs of length `4^k * base_fft.len()`
    pub fn new_with_base(k: u32, base_fft: Arc<dyn Fft<T>>) -> Self {
        let base_len = base_fft.len();
        let len = base_len * (1 << (k * 2));

        let direction = base_fft.fft_direction();

        const ROW_COUNT: usize = 4;
        let mut cross_fft_len = base_len;
        let mut twiddle_factors = Vec::with_capacity(len * 2);
        while cross_fft_len < len {
            let num_columns = cross_fft_len;
            cross_fft_len *= ROW_COUNT;

            for i in 0..num_columns {
                for k in 1..ROW_COUNT {
                    let twiddle = twiddles::compute_twiddle(i * k, cross_fft_len, direction);
                    twiddle_factors.push(twiddle);
                }
            }
        }

        let base_inplace_scratch = base_fft.get_inplace_scratch_len();
        let inplace_scratch_len = if base_inplace_scratch > cross_fft_len {
            cross_fft_len + base_inplace_scratch
        } else {
            cross_fft_len
        };
        let outofplace_scratch_len = if base_inplace_scratch > len {
            base_inplace_scratch
        } else {
            0
        };

        Self {
            twiddles: twiddle_factors.into_boxed_slice(),

            base_fft,
            base_len,

            len,
            direction,

            inplace_scratch_len,
            outofplace_scratch_len,
            immut_scratch_len: base_inplace_scratch,
        }
    }

    fn inplace_scratch_len(&self) -> usize {
        self.inplace_scratch_len
    }
    fn outofplace_scratch_len(&self) -> usize {
        self.outofplace_scratch_len
    }
    fn immut_scratch_len(&self) -> usize {
        self.immut_scratch_len
    }

    fn perform_fft_immut(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        if self.len() == self.base_len {
            output.copy_from_slice(input);
        } else {
            bitreversed_transpose::<Complex<T>, 4>(self.base_len, input, output);
        }

        self.base_fft.process_with_scratch(output, scratch);

        const ROW_COUNT: usize = 4;
        let mut cross_fft_len = self.base_len;
        let mut layer_twiddles: &[Complex<T>] = &self.twiddles;

        let butterfly4 = Butterfly4::new(self.direction);

        while cross_fft_len < output.len() {
            let num_columns = cross_fft_len;
            cross_fft_len *= ROW_COUNT;

            for data in output.chunks_exact_mut(cross_fft_len) {
                unsafe { butterfly_4(data, layer_twiddles, num_columns, &butterfly4) }
            }

            let twiddle_offset = num_columns * (ROW_COUNT - 1);
            layer_twiddles = &layer_twiddles[twiddle_offset..];
        }
    }

    fn perform_fft_out_of_place(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        if self.len() == self.base_len {
            output.copy_from_slice(input);
        } else {
            bitreversed_transpose::<Complex<T>, 4>(self.base_len, input, output);
        }

        let base_scratch = if scratch.len() > 0 { scratch } else { input };
        self.base_fft.process_with_scratch(output, base_scratch);

        const ROW_COUNT: usize = 4;
        let mut cross_fft_len = self.base_len;
        let mut layer_twiddles: &[Complex<T>] = &self.twiddles;

        let butterfly4 = Butterfly4::new(self.direction);

        while cross_fft_len < output.len() {
            let num_columns = cross_fft_len;
            cross_fft_len *= ROW_COUNT;

            for data in output.chunks_exact_mut(cross_fft_len) {
                unsafe { butterfly_4(data, layer_twiddles, num_columns, &butterfly4) }
            }

            let twiddle_offset = num_columns * (ROW_COUNT - 1);
            layer_twiddles = &layer_twiddles[twiddle_offset..];
        }
    }
}
boilerplate_fft_oop!(Radix4, |this: &Radix4<_>| this.len);
