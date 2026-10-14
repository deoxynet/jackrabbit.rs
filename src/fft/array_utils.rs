use crate::fft::common::RadixFactor;
use crate::fft::FftNum;
use num_complex::Complex;
use std::ops::{Deref, DerefMut};

/// Given an array of size width * height, representing a flattened 2D array,
/// transpose the rows and columns of that 2D array into the output
pub unsafe fn transpose_small<T: Copy>(width: usize, height: usize, input: &[T], output: &mut [T]) {
    for x in 0..width {
        for y in 0..height {
            let input_index = x + y * width;
            let output_index = y + x * height;

            *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
        }
    }
}

pub(crate) trait LoadStore<T: FftNum>: DerefMut {
    unsafe fn load(&self, idx: usize) -> Complex<T>;
    unsafe fn store(&mut self, val: Complex<T>, idx: usize);
}

impl<T: FftNum> LoadStore<T> for &mut [Complex<T>] {
    #[inline(always)]
    unsafe fn load(&self, idx: usize) -> Complex<T> {
        debug_assert!(idx < self.len());
        *self.get_unchecked(idx)
    }
    #[inline(always)]
    unsafe fn store(&mut self, val: Complex<T>, idx: usize) {
        debug_assert!(idx < self.len());
        *self.get_unchecked_mut(idx) = val;
    }
}
impl<T: FftNum, const N: usize> LoadStore<T> for &mut [Complex<T>; N] {
    #[inline(always)]
    unsafe fn load(&self, idx: usize) -> Complex<T> {
        debug_assert!(idx < self.len());
        *self.get_unchecked(idx)
    }
    #[inline(always)]
    unsafe fn store(&mut self, val: Complex<T>, idx: usize) {
        debug_assert!(idx < self.len());
        *self.get_unchecked_mut(idx) = val;
    }
}

pub(crate) struct DoubleBuf<'a, T> {
    pub input: &'a [Complex<T>],
    pub output: &'a mut [Complex<T>],
}
impl<'a, T> Deref for DoubleBuf<'a, T> {
    type Target = [Complex<T>];
    fn deref(&self) -> &Self::Target {
        self.input
    }
}
impl<'a, T> DerefMut for DoubleBuf<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.output
    }
}
impl<'a, T: FftNum> LoadStore<T> for DoubleBuf<'a, T> {
    #[inline(always)]
    unsafe fn load(&self, idx: usize) -> Complex<T> {
        debug_assert!(idx < self.input.len());
        *self.input.get_unchecked(idx)
    }
    #[inline(always)]
    unsafe fn store(&mut self, val: Complex<T>, idx: usize) {
        debug_assert!(idx < self.output.len());
        *self.output.get_unchecked_mut(idx) = val;
    }
}

pub(crate) trait Load<T: FftNum>: Deref {
    unsafe fn load(&self, idx: usize) -> Complex<T>;
}

impl<T: FftNum> Load<T> for &[Complex<T>] {
    #[inline(always)]
    unsafe fn load(&self, idx: usize) -> Complex<T> {
        debug_assert!(idx < self.len());
        *self.get_unchecked(idx)
    }
}
impl<T: FftNum, const N: usize> Load<T> for &[Complex<T>; N] {
    #[inline(always)]
    unsafe fn load(&self, idx: usize) -> Complex<T> {
        debug_assert!(idx < self.len());
        *self.get_unchecked(idx)
    }
}

pub fn validate_and_iter<T>(
    mut buffer: &mut [T],
    scratch: &mut [T],
    chunk_size: usize,
    required_scratch: usize,
    mut chunk_fn: impl FnMut(&mut [T], &mut [T]),
) -> Result<(), ()> {
    if scratch.len() < required_scratch {
        return Err(());
    }
    let scratch = &mut scratch[..required_scratch];

    while buffer.len() >= chunk_size {
        let (head, tail) = buffer.split_at_mut(chunk_size);
        buffer = tail;

        chunk_fn(head, scratch);
    }

    if buffer.len() == 0 {
        Ok(())
    } else {
        Err(())
    }
}

pub fn validate_and_zip<T>(
    mut buffer1: &[T],
    mut buffer2: &mut [T],
    scratch: &mut [T],
    chunk_size: usize,
    required_scratch: usize,
    mut chunk_fn: impl FnMut(&[T], &mut [T], &mut [T]),
) -> Result<(), ()> {
    if scratch.len() < required_scratch {
        return Err(());
    }
    let scratch = &mut scratch[..required_scratch];

    if buffer1.len() != buffer2.len() {
        return Err(());
    }

    while buffer1.len() >= chunk_size {
        let (head1, tail1) = buffer1.split_at(chunk_size);
        buffer1 = tail1;

        let (head2, tail2) = buffer2.split_at_mut(chunk_size);
        buffer2 = tail2;

        chunk_fn(head1, head2, scratch);
    }

    if buffer1.len() == 0 {
        Ok(())
    } else {
        Err(())
    }
}

pub fn validate_and_zip_mut<T>(
    mut buffer1: &mut [T],
    mut buffer2: &mut [T],
    scratch: &mut [T],
    chunk_size: usize,
    required_scratch: usize,
    mut chunk_fn: impl FnMut(&mut [T], &mut [T], &mut [T]),
) -> Result<(), ()> {
    if scratch.len() < required_scratch {
        return Err(());
    }
    let scratch = &mut scratch[..required_scratch];

    if buffer1.len() != buffer2.len() {
        return Err(());
    }

    while buffer1.len() >= chunk_size {
        let (head1, tail1) = buffer1.split_at_mut(chunk_size);
        buffer1 = tail1;

        let (head2, tail2) = buffer2.split_at_mut(chunk_size);
        buffer2 = tail2;

        chunk_fn(head1, head2, scratch);
    }

    if buffer1.len() == 0 {
        Ok(())
    } else {
        Err(())
    }
}

pub fn bitreversed_transpose<T: Copy, const D: usize>(
    height: usize,
    input: &[T],
    output: &mut [T],
) {
    let width = input.len() / height;

    assert!(D > 1 && input.len() % height == 0 && input.len() == output.len());

    let strided_width = width / D;
    let rev_digits = if D.is_power_of_two() {
        let width_bits = width.trailing_zeros();
        let d_bits = D.trailing_zeros();

        assert!(width_bits % d_bits == 0);
        width_bits / d_bits
    } else {
        compute_logarithm::<D>(width).unwrap()
    };

    for x in 0..strided_width {
        let mut i = 0;
        let x_fwd = [(); D].map(|_| {
            let value = D * x + i;
            i += 1;
            value
        });
        let x_rev = x_fwd.map(|x| reverse_bits::<D>(x, rev_digits));

        for r in x_rev {
            assert!(r < width);
        }
        for y in 0..height {
            for (fwd, rev) in x_fwd.iter().zip(x_rev.iter()) {
                let input_index = *fwd + y * width;
                let output_index = y + *rev * height;

                unsafe {
                    let temp = *input.get_unchecked(input_index);
                    *output.get_unchecked_mut(output_index) = temp;
                }
            }
        }
    }
}

pub fn reverse_bits<const D: usize>(value: usize, rev_digits: u32) -> usize {
    assert!(D > 1);

    let mut result: usize = 0;
    let mut value = value;
    for _ in 0..rev_digits {
        result = (result * D) + (value % D);
        value = value / D;
    }
    result
}

pub fn compute_logarithm<const D: usize>(value: usize) -> Option<u32> {
    if value == 0 || D < 2 {
        return None;
    }

    let mut current_exponent = 0;
    let mut current_value = value;

    while current_value % D == 0 {
        current_exponent += 1;
        current_value /= D;
    }

    if current_value == 1 {
        Some(current_exponent)
    } else {
        None
    }
}

pub(crate) struct TransposeFactor {
    pub factor: RadixFactor,
    pub count: u8,
}

pub(crate) fn factor_transpose<T: Copy, const D: usize>(
    height: usize,
    input: &[T],
    output: &mut [T],
    factors: &[TransposeFactor],
) {
    let width = input.len() / height;

    assert!(width % D == 0 && D > 1 && input.len() % width == 0 && input.len() == output.len());

    let strided_width = width / D;
    for x in 0..strided_width {
        let mut i = 0;
        let x_fwd = [(); D].map(|_| {
            let value = D * x + i;
            i += 1;
            value
        });
        let x_rev = x_fwd.map(|x| reverse_remainders(x, factors));

        for r in x_rev {
            assert!(r < width);
        }
        for y in 0..height {
            for (fwd, rev) in x_fwd.iter().zip(x_rev.iter()) {
                let input_index = *fwd + y * width;
                let output_index = y + *rev * height;

                unsafe {
                    let temp = *input.get_unchecked(input_index);
                    *output.get_unchecked_mut(output_index) = temp;
                }
            }
        }
    }
}

pub(crate) fn reverse_remainders(value: usize, factors: &[TransposeFactor]) -> usize {
    let mut result: usize = 0;
    let mut value = value;
    for f in factors.iter() {
        match f.factor {
            RadixFactor::Factor2 => {
                for _ in 0..f.count {
                    result = (result * 2) + (value % 2);
                    value = value / 2;
                }
            }
            RadixFactor::Factor3 => {
                for _ in 0..f.count {
                    result = (result * 3) + (value % 3);
                    value = value / 3;
                }
            }
            RadixFactor::Factor4 => {
                for _ in 0..f.count {
                    result = (result * 4) + (value % 4);
                    value = value / 4;
                }
            }
            RadixFactor::Factor5 => {
                for _ in 0..f.count {
                    result = (result * 5) + (value % 5);
                    value = value / 5;
                }
            }
            RadixFactor::Factor6 => {
                for _ in 0..f.count {
                    result = (result * 6) + (value % 6);
                    value = value / 6;
                }
            }
            RadixFactor::Factor7 => {
                for _ in 0..f.count {
                    result = (result * 7) + (value % 7);
                    value = value / 7;
                }
            }
        }
    }
    result
}
