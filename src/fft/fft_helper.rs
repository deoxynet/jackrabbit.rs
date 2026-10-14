use crate::fft::{
    array_utils,
    common::{fft_error_immut, fft_error_inplace, fft_error_outofplace},
};

#[inline(always)]
pub fn fft_helper_inplace<T>(
    buffer: &mut [T],
    scratch: &mut [T],
    chunk_size: usize,
    required_scratch: usize,
    chunk_fn: impl FnMut(&mut [T], &mut [T]),
) {
    if chunk_size == 0 {
        return;
    }

    let result =
        array_utils::validate_and_iter(buffer, scratch, chunk_size, required_scratch, chunk_fn);

    if result.is_err() {
        fft_error_inplace(chunk_size, buffer.len(), required_scratch, scratch.len());
    }
}

#[inline(always)]
pub fn fft_helper_immut<T>(
    input: &[T],
    output: &mut [T],
    scratch: &mut [T],
    chunk_size: usize,
    required_scratch: usize,
    chunk_fn: impl FnMut(&[T], &mut [T], &mut [T]),
) {
    if chunk_size == 0 {
        return;
    }

    let result = array_utils::validate_and_zip(
        input,
        output,
        scratch,
        chunk_size,
        required_scratch,
        chunk_fn,
    );

    if result.is_err() {
        fft_error_immut(
            chunk_size,
            input.len(),
            output.len(),
            required_scratch,
            scratch.len(),
        );
    }
}

#[inline(always)]
pub fn fft_helper_outofplace<T>(
    input: &mut [T],
    output: &mut [T],
    scratch: &mut [T],
    chunk_size: usize,
    required_scratch: usize,
    chunk_fn: impl FnMut(&mut [T], &mut [T], &mut [T]),
) {
    if chunk_size == 0 {
        return;
    }

    let result = array_utils::validate_and_zip_mut(
        input,
        output,
        scratch,
        chunk_size,
        required_scratch,
        chunk_fn,
    );

    if result.is_err() {
        fft_error_outofplace(
            chunk_size,
            input.len(),
            output.len(),
            required_scratch,
            scratch.len(),
        );
    }
}
