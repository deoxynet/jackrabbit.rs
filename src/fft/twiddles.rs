use crate::fft::common::FftNum;
use crate::fft::FftDirection;
use num_complex::Complex;

pub fn compute_twiddle<T: FftNum>(
    index: usize,
    fft_len: usize,
    direction: FftDirection,
) -> Complex<T> {
    let constant = -2f64 * std::f64::consts::PI / fft_len as f64;
    let angle = constant * index as f64;

    let result = Complex {
        re: T::from_f64(angle.cos()).unwrap(),
        im: T::from_f64(angle.sin()).unwrap(),
    };

    match direction {
        FftDirection::Forward => result,
        FftDirection::Inverse => result.conj(),
    }
}

pub fn rotate_90<T: FftNum>(value: Complex<T>, direction: FftDirection) -> Complex<T> {
    match direction {
        FftDirection::Forward => Complex {
            re: value.im,
            im: -value.re,
        },
        FftDirection::Inverse => Complex {
            re: -value.im,
            im: value.re,
        },
    }
}
