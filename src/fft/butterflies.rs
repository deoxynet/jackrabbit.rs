use num_complex::Complex;

use crate::fft::common::FftNum;
use crate::fft::FftDirection;

use crate::fft::array_utils::{DoubleBuf, LoadStore};
use crate::fft::twiddles;
use crate::fft::{Direction, Fft, Length};

#[allow(unused)]
macro_rules! boilerplate_fft_butterfly {
    ($struct_name:ident, $len:expr, $direction_fn:expr) => {
        impl<T: FftNum> $struct_name<T> {
            #[inline(always)]
            pub(crate) unsafe fn perform_fft_butterfly(&self, buffer: impl LoadStore<T>) {
                self.perform_fft_contiguous(buffer);
            }
        }
        impl<T: FftNum> Fft<T> for $struct_name<T> {
            #[inline]
            fn process_immutable_with_scratch(
                &self,
                input: &[Complex<T>],
                output: &mut [Complex<T>],
                _scratch: &mut [Complex<T>],
            ) {
                crate::fft::fft_helper::fft_helper_immut(
                    input,
                    output,
                    &mut [],
                    self.len(),
                    0,
                    |in_chunk, out_chunk, _| unsafe {
                        self.perform_fft_butterfly(DoubleBuf {
                            input: in_chunk,
                            output: out_chunk,
                        })
                    },
                );
            }
            fn process_outofplace_with_scratch(
                &self,
                input: &mut [Complex<T>],
                output: &mut [Complex<T>],
                _scratch: &mut [Complex<T>],
            ) {
                crate::fft::fft_helper::fft_helper_outofplace(
                    input,
                    output,
                    &mut [],
                    self.len(),
                    0,
                    |in_chunk, out_chunk, _| unsafe {
                        self.perform_fft_butterfly(DoubleBuf {
                            input: in_chunk,
                            output: out_chunk,
                        })
                    },
                );
            }
            fn process_with_scratch(&self, buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
                crate::fft::fft_helper::fft_helper_inplace(
                    buffer,
                    &mut [],
                    self.len(),
                    0,
                    |chunk, _| unsafe { self.perform_fft_butterfly(chunk) },
                );
            }
            #[inline(always)]
            fn get_inplace_scratch_len(&self) -> usize {
                0
            }
            #[inline(always)]
            fn get_outofplace_scratch_len(&self) -> usize {
                0
            }
            #[inline(always)]
            fn get_immutable_scratch_len(&self) -> usize {
                0
            }
        }
        impl<T> Length for $struct_name<T> {
            #[inline(always)]
            fn len(&self) -> usize {
                $len
            }
        }
        impl<T> Direction for $struct_name<T> {
            #[inline(always)]
            fn fft_direction(&self) -> FftDirection {
                $direction_fn(self)
            }
        }
    };
}

pub struct Butterfly1<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
}
impl<T: FftNum> Butterfly1<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            direction,
            _phantom: std::marker::PhantomData,
        }
    }
}
impl<T: FftNum> Fft<T> for Butterfly1<T> {
    fn process_immutable_with_scratch(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        _scratch: &mut [Complex<T>],
    ) {
        output.copy_from_slice(input);
    }

    fn process_outofplace_with_scratch(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        _scratch: &mut [Complex<T>],
    ) {
        output.copy_from_slice(input);
    }

    fn process_with_scratch(&self, _buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {}

    fn get_inplace_scratch_len(&self) -> usize {
        0
    }

    fn get_outofplace_scratch_len(&self) -> usize {
        0
    }

    fn get_immutable_scratch_len(&self) -> usize {
        0
    }
}
impl<T> Length for Butterfly1<T> {
    fn len(&self) -> usize {
        1
    }
}
impl<T> Direction for Butterfly1<T> {
    fn fft_direction(&self) -> FftDirection {
        self.direction
    }
}

pub struct Butterfly2<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_butterfly!(Butterfly2, 2, |this: &Butterfly2<_>| this.direction);
impl<T: FftNum> Butterfly2<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            direction,
            _phantom: std::marker::PhantomData,
        }
    }
    #[inline(always)]
    unsafe fn perform_fft_strided(left: &mut Complex<T>, right: &mut Complex<T>) {
        let temp = *left + *right;

        *right = *left - *right;
        *left = temp;
    }
    #[inline(always)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl LoadStore<T>) {
        let value0 = buffer.load(0);
        let value1 = buffer.load(1);
        buffer.store(value0 + value1, 0);
        buffer.store(value0 - value1, 1);
    }
}

pub struct Butterfly3<T> {
    pub twiddle: Complex<T>,
    direction: FftDirection,
}
boilerplate_fft_butterfly!(Butterfly3, 3, |this: &Butterfly3<_>| this.direction);
impl<T: FftNum> Butterfly3<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            twiddle: twiddles::compute_twiddle(1, 3, direction),
            direction,
        }
    }
    #[inline(always)]
    pub fn direction_of(fft: &Butterfly3<T>) -> Self {
        Self {
            twiddle: fft.twiddle.conj(),
            direction: fft.direction.opposite_direction(),
        }
    }
    #[inline(always)]
    unsafe fn perform_fft_strided(
        &self,
        val0: &mut Complex<T>,
        val1: &mut Complex<T>,
        val2: &mut Complex<T>,
    ) {
        let xp = *val1 + *val2;
        let xn = *val1 - *val2;
        let sum = *val0 + xp;

        let temp_a = *val0
            + Complex {
                re: self.twiddle.re * xp.re,
                im: self.twiddle.re * xp.im,
            };
        let temp_b = Complex {
            re: -self.twiddle.im * xn.im,
            im: self.twiddle.im * xn.re,
        };

        *val0 = sum;
        *val1 = temp_a + temp_b;
        *val2 = temp_a - temp_b;
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl LoadStore<T>) {
        let xp = buffer.load(1) + buffer.load(2);
        let xn = buffer.load(1) - buffer.load(2);
        let sum = buffer.load(0) + xp;

        let temp_a = buffer.load(0)
            + Complex {
                re: self.twiddle.re * xp.re,
                im: self.twiddle.re * xp.im,
            };
        let temp_b = Complex {
            re: -self.twiddle.im * xn.im,
            im: self.twiddle.im * xn.re,
        };

        buffer.store(sum, 0);
        buffer.store(temp_a + temp_b, 1);
        buffer.store(temp_a - temp_b, 2);
    }
}

pub struct Butterfly4<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
}
boilerplate_fft_butterfly!(Butterfly4, 4, |this: &Butterfly4<_>| this.direction);
impl<T: FftNum> Butterfly4<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            direction,
            _phantom: std::marker::PhantomData,
        }
    }
    #[inline(always)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl LoadStore<T>) {
        let mut value0 = buffer.load(0);
        let mut value1 = buffer.load(1);
        let mut value2 = buffer.load(2);
        let mut value3 = buffer.load(3);

        Butterfly2::perform_fft_strided(&mut value0, &mut value2);
        Butterfly2::perform_fft_strided(&mut value1, &mut value3);

        value3 = twiddles::rotate_90(value3, self.direction);

        Butterfly2::perform_fft_strided(&mut value0, &mut value1);
        Butterfly2::perform_fft_strided(&mut value2, &mut value3);

        buffer.store(value0, 0);
        buffer.store(value2, 1);
        buffer.store(value1, 2);
        buffer.store(value3, 3);
    }

    #[inline(always)]
    unsafe fn perform_fft_strided(
        &self,
        value0: &mut Complex<T>,
        value1: &mut Complex<T>,
        value2: &mut Complex<T>,
        value3: &mut Complex<T>,
    ) {
        Butterfly2::perform_fft_strided(value0, value2);
        Butterfly2::perform_fft_strided(value1, value3);

        *value3 = twiddles::rotate_90(*value3, self.direction);

        Butterfly2::perform_fft_strided(value0, value1);
        Butterfly2::perform_fft_strided(value2, value3);

        let temp = *value1;
        *value1 = *value2;
        *value2 = temp;
    }
}

pub struct Butterfly5<T> {
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    direction: FftDirection,
}
boilerplate_fft_butterfly!(Butterfly5, 5, |this: &Butterfly5<_>| this.direction);
impl<T: FftNum> Butterfly5<T> {
    pub fn new(direction: FftDirection) -> Self {
        Self {
            twiddle1: twiddles::compute_twiddle(1, 5, direction),
            twiddle2: twiddles::compute_twiddle(2, 5, direction),
            direction,
        }
    }

    #[inline(never)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl LoadStore<T>) {
        let x14p = buffer.load(1) + buffer.load(4);
        let x14n = buffer.load(1) - buffer.load(4);
        let x23p = buffer.load(2) + buffer.load(3);
        let x23n = buffer.load(2) - buffer.load(3);
        let sum = buffer.load(0) + x14p + x23p;
        let b14re_a = buffer.load(0).re + self.twiddle1.re * x14p.re + self.twiddle2.re * x23p.re;
        let b14re_b = self.twiddle1.im * x14n.im + self.twiddle2.im * x23n.im;
        let b23re_a = buffer.load(0).re + self.twiddle2.re * x14p.re + self.twiddle1.re * x23p.re;
        let b23re_b = self.twiddle2.im * x14n.im + -self.twiddle1.im * x23n.im;

        let b14im_a = buffer.load(0).im + self.twiddle1.re * x14p.im + self.twiddle2.re * x23p.im;
        let b14im_b = self.twiddle1.im * x14n.re + self.twiddle2.im * x23n.re;
        let b23im_a = buffer.load(0).im + self.twiddle2.re * x14p.im + self.twiddle1.re * x23p.im;
        let b23im_b = self.twiddle2.im * x14n.re + -self.twiddle1.im * x23n.re;

        let out1re = b14re_a - b14re_b;
        let out1im = b14im_a + b14im_b;
        let out2re = b23re_a - b23re_b;
        let out2im = b23im_a + b23im_b;
        let out3re = b23re_a + b23re_b;
        let out3im = b23im_a - b23im_b;
        let out4re = b14re_a + b14re_b;
        let out4im = b14im_a - b14im_b;
        buffer.store(sum, 0);
        buffer.store(
            Complex {
                re: out1re,
                im: out1im,
            },
            1,
        );
        buffer.store(
            Complex {
                re: out2re,
                im: out2im,
            },
            2,
        );
        buffer.store(
            Complex {
                re: out3re,
                im: out3im,
            },
            3,
        );
        buffer.store(
            Complex {
                re: out4re,
                im: out4im,
            },
            4,
        );
    }
}

pub struct Butterfly6<T> {
    butterfly3: Butterfly3<T>,
}
boilerplate_fft_butterfly!(Butterfly6, 6, |this: &Butterfly6<_>| this
    .butterfly3
    .fft_direction());
impl<T: FftNum> Butterfly6<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            butterfly3: Butterfly3::new(direction),
        }
    }
    #[inline(always)]
    pub fn direction_of(fft: &Butterfly6<T>) -> Self {
        Self {
            butterfly3: Butterfly3::direction_of(&fft.butterfly3),
        }
    }
    #[inline(always)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl LoadStore<T>) {
        let mut scratch_a = [buffer.load(0), buffer.load(2), buffer.load(4)];

        let mut scratch_b = [buffer.load(3), buffer.load(5), buffer.load(1)];

        self.butterfly3.perform_fft_contiguous(&mut scratch_a);
        self.butterfly3.perform_fft_contiguous(&mut scratch_b);

        Butterfly2::perform_fft_strided(&mut scratch_a[0], &mut scratch_b[0]);
        Butterfly2::perform_fft_strided(&mut scratch_a[1], &mut scratch_b[1]);
        Butterfly2::perform_fft_strided(&mut scratch_a[2], &mut scratch_b[2]);

        buffer.store(scratch_a[0], 0);
        buffer.store(scratch_b[1], 1);
        buffer.store(scratch_a[2], 2);
        buffer.store(scratch_b[0], 3);
        buffer.store(scratch_a[1], 4);
        buffer.store(scratch_b[2], 5);
    }
}

pub struct Butterfly7<T> {
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    direction: FftDirection,
}
boilerplate_fft_butterfly!(Butterfly7, 7, |this: &Butterfly7<_>| this.direction);
impl<T: FftNum> Butterfly7<T> {
    pub fn new(direction: FftDirection) -> Self {
        Self {
            twiddle1: twiddles::compute_twiddle(1, 7, direction),
            twiddle2: twiddles::compute_twiddle(2, 7, direction),
            twiddle3: twiddles::compute_twiddle(3, 7, direction),
            direction,
        }
    }
    #[inline(never)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl LoadStore<T>) {
        let x16p = buffer.load(1) + buffer.load(6);
        let x16n = buffer.load(1) - buffer.load(6);
        let x25p = buffer.load(2) + buffer.load(5);
        let x25n = buffer.load(2) - buffer.load(5);
        let x34p = buffer.load(3) + buffer.load(4);
        let x34n = buffer.load(3) - buffer.load(4);
        let sum = buffer.load(0) + x16p + x25p + x34p;

        let x16re_a = buffer.load(0).re
            + self.twiddle1.re * x16p.re
            + self.twiddle2.re * x25p.re
            + self.twiddle3.re * x34p.re;
        let x16re_b =
            self.twiddle1.im * x16n.im + self.twiddle2.im * x25n.im + self.twiddle3.im * x34n.im;
        let x25re_a = buffer.load(0).re
            + self.twiddle1.re * x34p.re
            + self.twiddle2.re * x16p.re
            + self.twiddle3.re * x25p.re;
        let x25re_b =
            -self.twiddle1.im * x34n.im + self.twiddle2.im * x16n.im - self.twiddle3.im * x25n.im;
        let x34re_a = buffer.load(0).re
            + self.twiddle1.re * x25p.re
            + self.twiddle2.re * x34p.re
            + self.twiddle3.re * x16p.re;
        let x34re_b =
            -self.twiddle1.im * x25n.im + self.twiddle2.im * x34n.im + self.twiddle3.im * x16n.im;
        let x16im_a = buffer.load(0).im
            + self.twiddle1.re * x16p.im
            + self.twiddle2.re * x25p.im
            + self.twiddle3.re * x34p.im;
        let x16im_b =
            self.twiddle1.im * x16n.re + self.twiddle2.im * x25n.re + self.twiddle3.im * x34n.re;
        let x25im_a = buffer.load(0).im
            + self.twiddle1.re * x34p.im
            + self.twiddle2.re * x16p.im
            + self.twiddle3.re * x25p.im;
        let x25im_b =
            -self.twiddle1.im * x34n.re + self.twiddle2.im * x16n.re - self.twiddle3.im * x25n.re;
        let x34im_a = buffer.load(0).im
            + self.twiddle1.re * x25p.im
            + self.twiddle2.re * x34p.im
            + self.twiddle3.re * x16p.im;
        let x34im_b =
            self.twiddle1.im * x25n.re - self.twiddle2.im * x34n.re - self.twiddle3.im * x16n.re;

        let out1re = x16re_a - x16re_b;
        let out1im = x16im_a + x16im_b;
        let out2re = x25re_a - x25re_b;
        let out2im = x25im_a + x25im_b;
        let out3re = x34re_a - x34re_b;
        let out3im = x34im_a - x34im_b;
        let out4re = x34re_a + x34re_b;
        let out4im = x34im_a + x34im_b;
        let out5re = x25re_a + x25re_b;
        let out5im = x25im_a - x25im_b;
        let out6re = x16re_a + x16re_b;
        let out6im = x16im_a - x16im_b;

        buffer.store(sum, 0);
        buffer.store(
            Complex {
                re: out1re,
                im: out1im,
            },
            1,
        );
        buffer.store(
            Complex {
                re: out2re,
                im: out2im,
            },
            2,
        );
        buffer.store(
            Complex {
                re: out3re,
                im: out3im,
            },
            3,
        );
        buffer.store(
            Complex {
                re: out4re,
                im: out4im,
            },
            4,
        );
        buffer.store(
            Complex {
                re: out5re,
                im: out5im,
            },
            5,
        );
        buffer.store(
            Complex {
                re: out6re,
                im: out6im,
            },
            6,
        );
    }
}

pub struct Butterfly8<T> {
    root2: T,
    direction: FftDirection,
}
boilerplate_fft_butterfly!(Butterfly8, 8, |this: &Butterfly8<_>| this.direction);
impl<T: FftNum> Butterfly8<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            root2: T::from_f64(0.5f64.sqrt()).unwrap(),
            direction,
        }
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl LoadStore<T>) {
        let butterfly4 = Butterfly4::new(self.direction);

        let mut scratch0 = [
            buffer.load(0),
            buffer.load(2),
            buffer.load(4),
            buffer.load(6),
        ];
        let mut scratch1 = [
            buffer.load(1),
            buffer.load(3),
            buffer.load(5),
            buffer.load(7),
        ];

        butterfly4.perform_fft_contiguous(&mut scratch0);
        butterfly4.perform_fft_contiguous(&mut scratch1);

        scratch1[1] = (twiddles::rotate_90(scratch1[1], self.direction) + scratch1[1]) * self.root2;
        scratch1[2] = twiddles::rotate_90(scratch1[2], self.direction);
        scratch1[3] = (twiddles::rotate_90(scratch1[3], self.direction) - scratch1[3]) * self.root2;

        for i in 0..4 {
            Butterfly2::perform_fft_strided(&mut scratch0[i], &mut scratch1[i]);
        }

        for i in 0..4 {
            buffer.store(scratch0[i], i);
        }
        for i in 0..4 {
            buffer.store(scratch1[i], i + 4);
        }
    }
}

pub struct Butterfly9<T> {
    pub(crate) butterfly3: Butterfly3<T>,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle4: Complex<T>,
}
boilerplate_fft_butterfly!(Butterfly9, 9, |this: &Butterfly9<_>| this
    .butterfly3
    .fft_direction());
impl<T: FftNum> Butterfly9<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            butterfly3: Butterfly3::new(direction),
            twiddle1: twiddles::compute_twiddle(1, 9, direction),
            twiddle2: twiddles::compute_twiddle(2, 9, direction),
            twiddle4: twiddles::compute_twiddle(4, 9, direction),
        }
    }
    #[inline(always)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl LoadStore<T>) {
        let mut scratch0 = [buffer.load(0), buffer.load(3), buffer.load(6)];
        let mut scratch1 = [buffer.load(1), buffer.load(4), buffer.load(7)];
        let mut scratch2 = [buffer.load(2), buffer.load(5), buffer.load(8)];

        self.butterfly3.perform_fft_contiguous(&mut scratch0);
        self.butterfly3.perform_fft_contiguous(&mut scratch1);
        self.butterfly3.perform_fft_contiguous(&mut scratch2);

        scratch1[1] = scratch1[1] * self.twiddle1;
        scratch1[2] = scratch1[2] * self.twiddle2;
        scratch2[1] = scratch2[1] * self.twiddle2;
        scratch2[2] = scratch2[2] * self.twiddle4;

        self.butterfly3
            .perform_fft_strided(&mut scratch0[0], &mut scratch1[0], &mut scratch2[0]);
        self.butterfly3
            .perform_fft_strided(&mut scratch0[1], &mut scratch1[1], &mut scratch2[1]);
        self.butterfly3
            .perform_fft_strided(&mut scratch0[2], &mut scratch1[2], &mut scratch2[2]);

        buffer.store(scratch0[0], 0);
        buffer.store(scratch0[1], 1);
        buffer.store(scratch0[2], 2);
        buffer.store(scratch1[0], 3);
        buffer.store(scratch1[1], 4);
        buffer.store(scratch1[2], 5);
        buffer.store(scratch2[0], 6);
        buffer.store(scratch2[1], 7);
        buffer.store(scratch2[2], 8);
    }
}

pub struct Butterfly12<T> {
    butterfly3: Butterfly3<T>,
    butterfly4: Butterfly4<T>,
}
boilerplate_fft_butterfly!(Butterfly12, 12, |this: &Butterfly12<_>| this
    .butterfly3
    .fft_direction());
impl<T: FftNum> Butterfly12<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            butterfly3: Butterfly3::new(direction),
            butterfly4: Butterfly4::new(direction),
        }
    }
    #[inline(always)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl LoadStore<T>) {
        let mut scratch0 = [
            buffer.load(0),
            buffer.load(3),
            buffer.load(6),
            buffer.load(9),
        ];
        let mut scratch1 = [
            buffer.load(4),
            buffer.load(7),
            buffer.load(10),
            buffer.load(1),
        ];
        let mut scratch2 = [
            buffer.load(8),
            buffer.load(11),
            buffer.load(2),
            buffer.load(5),
        ];

        self.butterfly4.perform_fft_contiguous(&mut scratch0);
        self.butterfly4.perform_fft_contiguous(&mut scratch1);
        self.butterfly4.perform_fft_contiguous(&mut scratch2);

        self.butterfly3
            .perform_fft_strided(&mut scratch0[0], &mut scratch1[0], &mut scratch2[0]);
        self.butterfly3
            .perform_fft_strided(&mut scratch0[1], &mut scratch1[1], &mut scratch2[1]);
        self.butterfly3
            .perform_fft_strided(&mut scratch0[2], &mut scratch1[2], &mut scratch2[2]);
        self.butterfly3
            .perform_fft_strided(&mut scratch0[3], &mut scratch1[3], &mut scratch2[3]);

        buffer.store(scratch0[0], 0);
        buffer.store(scratch1[1], 1);
        buffer.store(scratch2[2], 2);
        buffer.store(scratch0[3], 3);
        buffer.store(scratch1[0], 4);
        buffer.store(scratch2[1], 5);
        buffer.store(scratch0[2], 6);
        buffer.store(scratch1[3], 7);
        buffer.store(scratch2[0], 8);
        buffer.store(scratch0[1], 9);
        buffer.store(scratch1[2], 10);
        buffer.store(scratch2[3], 11);
    }
}

pub struct Butterfly16<T> {
    butterfly8: Butterfly8<T>,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
}
boilerplate_fft_butterfly!(Butterfly16, 16, |this: &Butterfly16<_>| this
    .butterfly8
    .fft_direction());
impl<T: FftNum> Butterfly16<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            butterfly8: Butterfly8::new(direction),
            twiddle1: twiddles::compute_twiddle(1, 16, direction),
            twiddle2: twiddles::compute_twiddle(2, 16, direction),
            twiddle3: twiddles::compute_twiddle(3, 16, direction),
        }
    }

    #[inline(never)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl LoadStore<T>) {
        let butterfly4 = Butterfly4::new(self.fft_direction());

        let mut scratch_evens = [
            buffer.load(0),
            buffer.load(2),
            buffer.load(4),
            buffer.load(6),
            buffer.load(8),
            buffer.load(10),
            buffer.load(12),
            buffer.load(14),
        ];

        let mut scratch_odds_n1 = [
            buffer.load(1),
            buffer.load(5),
            buffer.load(9),
            buffer.load(13),
        ];
        let mut scratch_odds_n3 = [
            buffer.load(15),
            buffer.load(3),
            buffer.load(7),
            buffer.load(11),
        ];

        self.butterfly8.perform_fft_contiguous(&mut scratch_evens);
        butterfly4.perform_fft_contiguous(&mut scratch_odds_n1);
        butterfly4.perform_fft_contiguous(&mut scratch_odds_n3);

        scratch_odds_n1[1] = scratch_odds_n1[1] * self.twiddle1;
        scratch_odds_n3[1] = scratch_odds_n3[1] * self.twiddle1.conj();

        scratch_odds_n1[2] = scratch_odds_n1[2] * self.twiddle2;
        scratch_odds_n3[2] = scratch_odds_n3[2] * self.twiddle2.conj();

        scratch_odds_n1[3] = scratch_odds_n1[3] * self.twiddle3;
        scratch_odds_n3[3] = scratch_odds_n3[3] * self.twiddle3.conj();

        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[0], &mut scratch_odds_n3[0]);
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[1], &mut scratch_odds_n3[1]);
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[2], &mut scratch_odds_n3[2]);
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[3], &mut scratch_odds_n3[3]);

        scratch_odds_n3[0] = twiddles::rotate_90(scratch_odds_n3[0], self.fft_direction());
        scratch_odds_n3[1] = twiddles::rotate_90(scratch_odds_n3[1], self.fft_direction());
        scratch_odds_n3[2] = twiddles::rotate_90(scratch_odds_n3[2], self.fft_direction());
        scratch_odds_n3[3] = twiddles::rotate_90(scratch_odds_n3[3], self.fft_direction());

        buffer.store(scratch_evens[0] + scratch_odds_n1[0], 0);
        buffer.store(scratch_evens[1] + scratch_odds_n1[1], 1);
        buffer.store(scratch_evens[2] + scratch_odds_n1[2], 2);
        buffer.store(scratch_evens[3] + scratch_odds_n1[3], 3);
        buffer.store(scratch_evens[4] + scratch_odds_n3[0], 4);
        buffer.store(scratch_evens[5] + scratch_odds_n3[1], 5);
        buffer.store(scratch_evens[6] + scratch_odds_n3[2], 6);
        buffer.store(scratch_evens[7] + scratch_odds_n3[3], 7);
        buffer.store(scratch_evens[0] - scratch_odds_n1[0], 8);
        buffer.store(scratch_evens[1] - scratch_odds_n1[1], 9);
        buffer.store(scratch_evens[2] - scratch_odds_n1[2], 10);
        buffer.store(scratch_evens[3] - scratch_odds_n1[3], 11);
        buffer.store(scratch_evens[4] - scratch_odds_n3[0], 12);
        buffer.store(scratch_evens[5] - scratch_odds_n3[1], 13);
        buffer.store(scratch_evens[6] - scratch_odds_n3[2], 14);
        buffer.store(scratch_evens[7] - scratch_odds_n3[3], 15);
    }
}

pub struct Butterfly24<T> {
    butterfly4: Butterfly4<T>,
    butterfly6: Butterfly6<T>,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle8: Complex<T>,
    twiddle10: Complex<T>,
    root2: T,
}
boilerplate_fft_butterfly!(Butterfly24, 24, |this: &Butterfly24<_>| this
    .butterfly4
    .fft_direction());
impl<T: FftNum> Butterfly24<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            butterfly4: Butterfly4::new(direction),
            butterfly6: Butterfly6::new(direction),
            twiddle1: twiddles::compute_twiddle(1, 24, direction),
            twiddle2: twiddles::compute_twiddle(2, 24, direction),
            twiddle4: twiddles::compute_twiddle(4, 24, direction),
            twiddle5: twiddles::compute_twiddle(5, 24, direction),
            twiddle8: twiddles::compute_twiddle(8, 24, direction),
            twiddle10: twiddles::compute_twiddle(10, 24, direction),
            root2: T::from_f64(0.5f64.sqrt()).unwrap(),
        }
    }
    #[inline(never)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl LoadStore<T>) {
        let mut scratch0 = [
            buffer.load(0),
            buffer.load(4),
            buffer.load(8),
            buffer.load(12),
            buffer.load(16),
            buffer.load(20),
        ];
        let mut scratch1 = [
            buffer.load(1),
            buffer.load(5),
            buffer.load(9),
            buffer.load(13),
            buffer.load(17),
            buffer.load(21),
        ];
        let mut scratch2 = [
            buffer.load(2),
            buffer.load(6),
            buffer.load(10),
            buffer.load(14),
            buffer.load(18),
            buffer.load(22),
        ];
        let mut scratch3 = [
            buffer.load(3),
            buffer.load(7),
            buffer.load(11),
            buffer.load(15),
            buffer.load(19),
            buffer.load(23),
        ];

        self.butterfly6.perform_fft_contiguous(&mut scratch0);
        self.butterfly6.perform_fft_contiguous(&mut scratch1);
        self.butterfly6.perform_fft_contiguous(&mut scratch2);
        self.butterfly6.perform_fft_contiguous(&mut scratch3);

        scratch1[1] = scratch1[1] * self.twiddle1;
        scratch1[2] = scratch1[2] * self.twiddle2;
        scratch1[3] =
            (twiddles::rotate_90(scratch1[3], self.fft_direction()) + scratch1[3]) * self.root2;
        scratch1[4] = scratch1[4] * self.twiddle4;
        scratch1[5] = scratch1[5] * self.twiddle5;
        scratch2[1] = scratch2[1] * self.twiddle2;
        scratch2[2] = scratch2[2] * self.twiddle4;
        scratch2[3] = twiddles::rotate_90(scratch2[3], self.fft_direction());
        scratch2[4] = scratch2[4] * self.twiddle8;
        scratch2[5] = scratch2[5] * self.twiddle10;
        scratch3[1] =
            (twiddles::rotate_90(scratch3[1], self.fft_direction()) + scratch3[1]) * self.root2;
        scratch3[2] = twiddles::rotate_90(scratch3[2], self.fft_direction());
        scratch3[3] =
            (twiddles::rotate_90(scratch3[3], self.fft_direction()) - scratch3[3]) * self.root2;
        scratch3[4] = -scratch3[4];
        scratch3[5] =
            (twiddles::rotate_90(scratch3[5], self.fft_direction()) + scratch3[5]) * -self.root2;

        self.butterfly4.perform_fft_strided(
            &mut scratch0[0],
            &mut scratch1[0],
            &mut scratch2[0],
            &mut scratch3[0],
        );
        self.butterfly4.perform_fft_strided(
            &mut scratch0[1],
            &mut scratch1[1],
            &mut scratch2[1],
            &mut scratch3[1],
        );
        self.butterfly4.perform_fft_strided(
            &mut scratch0[2],
            &mut scratch1[2],
            &mut scratch2[2],
            &mut scratch3[2],
        );
        self.butterfly4.perform_fft_strided(
            &mut scratch0[3],
            &mut scratch1[3],
            &mut scratch2[3],
            &mut scratch3[3],
        );
        self.butterfly4.perform_fft_strided(
            &mut scratch0[4],
            &mut scratch1[4],
            &mut scratch2[4],
            &mut scratch3[4],
        );
        self.butterfly4.perform_fft_strided(
            &mut scratch0[5],
            &mut scratch1[5],
            &mut scratch2[5],
            &mut scratch3[5],
        );

        buffer.store(scratch0[0], 0);
        buffer.store(scratch0[1], 1);
        buffer.store(scratch0[2], 2);
        buffer.store(scratch0[3], 3);
        buffer.store(scratch0[4], 4);
        buffer.store(scratch0[5], 5);
        buffer.store(scratch1[0], 6);
        buffer.store(scratch1[1], 7);
        buffer.store(scratch1[2], 8);
        buffer.store(scratch1[3], 9);
        buffer.store(scratch1[4], 10);
        buffer.store(scratch1[5], 11);
        buffer.store(scratch2[0], 12);
        buffer.store(scratch2[1], 13);
        buffer.store(scratch2[2], 14);
        buffer.store(scratch2[3], 15);
        buffer.store(scratch2[4], 16);
        buffer.store(scratch2[5], 17);
        buffer.store(scratch3[0], 18);
        buffer.store(scratch3[1], 19);
        buffer.store(scratch3[2], 20);
        buffer.store(scratch3[3], 21);
        buffer.store(scratch3[4], 22);
        buffer.store(scratch3[5], 23);
    }
}

pub struct Butterfly27<T> {
    butterfly9: Butterfly9<T>,
    twiddles: [Complex<T>; 12],
}
boilerplate_fft_butterfly!(Butterfly27, 27, |this: &Butterfly27<_>| this
    .butterfly9
    .fft_direction());
impl<T: FftNum> Butterfly27<T> {
    #[inline(always)]
    pub fn new(direction: FftDirection) -> Self {
        Self {
            butterfly9: Butterfly9::new(direction),
            twiddles: [
                twiddles::compute_twiddle(1, 27, direction),
                twiddles::compute_twiddle(2, 27, direction),
                twiddles::compute_twiddle(3, 27, direction),
                twiddles::compute_twiddle(4, 27, direction),
                twiddles::compute_twiddle(5, 27, direction),
                twiddles::compute_twiddle(6, 27, direction),
                twiddles::compute_twiddle(7, 27, direction),
                twiddles::compute_twiddle(8, 27, direction),
                twiddles::compute_twiddle(10, 27, direction),
                twiddles::compute_twiddle(12, 27, direction),
                twiddles::compute_twiddle(14, 27, direction),
                twiddles::compute_twiddle(16, 27, direction),
            ],
        }
    }

    #[inline(always)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl LoadStore<T>) {
        let mut scratch0 = [
            buffer.load(0),
            buffer.load(3),
            buffer.load(6),
            buffer.load(9),
            buffer.load(12),
            buffer.load(15),
            buffer.load(18),
            buffer.load(21),
            buffer.load(24),
        ];
        let mut scratch1 = [
            buffer.load(1 + 0),
            buffer.load(1 + 3),
            buffer.load(1 + 6),
            buffer.load(1 + 9),
            buffer.load(1 + 12),
            buffer.load(1 + 15),
            buffer.load(1 + 18),
            buffer.load(1 + 21),
            buffer.load(1 + 24),
        ];
        let mut scratch2 = [
            buffer.load(2 + 0),
            buffer.load(2 + 3),
            buffer.load(2 + 6),
            buffer.load(2 + 9),
            buffer.load(2 + 12),
            buffer.load(2 + 15),
            buffer.load(2 + 18),
            buffer.load(2 + 21),
            buffer.load(2 + 24),
        ];

        self.butterfly9.perform_fft_contiguous(&mut scratch0);
        self.butterfly9.perform_fft_contiguous(&mut scratch1);
        self.butterfly9.perform_fft_contiguous(&mut scratch2);

        scratch1[1] = scratch1[1] * self.twiddles[0];
        scratch1[2] = scratch1[2] * self.twiddles[1];
        scratch1[3] = scratch1[3] * self.twiddles[2];
        scratch1[4] = scratch1[4] * self.twiddles[3];
        scratch1[5] = scratch1[5] * self.twiddles[4];
        scratch1[6] = scratch1[6] * self.twiddles[5];
        scratch1[7] = scratch1[7] * self.twiddles[6];
        scratch1[8] = scratch1[8] * self.twiddles[7];
        scratch2[1] = scratch2[1] * self.twiddles[1];
        scratch2[2] = scratch2[2] * self.twiddles[3];
        scratch2[3] = scratch2[3] * self.twiddles[5];
        scratch2[4] = scratch2[4] * self.twiddles[7];
        scratch2[5] = scratch2[5] * self.twiddles[8];
        scratch2[6] = scratch2[6] * self.twiddles[9];
        scratch2[7] = scratch2[7] * self.twiddles[10];
        scratch2[8] = scratch2[8] * self.twiddles[11];

        self.butterfly9.butterfly3.perform_fft_strided(
            &mut scratch0[0],
            &mut scratch1[0],
            &mut scratch2[0],
        );
        self.butterfly9.butterfly3.perform_fft_strided(
            &mut scratch0[1],
            &mut scratch1[1],
            &mut scratch2[1],
        );
        self.butterfly9.butterfly3.perform_fft_strided(
            &mut scratch0[2],
            &mut scratch1[2],
            &mut scratch2[2],
        );
        self.butterfly9.butterfly3.perform_fft_strided(
            &mut scratch0[3],
            &mut scratch1[3],
            &mut scratch2[3],
        );
        self.butterfly9.butterfly3.perform_fft_strided(
            &mut scratch0[4],
            &mut scratch1[4],
            &mut scratch2[4],
        );
        self.butterfly9.butterfly3.perform_fft_strided(
            &mut scratch0[5],
            &mut scratch1[5],
            &mut scratch2[5],
        );
        self.butterfly9.butterfly3.perform_fft_strided(
            &mut scratch0[6],
            &mut scratch1[6],
            &mut scratch2[6],
        );
        self.butterfly9.butterfly3.perform_fft_strided(
            &mut scratch0[7],
            &mut scratch1[7],
            &mut scratch2[7],
        );
        self.butterfly9.butterfly3.perform_fft_strided(
            &mut scratch0[8],
            &mut scratch1[8],
            &mut scratch2[8],
        );

        buffer.store(scratch0[0], 0);
        buffer.store(scratch0[1], 1);
        buffer.store(scratch0[2], 2);
        buffer.store(scratch0[3], 3);
        buffer.store(scratch0[4], 4);
        buffer.store(scratch0[5], 5);
        buffer.store(scratch0[6], 6);
        buffer.store(scratch0[7], 7);
        buffer.store(scratch0[8], 8);

        buffer.store(scratch1[0], 9 + 0);
        buffer.store(scratch1[1], 9 + 1);
        buffer.store(scratch1[2], 9 + 2);
        buffer.store(scratch1[3], 9 + 3);
        buffer.store(scratch1[4], 9 + 4);
        buffer.store(scratch1[5], 9 + 5);
        buffer.store(scratch1[6], 9 + 6);
        buffer.store(scratch1[7], 9 + 7);
        buffer.store(scratch1[8], 9 + 8);

        buffer.store(scratch2[0], 18 + 0);
        buffer.store(scratch2[1], 18 + 1);
        buffer.store(scratch2[2], 18 + 2);
        buffer.store(scratch2[3], 18 + 3);
        buffer.store(scratch2[4], 18 + 4);
        buffer.store(scratch2[5], 18 + 5);
        buffer.store(scratch2[6], 18 + 6);
        buffer.store(scratch2[7], 18 + 7);
        buffer.store(scratch2[8], 18 + 8);
    }
}

pub struct Butterfly32<T> {
    butterfly16: Butterfly16<T>,
    butterfly8: Butterfly8<T>,
    twiddles: [Complex<T>; 7],
}
boilerplate_fft_butterfly!(Butterfly32, 32, |this: &Butterfly32<_>| this
    .butterfly8
    .fft_direction());
impl<T: FftNum> Butterfly32<T> {
    pub fn new(direction: FftDirection) -> Self {
        Self {
            butterfly16: Butterfly16::new(direction),
            butterfly8: Butterfly8::new(direction),
            twiddles: [
                twiddles::compute_twiddle(1, 32, direction),
                twiddles::compute_twiddle(2, 32, direction),
                twiddles::compute_twiddle(3, 32, direction),
                twiddles::compute_twiddle(4, 32, direction),
                twiddles::compute_twiddle(5, 32, direction),
                twiddles::compute_twiddle(6, 32, direction),
                twiddles::compute_twiddle(7, 32, direction),
            ],
        }
    }

    #[inline(never)]
    unsafe fn perform_fft_contiguous(&self, mut buffer: impl LoadStore<T>) {
        let mut scratch_evens = [
            buffer.load(0),
            buffer.load(2),
            buffer.load(4),
            buffer.load(6),
            buffer.load(8),
            buffer.load(10),
            buffer.load(12),
            buffer.load(14),
            buffer.load(16),
            buffer.load(18),
            buffer.load(20),
            buffer.load(22),
            buffer.load(24),
            buffer.load(26),
            buffer.load(28),
            buffer.load(30),
        ];

        let mut scratch_odds_n1 = [
            buffer.load(1),
            buffer.load(5),
            buffer.load(9),
            buffer.load(13),
            buffer.load(17),
            buffer.load(21),
            buffer.load(25),
            buffer.load(29),
        ];
        let mut scratch_odds_n3 = [
            buffer.load(31),
            buffer.load(3),
            buffer.load(7),
            buffer.load(11),
            buffer.load(15),
            buffer.load(19),
            buffer.load(23),
            buffer.load(27),
        ];

        self.butterfly16.perform_fft_contiguous(&mut scratch_evens);
        self.butterfly8.perform_fft_contiguous(&mut scratch_odds_n1);
        self.butterfly8.perform_fft_contiguous(&mut scratch_odds_n3);

        scratch_odds_n1[1] = scratch_odds_n1[1] * self.twiddles[0];
        scratch_odds_n3[1] = scratch_odds_n3[1] * self.twiddles[0].conj();

        scratch_odds_n1[2] = scratch_odds_n1[2] * self.twiddles[1];
        scratch_odds_n3[2] = scratch_odds_n3[2] * self.twiddles[1].conj();

        scratch_odds_n1[3] = scratch_odds_n1[3] * self.twiddles[2];
        scratch_odds_n3[3] = scratch_odds_n3[3] * self.twiddles[2].conj();

        scratch_odds_n1[4] = scratch_odds_n1[4] * self.twiddles[3];
        scratch_odds_n3[4] = scratch_odds_n3[4] * self.twiddles[3].conj();

        scratch_odds_n1[5] = scratch_odds_n1[5] * self.twiddles[4];
        scratch_odds_n3[5] = scratch_odds_n3[5] * self.twiddles[4].conj();

        scratch_odds_n1[6] = scratch_odds_n1[6] * self.twiddles[5];
        scratch_odds_n3[6] = scratch_odds_n3[6] * self.twiddles[5].conj();

        scratch_odds_n1[7] = scratch_odds_n1[7] * self.twiddles[6];
        scratch_odds_n3[7] = scratch_odds_n3[7] * self.twiddles[6].conj();

        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[0], &mut scratch_odds_n3[0]);
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[1], &mut scratch_odds_n3[1]);
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[2], &mut scratch_odds_n3[2]);
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[3], &mut scratch_odds_n3[3]);
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[4], &mut scratch_odds_n3[4]);
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[5], &mut scratch_odds_n3[5]);
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[6], &mut scratch_odds_n3[6]);
        Butterfly2::perform_fft_strided(&mut scratch_odds_n1[7], &mut scratch_odds_n3[7]);

        scratch_odds_n3[0] = twiddles::rotate_90(scratch_odds_n3[0], self.fft_direction());
        scratch_odds_n3[1] = twiddles::rotate_90(scratch_odds_n3[1], self.fft_direction());
        scratch_odds_n3[2] = twiddles::rotate_90(scratch_odds_n3[2], self.fft_direction());
        scratch_odds_n3[3] = twiddles::rotate_90(scratch_odds_n3[3], self.fft_direction());
        scratch_odds_n3[4] = twiddles::rotate_90(scratch_odds_n3[4], self.fft_direction());
        scratch_odds_n3[5] = twiddles::rotate_90(scratch_odds_n3[5], self.fft_direction());
        scratch_odds_n3[6] = twiddles::rotate_90(scratch_odds_n3[6], self.fft_direction());
        scratch_odds_n3[7] = twiddles::rotate_90(scratch_odds_n3[7], self.fft_direction());

        buffer.store(scratch_evens[0] + scratch_odds_n1[0], 0);
        buffer.store(scratch_evens[1] + scratch_odds_n1[1], 1);
        buffer.store(scratch_evens[2] + scratch_odds_n1[2], 2);
        buffer.store(scratch_evens[3] + scratch_odds_n1[3], 3);
        buffer.store(scratch_evens[4] + scratch_odds_n1[4], 4);
        buffer.store(scratch_evens[5] + scratch_odds_n1[5], 5);
        buffer.store(scratch_evens[6] + scratch_odds_n1[6], 6);
        buffer.store(scratch_evens[7] + scratch_odds_n1[7], 7);
        buffer.store(scratch_evens[8] + scratch_odds_n3[0], 8);
        buffer.store(scratch_evens[9] + scratch_odds_n3[1], 9);
        buffer.store(scratch_evens[10] + scratch_odds_n3[2], 10);
        buffer.store(scratch_evens[11] + scratch_odds_n3[3], 11);
        buffer.store(scratch_evens[12] + scratch_odds_n3[4], 12);
        buffer.store(scratch_evens[13] + scratch_odds_n3[5], 13);
        buffer.store(scratch_evens[14] + scratch_odds_n3[6], 14);
        buffer.store(scratch_evens[15] + scratch_odds_n3[7], 15);
        buffer.store(scratch_evens[0] - scratch_odds_n1[0], 16);
        buffer.store(scratch_evens[1] - scratch_odds_n1[1], 17);
        buffer.store(scratch_evens[2] - scratch_odds_n1[2], 18);
        buffer.store(scratch_evens[3] - scratch_odds_n1[3], 19);
        buffer.store(scratch_evens[4] - scratch_odds_n1[4], 20);
        buffer.store(scratch_evens[5] - scratch_odds_n1[5], 21);
        buffer.store(scratch_evens[6] - scratch_odds_n1[6], 22);
        buffer.store(scratch_evens[7] - scratch_odds_n1[7], 23);
        buffer.store(scratch_evens[8] - scratch_odds_n3[0], 24);
        buffer.store(scratch_evens[9] - scratch_odds_n3[1], 25);
        buffer.store(scratch_evens[10] - scratch_odds_n3[2], 26);
        buffer.store(scratch_evens[11] - scratch_odds_n3[3], 27);
        buffer.store(scratch_evens[12] - scratch_odds_n3[4], 28);
        buffer.store(scratch_evens[13] - scratch_odds_n3[5], 29);
        buffer.store(scratch_evens[14] - scratch_odds_n3[6], 30);
        buffer.store(scratch_evens[15] - scratch_odds_n3[7], 31);
    }
}
