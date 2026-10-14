use std::collections::HashMap;
use std::sync::Arc;

use crate::fft::common::RadixFactor;
use crate::fft::{common::FftNum, fft_cache::FftCache, FftDirection};

use crate::fft::butterflies::*;
use crate::fft::dft::Dft;
use crate::fft::radix4::Radix4;
use crate::fft::radixn::RadixN;
use crate::fft::Fft;

use crate::fft::math_utils::PrimeFactors;

const MAX_RADIXN_FACTOR: usize = 7;

#[derive(Debug, PartialEq, Clone)]
enum Recipe {
    Dft(usize),
    RadixN {
        factors: Box<[RadixFactor]>,
        base_fft: Arc<Recipe>,
    },
    Radix4 {
        k: u32,
        base_fft: Arc<Recipe>,
    },
    Butterfly2,
    Butterfly3,
    Butterfly4,
    Butterfly5,
    Butterfly6,
    Butterfly7,
    Butterfly8,
    Butterfly9,
    Butterfly12,
    Butterfly16,
    Butterfly24,
    Butterfly27,
    Butterfly32,
}

impl Recipe {
    pub fn len(&self) -> usize {
        match self {
            Recipe::Dft(length) => *length,
            Recipe::RadixN { factors, base_fft } => {
                base_fft.len() * factors.iter().map(|f| f.radix()).product::<usize>()
            }
            Recipe::Radix4 { k, base_fft } => base_fft.len() * (1 << (k * 2)),
            Recipe::Butterfly2 => 2,
            Recipe::Butterfly3 => 3,
            Recipe::Butterfly4 => 4,
            Recipe::Butterfly5 => 5,
            Recipe::Butterfly6 => 6,
            Recipe::Butterfly7 => 7,
            Recipe::Butterfly8 => 8,
            Recipe::Butterfly9 => 9,
            Recipe::Butterfly12 => 12,
            Recipe::Butterfly16 => 16,
            Recipe::Butterfly24 => 24,
            Recipe::Butterfly27 => 27,
            Recipe::Butterfly32 => 32,
        }
    }
}

/// The FFT planner creates new FFT algorithm instances.
pub struct FftPlanner<T: FftNum> {
    algorithm_cache: FftCache<T>,
    recipe_cache: HashMap<usize, Arc<Recipe>>,
}

impl<T: FftNum> FftPlanner<T> {
    /// Creates a new `FftPlanner` instance.
    pub fn new() -> Self {
        Self {
            algorithm_cache: FftCache::new(),
            recipe_cache: HashMap::new(),
        }
    }

    /// Returns a `Fft` instance which computes FFTs of size `len`.
    pub fn plan_fft(&mut self, len: usize, direction: FftDirection) -> Arc<dyn Fft<T>> {
        let recipe = self.design_fft_for_len(len);
        self.build_fft(&recipe, direction)
    }

    /// Returns a `Fft` instance which computes forward FFTs of size `len`
    pub fn plan_fft_forward(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(len, FftDirection::Forward)
    }

    /// Returns a `Fft` instance which computes inverse FFTs of size `len`
    pub fn plan_fft_inverse(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(len, FftDirection::Inverse)
    }

    fn design_fft_for_len(&mut self, len: usize) -> Arc<Recipe> {
        if len < 2 {
            Arc::new(Recipe::Dft(len))
        } else if let Some(recipe) = self.recipe_cache.get(&len) {
            Arc::clone(&recipe)
        } else {
            let factors = PrimeFactors::compute(len);
            let recipe = self.design_fft_with_factors(len, factors);
            self.recipe_cache.insert(len, Arc::clone(&recipe));
            recipe
        }
    }

    fn build_fft(&mut self, recipe: &Recipe, direction: FftDirection) -> Arc<dyn Fft<T>> {
        let len = recipe.len();
        if let Some(instance) = self.algorithm_cache.get(len, direction) {
            instance
        } else {
            let fft = self.build_new_fft(recipe, direction);
            self.algorithm_cache.insert(&fft);
            fft
        }
    }

    fn build_new_fft(&mut self, recipe: &Recipe, direction: FftDirection) -> Arc<dyn Fft<T>> {
        match recipe {
            Recipe::Dft(len) => Arc::new(Dft::new(*len, direction)) as Arc<dyn Fft<T>>,
            Recipe::RadixN { factors, base_fft } => {
                let base_fft = self.build_fft(base_fft, direction);
                Arc::new(RadixN::new(factors, base_fft)) as Arc<dyn Fft<T>>
            }
            Recipe::Radix4 { k, base_fft } => {
                let base_fft = self.build_fft(base_fft, direction);
                Arc::new(Radix4::new_with_base(*k, base_fft)) as Arc<dyn Fft<T>>
            }
            Recipe::Butterfly2 => Arc::new(Butterfly2::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly3 => Arc::new(Butterfly3::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly4 => Arc::new(Butterfly4::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly5 => Arc::new(Butterfly5::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly6 => Arc::new(Butterfly6::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly7 => Arc::new(Butterfly7::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly8 => Arc::new(Butterfly8::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly9 => Arc::new(Butterfly9::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly12 => Arc::new(Butterfly12::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly16 => Arc::new(Butterfly16::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly24 => Arc::new(Butterfly24::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly27 => Arc::new(Butterfly27::new(direction)) as Arc<dyn Fft<T>>,
            Recipe::Butterfly32 => Arc::new(Butterfly32::new(direction)) as Arc<dyn Fft<T>>,
        }
    }

    fn design_fft_with_factors(&mut self, len: usize, factors: PrimeFactors) -> Arc<Recipe> {
        if let Some(fft_instance) = self.design_butterfly_algorithm(len) {
            fft_instance
        } else if factors.has_factors_leq(MAX_RADIXN_FACTOR) {
            self.design_radixn(factors)
        } else {
            // Fallback to DFT for sizes with prime factors > 7
            Arc::new(Recipe::Dft(len))
        }
    }

    fn design_radixn(&mut self, factors: PrimeFactors) -> Arc<Recipe> {
        let p2 = factors.get_power_of_two();
        let p3 = factors.get_power_of_three();
        let p5 = factors
            .get_other_factors()
            .iter()
            .find_map(|f| if f.value == 5 { Some(f.count) } else { None })
            .unwrap_or(0);
        let p7 = factors
            .get_other_factors()
            .iter()
            .find_map(|f| if f.value == 7 { Some(f.count) } else { None })
            .unwrap_or(0);

        let base_len: usize = if factors.has_factors_gt(MAX_RADIXN_FACTOR) {
            factors.product_above(MAX_RADIXN_FACTOR)
        } else if p7 == 0 && p5 == 0 && p3 < 2 {
            if p3 == 0 {
                // pure power of 2
                assert!(p2 > 5);
                if p2 % 2 == 1 {
                    8
                } else {
                    16
                }
            } else {
                // 3 times a power of 2
                assert!(p2 > 3);
                if p2 % 2 == 1 {
                    24
                } else {
                    12
                }
            }
        } else if p2 > 0 && p3 > 0 {
            let excess_p2 = p2.saturating_sub(p3);
            match excess_p2 {
                0 => 6,
                1 => 12,
                _ => 24,
            }
        } else if p3 > 2 {
            27
        } else if p3 > 1 {
            9
        } else if p7 > 0 {
            7
        } else {
            assert!(p5 > 0);
            5
        };

        let base_fft = self.design_fft_for_len(base_len);
        let mut cross_len = factors.get_product() / base_len;

        // see if we can use radix4
        let cross_bits = cross_len.trailing_zeros();
        if cross_len.is_power_of_two() && cross_bits % 2 == 0 {
            let k = cross_bits / 2;
            return Arc::new(Recipe::Radix4 { k, base_fft });
        }

        // fall back to RadixN
        let mut factors = Vec::new();
        while cross_len % 7 == 0 {
            cross_len /= 7;
            factors.push(RadixFactor::Factor7);
        }
        while cross_len % 6 == 0 {
            cross_len /= 6;
            factors.push(RadixFactor::Factor6);
        }
        while cross_len % 5 == 0 {
            cross_len /= 5;
            factors.push(RadixFactor::Factor5);
        }
        while cross_len % 3 == 0 {
            cross_len /= 3;
            factors.push(RadixFactor::Factor3);
        }
        assert!(cross_len.is_power_of_two());

        let cross_bits = cross_len.trailing_zeros();
        if cross_bits % 2 == 1 {
            factors.push(RadixFactor::Factor2);
        }
        factors.extend(std::iter::repeat(RadixFactor::Factor4).take(cross_bits as usize / 2));

        Arc::new(Recipe::RadixN {
            factors: factors.into_boxed_slice(),
            base_fft,
        })
    }

    fn design_butterfly_algorithm(&mut self, len: usize) -> Option<Arc<Recipe>> {
        match len {
            2 => Some(Arc::new(Recipe::Butterfly2)),
            3 => Some(Arc::new(Recipe::Butterfly3)),
            4 => Some(Arc::new(Recipe::Butterfly4)),
            5 => Some(Arc::new(Recipe::Butterfly5)),
            6 => Some(Arc::new(Recipe::Butterfly6)),
            7 => Some(Arc::new(Recipe::Butterfly7)),
            8 => Some(Arc::new(Recipe::Butterfly8)),
            9 => Some(Arc::new(Recipe::Butterfly9)),
            12 => Some(Arc::new(Recipe::Butterfly12)),
            16 => Some(Arc::new(Recipe::Butterfly16)),
            24 => Some(Arc::new(Recipe::Butterfly24)),
            27 => Some(Arc::new(Recipe::Butterfly27)),
            32 => Some(Arc::new(Recipe::Butterfly32)),
            _ => None,
        }
    }
}
