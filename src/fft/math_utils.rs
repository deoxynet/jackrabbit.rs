#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct PrimeFactor {
    pub value: usize,
    pub count: u32,
}

#[derive(Clone, Debug)]
pub struct PrimeFactors {
    other_factors: Vec<PrimeFactor>,
    n: usize,
    power_two: u32,
    power_three: u32,
    total_factor_count: u32,
    distinct_factor_count: u32,
}
impl PrimeFactors {
    pub fn compute(mut n: usize) -> Self {
        let mut result = Self {
            other_factors: Vec::new(),
            n,
            power_two: 0,
            power_three: 0,
            total_factor_count: 0,
            distinct_factor_count: 0,
        };

        // compute powers of two separately
        result.power_two = n.trailing_zeros();
        result.total_factor_count += result.power_two;
        n >>= result.power_two;
        if result.power_two > 0 {
            result.distinct_factor_count += 1;
        }

        // also compute powers of three separately
        while n % 3 == 0 {
            result.power_three += 1;
            n /= 3;
        }
        result.total_factor_count += result.power_three;
        if result.power_three > 0 {
            result.distinct_factor_count += 1;
        }

        // if we have any other factors, gather them in the "other factors" vec
        if n > 1 {
            let mut divisor = 5;
            let mut limit = (n as f32).sqrt() as usize + 1;
            while divisor < limit {
                let mut count = 0;
                while n % divisor == 0 {
                    n /= divisor;
                    count += 1;
                }

                if count > 0 {
                    result.other_factors.push(PrimeFactor {
                        value: divisor,
                        count,
                    });
                    result.total_factor_count += count;
                    result.distinct_factor_count += 1;

                    limit = (n as f32).sqrt() as usize + 1;
                }

                divisor += 2;
            }

            if n > 1 {
                result
                    .other_factors
                    .push(PrimeFactor { value: n, count: 1 });
                result.total_factor_count += 1;
                result.distinct_factor_count += 1;
            }
        }

        result
    }

    pub fn is_prime(&self) -> bool {
        self.total_factor_count == 1
    }
    pub fn get_product(&self) -> usize {
        self.n
    }
    pub fn get_power_of_two(&self) -> u32 {
        self.power_two
    }
    pub fn get_power_of_three(&self) -> u32 {
        self.power_three
    }
    pub fn get_other_factors(&self) -> &[PrimeFactor] {
        &self.other_factors
    }

    pub fn has_factors_leq(&self, factor: usize) -> bool {
        self.power_two > 0
            || self.power_three > 0
            || self
                .other_factors
                .first()
                .map_or(false, |f| f.value <= factor)
    }

    pub fn has_factors_gt(&self, factor: usize) -> bool {
        (factor < 2 && self.power_two > 0)
            || (factor < 3 && self.power_three > 0)
            || self
                .other_factors
                .last()
                .map_or(false, |f| f.value > factor)
    }

    pub fn product_above(&self, min_factor: usize) -> usize {
        self.other_factors
            .iter()
            .skip_while(|f| f.value <= min_factor)
            .map(|f| f.value.pow(f.count))
            .product()
    }
}
