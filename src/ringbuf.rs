/// Power-of-2 ring buffer for streaming decode.

pub struct RingBuffer {
    data: Vec<f32>,
    mask: usize,
    write_pos: usize,
    read_pos: usize,
}

impl RingBuffer {
    pub fn new(size: usize) -> Self {
        assert!(size.is_power_of_two(), "Ring buffer size must be power of 2");
        RingBuffer {
            data: vec![0.0; size],
            mask: size - 1,
            write_pos: 0,
            read_pos: 0,
        }
    }

    pub fn push(&mut self, samples: &[f32]) {
        for &s in samples {
            self.data[self.write_pos & self.mask] = s;
            self.write_pos += 1;
        }
    }

    pub fn available(&self) -> usize {
        self.write_pos - self.read_pos
    }

    /// Read n samples without advancing the read pointer.
    pub fn peek(&self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.data[(self.read_pos + i) & self.mask]);
        }
        out
    }

    /// Read a single sample at offset from read_pos without advancing.
    pub fn peek_at(&self, offset: usize) -> f32 {
        self.data[(self.read_pos + offset) & self.mask]
    }

    /// Advance read pointer by n samples.
    pub fn advance(&mut self, n: usize) {
        self.read_pos += n;
    }

    /// Read n samples and advance.
    pub fn read(&mut self, n: usize) -> Vec<f32> {
        let out = self.peek(n);
        self.advance(n);
        out
    }

    pub fn clear(&mut self) {
        self.read_pos = self.write_pos;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut rb = RingBuffer::new(16);
        assert_eq!(rb.available(), 0);

        rb.push(&[1.0, 2.0, 3.0]);
        assert_eq!(rb.available(), 3);

        let data = rb.read(2);
        assert_eq!(data, vec![1.0, 2.0]);
        assert_eq!(rb.available(), 1);
    }

    #[test]
    fn test_wraparound() {
        let mut rb = RingBuffer::new(4);
        rb.push(&[1.0, 2.0, 3.0]);
        rb.advance(2);
        rb.push(&[4.0, 5.0, 6.0]);
        assert_eq!(rb.available(), 4);
        let data = rb.read(4);
        assert_eq!(data, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_peek() {
        let mut rb = RingBuffer::new(8);
        rb.push(&[10.0, 20.0, 30.0]);
        let peeked = rb.peek(3);
        assert_eq!(peeked, vec![10.0, 20.0, 30.0]);
        assert_eq!(rb.available(), 3); // not consumed
    }
}
