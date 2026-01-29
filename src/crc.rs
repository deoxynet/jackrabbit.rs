/// CRC-16/CCITT (polynomial 0x1021, init 0xFFFF).

pub fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &byte in data {
        crc ^= (byte as u16) << 8;
        for _ in 0..8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

/// Append CRC-16 to data (big-endian).
pub fn append_crc(data: &[u8]) -> Vec<u8> {
    let crc = crc16(data);
    let mut out = Vec::with_capacity(data.len() + 2);
    out.extend_from_slice(data);
    out.push((crc >> 8) as u8);
    out.push((crc & 0xFF) as u8);
    out
}

/// Verify CRC-16 on data (last 2 bytes are CRC). Returns payload without CRC if valid.
pub fn check_crc(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 3 {
        return None;
    }
    let payload = &data[..data.len() - 2];
    let received_crc = ((data[data.len() - 2] as u16) << 8) | data[data.len() - 1] as u16;
    if crc16(payload) == received_crc {
        Some(payload.to_vec())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc_roundtrip() {
        let data = b"Hello, world!";
        let with_crc = append_crc(data);
        assert_eq!(with_crc.len(), data.len() + 2);
        let recovered = check_crc(&with_crc).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_crc_corruption() {
        let data = b"test data";
        let mut with_crc = append_crc(data);
        with_crc[3] ^= 0x01; // flip a bit
        assert!(check_crc(&with_crc).is_none());
    }

    #[test]
    fn test_crc_empty() {
        assert!(check_crc(&[]).is_none());
        assert!(check_crc(&[0, 0]).is_none());
    }
}
