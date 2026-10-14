# 🐰 JackRabbit

**Hop the air gap.**

A from-scratch OFDM audio modem in Rust. Turns any audio channel into a data link — headphone jacks, Bluetooth, through the air, whatever. Because sometimes the only way out is through the speakers.

## What is this?

JackRabbit (`jackrabbit`) is a complete PHY/modem layer that encodes data into audio waveforms and decodes it back. It implements the same signal processing techniques used in WiFi and LTE, but over audio frequencies.

- **OFDM** with 960-point FFT at 48kHz
- **Modulation** from BPSK to 256-QAM
- **FEC** via convolutional codes (K=7 Viterbi) and QC-LDPC (from 802.11n)
- **Synchronization** using Schmidl-Cox preamble detection
- **Channel estimation and equalization**
- **16 protocol modes** tuned for different channel conditions

## Performance

Tested with 4000-byte payload through a Focusrite Scarlett 2i2 loopback (yes, using a shitty 30ft guitar cable):

| Mode | Modulation | Throughput | Use Case |
|------|------------|------------|----------|
| `UT_FASTEST` | 16-QAM | **27.1 kbps** | Best overall — wired or good acoustic |
| `UT_FAST` | QPSK | 12.8 kbps | Reliable wired connection |
| `MT_FAST` | QPSK | 6.3 kbps | Medium bandwidth acoustic |
| `DT_FASTEST` | QPSK | 4.6 kbps | Audible band, decent speed |
| `AUDIBLE_FASTEST` | QPSK | 2.8 kbps | Works through phone speakers |

Over-the-air through actual speakers/microphones: **~1 kbps** (varies with environment).

### Bulk Transfer (TCP/IP over TUN)

End-to-end throughput for a 10 KB file transfer via netcat through the TUN interface, measured over BlackHole 2ch loopback:

| Protocol | Burst | Time | Throughput | Status |
|----------|-------|------|------------|--------|
| `WIRED_TURBO` (256-QAM, LDPC 3/4) | 2 | 3.1s | **26.6 kbps** | md5 pass |
| `WIRED_TURBO` | 4 | 3.1s | 26.6 kbps | md5 pass |
| `WIRED_TURBO` | 8 | 5.6s | 14.6 kbps | md5 pass |
| `WIRED_MAX` (64-QAM, LDPC 3/4) | 2 | 5.6s | 14.6 kbps | md5 pass |
| `WIRED_MAX` | 4 | 5.7s | 14.3 kbps | md5 pass |
| `WIRED_MAX` | 8 | 6.1s | 13.4 kbps | md5 pass |

Burst pipelining (burst >= 2) disables stop-and-wait ARQ. The link saturates at burst 2 — higher burst counts add congestion without improving throughput. At 48 kHz sample rate, 26.6 kbps is close to the practical ceiling after OFDM overhead, FEC, MAC framing, and TCP/IP encapsulation.

## Protocol Modes

16 modes covering different tradeoffs between speed, robustness, and frequency range:

| ID | Name | Mod | FEC | Band | CP |
|----|------|-----|-----|------|-----|
| 0 | `AUDIBLE_NORMAL` | BPSK | Conv 1/2 | 1-4 kHz | 240 |
| 1 | `AUDIBLE_FAST` | QPSK | Conv 1/2 | 1-4 kHz | 240 |
| 2 | `AUDIBLE_FASTEST` | QPSK | Conv 2/3 | 1-4 kHz | 240 |
| 3 | `DT_NORMAL` | BPSK | Conv 1/2 | 1-6 kHz | 240 |
| 4 | `DT_FAST` | QPSK | Conv 1/2 | 1-6 kHz | 240 |
| 5 | `DT_FASTEST` | QPSK | Conv 2/3 | 1-6 kHz | 240 |
| 6 | `MT_NORMAL` | BPSK | Conv 1/2 | 1-8 kHz | 240 |
| 7 | `MT_FAST` | QPSK | Conv 2/3 | 1-8 kHz | 240 |
| 8 | `WIRED_FAST` | 64-QAM | LDPC 3/4 | 1-8 kHz | 60 |
| 9 | `UT_NORMAL` | BPSK | Conv 1/2 | 1-16 kHz | 240 |
| 10 | `UT_FAST` | QPSK | Conv 2/3 | 1-16 kHz | 240 |
| 11 | `UT_FASTEST` | 16-QAM | Conv 3/4 | 1-16 kHz | 120 |
| 12 | `DT_LDPC` | QPSK | LDPC 1/2 | 1-6 kHz | 240 |
| 13 | `WIRED_TURBO` | 256-QAM | LDPC 3/4 | wide | 30 |
| 14 | `WIRED_MAX` | 64-QAM | LDPC 3/4 | wide | 60 |
| 15 | `OTA_ROBUST` | BPSK | LDPC 1/2 | 1-4 kHz | 240 |

**AUDIBLE** modes stay within 1-4 kHz (comfortable to hear).  
**DT/MT/UT** modes progressively use more bandwidth.  
**WIRED** modes are aggressive — minimal cyclic prefix, high-order QAM, designed for direct cable connections with clean SNR.

## Technical Details

| Parameter | Value |
|-----------|-------|
| Sample rate | 48 kHz |
| FFT size | 960 |
| Subcarrier spacing | 50 Hz |
| Symbol duration | 20ms (+ cyclic prefix) |
| Max payload | 1023 bytes |
| CRC | CRC-16 |

### FEC Options

**Convolutional:** K=7 constraint length (same as 802.11a/g), Viterbi soft-decision decoding. Rates: 1/2, 2/3, 3/4 via puncturing.

**QC-LDPC:** Quasi-cyclic LDPC codes borrowed from 802.11n. Better performance at high SNR, enables 256-QAM and above.

## Installation

```bash
cargo add jackrabbit --git https://github.com/deoxynet/jackrabbit.rs
```

## Usage

```rust
use jackrabbit::{decoder::Decoder, encoder};

// Encode data to audio samples
let data = b"hop the air gap";
let samples = encoder::encode(data, 11, 1.0); // UT_FASTEST

// ... transmit samples over audio channel ...

// Decode received audio back to data
let mut decoder = Decoder::new();
let received = decoder.feed(&samples);
assert_eq!(received, data);
```

## Dependencies

Minimal:
- Bundled FFT implementation — FFT operations
- `num-complex` 0.4 — complex number support
- `num-traits` 0.2 — numeric traits for FFT operations
- `kcp` 0.6 — reliable transport

Optional:
- `wasm-bindgen` / `wasm-bindgen-futures` / `js-sys` — browser builds (behind `wasm` feature)
- `cpal` 0.15 / `libc` 0.2 — native audio I/O and TUN support (behind `cli` feature)

## Platform Support

Tested on macOS. Should work anywhere Rust compiles — it's pure DSP, no platform-specific audio code in the library itself.

## Why?

Because I can.

Also: air-gapped machines still have headphone jacks. Bluetooth headsets are everywhere. Sometimes the audio channel is the only channel. This gives you a way to move data through it.

## Future Ideas

- [x] TUN/TAP interface (TCP/IP over audio)
- [ ] Adaptive bitloading per-subcarrier

## License

MIT — Kevin Pham ([@deoxykev](https://github.com/deoxykev)) <k@vin-ph.am>

---

*They closed the ports. They forgot the jack.*
