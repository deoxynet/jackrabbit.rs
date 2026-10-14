/// JackRabbit — IP over audio.
///
/// Creates a macOS utun interface and bridges IP packets through the
/// OFDM acoustic modem. Two machines connected by speaker/mic (OTA)
/// or 3.5mm cable (wired) get a point-to-point IP link.
///
/// Usage:
///   sudo cargo run --features cli --bin jackrabbit -- [OPTIONS]
///
/// Audio-only mode (no root required):
///   cargo run --features cli --bin jackrabbit -- --no-tun

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SampleRate, StreamConfig};

use jackrabbit::mac::HalfDuplexMac;
use jackrabbit::params;

const SAMPLE_RATE: u32 = 48_000;
/// IPv4 minimum reassembly buffer; floor for TUN MTU.
const MIN_IPV4_MTU: u32 = 68;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Config {
    protocol_id: u8,
    volume: f32,
    wired: bool,
    cable: bool,
    input_device: Option<String>,
    output_device: Option<String>,
    tun_ip: String,
    tun_peer: String,
    tun_mtu: u32,
    mtu_is_auto: bool,
    target_frame_ms: u32,
    burst_limit: usize,
    no_tun: bool,
    list_devices: bool,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut protocol_id: u8 = 0;
    let mut volume: f32 = 0.8;
    let mut wired = false;
    let mut cable = false;
    let mut input_device: Option<String> = None;
    let mut output_device: Option<String> = None;
    let mut tun_ip = "10.73.0.1".to_string();
    let mut tun_peer = "10.73.0.2".to_string();
    let mut mtu_override: Option<u32> = None;
    let mut target_frame_ms_override: Option<u32> = None;
    let mut burst_override: Option<usize> = None;
    let mut no_tun = false;
    let mut list_devices = false;

    fn require_arg(args: &[String], i: &mut usize, flag: &str) -> String {
        *i += 1;
        if *i >= args.len() {
            eprintln!("Missing value for {}", flag);
            std::process::exit(1);
        }
        args[*i].clone()
    }

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--protocol" => {
                let value = require_arg(&args, &mut i, "--protocol");
                protocol_id = value.parse().expect("Invalid protocol number");
            }
            "--volume" => {
                let value = require_arg(&args, &mut i, "--volume");
                volume = value.parse().expect("Invalid volume");
            }
            "--wired" => wired = true,
            "--cable" => {
                cable = true;
                wired = true;
            }
            "--input" => {
                let value = require_arg(&args, &mut i, "--input");
                input_device = Some(value);
            }
            "--output" => {
                let value = require_arg(&args, &mut i, "--output");
                output_device = Some(value);
            }
            "--ip" => {
                let value = require_arg(&args, &mut i, "--ip");
                tun_ip = value;
            }
            "--peer" => {
                let value = require_arg(&args, &mut i, "--peer");
                tun_peer = value;
            }
            "--mtu" => {
                let value = require_arg(&args, &mut i, "--mtu");
                mtu_override = Some(value.parse().expect("Invalid MTU"));
            }
            "--target-frame-ms" => {
                let value = require_arg(&args, &mut i, "--target-frame-ms");
                target_frame_ms_override = Some(value.parse().expect("Invalid target frame ms"));
            }
            "--burst" => {
                let value = require_arg(&args, &mut i, "--burst");
                burst_override = Some(value.parse().expect("Invalid burst limit"));
            }
            "--no-tun" => no_tun = true,
            "--list-devices" => list_devices = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown option: {}", other);
                eprintln!("Try --help for usage.");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if protocol_id as usize >= params::PROTOCOLS.len() {
        eprintln!(
            "Invalid protocol ID {} (valid range: 0-{})",
            protocol_id,
            params::PROTOCOLS.len() - 1
        );
        std::process::exit(1);
    }

    if tun_ip == tun_peer {
        eprintln!("ERROR: --ip and --peer must be different (both are '{}')", tun_ip);
        eprintln!("  Example: --ip 10.73.0.1 --peer 10.73.0.2  (side A)");
        eprintln!("           --ip 10.73.0.2 --peer 10.73.0.1  (side B)");
        std::process::exit(1);
    }

    // --- Resolve defaults based on mode ---

    // Target frame duration: shorter for wired (clean channel), longer for OTA
    // (need to amortize preamble overhead).
    let target_frame_ms = target_frame_ms_override.unwrap_or(
        if wired { 200 } else { 400 }
    );
    if target_frame_ms == 0 {
        eprintln!("--target-frame-ms must be > 0");
        std::process::exit(1);
    }

    let proto_config = params::get_protocol(protocol_id);
    let overhead_cp_len = if cable {
        params::CP_LEN_CABLE
    } else if wired {
        proto_config.cp_len
    } else {
        params::CP_LEN
    };
    let data_cp_len = if cable {
        params::CP_LEN_CABLE
    } else {
        proto_config.cp_len
    };
    let preamble_symbols = if wired {
        params::PREAMBLE_SYMBOLS_MINIMAL
    } else {
        params::PREAMBLE_SYMBOLS
    };
    let training_symbols = if wired {
        params::TRAINING_SYMBOLS_SHORT
    } else {
        params::TRAINING_SYMBOLS
    };

    // MTU: auto-compute from protocol unless user overrides.
    let (tun_mtu, mtu_is_auto) = if let Some(mtu) = mtu_override {
        let clamped = mtu.min(params::MAX_PAYLOAD as u32);
        if clamped != mtu {
            eprintln!(
                "MTU {} exceeds MAX_PAYLOAD {} — clamped to {}",
                mtu, params::MAX_PAYLOAD, clamped
            );
        }
        (clamped, false)
    } else {
        let auto = if cable {
            recommended_mtu_custom(
                protocol_id,
                target_frame_ms,
                preamble_symbols,
                training_symbols,
                overhead_cp_len,
                data_cp_len,
            ) as u32
        } else {
            params::recommended_mtu(protocol_id, wired, target_frame_ms) as u32
        };
        let floored = auto.max(MIN_IPV4_MTU);
        if auto < MIN_IPV4_MTU {
            eprintln!(
                "WARNING: Protocol {} yields MTU {} for {}ms frames (below IPv4 min {}).",
                params::get_protocol(protocol_id).name, auto, target_frame_ms, MIN_IPV4_MTU
            );
            eprintln!(
                "         Frames will exceed {}ms target. Consider a faster protocol:",
                target_frame_ms
            );
            if wired {
                eprintln!("           --protocol 8 --wired   (WIRED_FAST, ~30-50 kbps)");
                eprintln!("           --protocol 13 --wired  (WIRED_TURBO, ~50-80 kbps)");
            } else {
                eprintln!("           --protocol 10  (UT_FAST, QPSK, ~2-5 kbps)");
                eprintln!("           --protocol 11  (UT_FASTEST, 16-QAM, ~10-27 kbps)");
            }
        }
        (floored, true)
    };

    // Burst limit: wired links are clean enough to pipeline; OTA uses
    // stop-and-wait for maximum reliability.
    let burst_limit = burst_override.unwrap_or(
        if wired { 4 } else { 1 }
    );

    Config {
        protocol_id,
        volume,
        wired,
        cable,
        input_device,
        output_device,
        tun_ip,
        tun_peer,
        tun_mtu,
        mtu_is_auto,
        target_frame_ms,
        burst_limit,
        no_tun,
        list_devices,
    }
}

fn print_help() {
    eprintln!("jackrabbit — IP over audio");
    eprintln!();
    eprintln!("Usage: sudo jackrabbit [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --protocol N         OFDM protocol ID (0-15, default: 0)");
    eprintln!("  --wired              Wired mode (short preamble, minimal CP)");
    eprintln!("  --cable              Cable mode (3.5mm jack, minimal CP)");
    eprintln!("  --volume V           TX volume 0.0-1.0 (default: 0.8)");
    eprintln!("  --input NAME         Audio input device (substring match)");
    eprintln!("  --output NAME        Audio output device (substring match)");
    eprintln!("  --ip ADDR            Local TUN IP (default: 10.73.0.1)");
    eprintln!("  --peer ADDR          Peer TUN IP (default: 10.73.0.2)");
    eprintln!("  --mtu N              TUN MTU override (default: auto from protocol)");
    eprintln!("  --target-frame-ms N  Target frame duration (default: 200 wired, 400 OTA)");
    eprintln!("  --burst N            Frames per TX burst (default: 4 wired, 1 OTA)");
    eprintln!("  --no-tun             Audio-only mode (no TUN, no root needed)");
    eprintln!("  --list-devices       List audio devices and exit");
    eprintln!();
    eprintln!("Recommended configurations:");
    eprintln!("  OTA robust:      --protocol 0                   (~280 bps, maximum range)");
    eprintln!("  OTA medium:      --protocol 10                  (~2-5 kbps, good SNR)");
    eprintln!("  OTA fast:        --protocol 11                  (~10-27 kbps, excellent SNR)");
    eprintln!("  Wired:           --protocol 8 --wired           (~30-50 kbps)");
    eprintln!("  Wired fast:      --protocol 13 --wired          (~50-80 kbps)");
    eprintln!("  Cable:           --protocol 13 --cable          (~50-80 kbps, short CP)");
    eprintln!();
    eprintln!("Environment:");
    eprintln!("  JACKRABBIT_DEBUG=1   Show decoder state in periodic stats");
}

// ---------------------------------------------------------------------------
// Framing helpers
// ---------------------------------------------------------------------------

fn total_samples_for_payload(
    config: &params::ProtocolConfig,
    payload_len: usize,
    preamble_symbols: usize,
    training_symbols: usize,
    overhead_cp_len: usize,
    data_cp_len: usize,
) -> usize {
    let n_data = config.data_symbols_for(payload_len);
    let overhead_sym_total = params::FFT_SIZE + overhead_cp_len;
    let data_sym_total = params::FFT_SIZE + data_cp_len;
    let fixed_symbols = preamble_symbols + training_symbols + params::HEADER_SYMBOLS;
    fixed_symbols * overhead_sym_total + n_data * data_sym_total
}

fn frame_duration_ms_custom(
    protocol_id: u8,
    payload_len: usize,
    preamble_symbols: usize,
    training_symbols: usize,
    overhead_cp_len: usize,
    data_cp_len: usize,
) -> f64 {
    let config = params::get_protocol(protocol_id);
    let samples = total_samples_for_payload(
        config,
        payload_len,
        preamble_symbols,
        training_symbols,
        overhead_cp_len,
        data_cp_len,
    );
    samples as f64 / SAMPLE_RATE as f64 * 1000.0
}

fn recommended_mtu_custom(
    protocol_id: u8,
    target_ms: u32,
    preamble_symbols: usize,
    training_symbols: usize,
    overhead_cp_len: usize,
    data_cp_len: usize,
) -> usize {
    let config = params::get_protocol(protocol_id);
    let target_samples = (SAMPLE_RATE as u64 * target_ms as u64 / 1000) as usize;
    let mut lo = 1usize;
    let mut hi = params::MAX_PAYLOAD;
    let mut best = 1usize;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let samples = total_samples_for_payload(
            config,
            mid,
            preamble_symbols,
            training_symbols,
            overhead_cp_len,
            data_cp_len,
        );
        if samples <= target_samples {
            best = mid;
            lo = mid + 1;
        } else {
            hi = mid.saturating_sub(1);
        }
    }
    best.min(params::MAX_PAYLOAD).max(1)
}

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------

fn find_config(
    ranges: impl Iterator<Item = cpal::SupportedStreamConfigRange>,
) -> Option<StreamConfig> {
    for cfg in ranges {
        if cfg.sample_format() == SampleFormat::F32
            && cfg.min_sample_rate() <= SampleRate(SAMPLE_RATE)
            && cfg.max_sample_rate() >= SampleRate(SAMPLE_RATE)
        {
            return Some(cfg.with_sample_rate(SampleRate(SAMPLE_RATE)).into());
        }
    }
    None
}

fn device_config(device: &cpal::Device, is_output: bool) -> StreamConfig {
    let from_ranges = if is_output {
        find_config(device.supported_output_configs().expect("No output configs"))
    } else {
        find_config(device.supported_input_configs().expect("No input configs"))
    };
    if let Some(config) = from_ranges {
        return config;
    }
    let default = if is_output {
        device.default_output_config()
    } else {
        device.default_input_config()
    }
    .expect("No default config");
    if default.sample_format() != SampleFormat::F32 {
        eprintln!(
            "Device does not support f32 at 48kHz (got {:?}).",
            default.sample_format()
        );
        std::process::exit(1);
    }
    assert_eq!(default.sample_rate().0, SAMPLE_RATE, "Device does not support 48 kHz");
    default.into()
}

fn find_device(
    host: &cpal::Host,
    name: &Option<String>,
    is_output: bool,
) -> cpal::Device {
    if let Some(ref name) = name {
        let devices = if is_output {
            host.output_devices()
        } else {
            host.input_devices()
        }
        .expect("Failed to enumerate devices");
        devices
            .into_iter()
            .find(|d| {
                d.name()
                    .unwrap_or_default()
                    .to_lowercase()
                    .contains(&name.to_lowercase())
            })
            .unwrap_or_else(|| {
                eprintln!("Device '{}' not found.", name);
                std::process::exit(1);
            })
    } else if is_output {
        host.default_output_device().expect("No default output device")
    } else {
        host.default_input_device().expect("No default input device")
    }
}

fn list_devices() {
    let host = cpal::default_host();
    eprintln!("Input devices:");
    if let Ok(devices) = host.input_devices() {
        for d in devices {
            eprintln!("  {}", d.name().unwrap_or_default());
        }
    }
    eprintln!("Output devices:");
    if let Ok(devices) = host.output_devices() {
        for d in devices {
            eprintln!("  {}", d.name().unwrap_or_default());
        }
    }
}

// ---------------------------------------------------------------------------
// macOS utun
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod tun {
    use std::io;
    use std::os::unix::io::{AsRawFd, FromRawFd, OwnedFd};
    use std::process::Command;

    // macOS system constants
    const PF_SYSTEM: libc::c_int = 32;
    const SYSPROTO_CONTROL: libc::c_int = 2;
    const AF_SYS_CONTROL: u16 = 2;
    const CTLIOCGINFO: libc::c_ulong = 0xc064_4e03;
    const UTUN_CONTROL_NAME: &[u8] = b"com.apple.net.utun_control";
    const AF_INET: u32 = 2;
    const AF_INET6: u32 = 30;

    #[repr(C)]
    struct CtlInfo {
        ctl_id: u32,
        ctl_name: [u8; 96],
    }

    #[repr(C)]
    struct SockaddrCtl {
        sc_len: u8,
        sc_family: u8,
        ss_sysaddr: u16,
        sc_id: u32,
        sc_unit: u32,
        sc_reserved: [u32; 5],
    }

    pub struct TunHandle {
        fd: OwnedFd,
        pub name: String,
        mtu: usize,
    }

    impl TunHandle {
        /// Read one IP packet (blocks or returns EAGAIN if non-blocking).
        pub fn read(&self) -> io::Result<Vec<u8>> {
            let buf_len = self.mtu.max(1500) + 4;
            let mut buf = vec![0u8; buf_len];
            let n = unsafe {
                libc::read(
                    self.fd.as_raw_fd(),
                    buf.as_mut_ptr() as *mut libc::c_void,
                    buf.len(),
                )
            };
            if n < 0 {
                return Err(io::Error::last_os_error());
            }
            let n = n as usize;
            if n <= 4 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "short read"));
            }
            // Strip 4-byte AF header, return raw IP packet
            Ok(buf[4..n].to_vec())
        }

        /// Write one IP packet (prepends AF header).
        pub fn write(&self, ip_packet: &[u8]) -> io::Result<()> {
            if ip_packet.is_empty() {
                return Ok(());
            }
            // Detect IP version from first nibble
            let af: u32 = if (ip_packet[0] >> 4) == 6 {
                AF_INET6
            } else {
                AF_INET
            };
            let mut buf = Vec::with_capacity(4 + ip_packet.len());
            buf.extend_from_slice(&af.to_be_bytes());
            buf.extend_from_slice(ip_packet);

            let n = unsafe {
                libc::write(
                    self.fd.as_raw_fd(),
                    buf.as_ptr() as *const libc::c_void,
                    buf.len(),
                )
            };
            if n < 0 {
                return Err(io::Error::last_os_error());
            }
            if (n as usize) < buf.len() {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    format!("short write: {} of {} bytes", n, buf.len()),
                ));
            }
            Ok(())
        }

        /// Set the fd to non-blocking mode.
        pub fn set_nonblocking(&self) {
            unsafe {
                let flags = libc::fcntl(self.fd.as_raw_fd(), libc::F_GETFL);
                libc::fcntl(self.fd.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK);
            }
        }
    }

    /// Create a utun interface. Returns the handle with the assigned name.
    pub fn open() -> io::Result<TunHandle> {
        // 1. Create PF_SYSTEM socket
        let fd = unsafe { libc::socket(PF_SYSTEM, libc::SOCK_DGRAM, SYSPROTO_CONTROL) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };

        // 2. Get control ID for utun
        let mut info = CtlInfo {
            ctl_id: 0,
            ctl_name: [0u8; 96],
        };
        info.ctl_name[..UTUN_CONTROL_NAME.len()].copy_from_slice(UTUN_CONTROL_NAME);

        let ret = unsafe {
            libc::ioctl(
                fd.as_raw_fd(),
                CTLIOCGINFO,
                &mut info as *mut CtlInfo,
            )
        };
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }

        // 3. Connect with unit=0 (auto-assign)
        let addr = SockaddrCtl {
            sc_len: std::mem::size_of::<SockaddrCtl>() as u8,
            sc_family: PF_SYSTEM as u8,
            ss_sysaddr: AF_SYS_CONTROL,
            sc_id: info.ctl_id,
            sc_unit: 0, // auto
            sc_reserved: [0; 5],
        };

        let ret = unsafe {
            libc::connect(
                fd.as_raw_fd(),
                &addr as *const SockaddrCtl as *const libc::sockaddr,
                std::mem::size_of::<SockaddrCtl>() as libc::socklen_t,
            )
        };
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }

        // 4. Get the assigned interface name via getsockopt
        let mut name_buf = [0u8; 32];
        let mut name_len: libc::socklen_t = name_buf.len() as libc::socklen_t;
        let ret = unsafe {
            libc::getsockopt(
                fd.as_raw_fd(),
                SYSPROTO_CONTROL,
                2, // UTUN_OPT_IFNAME
                name_buf.as_mut_ptr() as *mut libc::c_void,
                &mut name_len,
            )
        };
        let name = if ret == 0 {
            let end = name_buf.iter().position(|&b| b == 0).unwrap_or(name_len as usize);
            String::from_utf8_lossy(&name_buf[..end]).to_string()
        } else {
            "utun?".to_string()
        };

        Ok(TunHandle {
            fd,
            name,
            mtu: 1500,
        })
    }

    /// Configure the utun interface with IP addresses and MTU.
    /// Returns an error if ifconfig or route setup fails.
    pub fn configure(handle: &mut TunHandle, ip: &str, peer: &str, mtu: u32) -> io::Result<()> {
        handle.mtu = mtu as usize;
        let name = &handle.name;
        let status = Command::new("ifconfig")
            .args([name, ip, peer, "mtu", &mtu.to_string(), "up"])
            .status()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("ifconfig failed to run: {}", e)))?;
        if !status.success() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("ifconfig {} exited with {}", name, status),
            ));
        }

        // Ensure the host route points at this utun (remove any stale route first).
        let _ = Command::new("route")
            .args(["delete", "-host", peer])
            .status();
        let status = Command::new("route")
            .args(["add", "-host", peer, "-interface", name])
            .status()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("route failed to run: {}", e)))?;
        if !status.success() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("route add -host {} -interface {} exited with {}", peer, name, status),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let config = parse_args();

    if config.list_devices {
        list_devices();
        return;
    }

    let proto_config = params::get_protocol(config.protocol_id);
    let overhead_cp_len = if config.cable {
        params::CP_LEN_CABLE
    } else if config.wired {
        proto_config.cp_len
    } else {
        params::CP_LEN
    };
    let data_cp_len = if config.cable {
        params::CP_LEN_CABLE
    } else {
        proto_config.cp_len
    };
    let preamble_symbols = if config.wired {
        params::PREAMBLE_SYMBOLS_MINIMAL
    } else {
        params::PREAMBLE_SYMBOLS
    };
    let training_symbols = if config.wired {
        params::TRAINING_SYMBOLS_SHORT
    } else {
        params::TRAINING_SYMBOLS
    };

    // --- Create and configure MAC ---
    let mut mac = if config.wired {
        HalfDuplexMac::new_wired(config.protocol_id, config.volume, overhead_cp_len)
    } else {
        HalfDuplexMac::new(config.protocol_id, config.volume)
    };
    if config.cable {
        mac.set_data_cp_override(data_cp_len);
    }

    // Burst and flow control
    mac.set_burst_limit(config.burst_limit);
    // Wired with burst > 1: disable stop-and-wait to pipeline frames.
    // OTA with burst == 1: keep stop-and-wait for reliability.
    mac.set_stop_and_wait(config.burst_limit <= 1);

    // Timeouts: must account for the full path from TX end to ACK reception.
    //   data_frame + output_ring_drain + ack_quiet_threshold + ack_frame
    // We compute the actual ack_quiet below, so use a placeholder here and
    // set the final timeout after configuring the quiet threshold.
    let data_frame_ms = frame_duration_ms_custom(
        config.protocol_id,
        config.tun_mtu as usize,
        preamble_symbols,
        training_symbols,
        overhead_cp_len,
        data_cp_len,
    );
    let ack_frame_ms = frame_duration_ms_custom(
        config.protocol_id,
        1,
        preamble_symbols,
        training_symbols,
        overhead_cp_len,
        data_cp_len,
    );

    // Inter-frame gap, output ring sizing, and ACK quiet threshold.
    //
    // Key invariant: the output ring must drain within gap_samples so the
    // peer's carrier sense (gap_samples of silence) guarantees our echo
    // gate has reopened before their preamble arrives.
    //
    // The ack_quiet_threshold must exceed the sender's ring drain time so
    // the receiver waits for the sender's echo gate to open before
    // flushing the deferred ACK.
    let sym_total = params::FFT_SIZE + data_cp_len;
    let min_ring = 2048_usize; // 2 audio callback buffers — prevents underruns
    if config.wired {
        mac.set_gap(sym_total);
        mac.set_ack_quiet_threshold(sym_total.max(min_ring) + sym_total);
    } else {
        mac.set_ack_quiet_threshold(params::SYMBOL_TOTAL * 2 + sym_total);
    }

    // Timeout: data_frame + ack_quiet + ack_frame + safety margin.
    let gap_samples = if config.wired { sym_total } else { params::SYMBOL_TOTAL * 2 };
    let ack_quiet_ms = (gap_samples.max(min_ring) + sym_total) as f64 * 1000.0 / SAMPLE_RATE as f64;
    let round_trip_ms = data_frame_ms + ack_quiet_ms + ack_frame_ms;
    let timeout_ms = if config.wired {
        (round_trip_ms * 2.0).max(500.0) as u64
    } else {
        (round_trip_ms * 2.5).max(1500.0) as u64
    };
    mac.set_timeout(timeout_ms * SAMPLE_RATE as u64 / 1000);

    // --- Startup banner ---
    let mode_str = if config.cable {
        "cable"
    } else if config.wired {
        "wired"
    } else {
        "OTA"
    };
    let mtu_str = if config.mtu_is_auto {
        format!("{} (auto, {}ms target)", config.tun_mtu, config.target_frame_ms)
    } else {
        format!("{} (manual)", config.tun_mtu)
    };
    let flow_str = if config.burst_limit <= 1 {
        "stop-and-wait".to_string()
    } else {
        format!("burst={}, pipelined", config.burst_limit)
    };
    eprintln!("JackRabbit — IP over audio");
    eprintln!("  Protocol:  {} ({}, {}, {:?} {:?})",
        config.protocol_id, proto_config.name,
        mode_str,
        proto_config.modulation, proto_config.code_rate);
    eprintln!("  MTU:       {}", mtu_str);
    eprintln!("  Frame:     ~{:.0}ms data, ~{:.0}ms ACK, ~{:.0}ms round-trip",
        data_frame_ms, ack_frame_ms, round_trip_ms);
    eprintln!("  Flow:      {}", flow_str);
    eprintln!("  Timeout:   {}ms", timeout_ms);
    eprintln!("  Volume:    {}", config.volume);

    // --- Audio setup ---
    let host = cpal::default_host();
    let in_device = find_device(&host, &config.input_device, false);
    let out_device = find_device(&host, &config.output_device, true);
    let in_config = device_config(&in_device, false);
    let out_config = device_config(&out_device, true);
    let in_channels = in_config.channels as usize;
    let out_channels = out_config.channels as usize;

    eprintln!(
        "  Input:     {} ({} ch, {} Hz)",
        in_device.name().unwrap_or_default(),
        in_channels,
        in_config.sample_rate.0,
    );
    eprintln!(
        "  Output:    {} ({} ch, {} Hz)",
        out_device.name().unwrap_or_default(),
        out_channels,
        out_config.sample_rate.0,
    );

    let input_ring: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let output_ring: Arc<Mutex<VecDeque<f32>>> = Arc::new(Mutex::new(VecDeque::new()));

    // Input stream: mic → input_ring
    let ir = Arc::clone(&input_ring);
    let in_stream = in_device
        .build_input_stream(
            &in_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut ring = ir.lock().unwrap();
                if in_channels == 1 {
                    ring.extend_from_slice(data);
                } else {
                    for chunk in data.chunks(in_channels) {
                        ring.push(chunk[0]);
                    }
                }
            },
            |err| eprintln!("Input error: {}", err),
            None,
        )
        .expect("Failed to build input stream");

    // Output stream: output_ring → speaker
    let or = Arc::clone(&output_ring);
    let out_stream = out_device
        .build_output_stream(
            &out_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut ring = or.lock().unwrap();
                for frame in data.chunks_mut(out_channels) {
                    let sample = ring.pop_front().unwrap_or(0.0);
                    for s in frame.iter_mut() {
                        *s = sample;
                    }
                }
            },
            |err| eprintln!("Output error: {}", err),
            None,
        )
        .expect("Failed to build output stream");

    in_stream.play().expect("Failed to start input");
    out_stream.play().expect("Failed to start output");
    eprintln!("  Audio:     running");

    // --- TUN setup ---
    #[cfg(target_os = "macos")]
    let tun_handle = if !config.no_tun {
        let mut handle = match tun::open() {
            Ok(h) => h,
            Err(e) => {
                eprintln!("FATAL: Failed to create utun: {} (try sudo)", e);
                std::process::exit(1);
            }
        };
        if let Err(e) = tun::configure(&mut handle, &config.tun_ip, &config.tun_peer, config.tun_mtu) {
            eprintln!("FATAL: Failed to configure {}: {}", handle.name, e);
            std::process::exit(1);
        }
        eprintln!("  TUN:       {} ({} -> {}, MTU {})",
            handle.name, config.tun_ip, config.tun_peer, config.tun_mtu);
        handle.set_nonblocking();
        Some(handle)
    } else {
        None
    };

    #[cfg(not(target_os = "macos"))]
    let tun_handle: Option<()> = {
        if !config.no_tun {
            eprintln!("  TUN:       not supported (macOS only). Audio-only mode.");
        }
        None
    };

    let has_tun = tun_handle.is_some();
    eprintln!();
    if has_tun {
        eprintln!("Ready. ping {} to test.", config.tun_peer);
    } else {
        eprintln!("Ready. Audio-only mode — received payloads printed to stderr.");
    }

    // --- Main loop ---

    // Backpressure: limit TX queue depth relative to burst size.
    // For burst=1: queue up to 8 packets (stop-and-wait drains slowly).
    // For burst=4: queue up to 16 (pipeline keeps the link busy).
    let max_tx_queue = (config.burst_limit * 4).max(8);

    let mut last_poll = Instant::now();
    let mut stats_time = Instant::now();
    let mut stats_tx: u64 = 0;
    let mut stats_rx: u64 = 0;
    let mut stats_tx_pkts: u64 = 0;
    let mut stats_rx_pkts: u64 = 0;
    let mut stats_drops: u64 = 0;
    let mut last_output_backlog: usize = 0;
    let debug = std::env::var("JACKRABBIT_DEBUG").is_ok();

    loop {
        let now = Instant::now();
        let elapsed = now.duration_since(last_poll);
        let elapsed_samples =
            (elapsed.as_micros() as u64 * SAMPLE_RATE as u64 / 1_000_000) as usize;
        last_poll = now;

        // 1. Read IP packets from TUN → enqueue to MAC.
        // Backpressure: drop packets when the TX queue is deep to prevent
        // unbounded memory growth from applications that send faster than
        // the modem link (e.g. flood ping, TCP without proper congestion).
        #[cfg(target_os = "macos")]
        if let Some(ref handle) = tun_handle {
            loop {
                match handle.read() {
                    Ok(pkt) => {
                        if mac.tx_pending() < max_tx_queue {
                            stats_tx += pkt.len() as u64;
                            stats_tx_pkts += 1;
                            mac.enqueue(pkt);
                        } else {
                            stats_drops += 1;
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                    Err(e) => {
                        eprintln!("TUN read error: {} (kind: {:?})", e, e.kind());
                        break;
                    }
                }
            }
        }

        // 2. Drain input audio → feed MAC.
        // Echo gate: discard input while ANY of our audio is still
        // being played — both MAC output_buf AND the audio output ring.
        // The output ring holds up to 200ms of samples that are still
        // playing on the sound card (and loop back on devices like
        // BlackHole).  The ring tail can contain valid preamble symbols,
        // causing the decoder to false-lock on our own echo.
        //
        // This is safe because ACKs are always deferred: the peer waits
        // for silence (our ring fully drained) before sending its ACK,
        // so no real data arrives while we're suppressing.
        {
            let output_backlog = output_ring.lock().unwrap().len();
            let is_still_playing = mac.is_transmitting() || output_backlog > 0;
            let mut ring = input_ring.lock().unwrap();
            if !ring.is_empty() {
                if is_still_playing {
                    ring.clear(); // discard self-echo
                } else {
                    let samples: Vec<f32> = ring.drain(..).collect();
                    mac.feed(&samples);
                }
            }
        }

        // 3. Poll MAC
        mac.poll(elapsed_samples);

        // 4. Drip-feed MAC output → audio output ring.
        // Only move a small chunk per iteration so the MAC keeps the bulk
        // of the burst in its own output_buf.  This way is_transmitting()
        // stays true (and the echo gate stays active) for the duration of
        // the actual playback — not just the instant we copy into the ring.
        // The ring size matches gap_samples so it drains within one
        // carrier-sense period, keeping the echo gate blind period short.
        let max_ring_ahead = gap_samples.max(min_ring);
        let backlog = {
            let mut ring = output_ring.lock().unwrap();
            let headroom = max_ring_ahead.saturating_sub(ring.len());
            let avail = mac.output_available().min(headroom);
            if avail > 0 {
                let mut buf = vec![0.0f32; avail];
                mac.fill_output(&mut buf);
                ring.extend(buf.iter());
            }
            ring.len()
        };
        if backlog > 0 {
            mac.note_output_backlog(backlog);
        } else if last_output_backlog > 0 {
            mac.note_output_backlog(0);
        }
        last_output_backlog = backlog;

        // 5. Deliver received payloads → TUN or stderr
        while let Some(payload) = mac.recv() {
            stats_rx += payload.len() as u64;
            stats_rx_pkts += 1;
            #[cfg(target_os = "macos")]
            {
                if let Some(ref handle) = tun_handle {
                    if let Err(e) = handle.write(&payload) {
                        eprintln!("TUN write error: {} ({} bytes)", e, payload.len());
                    }
                } else {
                    print_packet(&payload);
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                print_packet(&payload);
            }
        }

        // 6. Periodic stats
        if now.duration_since(stats_time) >= Duration::from_secs(1) {
            let dt = now.duration_since(stats_time).as_secs_f64();
            let state = if mac.is_transmitting() {
                "TX"
            } else if !mac.arq().is_idle() {
                "WAIT"
            } else if !mac.is_idle() {
                "QUEUE"
            } else {
                "IDLE"
            };
            if debug {
                let di = mac.arq().decoder().debug_info();
                eprintln!(
                    "[{}] tx {:.0} bps ({} pkts) | rx {:.0} bps ({} pkts) | drop {} | q {} | arq {} | dec {} | last {} | pre {}",
                    state,
                    stats_tx as f64 * 8.0 / dt,
                    stats_tx_pkts,
                    stats_rx as f64 * 8.0 / dt,
                    stats_rx_pkts,
                    stats_drops,
                    mac.tx_pending(),
                    if mac.arq().is_idle() { "idle" } else { "fly" },
                    di.state,
                    di.last_event,
                    di.preamble_count,
                );
            } else {
                eprintln!(
                    "[{}] tx {:.0} bps | rx {:.0} bps | q {}{}",
                    state,
                    stats_tx as f64 * 8.0 / dt,
                    stats_rx as f64 * 8.0 / dt,
                    mac.tx_pending(),
                    if stats_drops > 0 { format!(" | drop {}", stats_drops) } else { String::new() },
                );
            }
            stats_tx = 0;
            stats_rx = 0;
            stats_tx_pkts = 0;
            stats_rx_pkts = 0;
            stats_drops = 0;
            stats_time = now;
        }

        std::thread::sleep(Duration::from_millis(1));
    }
}

fn print_packet(data: &[u8]) {
    if data.is_empty() {
        return;
    }
    let ip_ver = data[0] >> 4;
    let preview_len = data.len().min(40);
    eprintln!(
        "[RX] IPv{} {} bytes: {:02x?}{}",
        ip_ver,
        data.len(),
        &data[..preview_len],
        if data.len() > preview_len { "..." } else { "" },
    );
}
