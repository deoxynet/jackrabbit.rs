/// Hardware OTA loopback test.
///
/// Plays encoded audio through speakers, records from the microphone,
/// and attempts to decode. Requires a working audio device at 48 kHz.
///
/// Usage:
///   cargo run --example ota_loopback [-- --protocol N --payload "text"]

use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SampleRate, StreamConfig};

use aqua_soundbin::decoder::Decoder;
use aqua_soundbin::encoder;
use aqua_soundbin::params;

const TARGET_SAMPLE_RATE: u32 = 48_000;

fn main() {
    // Parse args
    let args: Vec<String> = std::env::args().collect();
    let mut protocol: u8 = 0;
    let mut payload_text = String::from("Hello, OTA loopback!");
    let mut listen_mode = false;
    let mut listen_secs: f64 = 10.0;
    let mut input_device_name: Option<String> = None;
    let mut output_device_name: Option<String> = None;


    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--protocol" => {
                i += 1;
                protocol = args[i].parse().expect("Invalid protocol number");
            }
            "--payload" => {
                i += 1;
                payload_text = args[i].clone();
            }
            "--input-device" => {
                i += 1;
                input_device_name = Some(args[i].clone());
            }
            "--output-device" => {
                i += 1;
                output_device_name = Some(args[i].clone());
            }
            "--listen" => {
                listen_mode = true;
                if i + 1 < args.len() {
                    if let Ok(s) = args[i + 1].parse::<f64>() {
                        listen_secs = s;
                        i += 1;
                    }
                }
            }

            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                eprintln!("Usage: ota_loopback [--protocol N] [--payload \"text\"] [--listen [secs]] [--input-device <name>] [--output-device <name>]");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let payload = payload_text.as_bytes();
    let config = params::get_protocol(protocol);
    eprintln!("Protocol: {} ({})", protocol, config.name);
    eprintln!("Payload: {:?} ({} bytes)", payload_text, payload.len());

    // Encode
    let waveform = encoder::encode(payload, protocol, 0.8);
    let waveform_secs = waveform.len() as f64 / TARGET_SAMPLE_RATE as f64;
    eprintln!("Encoded: {} samples ({:.2}s)", waveform.len(), waveform_secs);



    // In listen mode, loop the signal for the requested duration so the user
    // can hear what the OFDM waveform sounds like. No recording/decoding.
    if listen_mode {
        eprintln!("Listen mode: looping signal for {:.0}s", listen_secs);
        let gap = vec![0.0f32; (TARGET_SAMPLE_RATE as f64 * 0.3) as usize]; // 300ms gap
        let mut play_buf = Vec::new();
        let total_samples = (listen_secs * TARGET_SAMPLE_RATE as f64) as usize;
        while play_buf.len() < total_samples {
            play_buf.extend_from_slice(&waveform);
            play_buf.extend_from_slice(&gap);
        }
        play_buf.truncate(total_samples);
        play_only(&play_buf, listen_secs);
        return;
    }

    // --- Hardware OTA mode ---

    // Prepend 0.5s silence, append 1.0s silence
    let pre_silence = vec![0.0f32; (TARGET_SAMPLE_RATE / 2) as usize];
    let post_silence = vec![0.0f32; TARGET_SAMPLE_RATE as usize];
    let mut play_buf = Vec::with_capacity(pre_silence.len() + waveform.len() + post_silence.len());
    play_buf.extend_from_slice(&pre_silence);
    play_buf.extend_from_slice(&waveform);
    play_buf.extend_from_slice(&post_silence);

    let total_play_samples = play_buf.len();
    let total_play_secs = total_play_samples as f64 / TARGET_SAMPLE_RATE as f64;
    eprintln!("Total playback: {} samples ({:.2}s)", total_play_samples, total_play_secs);

    // Set up audio
    let host = cpal::default_host();

    let out_device = if let Some(name) = output_device_name {
        host.output_devices()
            .expect("Failed to get output devices")
            .find(|d| d.name().unwrap_or_default().to_lowercase().contains(&name.to_lowercase()))
            .expect(&format!(
                "Could not find output device '{}'. Available devices: {:?}",
                name,
                host.output_devices()
                    .unwrap()
                    .map(|d| d.name().unwrap_or_default())
                    .collect::<Vec<_>>()
            ))
    } else {
        host.default_output_device()
            .expect("No default output device found")
    };

    let in_device = if let Some(name) = input_device_name {
        host.input_devices()
            .expect("Failed to get input devices")
            .find(|d| d.name().unwrap_or_default().to_lowercase().contains(&name.to_lowercase()))
            .expect(&format!(
                "Could not find input device '{}'. Available devices: {:?}",
                name,
                host.input_devices()
                    .unwrap()
                    .map(|d| d.name().unwrap_or_default())
                    .collect::<Vec<_>>()
            ))
    } else {
        host.default_input_device()
            .expect("No default input device found")
    };

    eprintln!("Output device: {}", out_device.name().unwrap_or_default());
    eprintln!("Input device: {}", in_device.name().unwrap_or_default());

    // Find a suitable output config
    let out_config = find_mono_config(&out_device, true);
    let in_config = find_mono_config(&in_device, false);

    let out_channels = out_config.channels as usize;
    let in_channels = in_config.channels as usize;

    eprintln!("Output config: {} ch, {} Hz", out_channels, out_config.sample_rate.0);
    eprintln!("Input config: {} ch, {} Hz", in_channels, in_config.sample_rate.0);

    assert_eq!(
        out_config.sample_rate.0, TARGET_SAMPLE_RATE,
        "Output device does not support 48 kHz"
    );
    assert_eq!(
        in_config.sample_rate.0, TARGET_SAMPLE_RATE,
        "Input device does not support 48 kHz"
    );

    // Shared state
    let play_pos = Arc::new(Mutex::new(0usize));
    let play_done = Arc::new(Mutex::new(false));
    let rec_buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let rec_done = Arc::new(Mutex::new(false));

    // Recording stream (start first so we don't miss anything)
    let rec_buf_clone = Arc::clone(&rec_buf);
    let rec_done_clone = Arc::clone(&rec_done);
    let in_stream = in_device.build_input_stream(
        &in_config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            if *rec_done_clone.lock().unwrap() {
                return;
            }
            let mut buf = rec_buf_clone.lock().unwrap();
            if in_channels == 1 {
                buf.extend_from_slice(data);
            } else {
                // Take left channel only
                for chunk in data.chunks(in_channels) {
                    buf.push(chunk[0]);
                }
            }
        },
        move |err| {
            eprintln!("Input stream error: {}", err);
        },
        None,
    ).expect("Failed to build input stream");

    // Playback stream
    let play_buf = Arc::new(play_buf);
    let play_buf_clone = Arc::clone(&play_buf);
    let play_pos_clone = Arc::clone(&play_pos);
    let play_done_clone = Arc::clone(&play_done);
    let out_stream = out_device.build_output_stream(
        &out_config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let mut pos = play_pos_clone.lock().unwrap();
            let buf = &play_buf_clone;

            for frame in data.chunks_mut(out_channels) {
                let sample = if *pos < buf.len() {
                    buf[*pos]
                } else {
                    0.0
                };
                // Duplicate mono to all channels
                for s in frame.iter_mut() {
                    *s = sample;
                }
                *pos += 1;
            }

            if *pos >= buf.len() {
                *play_done_clone.lock().unwrap() = true;
            }
        },
        move |err| {
            eprintln!("Output stream error: {}", err);
        },
        None,
    ).expect("Failed to build output stream");

    // Start streams
    in_stream.play().expect("Failed to start input stream");
    out_stream.play().expect("Failed to start output stream");

    eprintln!("Playing and recording...");

    // Wait for playback to finish + 1s margin
    let wait_secs = total_play_secs + 1.0;
    std::thread::sleep(std::time::Duration::from_secs_f64(wait_secs));

    // Stop
    *rec_done.lock().unwrap() = true;
    drop(out_stream);
    drop(in_stream);

    let recording = rec_buf.lock().unwrap().clone();
    eprintln!("Recorded: {} samples ({:.2}s)",
        recording.len(),
        recording.len() as f32 / TARGET_SAMPLE_RATE as f32);

    if recording.is_empty() {
        eprintln!("ERROR: No audio recorded. Check microphone permissions.");
        std::process::exit(1);
    }

    // Decode
    eprintln!("Decoding...");
    let mut decoder = Decoder::new();
    let result = decoder.feed(&recording);

    if result.is_empty() {
        eprintln!("FAIL: Decoder returned empty result.");
        eprintln!("  Recording RMS: {:.6}", rms(&recording));
        eprintln!("  Recording peak: {:.6}", recording.iter().map(|s| s.abs()).fold(0.0f32, f32::max));
        std::process::exit(1);
    }

    if result == payload {
        eprintln!("SUCCESS: Decoded payload matches!");
    } else {
        eprintln!("FAIL: Payload mismatch.");
        eprintln!("  Expected: {:?}", std::str::from_utf8(payload).unwrap_or("<binary>"));
        eprintln!("  Got:      {:?}", std::str::from_utf8(&result).unwrap_or("<binary>"));
        eprintln!("  Expected len: {}, Got len: {}", payload.len(), result.len());
    }

    // Channel report
    if let Some(report) = decoder.channel_report() {
        eprintln!("\nChannel Report:");
        let band_names = ["Audible (1-4 kHz)", "DT (1-6 kHz)", "MT (1-8 kHz)", "UT (1-16 kHz)"];
        for (i, band) in report.bands.iter().enumerate() {
            if band.snr_db.is_finite() {
                eprintln!("  {}: avg SNR = {:.1} dB, min SNR = {:.1} dB ({} carriers)",
                    band_names.get(i).unwrap_or(&"?"),
                    band.snr_db, band.snr_min_db, band.num_carriers);
            }
        }
        eprintln!("  Recommended protocol: {} ({})",
            report.recommended_protocol,
            params::get_protocol(report.recommended_protocol).name);
        eprintln!("  Viable protocols: {:?}", report.viable_protocols);
    }

    if result == payload {
        std::process::exit(0);
    } else {
        std::process::exit(1);
    }
}

/// Play audio without recording — just output to speakers.
fn play_only(play_buf: &[f32], duration_secs: f64) {
    let host = cpal::default_host();
    let out_device = host.default_output_device().expect("No output device found");
    let out_config = find_mono_config(&out_device, true);
    let out_channels = out_config.channels as usize;
    eprintln!("Output device: {} ({} ch, {} Hz)",
        out_device.name().unwrap_or_default(), out_channels, out_config.sample_rate.0);

    let play_buf = Arc::new(play_buf.to_vec());
    let play_buf_clone = Arc::clone(&play_buf);
    let play_pos = Arc::new(Mutex::new(0usize));
    let play_pos_clone = Arc::clone(&play_pos);

    let out_stream = out_device.build_output_stream(
        &out_config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let mut pos = play_pos_clone.lock().unwrap();
            for frame in data.chunks_mut(out_channels) {
                let sample = if *pos < play_buf_clone.len() {
                    play_buf_clone[*pos]
                } else {
                    0.0
                };
                for s in frame.iter_mut() {
                    *s = sample;
                }
                *pos += 1;
            }
        },
        |err| eprintln!("Output stream error: {}", err),
        None,
    ).expect("Failed to build output stream");

    out_stream.play().expect("Failed to start output stream");
    eprintln!("Playing...");
    std::thread::sleep(std::time::Duration::from_secs_f64(duration_secs + 0.5));
    drop(out_stream);
    eprintln!("Done.");
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() { return 0.0; }
    let power: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
    power.sqrt()
}

/// Find a f32 config at 48 kHz from supported config ranges.
fn find_config_from_ranges(
    ranges: impl Iterator<Item = cpal::SupportedStreamConfigRange>,
) -> Option<StreamConfig> {
    for cfg in ranges {
        if cfg.sample_format() == SampleFormat::F32
            && cfg.min_sample_rate() <= SampleRate(TARGET_SAMPLE_RATE)
            && cfg.max_sample_rate() >= SampleRate(TARGET_SAMPLE_RATE)
        {
            return Some(cfg.with_sample_rate(SampleRate(TARGET_SAMPLE_RATE)).into());
        }
    }
    None
}

/// Find a mono (or stereo) f32 config at 48 kHz.
fn find_mono_config(device: &cpal::Device, is_output: bool) -> StreamConfig {
    // Try to find a config with 48 kHz support and f32 format
    let from_ranges = if is_output {
        find_config_from_ranges(
            device.supported_output_configs()
                .expect("Failed to query supported output configs"),
        )
    } else {
        find_config_from_ranges(
            device.supported_input_configs()
                .expect("Failed to query supported input configs"),
        )
    };

    if let Some(config) = from_ranges {
        return config;
    }

    // Fallback: use default config
    let default = if is_output {
        device.default_output_config()
    } else {
        device.default_input_config()
    }.expect("No default config available");

    assert_eq!(
        default.sample_rate().0, TARGET_SAMPLE_RATE,
        "Device does not support 48 kHz"
    );

    default.into()
}
