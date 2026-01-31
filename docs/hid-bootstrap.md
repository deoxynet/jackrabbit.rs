# HID Bootstrap: Air-Gapped File Transfer via Audio

## Concept

Use a USB HID keyboard device (Rubber Ducky-style) to bootstrap jackrabbit's
WASM audio modem onto a locked-down target machine. Once loaded, files transfer
bidirectionally over a 3.5mm audio cable using OFDM.

## Threat Model

Target machine is enterprise/government locked-down:

- USB mass storage: **blocked**
- Network (WiFi, Ethernet, Bluetooth): **restricted/monitored**
- Browser dev tools (F12): **may be blocked**
- USB HID keyboard: **allowed** (must accept keyboard input)
- Audio jack (mic + speakers): **allowed** (headset/conferencing use)

## Architecture

```
Attacker Laptop                              Target Machine
[jackrabbit native] <-- audio cable --> [jackrabbit WASM in browser]
     speaker out    ------>             mic in
     mic in         <------            speaker out
```

The attacker side runs jackrabbit natively (or in a browser). The target side
runs jackrabbit as a WASM module bootstrapped via HID keystroke injection.

## Bootstrap Sequence

### Stage 1: Bookmarklet (~1-2 KB, typed in seconds)

HID device types a `javascript:` bookmarklet into the browser URL bar. This is
small enough to fit in the browser's URL length limit and executes instantly.

The bookmarklet:
1. Creates a full-screen textarea for receiving the WASM payload
2. Sets up base64 decode + WASM instantiation pipeline
3. Initializes Web Audio API (mic + speaker access)
4. Wires audio I/O to the WASM module once loaded

```
HID types into URL bar:
javascript:void((...bookmarklet code...))
```

Note: If `javascript:` URLs are blocked, fall back to a `data:text/html,`
URI which some browsers still accept in the address bar.

### Stage 2: WASM Payload (~205K chars, ~3.2 min at max HID speed)

HID device pastes the gzipped+base64-encoded WASM binary into the textarea
created by the bookmarklet. At USB HID maximum throughput (~1000 chars/sec),
this takes approximately 3.2 minutes.

### Stage 3: Activation

Bookmarklet detects paste completion, decodes base64, decompresses gzip,
instantiates the WASM module, requests microphone permission, and opens the
audio link. The target machine is now a jackrabbit modem.

### Stage 4: File Transfer

Files transfer over the audio cable using the full jackrabbit stack:
```
File -> framing protocol -> ARQ -> MAC -> PHY (OFDM) -> Audio
```

## WASM Binary Size Budget

Current measurements (2025-01):

| Build Config | Raw WASM | gzipped | base64 | HID time @1000c/s |
|---|---|---|---|---|
| opt-level 3 (default) | 659 KB | 192 KB | 262K chars | 4.4 min |
| opt-level "z" + strip | 493 KB | 163 KB | 217K chars | 3.6 min |
| + wasm-opt -Oz | 418 KB | 153 KB | 205K chars | 3.4 min |
| Theoretical floor | ~350 KB | ~140 KB | ~190K chars | 3.2 min |

### Size Breakdown

| Component | Size in WASM | Notes |
|---|---|---|
| rustfft + transpose + deps | ~156 KB | Dominates binary. General-purpose FFT planner. |
| decoder | 11 KB | Core, required |
| preamble | 9 KB | Core, required |
| ldpc | 7 KB | Required for cable protocols |
| modulation (all schemes) | 6.5 KB | Could trim to cable-only (QAM64/256/1024) |
| fec (Viterbi/convolutional) | 6 KB | OTA-only, not needed for cable |
| encoder | 4.4 KB | Core, required |
| adaptive | 2.2 KB | OTA-only, not needed for cable |
| impairments | 0 KB | Test-only, already dead-code eliminated |

### Potential Further Reductions

- **Feature-gate OTA modules** (`fec`, `adaptive`): saves ~8 KB raw, ~3 KB gzip
- **Hand-rolled FFT**: jackrabbit only needs FFT size 960 (= 64 x 15). A
  dedicated mixed-radix implementation could replace the general-purpose rustfft
  planner, potentially saving 50-100 KB
- **Simpler bootstrap modulation**: use a minimal FSK/BPSK-only first stage,
  then upgrade to full OFDM. Smaller initial binary, but adds complexity

## Hardware: Full-Duplex Crossover Cable

Standard aux cable won't work (connects output-to-output). Need crossed signals
with voltage dividers to attenuate headphone-level output to mic-level input.

```
LAPTOP A                                              LAPTOP B

                    Channel 1 (A -> B)
Tip (L out) ------- R1 [10k] ---+--- C1 [10uF] ------ Sleeve (Mic in)
                                |
                            R2 [1k]
                                |
                               GND

                    Channel 2 (B -> A)
Sleeve (Mic in) --- C2 [10uF] --+--- R3 [10k] ------- Tip (L out)
                                |
                            R4 [1k]
                                |
                               GND

                    Ground
Ring2 (GND) ------------------------------------------ Ring2 (GND)
Ring1 (R out) --- NC
Ring1 (R out) --- NC
```

**BOM:** 2x 10k resistors, 2x 1k resistors, 2x 10uF caps, 2x TRRS plugs (~$3-5).

**Alternative:** Two USB audio adapters ($8 each) with separate mic/headphone
jacks + two standard aux cables. No soldering, full duplex.

## Productization Ideas

### USB-C Audio Adapter Kit (Hak5-style)

Sold in pairs. Each unit is a USB-C audio adapter with separate line-in/out
jacks. Ship with a pre-made crossover cable. Total BOM ~$25/pair at volume.

### Raspberry Pi Standalone Kit

Pi Zero 2 W + USB audio adapter + SD card with jackrabbit OS image. ~$25 BOM
per unit. Hosts a local web UI over WiFi for drag-and-drop file transfer.
The audio link is the air-gapped transport between two Pi boxes.

## Fileless Browser Injection Techniques

All techniques below assume: no dev tools (F12 blocked), no file system access,
no network egress. The HID keyboard is the only input vector. Goal is to get
JavaScript execution in a browser context.

### 1. `javascript:` Bookmarklet (Primary)

HID types a `javascript:` URL into the browser address bar and hits Enter.
~2 KB payload limit in practice. Sufficient for a WASM loader stub.

**Compat:** Chrome strips `javascript:` when pasting but allows typed input.
Firefox and Safari still allow it. Some enterprise policies block it entirely.

### 2. `data:text/html;base64,` URI (Best Fallback)

HID types a full HTML page as a data URI into the address bar:

```
data:text/html;base64,PGh0bWw+PHNjcmlwdD4uLi48L3NjcmlwdD48L2h0bWw+
```

More compatible than `javascript:` — browsers treat it as a page navigation,
not script execution from the URL bar. Base64 encoding avoids URL-escaping
issues. Larger than a bookmarklet but the loader HTML can be more complete
(include a textarea, status display, audio visualizer).

**Compat:** Chrome, Firefox, Safari all support data URIs in the address bar.
Some enterprise policies block `data:` scheme navigation.

### 3. `data:text/html,` URI (Unencoded Variant)

Same as above but without base64. HID types raw HTML:

```
data:text/html,<script>document.write('<textarea id=t></textarea>');</script>
```

Smaller for simple payloads but special characters need URL encoding.

### 4. Browser Console via Alternative Shortcuts

Even with F12 disabled, alternative console access may work:

- `Ctrl+Shift+J` (Chrome DevTools console directly)
- `Ctrl+Shift+K` (Firefox Web Console)
- `Cmd+Option+J` (Chrome on macOS)
- `Ctrl+Shift+I` (alternative to F12, sometimes blocked separately)

Enterprise policies that block F12 don't always block all of these.
HID tries each in sequence until one opens.

### 5. Webmail / Internal Wiki Textarea

Many locked-down machines have browser access to webmail (OWA, Gmail) or
internal wikis (Confluence, SharePoint). Some rich text editors execute or
preview JavaScript:

- Confluence: preview mode can render HTML/JS in macros
- OWA: HTML compose mode with certain mailbox configurations
- SharePoint: Script Editor web parts

Fragile and target-specific. Requires reconnaissance on what internal apps
are available.

### 6. Service Worker Persistence

Once JavaScript execution is achieved via any method above, register a
Service Worker. It persists across page loads and browser restarts until
manually cleared from `chrome://serviceworker-internals` or equivalent.

```javascript
navigator.serviceWorker.register('data:text/javascript,'+encodeURIComponent(swCode))
```

This makes the bootstrap survive tab closure. The user can re-open the
browser and the audio modem is still running. Note: Service Workers require
a secure context (HTTPS or localhost), so this only works with `data:` origins
in some browsers.

### 7. WebRTC SDP Paste (Alternative Data Channel)

Two browsers can establish a direct peer connection by manually exchanging
SDP offers. No signaling server needed:

1. Initiator browser generates SDP offer, displays in textarea
2. User copies offer to other machine (via HID, or read aloud, or QR code)
3. Other machine generates SDP answer, pastes back
4. Direct data channel established

Doesn't apply to the audio use case but demonstrates another fileless
bidirectional channel. Could complement audio for out-of-band signaling.

### Recommended Priority

1. **`data:text/html;base64,`** — most compatible, most robust
2. **`javascript:` bookmarklet** — smaller, faster to type, but increasingly blocked
3. **Alternative console shortcuts** — opportunistic, try if the above fail
4. **Service Worker** — for persistence after initial injection
5. **Internal app injection** — last resort, requires target-specific recon

## Open Questions

- [ ] Bookmarklet size: can the full Web Audio setup + WASM loader fit in ~2 KB?
- [ ] Browser compat: which browsers still allow `javascript:` URLs pasted into
  the address bar? Chrome has been restricting this.
- [ ] `data:text/html,` fallback: encode a minimal HTML page as a data URI
  that the HID types into the URL bar. Larger but more compatible.
- [ ] Mic permission prompt: requires user click on the target machine. HID can
  simulate Tab + Enter to accept, but timing is browser-dependent.
- [ ] Can we use Web Serial API as an alternative to Web Audio on machines where
  serial ports are available but audio is not?
- [ ] Investigate Brotli instead of gzip for the WASM payload (better
  compression ratio, ~10-15% smaller, but needs a JS decompressor in the
  bookmarklet)
