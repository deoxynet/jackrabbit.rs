# Claude Code Guide — aqua-soundbin.rs

## Build & Deploy

```bash
# Run tests
cargo test

# Build WASM and deploy to ofdm.soundbin.io
cd ~/git/aqua-soundbin.rs && wasm-pack build --release --target web --features wasm \
  && cp pkg/aqua_soundbin.js pkg/aqua_soundbin_bg.wasm \
  ~/git/soundbin.io-ofdm/public/wasm/ && cd ~/git/soundbin.io-ofdm && make deploy
```
