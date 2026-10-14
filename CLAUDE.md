# Claude Code Guide — jackrabbit.rs

## Build

```bash
# Run tests
cargo test

# Build the native CLI
cargo build --release --features cli --bin jackrabbit

# Build the browser WASM package
wasm-pack build --release --target web --features wasm
```

The browser build writes `pkg/jackrabbit.js` and `pkg/jackrabbit_bg.wasm`.
