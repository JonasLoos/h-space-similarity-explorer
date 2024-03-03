# WASM Webworker for heavy computations

This is a subproject in rust that is compiled to wasm and run in a webworker. It is used to offload heavy computations from the main thread to a separate thread.

## Build
```bash
wasm-pack build --target web --release --no-typescript --no-pack
```
