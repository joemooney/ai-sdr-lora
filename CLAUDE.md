# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**R4W - Rust for Waveforms** - A platform for developing, testing, and deploying SDR waveforms in Rust. Provides reusable DSP libraries, educational tools, and production-ready components for waveform development.

### Architecture

- **r4w-core**: Core DSP algorithms, timing, RT primitives, configuration
  - `waveform/`: 38+ waveform implementations
  - `analysis/`: Spectrum analyzer, waterfall generator, signal statistics, peak detection
  - `timing.rs`: Multi-clock model (SampleClock, WallClock, HardwareClock, SyncedTime)
  - `rt/`: Lock-free ring buffers, buffer pools, RT thread spawning
  - `config.rs`: YAML-based configuration system
- **r4w-sim**: SDR simulation, HAL traits, channel models
  - `hal/`: StreamHandle, TunerControl, ClockControl, SdrDeviceExt traits
  - `channel.rs`: AWGN, Rayleigh, Rician, CFO, TDL multipath (EPA/EVA/ETU), Jake's Doppler
  - `doppler.rs`: Jake's/Clarke's, Flat, and Gaussian Doppler models
  - `simulator.rs`: Software SDR simulator
- **r4w-fpga**: FPGA acceleration (Xilinx Zynq, Lattice iCE40/ECP5)
- **r4w-sandbox**: Waveform isolation (8 security levels)
- **r4w-gui**: Educational egui application (run with `cargo run --bin r4w-explorer`)
- **r4w-cli**: Command-line interface (run with `cargo run --bin r4w`)
- **r4w-web**: WebAssembly entry point for browser deployment

### Key Commands

```bash
# Run GUI
cargo run --bin r4w-explorer

# CLI examples
cargo run --bin r4w -- info --sf 7 --bw 125
cargo run --bin r4w -- simulate --message "Hello R4W!" --snr 10.0
cargo run --bin r4w -- waveform --list

# Mesh networking
cargo run --bin r4w -- mesh info
cargo run --bin r4w -- mesh status --preset LongFast --region US
cargo run --bin r4w -- mesh send -m "Hello mesh!" --dest broadcast
cargo run --bin r4w -- mesh simulate --nodes 4 --messages 10

# Waveform comparison (BER vs SNR)
cargo run --bin r4w -- compare -w BPSK,QPSK,16QAM --snr-min -5 --snr-max 20
cargo run --bin r4w -- compare --list  # Show available waveforms

# Shell completions (bash/zsh/fish/powershell)
cargo run --bin r4w -- completions bash > ~/.local/share/bash-completion/completions/r4w

# Record/Playback signals (SigMF format)
cargo run --bin r4w -- record -o test.sigmf --generate tone --duration 5.0
cargo run --bin r4w -- playback -i test.sigmf --info

# Signal analysis
cargo run --bin r4w -- analyze spectrum -i file.sigmf-meta --fft-size 1024
cargo run --bin r4w -- analyze waterfall -i file.sigmf-meta --output waterfall.png
cargo run --bin r4w -- analyze stats -i file.sigmf-meta --format json
cargo run --bin r4w -- analyze peaks -i file.sigmf-meta --threshold 10

# Generate gallery images
cargo run --example gallery_generate -p r4w-sim --features image

# Prometheus metrics
cargo run --bin r4w -- metrics --format prometheus
cargo run --bin r4w -- metrics --serve --port 9090

# Run in browser (WASM)
cd crates/r4w-web && trunk serve

# Docker
docker run -it r4w:latest r4w --help
docker-compose run r4w-dev  # Development container

# cargo-binstall (pre-built binaries)
cargo binstall r4w-cli

# Cross-compile for ARM
make build-cli-arm64
make deploy-arm64 REMOTE_HOST=joe@raspberrypi
```

### Waveform Development

All waveforms implement the `Waveform` trait:
- `modulate(&self, bits: &[bool]) -> Vec<IQSample>`
- `demodulate(&self, samples: &[IQSample]) -> Vec<bool>`
- `constellation_points(&self) -> Vec<IQSample>`
- `get_modulation_stages()` / `get_demodulation_steps()` for education

See OVERVIEW.md for the full Waveform Developer's Guide and Porting Guide.

### Technical Notes

- LoRa uses Chirp Spread Spectrum (CSS) modulation
- FFT-based demodulation (multiply by downchirp, find peak)
- Pipeline: Whitening → Hamming FEC → Interleaving → Gray Code → CSS
- Spreading factors 5-12, bandwidths 125/250/500 kHz
- PSK/FSK/QAM waveforms for comparison and education

### Recent Updates

- **Enhanced Channel Simulation** - Jake's/Clarke's Doppler model, Tapped Delay Line (TDL) multipath with 3GPP profiles (EPA, EVA, ETU)
- **CLI Analysis Tools** - `r4w analyze` subcommands: spectrum, waterfall, stats, peaks
- **Signal Gallery** - 23 PNG images: constellations, spectra, channel effects (`gallery/`)
- **Jupyter Notebooks** - 8 interactive tutorials with Python wrapper (`notebooks/`)
- **Deployment Options** - Docker image, cargo-binstall manifest, GitHub Actions release workflow
- **CLI Enhancements** - Record/Playback (SigMF), Prometheus metrics, shell completions, waveform comparison
- **Example Gallery** - Getting-started examples for modulation, channels, LoRa, mesh networking
- **GitHub Actions CI** - Automated testing, cross-platform builds, WASM builds, performance regression tracking
- **Mesh CLI Commands** - `r4w mesh` subcommands for LoRa mesh networking (status, send, neighbors, simulate, info)
- **Mesh Networking Module** - Full mesh stack with MeshtasticNode, FloodRouter, CSMA/CA MAC, LoRaMesh integration
- **Physical Layer Architecture** (Session 18):
  - Multi-clock timing model (SampleClock, WallClock, HardwareClock, GPS/PTP SyncedTime)
  - Real-time primitives (lock-free SPSC ring buffers, buffer pools, RT thread spawning)
  - YAML configuration system with hardware profiles
  - Enhanced HAL traits (StreamHandle, TunerControl, ClockControl)
- Added TETRA, DMR, and 3G ALE waveform implementations
- Enhanced waveform-spec schema with TDMA/FDMA, radar, classified components
- Renamed project to "R4W - Rust for Waveforms"
- Created platform vision with Waveform Developer's Guide
- Documented FPGA integration architecture
- Remote Lab for distributed testing on Raspberry Pis

## Requirements Management

This project uses AIDA for requirements tracking. **Do NOT maintain a separate REQUIREMENTS.md file.**

Requirements database: `requirements.yaml`

### CLI Commands
```bash
aida list                              # List all requirements
aida list --status draft               # Filter by status
aida show <ID>                         # Show requirement details (e.g., FR-0042)
aida add --title "..." --description "..." --status draft  # Add new requirement
aida edit <ID> --status completed      # Update status
aida comment add <ID> "..."            # Add implementation note
```

### During Development
- When implementing a feature, update its requirement status
- Add comments to requirements with implementation decisions
- Create child requirements for sub-tasks discovered during implementation
- Link related requirements with: `aida rel add --from <FROM> --to <TO> --type <Parent|Verifies|References>`

### Session Workflow
If you work conversationally without explicit /aida-req calls, use `/aida-capture` at session end to review and capture any requirements that were discussed but not yet added to the database.

## Code Traceability

### Inline Trace Comments
When implementing requirements, add inline trace comments:

```rust
// trace:FR-0042 | ai:claude
fn implement_feature() {
    // Implementation
}
```

Format: `// trace:<SPEC-ID> | ai:<tool>[:<confidence>]`

### Commit Message Format
**Standard format:**
```
[AI:tool] type(scope): description (REQ-ID)
```

**Examples:**
```
[AI:claude] feat(auth): add login validation (FR-0042)
[AI:claude:med] fix(api): handle null response (BUG-0023)
chore(deps): update dependencies
docs: update README
```

**Rules:**
- `[AI:tool]` - Required when commit includes AI-assisted code
- `type` - Required: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert
- `(scope)` - Optional: component or area affected
- `(REQ-ID)` - Required for feat/fix commits, optional for chore/docs

**Confidence levels:**
- `[AI:claude]` - High confidence (implied, >80% AI-generated)
- `[AI:claude:med]` - Medium (40-80% AI with modifications)
- `[AI:claude:low]` - Low (<40% AI, mostly human)

**Configuration:**
Set `AIDA_COMMIT_STRICT=true` to reject non-conforming commits, or create `.aida/commit-config`.

## Claude Code Skills

This project uses AIDA requirements-driven development:

### /aida-req
Add new requirements with AI evaluation:
- Interactive requirement gathering
- Immediate database storage with draft status
- Background AI evaluation for quality feedback
- Follow-up actions: improve, split, link, accept

### /aida-implement
Implement requirements with traceability:
- Load and display requirement context
- Break down into child requirements as needed
- Update requirements during implementation
- Add inline traceability comments to code

### /aida-capture
Review session and capture missed requirements:
- Scan conversation for discussed features/bugs/ideas
- Identify implemented work not yet in requirements database
- Prompt to add missing requirements or update statuses
- Use at end of conversational sessions as a safety net
