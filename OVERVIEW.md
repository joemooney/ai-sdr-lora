# R4W - Rust for Waveforms

**A platform for developing, testing, and deploying SDR waveforms in Rust.**

```
    ██████╗ ██╗  ██╗██╗    ██╗
    ██╔══██╗██║  ██║██║    ██║
    ██████╔╝███████║██║ █╗ ██║     Rust 4 Waveforms
    ██╔══██╗╚════██║██║███╗██║     SDR Developer Studio
    ██║  ██║     ██║╚███╔███╔╝
    ╚═╝  ╚═╝     ╚═╝ ╚══╝╚══╝
```

## Vision

R4W is a **waveform development platform** that brings the power of Rust to Software Defined Radio. We provide:

1. **A Foundation of Reusable Libraries** - Core DSP primitives, modulation/demodulation frameworks, and signal processing building blocks
2. **Cross-Platform Deployment** - Write once, deploy everywhere: x86, ARM, WASM, FPGA
3. **Educational Tools** - Interactive visualization for learning SDR concepts
4. **Production-Ready Components** - From prototyping to field deployment

### Why Rust for SDR?

| Feature | Benefit |
|---------|---------|
| **Memory Safety** | No buffer overflows, data races, or undefined behavior in signal processing |
| **Zero-Cost Abstractions** | High-level APIs with C-level performance |
| **Fearless Concurrency** | Safe parallel processing for real-time DSP |
| **Cross-Compilation** | Single codebase for ARM, x86, embedded, and WASM |
| **Cargo Ecosystem** | Rich library ecosystem: FFT, linear algebra, async I/O |
| **SIMD Support** | Portable SIMD for vectorized operations |
| **WASM Target** | Run in browsers for education and demos |
| **No Runtime** | No garbage collection pauses - predictable real-time behavior |

### Proven Performance

R4W significantly outperforms GNU Radio on core DSP operations:

| Operation | R4W | GNU Radio | Speedup |
|-----------|-----|-----------|---------|
| **FFT 1024-pt** | 371 M samples/sec | 50 M samples/sec | **7.4x faster** |
| **FFT 4096-pt** | 330 M samples/sec | 12 M samples/sec | **27x faster** |
| **FFT 2048-pt** | 179 M samples/sec | ~25 M samples/sec | **7x faster** |

*Benchmarks: R4W with rustfft. GNU Radio baseline: i7-10700K, FFTW3+VOLK.*

### C/C++ Integration (Phase 4 Complete ✓)

For teams migrating from GNU Radio, R4W provides complete C/C++ interoperability:

- **C headers** - Auto-generated via cbindgen (`include/r4w.h`)
- **C++ RAII wrappers** - Modern C++ interface (`include/r4w.hpp`)
- **CMake integration** - `find_package(R4W)` support
- **Example programs** - C and C++ examples in `examples/c/`

```cpp
#include <r4w.hpp>
auto waveform = r4w::Waveform::bpsk(48000.0, 1200.0);
auto samples = waveform.modulate({1, 0, 1, 1, 0, 0, 1, 0});
```

## Platform Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           R4W Platform Stack                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                        Applications Layer                               │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐   │   │
│   │  │ r4w-explorer│  │     r4w     │  │  r4w-web    │  │ Your Waveform │   │   │
│   │  │   (GUI)     │  │    (CLI)    │  │   (WASM)    │  │  Application  │   │   │
│   │  └─────────────┘  └─────────────┘  └─────────────┘  └───────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                      ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                        Waveform Framework                               │   │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │   │
│   │  │   LoRa   │  │PSK/QAM   │  │   FSK    │  │ SINCGARS │  │HAVEQUICK │   │   │
│   │  │   CSS    │  │BPSK/QPSK │  │ 2/4-FSK  │  │   FHSS   │  │UHF AM/FH │   │   │
│   │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                      ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                          Core Libraries                                 │   │
│   │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────────┐    │   │
│   │  │   r4w-core    │  │   r4w-sim     │  │      r4w-gui (lib)        │    │   │
│   │  │  DSP Kernels  │  │ Channel Models│  │  Visualization Components │    │   │
│   │  │  FFT, Filters │  │ AWGN, Fading  │  │  egui Widgets             │    │   │
│   │  │  Coding, FEC  │  │ UDP Transport │  │  Plot/Spectrum/Waterfall  │    │   │
│   │  └───────────────┘  └───────────────┘  └───────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                      ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                       Hardware Abstraction                              │   │
│   │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────────┐    │   │
│   │  │    SdrDevice  │  │   UDP I/Q     │  │       r4w-fpga            │    │   │
│   │  │     Trait     │  │   Transport   │  │    Xilinx Zynq mmap       │    │   │
│   │  │  USRP/RTL-SDR │  │  GNU Radio    │  │    Lattice FTDI/SPI       │    │   │
│   │  └───────────────┘  └───────────────┘  └───────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Crate Overview

| Crate | Purpose | Key Features |
|-------|---------|--------------|
| **r4w-core** | DSP algorithms and waveform trait | FFT, chirp gen, PSK/FSK/QAM, FEC, Gray coding, benchmarking |
| **r4w-sim** | Channel simulation and transport | AWGN/Rayleigh/Rician, UDP I/Q, device abstraction |
| **r4w-fpga** | FPGA hardware acceleration | Xilinx Zynq, Lattice iCE40/ECP5, simulated backend |
| **r4w-sandbox** | Waveform isolation | Secure memory, namespaces, seccomp, container/VM support |
| **r4w-gui** | Visualization library + app | egui widgets, spectrum plots, constellation diagrams |
| **r4w-cli** | Command-line tool (`r4w`) | TX/RX, benchmarking, remote agents, waveform simulation |
| **r4w-web** | WebAssembly entry point | Browser-based demo and education |

## Plugin System

R4W supports dynamic loading of waveform plugins at runtime. Plugins are shared libraries (`.so` on Linux, `.dll` on Windows, `.dylib` on macOS) that implement the R4W plugin ABI.

### Creating a Plugin

```rust
use r4w_core::plugin::{PluginInfo, PLUGIN_API_VERSION};
use std::ffi::c_char;

#[no_mangle]
pub extern "C" fn r4w_plugin_api_version() -> u32 {
    PLUGIN_API_VERSION
}

#[no_mangle]
pub extern "C" fn r4w_plugin_info() -> *const PluginInfo {
    static INFO: PluginInfo = PluginInfo {
        name: b"my_waveform\0".as_ptr() as *const c_char,
        version: b"1.0.0\0".as_ptr() as *const c_char,
        description: b"Custom waveform plugin\0".as_ptr() as *const c_char,
        author: b"Author Name\0".as_ptr() as *const c_char,
        waveform_count: 1,
    };
    &INFO
}
```

### Loading Plugins

```rust
use r4w_core::plugin::PluginManager;

let mut manager = PluginManager::new();
manager.add_search_path("/usr/lib/r4w/plugins");
manager.discover_plugins()?;

for wf in manager.list_waveforms() {
    println!("{}: {}", wf.name, wf.description);
}
```

### Building Plugins

```bash
# Build the example plugin
cargo build --release -p r4w-example-plugin

# Output: target/release/libr4w_example_plugin.so
```

Enable the `plugins` feature in r4w-core for real dynamic loading:

```toml
r4w-core = { version = "0.1", features = ["plugins"] }
```

## Quick Start

```bash
# Clone and build
git clone https://github.com/joemooney/r4w
cd r4w
cargo build --release

# Run the GUI explorer
cargo run --bin r4w-explorer

# List available waveforms
cargo run --bin r4w -- waveform --list

# Simulate LoRa transmission
cargo run --bin r4w -- simulate --message "Hello R4W!" --snr 10.0

# Run in browser (WASM)
cd crates/r4w-web && trunk serve
```

## Available Waveforms

R4W includes 38+ waveform implementations:

```
Simple:       CW, OOK, PPM, ADS-B
Analog:       AM-Broadcast, FM-Broadcast, NBFM
Amplitude:    ASK, 4-ASK
Frequency:    BFSK, 4-FSK
Phase:        BPSK, QPSK, 8-PSK
QAM:          16-QAM, 64-QAM, 256-QAM
Multi-carrier: OFDM
Spread:       DSSS, DSSS-QPSK, FHSS, LoRa (SF7-SF12)
IoT/Radar:    Zigbee (802.15.4), UWB, FMCW
HF/Military:  STANAG 4285, ALE, 3G-ALE, MIL-STD-188-110
              SINCGARS*, HAVEQUICK*, Link-16*, P25*
PMR:          TETRA, DMR (Tier II/III)
```

**\* Framework implementations** - These waveforms use a trait-based architecture where classified/proprietary components (frequency hopping algorithms, TRANSEC, voice codecs) are represented by simulator stubs. The unclassified signal processing, modulation, and framing are fully implemented. See [docs/PORTING_GUIDE_MILITARY.md](./docs/PORTING_GUIDE_MILITARY.md) for details.

## Test Status

All tests pass across the workspace:

| Crate | Tests | Status |
|-------|-------|--------|
| r4w-core | 421 | ✅ Pass |
| r4w-sim | 50 | ✅ Pass |
| r4w-gui | 22 | ✅ Pass |
| r4w-sandbox | 14 | ✅ Pass |
| r4w-fpga | 7 | ✅ Pass |
| r4w-ffi | 7 | ✅ Pass |
| r4w-example-plugin | 6 | ✅ Pass |
| **Total** | **527** | **✅ All Pass** |

Run tests: `cargo test --workspace`

## Code Metrics

Measured with `tokei`:

| Language | Code Lines | Files | Purpose |
|----------|------------|-------|---------|
| **Rust** | 66,572 | 217 | Core implementation (79%) |
| **Coq** | 6,324 | 27 | Formal verification proofs |
| **YAML** | 3,445 | 11 | Requirements, configs, waveform specs |
| **HTML/CSS/JS** | 3,378 | 4 | Web interface |
| **C/C++** | 2,037 | 5 | FFI bindings, hardware interfaces |
| **Makefile** | 829 | 4 | Build automation |
| **TOML** | 494 | 17 | Cargo configs |
| **CMake** | 402 | 12 | C/C++ build system |
| **TCL** | 198 | 4 | FPGA tooling scripts |
| **Total** | **84,467** | **359** | |

**Highlights:**
- Rust dominates (79%) as expected for an SDR platform
- 6,324 lines of Coq proofs demonstrates commitment to formal verification
- Extensive documentation: 9,690 comment lines in Markdown code blocks
- Test density: 527 tests covering 66k lines = ~1 test per 126 lines of Rust

## Recent Updates

### December 2024

- **LoRa Mesh Integration Complete** - Full integration of LoRa waveform with mesh networking:
  - `LoRaMeshPhy`: Adapts LoRa waveform to implement `MeshPhy` trait
  - `LoRaMesh`: Complete mesh node combining PHY, MAC, and routing layers
  - Channel Activity Detection (CAD) via signal power estimation
  - Sample-based processing for SDR integration
  - 42 passing mesh tests, 11 of 20 MESH requirements completed
- **Mesh Networking Module Implemented** - Full mesh networking stack in `r4w-core/src/mesh/`:
  - `MeshNetwork` and `MeshPhy` traits for protocol abstraction
  - CSMA/CA MAC layer with contention window scaling
  - FloodRouter (SNR-based delays) and NextHopRouter with route caching
  - NeighborTable with link quality metrics (RSSI/SNR/PDR)
  - MeshtasticNode implementation with regional frequency support
- **License Simplified** - Changed from dual MIT/Apache-2.0 to MIT only for maximum permissiveness
- **License Files Added** - `LICENSE` and `THIRD_PARTY.md` with proper attributions
- **Physical Layer Complete** - All 6 phases implemented:
  - Phase 1: Timing model (multi-clock architecture)
  - Phase 2: HAL enhancement (SigMF file I/O, driver stubs)
  - Phase 3: Observability (structured logging, metrics, capture)
  - Phase 4: Real-time primitives (lock-free buffers, thread priorities)
  - Phase 5: Plugin system (dynamic waveform loading)
  - Phase 6: Hardware drivers (RTL-SDR, SoapySDR, UHD stubs)
- **Lattice FPGA Support** - Open-source iCE40/ECP5 with Yosys/nextpnr toolchain
- **Crypto Boundary Design** - CSI architecture for RED/BLACK separation
- **Waveform Specification System** - YAML schema for AI-assisted waveform generation

---

# Waveform Wizard

The Waveform Wizard is an interactive GUI tool for designing new waveforms and generating AI implementation prompts.

## Features

- **Interactive Configuration**: Step-by-step wizard with presets for common waveform types
- **YAML Specification**: Generates complete waveform specification following the R4W schema
- **AI Implementation Prompt**: Exports self-contained prompt for Claude to implement the waveform
- **Export Options**: Choose between spec-only (.yaml) or full implementation prompt (.md)

## Workflow

1. **Open the Wizard**: Run `cargo run --bin r4w-explorer` and navigate to "Waveform Wizard"
2. **Select Preset or Configure**: Choose from presets (BPSK, QPSK, LoRa-like, etc.) or configure manually
3. **Review Specification**: Preview the generated YAML specification
4. **Export**: Click "Export" and enable "Include R4W Implementation Prompt"
5. **Implement with AI**: Paste the exported content into a fresh Claude Code session
6. **Test**: Claude generates the waveform module, factory registration, and tests

## Export Modes

| Mode | Extension | Contents | Use Case |
|------|-----------|----------|----------|
| Spec Only | `.yaml` | Just the waveform specification | Storage, reference, manual implementation |
| With Prompt | `.md` | Full R4W context + specification | AI-assisted implementation with Claude |

## Registration Requirements

When implementing a new waveform, two registrations are needed:

1. **WaveformFactory** (`mod.rs`): Makes waveform available to CLI and core
2. **WaveformGroup** (`app.rs`): Makes waveform appear in GUI dropdown

The implementation prompt documents both registration points.

## Files

| File | Purpose |
|------|---------|
| `waveform-spec/schema.yaml` | Complete specification schema |
| `waveform-spec/IMPLEMENTATION_PROMPT.md` | AI implementation context template |
| `waveform-spec/examples/` | Example waveform specifications |

---

# Waveform Developer's Guide

This section covers how to implement new waveforms in R4W.

## The Waveform Trait

Every waveform implements the `Waveform` trait from `r4w-core`:

```rust
pub trait Waveform: Send + Sync {
    /// Get information about this waveform
    fn info(&self) -> WaveformInfo;

    /// Modulate bits to I/Q samples
    fn modulate(&self, bits: &[bool]) -> Vec<IQSample>;

    /// Demodulate I/Q samples to bits
    fn demodulate(&self, samples: &[IQSample]) -> Vec<bool>;

    /// Get constellation points for visualization
    fn constellation_points(&self) -> Vec<IQSample>;

    /// Get educational pipeline stages
    fn get_modulation_stages(&self, bits: &[bool]) -> Vec<ModulationStage>;
    fn get_demodulation_steps(&self, samples: &[IQSample]) -> Vec<DemodulationStep>;
}

pub struct WaveformInfo {
    pub name: &'static str,           // Short name (e.g., "QPSK")
    pub full_name: &'static str,      // Full name (e.g., "Quadrature Phase Shift Keying")
    pub bits_per_symbol: u8,          // Spectral efficiency
    pub sample_rate: f64,             // Operating sample rate
    pub symbol_rate: f64,             // Symbol rate in Hz
    pub carries_data: bool,           // Does this waveform carry data?
}
```

## Creating a New Waveform

### Step 1: Create the Waveform Struct

```rust
// crates/r4w-core/src/waveform/my_waveform.rs

use crate::types::IQSample;
use crate::waveform::{Waveform, WaveformInfo, ModulationStage, DemodulationStep};

pub struct MyWaveform {
    sample_rate: f64,
    symbol_rate: f64,
    // ... your parameters
}

impl MyWaveform {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            symbol_rate: sample_rate / 10.0,  // 10 samples per symbol
        }
    }
}
```

### Step 2: Implement the Trait

```rust
impl Waveform for MyWaveform {
    fn info(&self) -> WaveformInfo {
        WaveformInfo {
            name: "MyWave",
            full_name: "My Custom Waveform",
            bits_per_symbol: 2,
            sample_rate: self.sample_rate,
            symbol_rate: self.symbol_rate,
            carries_data: true,
        }
    }

    fn modulate(&self, bits: &[bool]) -> Vec<IQSample> {
        let samples_per_symbol = (self.sample_rate / self.symbol_rate) as usize;
        let mut samples = Vec::new();

        // Process bits in groups based on bits_per_symbol
        for chunk in bits.chunks(2) {
            let symbol = match (chunk.get(0), chunk.get(1)) {
                (Some(&false), Some(&false)) => IQSample::new(-1.0, -1.0),
                (Some(&false), Some(&true))  => IQSample::new(-1.0,  1.0),
                (Some(&true),  Some(&false)) => IQSample::new( 1.0, -1.0),
                (Some(&true),  Some(&true))  => IQSample::new( 1.0,  1.0),
                _ => IQSample::new(1.0, 1.0),
            };

            // Repeat symbol for samples_per_symbol
            samples.extend(std::iter::repeat(symbol).take(samples_per_symbol));
        }

        samples
    }

    fn demodulate(&self, samples: &[IQSample]) -> Vec<bool> {
        let samples_per_symbol = (self.sample_rate / self.symbol_rate) as usize;
        let mut bits = Vec::new();

        for chunk in samples.chunks(samples_per_symbol) {
            // Average samples in symbol period
            let avg = chunk.iter().fold(IQSample::new(0.0, 0.0), |acc, s| {
                IQSample::new(acc.re + s.re, acc.im + s.im)
            });
            let avg = IQSample::new(
                avg.re / chunk.len() as f32,
                avg.im / chunk.len() as f32,
            );

            // Decision regions
            bits.push(avg.re > 0.0);  // Bit 0
            bits.push(avg.im > 0.0);  // Bit 1
        }

        bits
    }

    fn constellation_points(&self) -> Vec<IQSample> {
        vec![
            IQSample::new(-1.0, -1.0),  // 00
            IQSample::new(-1.0,  1.0),  // 01
            IQSample::new( 1.0, -1.0),  // 10
            IQSample::new( 1.0,  1.0),  // 11
        ]
    }

    fn get_modulation_stages(&self, bits: &[bool]) -> Vec<ModulationStage> {
        // Return educational visualization stages
        vec![
            ModulationStage {
                name: "Bit Grouping".to_string(),
                description: "Group bits into dibits".to_string(),
                samples: vec![],  // Intermediate data
            },
            ModulationStage {
                name: "Symbol Mapping".to_string(),
                description: "Map dibits to constellation points".to_string(),
                samples: self.modulate(bits),
            },
        ]
    }

    fn get_demodulation_steps(&self, samples: &[IQSample]) -> Vec<DemodulationStep> {
        vec![
            DemodulationStep {
                name: "Symbol Detection".to_string(),
                description: "Find nearest constellation point".to_string(),
                data: samples.to_vec(),
                bits: self.demodulate(samples),
            },
        ]
    }
}
```

### Step 3: Register with WaveformFactory

```rust
// In crates/r4w-core/src/waveform/factory.rs

impl WaveformFactory {
    pub fn create(name: &str, sample_rate: f64) -> Option<Box<dyn Waveform>> {
        match name.to_uppercase().as_str() {
            // ... existing waveforms ...
            "MYWAVE" | "MY-WAVE" => Some(Box::new(MyWaveform::new(sample_rate))),
            _ => None,
        }
    }

    pub fn list() -> &'static [&'static str] {
        &[
            // ... existing ...
            "MyWave",
        ]
    }
}
```

### Step 4: Add Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let waveform = MyWaveform::new(48000.0);
        let original_bits = vec![true, false, true, true, false, false];

        let samples = waveform.modulate(&original_bits);
        let recovered_bits = waveform.demodulate(&samples);

        assert_eq!(original_bits, recovered_bits);
    }

    #[test]
    fn test_with_noise() {
        use r4w_sim::channel::AwgnChannel;

        let waveform = MyWaveform::new(48000.0);
        let bits = vec![true; 100];

        let samples = waveform.modulate(&bits);
        let noisy = AwgnChannel::new(10.0).apply(&samples);  // 10 dB SNR
        let recovered = waveform.demodulate(&noisy);

        let errors = bits.iter().zip(&recovered)
            .filter(|(a, b)| a != b)
            .count();
        let ber = errors as f64 / bits.len() as f64;

        assert!(ber < 0.1, "BER too high: {:.2}%", ber * 100.0);
    }
}
```

---

# Porting Guide

This section covers how to port existing waveform implementations to R4W.

## From C/C++ SDR Code

### Step 1: Identify the Core Functions

Look for these patterns in the original code:
- `modulate()`, `encode()`, `tx()` → maps to `Waveform::modulate()`
- `demodulate()`, `decode()`, `rx()` → maps to `Waveform::demodulate()`
- Complex number types → use `num_complex::Complex<f32>` or `IQSample`

### Step 2: Replace C Idioms with Rust

| C/C++ Pattern | Rust Equivalent |
|---------------|-----------------|
| `float* samples` | `&[IQSample]` or `Vec<IQSample>` |
| `malloc`/`free` | Stack allocation or `Vec` |
| `fftw_plan` | `rustfft::FftPlanner` |
| `memcpy` | `.clone()` or `.copy_from_slice()` |
| Raw pointer arithmetic | Iterator methods |
| `#define SYMBOL_SIZE 128` | `const SYMBOL_SIZE: usize = 128;` |

### Example: Porting a Simple FSK Modulator

**Original C code:**
```c
void fsk_modulate(float* samples, int* bits, int n_bits, float sample_rate) {
    float f0 = 1200.0, f1 = 2200.0;
    int samples_per_bit = (int)(sample_rate / 300.0);  // 300 baud

    for (int i = 0; i < n_bits; i++) {
        float freq = bits[i] ? f1 : f0;
        for (int j = 0; j < samples_per_bit; j++) {
            int idx = i * samples_per_bit + j;
            float t = (float)idx / sample_rate;
            samples[idx * 2] = cosf(2 * M_PI * freq * t);      // I
            samples[idx * 2 + 1] = sinf(2 * M_PI * freq * t);  // Q
        }
    }
}
```

**Ported Rust code:**
```rust
use std::f32::consts::PI;
use num_complex::Complex;

pub struct BfskWaveform {
    sample_rate: f64,
    baud_rate: f64,
    f0: f32,  // Mark frequency
    f1: f32,  // Space frequency
}

impl BfskWaveform {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            baud_rate: 300.0,
            f0: 1200.0,
            f1: 2200.0,
        }
    }
}

impl Waveform for BfskWaveform {
    fn modulate(&self, bits: &[bool]) -> Vec<IQSample> {
        let samples_per_bit = (self.sample_rate / self.baud_rate) as usize;

        bits.iter()
            .enumerate()
            .flat_map(|(bit_idx, &bit)| {
                let freq = if bit { self.f1 } else { self.f0 };

                (0..samples_per_bit).map(move |j| {
                    let sample_idx = bit_idx * samples_per_bit + j;
                    let t = sample_idx as f32 / self.sample_rate as f32;
                    let phase = 2.0 * PI * freq * t;
                    IQSample::new(phase.cos(), phase.sin())
                })
            })
            .collect()
    }

    fn demodulate(&self, samples: &[IQSample]) -> Vec<bool> {
        // Use Goertzel algorithm for each frequency
        let samples_per_bit = (self.sample_rate / self.baud_rate) as usize;

        samples.chunks(samples_per_bit)
            .map(|chunk| {
                let power_f0 = goertzel_power(chunk, self.f0, self.sample_rate as f32);
                let power_f1 = goertzel_power(chunk, self.f1, self.sample_rate as f32);
                power_f1 > power_f0
            })
            .collect()
    }

    // ... rest of trait implementation
}
```

## From GNU Radio

GNU Radio blocks can be ported by:

1. **Identify the block type**: Source, Sink, or Signal Processing
2. **Map GR types to Rust types**:
   - `gr_complex` → `Complex<f32>` / `IQSample`
   - `pmt::pmt_t` → Rust enums or structs
   - `work()` function → iterator-based processing

### Example: Porting a GNU Radio Block

**GNU Radio Python block:**
```python
class my_block(gr.sync_block):
    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]

        for i in range(len(in0)):
            out[i] = in0[i] * self.gain

        return len(out)
```

**Rust equivalent:**
```rust
pub struct GainBlock {
    gain: f32,
}

impl GainBlock {
    pub fn new(gain: f32) -> Self {
        Self { gain }
    }

    pub fn process(&self, input: &[IQSample]) -> Vec<IQSample> {
        input.iter()
            .map(|s| IQSample::new(s.re * self.gain, s.im * self.gain))
            .collect()
    }

    // For streaming/real-time, use an iterator adapter:
    pub fn process_stream<'a>(&'a self, input: impl Iterator<Item = IQSample> + 'a)
        -> impl Iterator<Item = IQSample> + 'a
    {
        input.map(move |s| IQSample::new(s.re * self.gain, s.im * self.gain))
    }
}
```

---

# FPGA Integration Architecture

R4W is designed for FPGA acceleration, with **Xilinx Zynq as the primary target platform**.

## Priority Roadmap

| Priority | Platform | Use Case | Status |
|----------|----------|----------|--------|
| **1st** | **Xilinx Zynq** | Production SDR, high-performance | **Active Development** |
| **2nd** | **Lattice iCE40/ECP5** | Low-cost prototyping, education | **Implemented** |
| 3rd | Intel/Altera | Enterprise, high-end | Future |
| 4th | LiteX SoC | Open-source FPGA SoCs | Exploratory |

---

## Xilinx Zynq Integration (Primary Target)

The Zynq combines ARM Cortex-A cores with FPGA fabric, making it ideal for R4W:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Xilinx Zynq SoC                                     │
├─────────────────────────────────┬───────────────────────────────────────────┤
│      Processing System (PS)     │        Programmable Logic (PL)            │
│  ┌───────────────────────────┐  │  ┌─────────────────────────────────────┐  │
│  │   ARM Cortex-A9/A53       │  │  │         DSP Accelerators            │  │
│  │   Running Linux + R4W     │  │  │  ┌──────────┐  ┌──────────────────┐ │  │
│  │                           │  │  │  │ FFT Core │  │ Chirp Correlator │ │  │
│  │  ┌─────────────────────┐  │  │  │  │ (Radix-4)│  │ (LoRa demod)     │ │  │
│  │  │    r4w-core         │  │  │  │  └──────────┘  └──────────────────┘ │  │
│  │  │    r4w-cli          │◄─┼──┼──┤  ┌──────────┐  ┌──────────────────┐ │  │
│  │  │    r4w-explorer     │  │  │  │  │FIR Filter│  │ Symbol Detector  │ │  │
│  │  └─────────────────────┘  │  │  │  │(up to    │  │ (matched filter) │ │  │
│  │           │               │  │  │  │ 256 taps)│  │                  │ │  │
│  │           │ mmap()        │  │  │  └──────────┘  └──────────────────┘ │  │
│  │           ▼               │  │  │  ┌──────────┐  ┌──────────────────┐ │  │
│  │  ┌─────────────────────┐  │  │  │  │ NCO/DDS  │  │  CORDIC          │ │  │
│  │  │    /dev/mem         │  │  │  │  │(carrier) │  │  (sin/cos)       │ │  │
│  │  │    /dev/uio*        │◄─┼──┼──┤  └──────────┘  └──────────────────┘ │  │
│  │  └─────────────────────┘  │  │  └─────────────────────────────────────┘  │
│  │           │               │  │           ▲           │                   │
│  └───────────┼───────────────┘  │           │           ▼                   │
│              │                  │  ┌─────────────────────────────────────┐  │
│              │   AXI-Lite       │  │        AXI DMA Engine               │  │
│              └──────────────────┼──┤  (scatter-gather, up to 8 channels) │  │
│                                 │  └─────────────────────────────────────┘  │
│                                 │           │           │                   │
│                                 │           ▼           ▼                   │
│                                 │  ┌─────────────┐ ┌─────────────────────┐  │
│                                 │  │  ADC/DAC    │ │  RF Frontend        │  │
│                                 │  │  Interface  │ │  (AD9361/AD9364)    │  │
│                                 │  └─────────────┘ └─────────────────────┘  │
└─────────────────────────────────┴───────────────────────────────────────────┘
```

### Zynq Target Boards

| Board | Zynq Device | Use Case | Approx. Cost |
|-------|-------------|----------|--------------|
| **PYNQ-Z2** | Zynq-7020 | Learning, prototyping | $120 |
| **ZedBoard** | Zynq-7020 | Development | $500 |
| **ADALM-PLUTO** | Zynq-7010 + AD9363 | Complete SDR | $150 |
| **ZCU102** | Zynq UltraScale+ | High-performance | $3,000 |
| **Red Pitaya** | Zynq-7010 | Test equipment + SDR | $300 |

### Zynq Communication Methods

```rust
/// r4w-fpga crate (planned)
pub mod zynq {
    use std::fs::OpenOptions;
    use std::os::unix::io::AsRawFd;

    /// Memory-mapped register access via /dev/mem
    pub struct ZynqMmap {
        base_addr: usize,
        size: usize,
        ptr: *mut u8,
    }

    impl ZynqMmap {
        /// Map FPGA registers into userspace
        pub fn new(base_addr: usize, size: usize) -> Result<Self, std::io::Error> {
            let fd = OpenOptions::new()
                .read(true)
                .write(true)
                .open("/dev/mem")?;

            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd.as_raw_fd(),
                    base_addr as libc::off_t,
                )
            };

            if ptr == libc::MAP_FAILED {
                return Err(std::io::Error::last_os_error());
            }

            Ok(Self { base_addr, size, ptr: ptr as *mut u8 })
        }

        /// Write to FPGA register
        pub fn write_reg(&self, offset: usize, value: u32) {
            unsafe {
                let reg = self.ptr.add(offset) as *mut u32;
                std::ptr::write_volatile(reg, value);
            }
        }

        /// Read from FPGA register
        pub fn read_reg(&self, offset: usize) -> u32 {
            unsafe {
                let reg = self.ptr.add(offset) as *const u32;
                std::ptr::read_volatile(reg)
            }
        }
    }

    /// UIO-based interrupt handling
    pub struct ZynqUio {
        fd: std::fs::File,
        irq_count: u32,
    }

    impl ZynqUio {
        pub fn new(uio_device: &str) -> Result<Self, std::io::Error> {
            let fd = OpenOptions::new()
                .read(true)
                .write(true)
                .open(uio_device)?;
            Ok(Self { fd, irq_count: 0 })
        }

        /// Wait for interrupt from FPGA
        pub fn wait_irq(&mut self) -> Result<u32, std::io::Error> {
            use std::io::Read;
            let mut buf = [0u8; 4];
            self.fd.read_exact(&mut buf)?;
            self.irq_count = u32::from_ne_bytes(buf);
            Ok(self.irq_count)
        }
    }
}
```

### Zynq Vivado IP Cores (Implemented)

Located in `vivado/ip/`:

| IP Core | Function | Interface | Resource Est. | Status |
|---------|----------|-----------|---------------|--------|
| `r4w_fft` | 1024-pt Radix-4 FFT | AXI-Stream + AXI-Lite | ~15k LUTs | ✅ Done |
| `r4w_fir` | 256-tap FIR filter | AXI-Stream + AXI-Lite | ~8k LUTs | ✅ Done |
| `r4w_chirp_gen` | LoRa chirp generator | AXI-Lite + AXI-Stream | ~2k LUTs | ✅ Done |
| `r4w_chirp_corr` | Chirp correlator | AXI-Stream + AXI-Lite | ~12k LUTs | ✅ Done |
| `r4w_nco` | Numerically Controlled Oscillator | AXI-Lite | ~1.5k LUTs | ✅ Done |
| `r4w_dma` | DMA controller for PS↔PL | AXI-Stream + AXI-Lite | ~3k LUTs | ✅ Done |

**Build with Vivado:**
```bash
cd vivado
vivado -mode batch -source scripts/build_project.tcl
vivado -mode batch -source scripts/build_bitstream.tcl
```

**Deploy to PYNQ-Z2:**
```bash
scp output/r4w_design.bit xilinx@pynq:/home/xilinx/
# On PYNQ:
sudo fpgautil -b r4w_design.bit -o r4w-overlay.dtbo
```

---

## Lattice FPGA Integration (Secondary Target)

Lattice FPGAs are ideal for **low-cost, low-power** applications and **education**.

### Lattice Product Families

| Family | Logic Cells | DSP Blocks | Use Case | Open Tools? |
|--------|-------------|------------|----------|-------------|
| **iCE40 UP5K** | 5,280 | 8 DSPs | IoT, battery-powered | **Yes** (Yosys+nextpnr) |
| **iCE40 HX8K** | 7,680 | 0 | Prototyping | **Yes** |
| **ECP5** | 12k-85k | 12-156 DSPs | Mid-range SDR | **Yes** (Yosys+nextpnr) |
| **CrossLink-NX** | 17k-40k | 28-56 DSPs | High-speed I/O | Partial |
| **Certus-NX** | 17k-39k | 28-56 DSPs | General purpose | No |

### Why Lattice for R4W?

1. **Open-Source Toolchain**: iCE40 and ECP5 work with fully open-source tools:
   - **Yosys**: Synthesis
   - **nextpnr**: Place and route
   - **Project IceStorm/Trellis**: Bitstream generation
   - No expensive Xilinx/Intel licenses needed!

2. **Low Cost**: iCE40 UP5K boards start at $12 (Upduino), ECP5 at $45 (OrangeCrab)

3. **USB Programming**: Simple `iceprog` or `openFPGALoader` - no JTAG dongles

4. **Rust Integration**: The open toolchain integrates well with Rust build systems

### Lattice Target Boards

| Board | FPGA | Features | Approx. Cost |
|-------|------|----------|--------------|
| **Upduino v3** | iCE40 UP5K | 5k LUTs, 1Mbit SPRAM, RGB LED | $12 |
| **iCEBreaker** | iCE40 UP5K | PMOD, USB-C | $70 |
| **OrangeCrab** | ECP5-25F | 24k LUTs, DDR3, USB-C | $45 |
| **ULX3S** | ECP5-12F/85F | WiFi, HDMI, buttons | $100-200 |
| **Colorlight i5** | ECP5-25F | Cheap (LED controller) | $15 |

### Lattice Communication (SPI/USB)

```rust
/// r4w-fpga crate (planned) - Lattice support
pub mod lattice {
    use std::io::{Read, Write};

    /// SPI-based communication with iCE40/ECP5
    pub struct LatticeSpi {
        device: String,
        speed_hz: u32,
    }

    impl LatticeSpi {
        pub fn new(spi_device: &str, speed_hz: u32) -> Result<Self, std::io::Error> {
            // Uses spidev for Linux SPI access
            Ok(Self {
                device: spi_device.to_string(),
                speed_hz,
            })
        }

        /// Send samples to FPGA for processing
        pub fn send_samples(&mut self, samples: &[u8]) -> Result<(), std::io::Error> {
            // SPI transaction to FPGA
            todo!()
        }

        /// Receive processed samples from FPGA
        pub fn recv_samples(&mut self, buffer: &mut [u8]) -> Result<usize, std::io::Error> {
            todo!()
        }
    }

    /// USB-based communication via FTDI
    pub struct LatticeFtdi {
        // Uses libftdi or ftd2xx
    }
}
```

### Open-Source Toolchain Integration

```makefile
# Example: Building FPGA bitstream with open tools
YOSYS := yosys
NEXTPNR := nextpnr-ice40
ICEPACK := icepack
ICEPROG := iceprog

# Synthesize Verilog to JSON
%.json: %.v
	$(YOSYS) -p "synth_ice40 -top top -json $@" $<

# Place and route
%.asc: %.json %.pcf
	$(NEXTPNR) --up5k --package sg48 --json $< --pcf $(word 2,$^) --asc $@

# Generate bitstream
%.bin: %.asc
	$(ICEPACK) $< $@

# Program FPGA
program: design.bin
	$(ICEPROG) $<
```

### Lattice IP Cores (Implemented)

Located in `lattice/ip/`:

| IP Core | FPGA | Function | Est. LUTs | Status |
|---------|------|----------|-----------|--------|
| `r4w_spi_slave` | iCE40/ECP5 | SPI slave interface | ~200 | Done |
| `r4w_nco` | iCE40/ECP5 | NCO/DDS (LUT or CORDIC) | ~200 | Done |
| `r4w_chirp_gen` | iCE40/ECP5 | LoRa chirp generator | ~500 | Done |

**Build with open-source toolchain:**
```bash
cd lattice/scripts
make ice40    # Build for iCE40-HX8K
make ecp5     # Build for ECP5-25K
make sim      # Run simulation
make lint     # Verilator lint check
```

**Future Lattice IP:**
| IP Core | FPGA | Function | Est. LUTs |
|---------|------|----------|-----------|
| `r4w_fft_256` | iCE40/ECP5 | 256-pt FFT | ~3k |
| `r4w_fir_32` | iCE40 | 32-tap FIR | ~500 |
| `r4w_fir_128` | ECP5 | 128-tap FIR | ~2k |

---

## FPGA Bridge Trait

All FPGA platforms implement a common trait:

```rust
/// Trait for FPGA-accelerated operations
pub trait FpgaAccelerator: Send + Sync {
    /// Get platform info
    fn info(&self) -> FpgaInfo;

    /// Check if FPGA is available and configured
    fn is_available(&self) -> bool;

    /// Get FPGA capabilities
    fn capabilities(&self) -> FpgaCapabilities;

    /// Offload FFT computation
    fn fft(&self, samples: &[IQSample], inverse: bool) -> Result<Vec<IQSample>, FpgaError>;

    /// Offload FIR filtering
    fn fir_filter(&self, samples: &[IQSample], taps: &[f32]) -> Result<Vec<IQSample>, FpgaError>;

    /// Offload complete modulation
    fn modulate(&self, waveform_id: u32, bits: &[bool]) -> Result<Vec<IQSample>, FpgaError>;

    /// Offload complete demodulation
    fn demodulate(&self, waveform_id: u32, samples: &[IQSample]) -> Result<Vec<bool>, FpgaError>;

    /// Stream processing (for real-time)
    fn start_stream(&mut self, config: StreamConfig) -> Result<StreamHandle, FpgaError>;
    fn stop_stream(&mut self, handle: StreamHandle) -> Result<(), FpgaError>;
}

pub struct FpgaInfo {
    pub platform: FpgaPlatform,
    pub device: String,
    pub bitstream_version: Option<String>,
}

pub enum FpgaPlatform {
    XilinxZynq { part: String },
    LatticeIce40 { variant: String },
    LatticeEcp5 { variant: String },
    IntelCyclone { variant: String },
    Other(String),
}

pub struct FpgaCapabilities {
    pub max_fft_size: usize,
    pub max_fir_taps: usize,
    pub supported_waveforms: Vec<String>,
    pub dma_buffer_size: usize,
    pub clock_frequency_hz: u64,
    pub dsp_blocks: usize,
    pub logic_cells: usize,
}
```

## High-Level Synthesis (HLS) Integration

R4W DSP kernels are designed to be HLS-friendly for Vitis HLS (Xilinx):

```rust
// DSP kernel designed for potential HLS transpilation
#[inline(never)]  // Preserve function boundary for HLS
pub fn chirp_correlate_kernel(
    samples: &[IQSample; 1024],
    chirp: &[IQSample; 1024],
) -> [IQSample; 1024] {
    let mut result = [IQSample::new(0.0, 0.0); 1024];

    // HLS pragma: pipeline this loop
    for i in 0..1024 {
        result[i] = IQSample::new(
            samples[i].re * chirp[i].re + samples[i].im * chirp[i].im,
            samples[i].im * chirp[i].re - samples[i].re * chirp[i].im,
        );
    }

    result
}
```

For Lattice (no HLS), we provide hand-written Verilog:
```verilog
// r4w_correlator.v - Hand-optimized for iCE40/ECP5
module r4w_correlator #(
    parameter N = 256
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,
    input  wire [31:0] sample_re,  // Q15.16 fixed-point
    input  wire [31:0] sample_im,
    input  wire [31:0] chirp_re,
    input  wire [31:0] chirp_im,
    output reg         valid_out,
    output reg  [31:0] result_re,
    output reg  [31:0] result_im
);
    // Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    always @(posedge clk) begin
        if (rst) begin
            valid_out <= 0;
        end else if (valid_in) begin
            result_re <= (sample_re * chirp_re - sample_im * chirp_im) >>> 16;
            result_im <= (sample_re * chirp_im + sample_im * chirp_re) >>> 16;
            valid_out <= 1;
        end else begin
            valid_out <= 0;
        end
    end
endmodule
```

---

# Waveform Performance Benchmarks

Real-world benchmark data from distributed TX/RX testing (Pi 500 → Pi 3, 125 kHz):

## Clean Channel Performance

| Metric | BPSK | QPSK | LoRa (SF7) |
|--------|------|------|------------|
| Throughput | 85,771 Sps | 77,224 Sps | 83,129 Sps |
| Bits/symbol | 1 | 2 | 7 |
| Demod rate | 168 bps | 228 bps | **322 bps** |
| Avg latency | **29 μs** | 38 μs | 395 μs |
| P99 latency | **56 μs** | 61 μs | 565 μs |

## Performance Under Noise (Bits Decoded, 8-second test)

| SNR | BPSK | QPSK | LoRa (SF7) |
|-----|------|------|------------|
| 20 dB | 1,296 | 1,861 | **2,616** |
| 10 dB | 1,266 | 1,895 | **2,620** |
| 5 dB | 1,336 | 1,904 | **2,580** |
| 0 dB | 1,221 | 1,892 | **2,428** |

## LoRa Spreading Factor Comparison

| Parameter | SF7 | SF12 | Ratio |
|-----------|-----|------|-------|
| Symbol time @ 125kHz | 1.02 ms | 32.77 ms | 32x slower |
| Data rate | ~5.5 kbps | ~293 bps | 19x slower |
| Processing gain | 7 dB | 21 dB | +14 dB |
| Typical range | 2-5 km | 10-15 km | ~3x farther |

---

# Remote Lab (Distributed Testing)

Deploy R4W agents to Raspberry Pis for distributed TX/RX testing:

```
┌─────────────────┐         ┌─────────────────┐
│   Development   │         │   TX Agent      │
│   Machine       │  TCP    │   (Raspberry Pi)│
│                 │ ────────│                 │
│   r4w-explorer  │  6000   │   r4w agent     │
│   (GUI)         │         │                 │
│                 │         └────────┬────────┘
│                 │                  │ UDP I/Q
│                 │         ┌────────▼────────┐
│                 │  TCP    │   RX Agent      │
│                 │ ────────│   (Raspberry Pi)│
│                 │  6000   │   r4w agent     │
└─────────────────┘         └─────────────────┘
```

## Deployment

```bash
# Build for ARM
make build-cli-arm64

# Deploy to both Pis
make deploy-both TX_HOST=joe@192.168.1.100 RX_HOST=joe@192.168.1.101

# Start testing from GUI or CLI
r4w remote -a 192.168.1.100 start-tx -w BPSK -t 192.168.1.101:5000
r4w remote -a 192.168.1.101 start-rx -w BPSK -p 5000
```

---

# Security & Waveform Isolation

R4W addresses a critical need in SDR deployments: **preventing interference between waveforms** and **separating sensitive communications**. The `r4w-sandbox` crate provides 8 levels of isolation, from basic memory safety to complete air-gapped systems.

## Why Isolation Matters

- **Multi-classification environments**: Run SECRET and UNCLASSIFIED waveforms on shared hardware
- **Multi-tenant systems**: Prevent one customer's waveform from affecting another
- **Cryptographic separation**: Isolate encrypted from plaintext processing
- **Fault containment**: A bug in one waveform cannot crash others
- **Regulatory compliance**: Meet government/defense isolation requirements

## Isolation Levels Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         Isolation Levels                                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  L1  ┌──────────────┐   Rust memory safety                   TURN-KEY      │
│      │  No Sandbox  │   - Zero-cost, always-on                             │
│      └──────────────┘   - Prevents buffer overflows, use-after-free        │
│                                                                            │
│ L1.5 ┌──────────────┐   WebAssembly sandbox                  TURN-KEY      │
│      │ WASM Sandbox │   - Language-agnostic isolation (Rust, C, C++)       │
│      └──────────────┘   - Native DSP host functions for performance        │
│                                                                            │
│  L2  ┌──────────────┐   Linux namespaces                     TURN-KEY      │
│      │  Namespaces  │   - Process, network, mount isolation                │
│      └──────────────┘   - Separate /proc, network stack per waveform       │
│                                                                            │
│  L3  ┌──────────────┐   Seccomp + LSM (SELinux/AppArmor)     TURN-KEY      │
│      │ Syscall Lock │   - Restrict system calls to DSP operations          │
│      └──────────────┘   - Mandatory access control policies                │
│                                                                            │
│  L4  ┌──────────────┐   Container (Docker/Podman)            TEMPLATES     │
│      │  Container   │   - cgroups for resource limits                      │
│      └──────────────┘   - Pre-built Dockerfile provided                    │
│                                                                            │
│  L5  ┌──────────────┐   MicroVM (Firecracker)                CONFIG        │
│      │   MicroVM    │   - VM-level isolation, minimal overhead             │
│      └──────────────┘   - Sub-second boot times                            │
│                                                                            │
│  L6  ┌──────────────┐   Full VM (KVM/QEMU)                   CONFIG        │
│      │   Full VM    │   - Complete virtual machine isolation               │
│      └──────────────┘   - For certification requirements                   │
│                                                                            │
│  L7  ┌──────────────┐   Hardware isolation                   SETUP         │
│      │   Hardware   │   - FPGA partitions with AXI firewalls               │
│      └──────────────┘   - CPU pinning, IOMMU, memory encryption            │
│                                                                            │
│  L8  ┌──────────────┐   Air gap                              SETUP         │
│      │   Air Gap    │   - Physically separate systems                      │
│      └──────────────┘   - Data diodes for one-way transfer                 │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

## Turn-Key Features

The following security features work out of the box:

| Feature | Description | Crate |
|---------|-------------|-------|
| **WASM Sandbox** | WebAssembly isolation with wasmtime runtime | `r4w-sandbox` |
| **DSP Host Functions** | Native FFT, complex ops, windows callable from WASM | `r4w-sandbox` |
| **SecureBuffer** | Memory zeroization on drop, mlock to prevent swap | `r4w-sandbox` |
| **EncryptedBuffer** | AES-GCM encrypted memory for keys | `r4w-sandbox` |
| **GuardedBuffer** | Guard pages to detect overflows | `r4w-sandbox` |
| **Namespace Isolation** | PID/NET/MOUNT/USER separation | `r4w-sandbox` |
| **Seccomp Profiles** | DSP-optimized syscall allowlists | `r4w-sandbox` |
| **Shared Memory IPC** | Zero-copy sample transfer between sandboxes | `r4w-sandbox` |
| **Control Channels** | Unix socket communication for isolated waveforms | `r4w-sandbox` |
| **FPGA Partitions** | AXI firewall configuration for PL isolation | `r4w-sandbox` |

## Quick Example

```rust
use r4w_sandbox::{Sandbox, IsolationLevel, SecureBuffer};
use r4w_sandbox::policy::{SeccompProfile, Capability};

// Create a sandbox for classified waveform processing
let sandbox = Sandbox::builder()
    .isolation_level(IsolationLevel::L3_LSM)
    .waveform("SINCGARS")
    .memory_limit(512 * 1024 * 1024)    // 512 MB limit
    .cpu_limit(100)                       // 1 CPU core
    .seccomp_profile(SeccompProfile::DSP) // DSP-only syscalls
    .capabilities(&[Capability::IpcLock]) // Allow mlock for secure memory
    .allow_network(false)                 // No network access
    .build()?;

// Run classified processing in isolation
sandbox.run(|| {
    // This code runs with:
    // - Isolated PID namespace (can't see other processes)
    // - Restricted syscalls (only DSP operations)
    // - Memory limits enforced
    // - No network access
    let key = SecureBuffer::new(32);  // Auto-zeroized on drop
    process_sincgars(&samples, &key);
})?;
```

For complete documentation, see:
- [docs/ISOLATION_GUIDE.md](./docs/ISOLATION_GUIDE.md) - Comprehensive isolation architecture
- [docs/SECURITY_GUIDE.md](./docs/SECURITY_GUIDE.md) - Security best practices

## WASM Sandbox with DSP Host Functions

The L1.5 WASM sandbox enables hybrid architecture: **WASM for isolated logic, native code for fast DSP**:

```rust
use r4w_sandbox::wasm::{WasmSandbox, WasmConfig};

// Load waveform from untrusted source with DSP acceleration
let config = WasmConfig::dsp();  // Enable SIMD, larger memory
let mut sandbox = WasmSandbox::new(config)?;
sandbox.load_module(&waveform_wasm_bytes)?;

// WASM waveform calls native DSP host functions:
// - fft/ifft (Forward/inverse FFT via rustfft)
// - complex_multiply (SIMD-accelerated complex ops)
// - find_peak (Peak detection in spectrum)
// - hann_window, hamming_window (Window generation)
// - total_power, compute_magnitudes (Signal analysis)

let result = sandbox.call_demodulate(input_ptr, reference_ptr, len)?;
```

In WASM module (Rust/C/C++):
```rust
#[link(wasm_import_module = "r4w_dsp")]
extern "C" {
    fn fft(in_ptr: *const f32, out_ptr: *mut f32, len: i32);
    fn complex_multiply(a: *const f32, b: *const f32, out: *mut f32, len: i32);
    fn find_peak(ptr: *const f32, len: i32) -> i32;
}

// Hybrid demodulation: WASM logic + native DSP
pub fn demodulate(input: &[f32], reference: &[f32]) -> i32 {
    unsafe {
        complex_multiply(input.as_ptr(), reference.as_ptr(), mixed.as_mut_ptr(), len);
        fft(mixed.as_ptr(), spectrum.as_mut_ptr(), len);
        find_peak(spectrum.as_ptr(), len)  // Returns peak bin
    }
}
```

---

# Crypto Boundary Architecture (Commercial Secure SDR)

For commercial secure SDR applications requiring formal separation between trusted and untrusted domains, R4W supports integration with a **Crypto Service Interface (CSI)**.

## Architecture Overview

```
┌─────────────────────────────────────────┐
│ Application (Voice/Data)                │  RED (Trusted)
│   PlaintextIn { payload, policy_id }    │
├═════════════════════════════════════════┤
│ Crypto Service Interface (CSI)          │  ← CRYPTO BOUNDARY
│   - AEAD encryption/decryption          │
│   - Replay protection (sliding window)  │
│   - Key references (no raw keys)        │
│   - Zeroization with observable state   │
├═════════════════════════════════════════┤
│ Waveform Layer (sees only ciphertext)   │  BLACK (Untrusted)
│   modulate(&[u8]) -> Vec<IQSample>      │
├─────────────────────────────────────────┤
│ HAL / RF Hardware                       │  BLACK
└─────────────────────────────────────────┘
```

## Why R4W is Already Compatible

1. **Waveform trait takes `&[u8]`** - opaque bytes, whether plaintext or ciphertext
2. **HAL is already "untrusted"** - only moves IQ samples, no plaintext exposure
3. **Real-time infrastructure aligns** - lock-free queues, `no_std` readiness

## Key Security Properties

| Property | Description |
|----------|-------------|
| **Directionality** | Plaintext only enters CSI; ciphertext only exits |
| **No RF metadata with plaintext** | Frequency, modulation, etc. never cross boundary |
| **Replay protection** | Per-flow sliding window (64-128 packets) |
| **Key references** | Never raw key material in waveform code |
| **Zeroization** | Explicit command with observable state transition |

## Integration Strategy

CSI is designed as an **optional layer** that sits above the waveform:

```rust
// Without CSI (educational/hobby)
let samples = waveform.modulate(plaintext_bytes);

// With CSI (commercial secure)
csi.submit_plaintext(plaintext_msg)?;
if let Some(ct) = csi.poll_ciphertext() {
    let samples = waveform.modulate(&ct.ciphertext);
}
```

## Implementation Status

| Phase | Status | Description |
|-------|--------|-------------|
| Architecture Design | **Complete** | Documented in `docs/CRYPTO_BOUNDARY.md` |
| CSI Specification | **Complete** | Flow management, replay, zeroization |
| R4W Compatibility | **Ready** | No changes needed to existing code |
| CSI Implementation | Future | `csi-core`, `csi-queues`, `csi-backend-soft` |
| Embedded Target | Future | STM32H7 (no_std), Zynq |

## Target Platforms

| Platform | Use Case | CSI Backend |
|----------|----------|-------------|
| Desktop SDR | Educational, development | Software (ChaCha20-Poly1305) |
| STM32H7 | Embedded radio control plane | Software or secure element |
| Zynq | Production SDR with FPGA PHY | Hardware crypto acceleration |

For complete documentation: [docs/CRYPTO_BOUNDARY.md](./docs/CRYPTO_BOUNDARY.md)

---

# Mesh Networking

R4W is adding mesh networking capability, starting with **Meshtastic protocol support** to enable participation in the global off-grid mesh communication network with 40,000+ active nodes.

## Why Mesh Networking?

| Benefit | Description |
|---------|-------------|
| **Off-Grid Communication** | Text messaging and position sharing without cellular/internet |
| **Disaster Resilience** | Self-healing network topology |
| **Long Range** | LoRa enables 10+ km links with low power |
| **Education** | Learn real-world mesh routing algorithms |
| **Interoperability** | Join existing Meshtastic network immediately |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         R4W Mesh Stack                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      Application Layer                               │   │
│   │  ┌───────────────┐  ┌───────────────┐  ┌─────────────────────────┐   │   │
│   │  │ Text Messaging│  │Position Share │  │     Telemetry           │   │   │
│   │  │   (228 bytes) │  │  (GPS coords) │  │  (battery, sensors)     │   │   │
│   │  └───────────────┘  └───────────────┘  └─────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                      ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      Mesh Routing Layer                              │   │
│   │  ┌───────────────────────────┐  ┌─────────────────────────────────┐  │   │
│   │  │   Managed Flood Routing   │  │    Next-Hop Routing             │  │   │
│   │  │   (broadcasts, SNR-based) │  │    (direct messages, cached)    │  │   │
│   │  └───────────────────────────┘  └─────────────────────────────────┘  │   │
│   │  ┌───────────────────────────┐  ┌─────────────────────────────────┐  │   │
│   │  │  Duplicate Detection      │  │    Neighbor Discovery           │  │   │
│   │  │  (packet ID cache)        │  │    (NodeInfo exchange)          │  │   │
│   │  └───────────────────────────┘  └─────────────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                      ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         MAC Layer                                    │   │
│   │  ┌───────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │   │
│   │  │ CSMA/CA           │  │ Packet Framing   │  │ AES Encryption   │   │   │
│   │  │ (contention window│  │ (header+payload) │  │ (AES-128/256-CTR)│   │   │
│   │  │  scales w/ util)  │  │                  │  │                  │   │   │
│   │  └───────────────────┘  └──────────────────┘  └──────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                      ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      Physical Layer (LoRa)                           │   │
│   │  ┌───────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │   │
│   │  │ CSS Modulation    │  │ CAD (Channel     │  │ Regional Freq    │   │   │
│   │  │ (existing r4w-core│  │  Activity Det.)  │  │ (US/EU/AU/etc)   │   │   │
│   │  │  LoRa waveform)   │  │                  │  │                  │   │   │
│   │  └───────────────────┘  └──────────────────┘  └──────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Meshtastic Protocol Overview

| Layer | Component | Description |
|-------|-----------|-------------|
| **Physical** | LoRa CSS | 16-symbol preamble, sync word 0x2B, SF7-SF12 |
| **MAC** | CSMA/CA | Channel activity detection, contention window |
| **Routing** | Managed Flood | SNR-based rebroadcast delay (distant nodes first) |
| **Messages** | Protobuf | Position, Text, NodeInfo, Routing, Telemetry |
| **Security** | AES-CTR | Channel PSK-derived keys, optional encryption |

## Generic Mesh Trait Design

```rust
/// Trait for mesh-capable protocols
pub trait MeshNetwork: Send + Sync {
    type NodeId;
    type Packet;

    /// Discover neighboring nodes
    fn discover_neighbors(&mut self) -> Vec<Neighbor<Self::NodeId>>;

    /// Get route to destination
    fn route(&self, dest: Self::NodeId) -> Option<Route<Self::NodeId>>;

    /// Forward packet through mesh
    fn forward(&mut self, packet: Self::Packet) -> Result<(), MeshError>;

    /// Handle received packet
    fn on_receive(&mut self, packet: Self::Packet, rssi: f32, snr: f32);
}

/// Extension trait for mesh-capable waveforms
pub trait MeshPhy: Waveform {
    /// Check if channel is busy (CAD)
    fn channel_busy(&self) -> bool;

    /// Get received signal strength
    fn rssi(&self) -> f32;

    /// Get signal-to-noise ratio
    fn snr(&self) -> f32;
}
```

## Requirements Summary (MESH-001 to MESH-020)

| Category | Requirements | Priority |
|----------|--------------|----------|
| **Physical Layer** | Symbol encoding, CAD, regional frequencies | High |
| **MAC Layer** | CSMA/CA, packet framing, channel utilization | High |
| **Mesh Routing** | Flood routing, next-hop, deduplication, discovery | High |
| **Interoperability** | Protobuf messages, AES encryption, channels | High |
| **Applications** | Text messaging, position sharing | High/Medium |
| **Integration** | MeshNetwork trait, hardware support, simulation | Medium |

## Implementation Status

The mesh networking module is now implemented in `crates/r4w-core/src/mesh/`:

| File | Description | Status |
|------|-------------|--------|
| `traits.rs` | `MeshNetwork` and `MeshPhy` traits, `MeshStats`, `MeshConfig` | ✅ Complete |
| `packet.rs` | `NodeId`, `MeshPacket`, `PacketHeader`, `PacketFlags`, CRC-16 | ✅ Complete |
| `neighbor.rs` | `NeighborTable`, `NodeInfo`, `LinkQuality` with RSSI/SNR/PDR | ✅ Complete |
| `routing.rs` | `FloodRouter`, `NextHopRouter`, `DuplicateCache`, SNR-based delays | ✅ Complete |
| `mac.rs` | CSMA/CA `MacLayer`, `CsmaConfig`, `ChannelUtilization` | ✅ Complete |
| `meshtastic.rs` | `MeshtasticNode`, `ModemPreset`, `Region`, channel config | ✅ Complete |
| `lora_mesh.rs` | `LoRaMesh`, `LoRaMeshPhy` - LoRa waveform with mesh integration | ✅ Complete |

### Completed Requirements
- MESH-002: LoRa symbol encoding (via `LoRaMeshPhy` integrating existing LoRa waveform)
- MESH-003: Channel Activity Detection (CAD) via signal power estimation
- MESH-004: Regional frequency configuration
- MESH-005: CSMA/CA with contention window
- MESH-006: Packet framing
- MESH-007: Channel utilization tracking
- MESH-008: Managed flood routing
- MESH-009: Next-hop routing
- MESH-010: Duplicate packet detection
- MESH-011: Node discovery and neighbor table
- MESH-017: MeshNetwork trait implementation

### Remaining Work
- MESH-012, MESH-013: Protobuf and AES encryption for full Meshtastic interoperability
- MESH-015, MESH-016: Application layer (text messaging, position sharing)
- MESH-018: SX126x hardware integration
- MESH-019: Multi-node simulation framework

See `requirements.yaml` for complete requirement details.

---

# Documentation

For comprehensive documentation, see the [docs/](./docs/) directory:

| Document | Description |
|----------|-------------|
| [docs/README.md](./docs/README.md) | Documentation index and navigation guide |
| [docs/WAVEFORM_DEVELOPERS_GUIDE.md](./docs/WAVEFORM_DEVELOPERS_GUIDE.md) | Complete guide for waveform developers: debugging, testing, deployment |
| [docs/PHYSICAL_LAYER_GUIDE.md](./docs/PHYSICAL_LAYER_GUIDE.md) | Timing model, HAL, RT primitives, configuration, observability |
| [docs/TICK_SCHEDULER_GUIDE.md](./docs/TICK_SCHEDULER_GUIDE.md) | Discrete event simulation and time control |
| [docs/REALTIME_SCHEDULER_GUIDE.md](./docs/REALTIME_SCHEDULER_GUIDE.md) | TX/RX coordination, FHSS, TDMA timing |
| [docs/FPGA_DEVELOPERS_GUIDE.md](./docs/FPGA_DEVELOPERS_GUIDE.md) | FPGA engineer's guide: IP cores, register maps, collaboration |
| [docs/SECURITY_GUIDE.md](./docs/SECURITY_GUIDE.md) | Security: memory safety, crypto, isolation, secure deployment |
| [docs/ISOLATION_GUIDE.md](./docs/ISOLATION_GUIDE.md) | Waveform isolation: containers, VMs, hardware separation |
| [docs/PORTING_GUIDE_MILITARY.md](./docs/PORTING_GUIDE_MILITARY.md) | Military waveform porting: SINCGARS, HAVEQUICK, Link-16, P25 |
| [MISSING_FEATURES.md](./MISSING_FEATURES.md) | Production readiness assessment and roadmap |

---

# References

- SDR-LoRa Paper: "SDR-LoRa, an open-source, full-fledged implementation of LoRa on Software-Defined-Radios"
- Semtech LoRa Patent: "Chirp Spread Spectrum" modulation
- GNU Radio Project: https://gnuradio.org
- Rust SDR Ecosystem: uhd-rs, soapysdr-rs, rustfft
