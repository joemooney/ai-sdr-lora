# R4W Waveform Isolation Guide

This document provides comprehensive guidance for isolating waveforms from each other in R4W deployments. It covers scenarios from basic process separation to hardware-enforced isolation for multi-level security (MLS) environments.

## Table of Contents

1. [Overview](#overview)
2. [Threat Model](#threat-model)
3. [Isolation Levels](#isolation-levels)
4. [Level 1-3: Process Isolation](#level-1-3-process-isolation)
5. [Level 1.5: WebAssembly Isolation](#level-15-webassembly-isolation)
6. [Level 4-5: Container Isolation](#level-4-5-container-isolation)
7. [Level 6: Virtual Machine Isolation](#level-6-virtual-machine-isolation)
8. [Level 7: Hardware Isolation](#level-7-hardware-isolation)
9. [Level 8: Air-Gap Isolation](#level-8-air-gap-isolation)
10. [FPGA Isolation](#fpga-isolation)
11. [Memory Protection](#memory-protection)
12. [Cross-Domain Solutions](#cross-domain-solutions)
13. [Implementation Guide](#implementation-guide)
14. [Deployment Configurations](#deployment-configurations)

---

## Overview

### Why Waveform Isolation?

In SDR systems processing multiple waveforms, several security concerns arise:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Multi-Waveform Security Concerns                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Data Leakage                                                            │
│     • Encrypted traffic mixed with unencrypted                              │
│     • Classified waveform data accessible to unclassified processes         │
│     • I/Q samples from one waveform visible to another                      │
│                                                                             │
│  2. Cross-Contamination                                                     │
│     • Bug in one waveform affecting others                                  │
│     • Malicious waveform attacking system                                   │
│     • Resource exhaustion (CPU, memory, FPGA)                               │
│                                                                             │
│  3. Covert Channels                                                         │
│     • Timing side-channels between waveforms                                │
│     • Cache-based information leakage                                       │
│     • Shared resource contention as communication                           │
│                                                                             │
│  4. Privilege Escalation                                                    │
│     • Low-security waveform gaining access to high-security resources       │
│     • FPGA reconfiguration attacks                                          │
│     • Key material exposure                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Isolation Spectrum

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Isolation Level Spectrum                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Overhead ──────────────────────────────────────────────────────► Security  │
│  (Low)                                                              (High)  │
│                                                                             │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │
│  │ L1  │ │L1.5 │ │ L2  │ │ L3  │ │ L4  │ │ L5  │ │ L6  │ │ L7  │ │ L8  │   │
│  │     │ │     │ │     │ │     │ │     │ │     │ │     │ │     │ │     │   │
│  │Proc │ │WASM │ │ NS  │ │LSM  │ │Cont │ │uVM  │ │ VM  │ │ HW  │ │ Air │   │
│  │     │ │     │ │     │ │     │ │     │ │     │ │     │ │     │ │ Gap │   │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘   │
│                                                                             │
│  L1:   Process        - Separate processes, shared kernel                   │
│  L1.5: WebAssembly    - WASM sandbox with WASI capability control           │
│  L2:   Namespace      - + Linux namespaces (pid, net, mount, user)          │
│  L3:   LSM            - + seccomp + SELinux/AppArmor                        │
│  L4:   Container      - Docker/Podman with security profiles                │
│  L5:   MicroVM        - Firecracker/gVisor (lightweight VMs)                │
│  L6:   Full VM        - KVM/QEMU with dedicated resources                   │
│  L7:   Hardware       - CPU pinning + IOMMU + memory encryption             │
│  L8:   Air Gap        - Physically separate systems with data diodes        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Threat Model

### Adversary Capabilities

| Threat Level | Adversary | Capabilities |
|--------------|-----------|--------------|
| **T1** | Buggy Waveform | Unintentional memory corruption, resource exhaustion |
| **T2** | Malicious User | Intentional attacks within waveform code |
| **T3** | Sophisticated | Side-channel attacks, timing analysis |
| **T4** | Nation State | Hardware attacks, supply chain compromise |

### Assets to Protect

```
┌─────────────────────────────────────────────────────────────────┐
│                      Asset Classification                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Critical (must never leak)                                     │
│  ├── Cryptographic keys (TRANSEC, COMSEC)                       │
│  ├── Classified message content                                 │
│  └── Frequency hopping sequences                                │
│                                                                 │
│  Sensitive (controlled access)                                  │
│  ├── Waveform parameters and configurations                     │
│  ├── Network topology and routing                               │
│  └── User credentials and certificates                          │
│                                                                 │
│  Operational (availability matters)                             │
│  ├── I/Q sample streams                                         │
│  ├── Timing synchronization                                     │
│  └── System health metrics                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Isolation Levels

### Quick Reference

| Level | Mechanism | Use Case | Latency Impact |
|-------|-----------|----------|----------------|
| L1 | Process | Development, testing | ~0% |
| L1.5 | WebAssembly | Plugin isolation, portability | 10-50% |
| L2 | Namespace | Multi-tenant non-critical | ~0% |
| L3 | LSM | Production single-security | <1% |
| L4 | Container | Multi-tenant production | 1-5% |
| L5 | MicroVM | High-security multi-tenant | 5-10% |
| L6 | Full VM | MLS environments | 10-20% |
| L7 | Hardware | Critical infrastructure | 20-50% |
| L8 | Air Gap | Absolute separation | N/A |

---

## Level 1-3: Process Isolation

### Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Process-Level Isolation (L1-L3)                         │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Linux Kernel                                │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │                    Security Modules                          │   │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │   │   │
│  │  │  │  seccomp    │  │  SELinux    │  │     AppArmor        │   │   │   │
│  │  │  │  BPF filter │  │  MAC policy │  │   Path-based MAC    │   │   │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                    │
│  │  Waveform A  │   │  Waveform B  │   │  Waveform C  │                    │
│  │              │   │              │   │              │                    │
│  │ PID NS: 1000 │   │ PID NS: 2000 │   │ PID NS: 3000 │                    │
│  │ NET NS: wf-a │   │ NET NS: wf-b │   │ NET NS: wf-c │                    │
│  │ User: wf-a   │   │ User: wf-b   │   │ User: wf-c   │                    │
│  │              │   │              │   │              │                    │
│  │ seccomp: dsp │   │ seccomp: dsp │   │ seccomp: dsp │                    │
│  │ caps: NICE   │   │ caps: NICE   │   │ caps: NICE   │                    │
│  └──────────────┘   └──────────────┘   └──────────────┘                    │
│         │                  │                  │                            │
│         └─────────────────┬┴──────────────────┘                            │
│                           │                                                │
│                    ┌──────▼──────┐                                         │
│                    │   Control   │                                         │
│                    │   Process   │                                         │
│                    │ (r4w-ctl)   │                                         │
│                    └─────────────┘                                         │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### r4w-sandbox API

```rust
use r4w_sandbox::{Sandbox, IsolationLevel, WaveformConfig};

// Create sandbox for a waveform
let sandbox = Sandbox::builder()
    .isolation_level(IsolationLevel::L3_LSM)
    .waveform("BPSK")
    .user("wf-bpsk")
    .namespaces(Namespaces::PID | Namespaces::NET | Namespaces::MOUNT)
    .seccomp_profile(SeccompProfile::DSP)
    .capabilities(&[Capability::SYS_NICE, Capability::IPC_LOCK])
    .memory_limit(512 * 1024 * 1024)  // 512 MB
    .cpu_quota(200)  // 200% = 2 cores
    .build()?;

// Spawn isolated waveform process
let handle = sandbox.spawn(|| {
    // This runs in isolated context
    let waveform = Bpsk::new(48000.0);
    waveform.run_loop()
})?;

// IPC between control and waveform
sandbox.send_command(Command::SetFrequency(915_000_000))?;
let status = sandbox.recv_status()?;
```

### Seccomp Profile for DSP

```rust
/// Minimal syscall allowlist for DSP processing
pub fn dsp_seccomp_profile() -> SeccompProfile {
    SeccompProfile::new()
        // Memory operations (required for signal processing)
        .allow(SYS_mmap)
        .allow(SYS_munmap)
        .allow(SYS_mprotect)
        .allow(SYS_mlock)
        .allow(SYS_madvise)

        // Basic I/O (for IPC with control process)
        .allow(SYS_read)
        .allow(SYS_write)
        .allow(SYS_close)
        .allow(SYS_recvmsg)
        .allow(SYS_sendmsg)

        // Shared memory (for sample transfer)
        .allow(SYS_shmat)
        .allow(SYS_shmget)
        .allow(SYS_shmdt)
        .allow(SYS_shmctl)

        // Time (for real-time scheduling)
        .allow(SYS_clock_gettime)
        .allow(SYS_clock_nanosleep)
        .allow(SYS_nanosleep)

        // Thread operations
        .allow(SYS_futex)
        .allow(SYS_clone)  // Restricted to CLONE_THREAD only

        // FPGA access (restricted paths)
        .allow_with_args(SYS_openat, |args| {
            // Only allow /dev/uio* and /dev/mem
            args.path_matches("/dev/uio*") ||
            args.path_matches("/dev/mem")
        })
        .allow(SYS_ioctl)  // For UIO

        // Default: kill process on violation
        .default_action(SeccompAction::Kill)
}
```

### SELinux Policy Module

```
# r4w_waveform.te - SELinux Type Enforcement for R4W Waveforms

policy_module(r4w_waveform, 1.0.0)

# Define types for each security level
type r4w_unclass_t;      # Unclassified waveforms
type r4w_secret_t;       # Secret waveforms
type r4w_topsecret_t;    # Top Secret waveforms
type r4w_control_t;      # Control process

# File contexts
type r4w_samples_t;      # I/Q sample files
type r4w_keys_t;         # Cryptographic keys
type r4w_config_t;       # Configuration files

# Domain transitions
domain_auto_trans(r4w_control_t, r4w_unclass_exec_t, r4w_unclass_t)
domain_auto_trans(r4w_control_t, r4w_secret_exec_t, r4w_secret_t)

# Isolation rules - prevent cross-level access
neverallow r4w_unclass_t r4w_secret_t:process signal;
neverallow r4w_unclass_t r4w_topsecret_t:process signal;
neverallow r4w_secret_t r4w_topsecret_t:process signal;

# Prevent reading higher classification samples
neverallow r4w_unclass_t { r4w_secret_t r4w_topsecret_t }:shm { read write };

# Key material access
allow r4w_secret_t r4w_keys_t:file { read };
neverallow r4w_unclass_t r4w_keys_t:file *;

# FPGA access controlled by classification
allow r4w_topsecret_t fpga_device_t:chr_file { read write ioctl };
allow r4w_secret_t fpga_device_t:chr_file { read write ioctl };
neverallow r4w_unclass_t fpga_device_t:chr_file *;
```

---

## Level 1.5: WebAssembly Isolation

WebAssembly (WASM) provides a lightweight isolation mechanism that sits between basic process isolation and Linux namespaces. It's particularly well-suited for:

- **Plugin/waveform isolation**: Run untrusted waveform code safely
- **Portable deployment**: Same binary runs on any platform
- **Fast cold-start**: ~10-15x faster than containers
- **Capability-based security**: Deny-by-default via WASI

### Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    WebAssembly Isolation (L1.5)                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Host Process                                │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                     Wasmtime Runtime                          │  │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐    │  │   │
│  │  │  │  Waveform A │  │  Waveform B │  │    Waveform C       │    │  │   │
│  │  │  │  (WASM)     │  │  (WASM)     │  │    (WASM)           │    │  │   │
│  │  │  │             │  │             │  │                     │    │  │   │
│  │  │  │ Linear Mem  │  │ Linear Mem  │  │   Linear Memory     │    │  │   │
│  │  │  │ (isolated)  │  │ (isolated)  │  │   (isolated)        │    │  │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────┘    │  │   │
│  │  │                                                               │  │   │
│  │  │  WASI Capabilities (per-module):                              │  │   │
│  │  │  • stdio: stdout/stderr for logging                           │  │   │
│  │  │  • clocks: timing for DSP                                     │  │   │
│  │  │  • random: crypto operations                                  │  │   │
│  │  │  • filesystem: DENIED                                         │  │   │
│  │  │  • network: DENIED                                            │  │   │
│  │  │  • env vars: DENIED                                           │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### r4w-sandbox WASM API

```rust
use r4w_sandbox::{WasmSandbox, WasmConfig, WasiCapabilities};

// Create sandbox with DSP-appropriate capabilities
let config = WasmConfig::dsp()
    .max_memory(512 * 1024 * 1024)  // 512 MB for sample buffers
    .fuel_limit(10_000_000_000);     // Limit execution time

let sandbox = WasmSandbox::new(config)?;

// Load waveform compiled to WASM
let module = sandbox.load_module("waveforms/bpsk.wasm")?;
let mut instance = sandbox.instantiate(&module)?;

// Call waveform functions
let result = instance.call_i32("modulate")?;
println!("Executed in {}us, consumed {} fuel",
    result.execution_time_us,
    result.fuel_consumed.unwrap_or(0));

// Direct memory access for sample buffers
instance.write_memory(0x1000, &samples)?;
let output = instance.read_memory(0x2000, output_len)?;
```

### WASI Capability Presets

```rust
// Maximum isolation - no capabilities
let caps = WasiCapabilities::none();

// DSP workloads - stdout/stderr for logging, clocks for timing
let caps = WasiCapabilities::dsp();  // Recommended for waveforms

// Development - full stdio for debugging
let caps = WasiCapabilities::with_stdio();

// Custom capabilities
let caps = WasiCapabilities::none()
    .stdout(true)
    .stderr(true)
    .clocks(true)
    .random(true)  // For crypto operations
    .preopened_dir_ro("/data/samples");  // Read-only sample files
```

### Benchmark Results (Measured)

The following benchmarks were measured on an x86_64 Linux system using release builds:

#### Startup Costs

| Operation | Time | Notes |
|-----------|------|-------|
| Module loading | ~9ms | One-time cost per module |
| Instantiation | ~70μs | Per-instance cost |
| Total cold start | <10ms | Module load + instantiate |

#### Function Call Overhead

| Signature | Latency | Notes |
|-----------|---------|-------|
| `() -> i32` | <1μs | Simple getters |
| `(i32) -> i32` | <1μs | Single-arg functions |
| `(i32, i32) -> i32` | <1μs | Two-arg functions |
| DSP function (modulate) | ~0.2μs | Includes memory allocation |

#### Memory Operations

| Operation | Throughput | Notes |
|-----------|------------|-------|
| Memory read | 80 GB/s | Host reading WASM memory |
| Memory write | 48 GB/s | Host writing WASM memory |
| alloc(64 bytes) | ~380ns | In-WASM allocation |
| alloc(4KB) | ~2μs | Larger allocations |

#### DSP Performance

| Metric | Value |
|--------|-------|
| Symbol rate | 650k symbols/sec |
| Sample throughput | 400 Msamples/sec |
| BPSK modulation | 1.5ms / 1000 bits |

#### Native vs WASM Comparison

| Operation | Native | WASM | Overhead |
|-----------|--------|------|----------|
| Simple `add(a,b)` | 1ns | 302ns | **300x** |
| BPSK modulate | 0.5μs | 1.6μs | **2.8x** |

**Key Insight**: The 300x overhead on trivial operations is due to the WASM call trampoline. For real DSP work where computation dominates, overhead drops to 2.8x because actual work amortizes the call cost.

### Understanding the WASM Call Trampoline

The 300x overhead for simple operations like `add(a, b)` is not a bug or inefficiency—it's an inherent cost of the isolation boundary. Understanding this is critical for designing WASM-based systems.

#### What is the Call Trampoline?

When the host calls a WASM function, the runtime must perform several steps:

```
Host Code                     WASM Sandbox
    │                              │
    │  1. Marshal arguments        │
    │  ─────────────────────►      │
    │                              │
    │  2. Validate types           │
    │  ─────────────────────►      │
    │                              │
    │  3. Set up WASM stack        │
    │  ─────────────────────►      │
    │                              │
    │  4. Bounds check setup       │
    │  ─────────────────────►      │
    │                              │
    │  5. Context switch           │
    │  ═══════════════════════►    │
    │                              │  ← WASM code executes here
    │  6. Context switch back      │
    │  ◄═══════════════════════    │
    │                              │
    │  7. Validate return value    │
    │  ◄─────────────────────      │
    │                              │
    │  8. Unmarshal result         │
    │  ◄─────────────────────      │
    │                              │
```

For `add(a, b)` which is a single CPU instruction (~1ns), these 8 steps take ~300ns. The actual computation is less than 1% of the total time.

#### Why Can't This Be Optimized Away?

The trampoline exists to enforce isolation guarantees:

| Step | Purpose | Cannot Skip Because... |
|------|---------|------------------------|
| Marshal args | Convert host types to WASM types | Type system boundary enforcement |
| Validate types | Ensure WASM ABI compliance | Prevents type confusion attacks |
| Stack setup | Initialize WASM stack frame | Sandboxed stack isolation |
| Bounds check | Configure linear memory guards | Memory safety is fundamental |
| Context switch | Enter sandboxed execution | This IS the isolation |

If you skip these steps, you lose the isolation guarantees. The overhead is the **cost of isolation**, not inefficiency.

#### When Overhead Doesn't Matter

The key insight is that call overhead is **fixed per call**, not per operation. This means:

```
Overhead Impact = Trampoline_Time / Total_Work_Time

Example 1: Single add
  Work: 1ns, Trampoline: 300ns → Overhead = 300x (99.7% overhead)

Example 2: 1000 adds in a loop (inside WASM)
  Work: 1000ns, Trampoline: 300ns → Overhead = 1.3x (23% overhead)

Example 3: FFT of 1024 samples
  Work: 50000ns, Trampoline: 300ns → Overhead = 1.006x (0.6% overhead)
```

**Design Rule**: Minimize host-WASM boundary crossings. Do more work per call.

### Is WASM Viable for Production Waveforms?

**Yes, but with the right architecture.** Here's the decision framework:

#### WASM is Viable When:

| Pattern | Why It Works | Example |
|---------|--------------|---------|
| **Batch Processing** | Amortizes call overhead over many samples | Process 1000 samples per call instead of 1 |
| **Complex Per-Call Work** | Computation dominates trampoline cost | FFT, FIR filter, correlation |
| **Control Flow** | Logic is cheap, computation happens in batches | State machine, protocol parsing |
| **Plugin Architecture** | Safety trumps raw performance | Third-party waveforms |

#### WASM is NOT Viable When:

| Pattern | Why It Fails | Example |
|---------|--------------|---------|
| **Sample-by-Sample** | Every sample pays 300ns overhead | Real-time audio callback per sample |
| **Tight Inner Loops** | Call overhead dominates | Calling host for each multiply |
| **Hard Real-Time** | Overhead adds jitter | <1μs latency requirements |
| **Frequent Host Interaction** | Boundary crossings kill performance | Reading host state every sample |

### Hybrid Architecture: Native DSP + WASM Logic

The optimal architecture separates concerns:

- **WASM**: Waveform logic, configuration, protocol handling (safe, portable)
- **Native**: DSP primitives, SIMD operations, FFT (fast, reusable)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Hybrid WASM/Native Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    WASM Sandbox (Isolated)                            │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │              Waveform Plugin (untrusted code)                   │  │  │
│  │  │                                                                 │  │  │
│  │  │  • State machine logic                                          │  │  │
│  │  │  • Protocol parsing/generation                                  │  │  │
│  │  │  • Configuration handling                                       │  │  │
│  │  │  • Symbol mapping                                               │  │  │
│  │  │  • Error correction control flow                                │  │  │
│  │  │                                                                 │  │  │
│  │  │         │                │                │                     │  │  │
│  │  │         │ call           │ call           │ call                │  │  │
│  │  │         ▼                ▼                ▼                     │  │  │
│  │  └─────────┼────────────────┼────────────────┼─────────────────────┘  │  │
│  │            │                │                │                        │  │
│  └────────────┼────────────────┼────────────────┼────────────────────────┘  │
│               │                │                │                           │
│  ┌────────────▼────────────────▼────────────────▼────────────────────────┐  │
│  │                 Native Host Functions (trusted, fast)                 │  │
│  │                                                                       │  │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────────┐  │  │
│  │  │   fft_1024    │  │  fir_filter   │  │   simd_complex_multiply   │  │  │
│  │  │               │  │               │  │                           │  │  │
│  │  │ • FFTW/MKL    │  │ • AVX-512     │  │ • ARM NEON                │  │  │
│  │  │ • SIMD        │  │ • Optimized   │  │ • Vectorized              │  │  │
│  │  │ • Native      │  │ • Native      │  │ • Native                  │  │  │
│  │  └───────────────┘  └───────────────┘  └───────────────────────────┘  │  │
│  │                                                                       │  │
│  │  Additional host functions:                                           │  │
│  │  • correlate(samples, pattern) → peak position                        │  │
│  │  • demodulate_batch(samples, constellation) → symbols                 │  │
│  │  • apply_agc(samples, target_level) → normalized samples              │  │
│  │  • resample(samples, ratio) → resampled                               │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Implementing Host Functions

Host functions are native Rust functions exposed to WASM:

```rust
use wasmtime::*;

/// Native DSP library exposed to WASM
pub struct DspHostFunctions {
    fft_planner: FftPlanner<f32>,
}

impl DspHostFunctions {
    /// Register all DSP host functions with the linker
    pub fn register(linker: &mut Linker<WasmHostState>) -> Result<()> {
        // FFT: (input_ptr, input_len, output_ptr) -> ()
        linker.func_wrap(
            "r4w_dsp",
            "fft_1024",
            |mut caller: Caller<'_, WasmHostState>,
             input_ptr: i32,
             output_ptr: i32| {
                // Read input from WASM memory
                let memory = caller.get_export("memory")
                    .unwrap().into_memory().unwrap();
                let input = read_complex_slice(&memory, &caller, input_ptr, 1024);

                // Perform FFT using native SIMD-optimized code
                let mut output = input.clone();
                caller.data().dsp.fft_1024(&mut output);

                // Write output back to WASM memory
                write_complex_slice(&memory, &mut caller, output_ptr, &output);
            },
        )?;

        // FIR Filter: (samples_ptr, len, coeffs_ptr, coeffs_len, output_ptr) -> ()
        linker.func_wrap(
            "r4w_dsp",
            "fir_filter",
            |mut caller: Caller<'_, WasmHostState>,
             samples_ptr: i32, samples_len: i32,
             coeffs_ptr: i32, coeffs_len: i32,
             output_ptr: i32| {
                // Read from WASM memory
                let memory = caller.get_export("memory")
                    .unwrap().into_memory().unwrap();
                let samples = read_f32_slice(&memory, &caller, samples_ptr, samples_len);
                let coeffs = read_f32_slice(&memory, &caller, coeffs_ptr, coeffs_len);

                // Native SIMD-optimized FIR filter
                let output = simd_fir_filter(&samples, &coeffs);

                // Write back
                write_f32_slice(&memory, &mut caller, output_ptr, &output);
            },
        )?;

        // Batch complex multiply: (a_ptr, b_ptr, len, out_ptr) -> ()
        linker.func_wrap(
            "r4w_dsp",
            "complex_multiply_batch",
            |mut caller: Caller<'_, WasmHostState>,
             a_ptr: i32, b_ptr: i32, len: i32, out_ptr: i32| {
                let memory = caller.get_export("memory")
                    .unwrap().into_memory().unwrap();
                let a = read_complex_slice(&memory, &caller, a_ptr, len);
                let b = read_complex_slice(&memory, &caller, b_ptr, len);

                // AVX-512 / NEON optimized
                let output = simd_complex_multiply(&a, &b);

                write_complex_slice(&memory, &mut caller, out_ptr, &output);
            },
        )?;

        Ok(())
    }
}
```

#### WASM Waveform Using Host Functions

```rust
// In the WASM waveform module (compiled to wasm32-wasip1)

// Import host functions
#[link(wasm_import_module = "r4w_dsp")]
extern "C" {
    fn fft_1024(input_ptr: *const Complex, output_ptr: *mut Complex);
    fn fir_filter(
        samples_ptr: *const f32, samples_len: i32,
        coeffs_ptr: *const f32, coeffs_len: i32,
        output_ptr: *mut f32,
    );
    fn complex_multiply_batch(
        a_ptr: *const Complex, b_ptr: *const Complex,
        len: i32, out_ptr: *mut Complex,
    );
}

/// LoRa demodulation - logic in WASM, heavy math in native
#[no_mangle]
pub extern "C" fn demodulate_symbol(samples_ptr: *const Complex, samples_len: i32) -> i32 {
    // Allocate buffers
    let mut fft_input = vec![Complex::zero(); 1024];
    let mut fft_output = vec![Complex::zero(); 1024];
    let mut downchirp = vec![Complex::zero(); 1024];

    // Generate downchirp (simple math, stays in WASM)
    generate_downchirp(&mut downchirp, samples_len as usize);

    // Multiply by downchirp - USE NATIVE HOST FUNCTION
    unsafe {
        complex_multiply_batch(
            samples_ptr,
            downchirp.as_ptr(),
            samples_len,
            fft_input.as_mut_ptr(),
        );
    }

    // FFT to find peak - USE NATIVE HOST FUNCTION
    unsafe {
        fft_1024(fft_input.as_ptr(), fft_output.as_mut_ptr());
    }

    // Find peak (simple loop, stays in WASM)
    let symbol = find_peak_bin(&fft_output);

    // Gray decode (simple lookup, stays in WASM)
    gray_decode(symbol)
}
```

#### Performance Comparison

| Approach | Call Pattern | Overhead | Viable? |
|----------|--------------|----------|---------|
| Pure WASM | Per-sample calls to host | 300x | ❌ |
| Pure WASM | Batch processing | 2-3x | ✅ |
| Hybrid | Logic in WASM, DSP in native | 1.1-1.5x | ✅ |
| Pure Native | No isolation | 1x | ⚠️ Security risk |

#### Standard DSP Host Function Library

For production use, R4W provides a standard library of host functions that waveforms can use:

| Category | Functions | Notes |
|----------|-----------|-------|
| **FFT** | `fft_64`, `fft_128`, `fft_256`, `fft_512`, `fft_1024`, `fft_2048`, `fft_4096` | FFTW/MKL backend |
| **Filters** | `fir_filter`, `iir_filter`, `polyphase_filter` | SIMD optimized |
| **Resampling** | `resample`, `interpolate`, `decimate` | Fractional rate support |
| **Modulation** | `qam_mod`, `qam_demod`, `psk_mod`, `psk_demod` | Batch operations |
| **Math** | `complex_multiply`, `complex_add`, `magnitude`, `phase` | Vectorized |
| **Correlation** | `correlate`, `cross_correlate`, `auto_correlate` | For sync detection |
| **AGC** | `agc_process`, `power_estimate` | Automatic gain control |

This library is:
- **Generic**: Works with any waveform
- **Trusted**: Part of the R4W core, audited
- **Fast**: Native SIMD implementations
- **Safe**: WASM can only call these specific functions

### Summary: WASM Production Viability

| Question | Answer |
|----------|--------|
| Can WASM achieve native performance? | No, there is an inherent isolation cost |
| Can WASM be viable for production? | **Yes**, with hybrid architecture |
| What's the practical overhead? | 10-50% with proper design (batch + host functions) |
| When should I use pure WASM? | Prototyping, untrusted plugins, portability |
| When should I use hybrid? | Production waveforms needing both isolation and performance |

The hybrid architecture gives you:
- **Isolation** for untrusted waveform logic
- **Performance** for DSP operations
- **Portability** across platforms
- **Security** with deny-by-default capabilities

#### Fuel Metering

| Metric | Value |
|--------|-------|
| Overhead | Negligible (<1%) |
| Fuel per `add()` | 14 units |

### Trade-offs

| Aspect | WASM (L1.5) | Namespaces (L2) | Containers (L4) |
|--------|-------------|-----------------|-----------------|
| **Cold start** | <10ms | ~10ms | ~100-500ms |
| **Call overhead** | 300ns | ~0ns | ~0ns |
| **DSP overhead** | 2.8x | ~1x | ~1x |
| **Memory bandwidth** | 48-80 GB/s | Native | Native |
| **Memory isolation** | Linear memory | Virtual memory | cgroups |
| **Syscall filtering** | WASI only | seccomp | seccomp |
| **Portability** | Cross-platform | Linux only | Linux/Docker |
| **Root required** | No | Sometimes | Usually |

### When to Use WASM Isolation

**Good fit:**
- Plugin architecture for third-party waveforms
- Non-real-time DSP where 2-3x overhead is acceptable
- Cross-platform deployment (Linux, macOS, Windows, embedded)
- Untrusted code execution without root privileges
- Rapid prototyping (fast compilation, instant feedback)
- Batch processing where call overhead is amortized

**Poor fit:**
- Hard real-time DSP requiring <1μs latency
- Tight inner loops with frequent function calls
- Multi-level security requiring kernel-level isolation
- Direct hardware access (FPGA, SDR devices)
- Sample-by-sample processing (call overhead dominates)

### Compiling Waveforms to WASM

```bash
# Install WASM target
rustup target add wasm32-wasip1

# Compile waveform to WASM
cargo build --target wasm32-wasip1 --release -p my-waveform

# Optimize (optional)
wasm-opt -O3 target/wasm32-wasip1/release/my_waveform.wasm \
    -o waveforms/my_waveform.wasm
```

### Benchmark Example

```rust
use r4w_sandbox::{WasmSandbox, WasmConfig, WasmBenchmark};

let sandbox = WasmSandbox::new(WasmConfig::dsp())?;
let module = sandbox.load_module("waveforms/bpsk.wasm")?;
let mut instance = sandbox.instantiate(&module)?;

// Benchmark modulation function
let mut bench = WasmBenchmark::new();
for _ in 0..1000 {
    let result = instance.call_i32("process_samples")?;
    bench.record(result.execution_time_us);
}

println!("WASM benchmark: {}", bench.summary());
// Output: n=1000 min=45us mean=52.3us p50=51us p99=89us max=156us
```

---

## Level 4-5: Container Isolation

### Docker Compose for Multi-Waveform Deployment

```yaml
# docker-compose.isolation.yml
version: '3.8'

x-waveform-common: &waveform-common
  security_opt:
    - no-new-privileges:true
    - seccomp:seccomp-dsp.json
  cap_drop:
    - ALL
  cap_add:
    - SYS_NICE
    - IPC_LOCK
  read_only: true
  tmpfs:
    - /tmp:noexec,nosuid,size=64M

services:
  # Unclassified waveform - maximum restrictions
  waveform-unclass:
    <<: *waveform-common
    image: r4w/waveform:latest
    container_name: r4w-unclass
    environment:
      - R4W_WAVEFORM=BPSK
      - R4W_SECURITY_LEVEL=UNCLASS
    mem_limit: 256m
    cpus: 1
    pids_limit: 50
    networks:
      - unclass-net
    # No FPGA access
    # No shared memory with other waveforms

  # Secret waveform - moderate restrictions
  waveform-secret:
    <<: *waveform-common
    image: r4w/waveform:latest
    container_name: r4w-secret
    security_opt:
      - no-new-privileges:true
      - seccomp:seccomp-dsp.json
      - apparmor:r4w-secret-profile
    environment:
      - R4W_WAVEFORM=SINCGARS
      - R4W_SECURITY_LEVEL=SECRET
    mem_limit: 512m
    cpus: 2
    pids_limit: 100
    devices:
      - /dev/uio0:/dev/uio0:rw  # FPGA access
    volumes:
      - secret-keys:/keys:ro
    networks:
      - secret-net
    # Isolated network, no route to unclass

  # Top Secret waveform - minimal restrictions for performance
  waveform-topsecret:
    <<: *waveform-common
    image: r4w/waveform:latest
    container_name: r4w-topsecret
    security_opt:
      - no-new-privileges:true
      - seccomp:seccomp-dsp.json
      - apparmor:r4w-topsecret-profile
    environment:
      - R4W_WAVEFORM=LINK16
      - R4W_SECURITY_LEVEL=TOPSECRET
    mem_limit: 1g
    cpus: 4
    pids_limit: 200
    devices:
      - /dev/uio0:/dev/uio0:rw
      - /dev/uio1:/dev/uio1:rw
    volumes:
      - topsecret-keys:/keys:ro
    networks:
      - topsecret-net
    # Dedicated FPGA partition

  # Control process - orchestrates all waveforms
  r4w-control:
    image: r4w/control:latest
    container_name: r4w-control
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    networks:
      - unclass-net
      - secret-net
      - topsecret-net
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro

networks:
  unclass-net:
    driver: bridge
    internal: true
    driver_opts:
      com.docker.network.bridge.enable_icc: "false"
  secret-net:
    driver: bridge
    internal: true
    driver_opts:
      com.docker.network.bridge.enable_icc: "false"
  topsecret-net:
    driver: bridge
    internal: true
    driver_opts:
      com.docker.network.bridge.enable_icc: "false"

volumes:
  secret-keys:
    driver: local
    driver_opts:
      type: tmpfs
      o: size=1m,mode=0400
  topsecret-keys:
    driver: local
    driver_opts:
      type: tmpfs
      o: size=1m,mode=0400
```

### Firecracker MicroVM (Level 5)

```rust
use firecracker_sdk::{VmConfig, NetworkInterface, Drive};

/// Create a Firecracker microVM for isolated waveform execution
pub fn create_waveform_microvm(
    waveform: &str,
    security_level: SecurityLevel,
) -> Result<VmHandle> {
    let config = VmConfig::builder()
        // Minimal kernel for DSP workloads
        .kernel_image("/var/lib/r4w/vmlinux-dsp")
        .kernel_args("console=ttyS0 quiet")

        // Root filesystem with waveform
        .root_drive(Drive::builder()
            .path(format!("/var/lib/r4w/rootfs-{}.ext4", waveform))
            .read_only(true)
            .build())

        // Resource limits based on security level
        .vcpu_count(match security_level {
            SecurityLevel::Unclass => 1,
            SecurityLevel::Secret => 2,
            SecurityLevel::TopSecret => 4,
        })
        .mem_size_mib(match security_level {
            SecurityLevel::Unclass => 256,
            SecurityLevel::Secret => 512,
            SecurityLevel::TopSecret => 1024,
        })

        // Network isolation
        .network_interface(NetworkInterface::builder()
            .iface_id("eth0")
            .host_dev_name(format!("tap-{}", waveform))
            .build())

        // FPGA passthrough (for high-security only)
        .mmio_device(if security_level >= SecurityLevel::Secret {
            Some(MmioDevice::builder()
                .type_("virtio-fpga")
                .base_addr(0x40000000)
                .size(0x10000)
                .irq(5)
                .build())
        } else {
            None
        })

        .build()?;

    VmHandle::spawn(config)
}
```

---

## Level 6: Virtual Machine Isolation

### KVM/QEMU with VFIO Passthrough

```bash
#!/bin/bash
# launch-isolated-vm.sh - Launch VM with FPGA passthrough

WAVEFORM=$1
SECURITY_LEVEL=$2

# Determine IOMMU group for FPGA
FPGA_PCI="0000:01:00.0"
IOMMU_GROUP=$(readlink /sys/bus/pci/devices/$FPGA_PCI/iommu_group | basename)

# Unbind from host driver
echo $FPGA_PCI > /sys/bus/pci/drivers/xilinx_dma/unbind
echo $FPGA_PCI > /sys/bus/pci/drivers/vfio-pci/bind

# Launch QEMU with isolated resources
qemu-system-x86_64 \
    -name "r4w-${WAVEFORM}-${SECURITY_LEVEL}" \
    -machine q35,accel=kvm,kernel-irqchip=split \
    -cpu host,+invtsc \
    -m 2G \
    -smp 4,sockets=1,cores=4,threads=1 \
    \
    # Memory isolation
    -object memory-backend-memfd,id=mem0,size=2G,share=off \
    -numa node,memdev=mem0 \
    \
    # CPU pinning for deterministic performance
    -realtime mlock=on \
    -overcommit cpu-pm=on \
    \
    # FPGA passthrough via VFIO
    -device vfio-pci,host=$FPGA_PCI,id=fpga0 \
    \
    # Isolated network
    -netdev tap,id=net0,ifname=tap-${WAVEFORM},script=no \
    -device virtio-net-pci,netdev=net0,mac=52:54:00:${SECURITY_LEVEL}:00:01 \
    \
    # Secure boot (optional)
    -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE.fd \
    -drive if=pflash,format=raw,file=/var/lib/r4w/OVMF_VARS_${WAVEFORM}.fd \
    \
    # Root filesystem
    -drive file=/var/lib/r4w/vm-${WAVEFORM}.qcow2,if=virtio,format=qcow2 \
    \
    # Console
    -serial stdio \
    -display none
```

### libvirt Configuration

```xml
<!-- /etc/libvirt/qemu/r4w-secret.xml -->
<domain type='kvm'>
  <name>r4w-secret-sincgars</name>
  <uuid>12345678-1234-1234-1234-123456789012</uuid>
  <memory unit='GiB'>2</memory>
  <vcpu placement='static' cpuset='4-7'>4</vcpu>

  <cputune>
    <vcpupin vcpu='0' cpuset='4'/>
    <vcpupin vcpu='1' cpuset='5'/>
    <vcpupin vcpu='2' cpuset='6'/>
    <vcpupin vcpu='3' cpuset='7'/>
    <emulatorpin cpuset='4-7'/>
  </cputune>

  <numatune>
    <memory mode='strict' nodeset='1'/>
  </numatune>

  <memoryBacking>
    <hugepages/>
    <locked/>
    <nosharepages/>
  </memoryBacking>

  <os>
    <type arch='x86_64'>hvm</type>
    <loader readonly='yes' type='pflash'>/usr/share/OVMF/OVMF_CODE.fd</loader>
    <nvram>/var/lib/libvirt/qemu/nvram/r4w-secret_VARS.fd</nvram>
    <boot dev='hd'/>
  </os>

  <features>
    <acpi/>
    <apic/>
    <ioapic driver='kvm'/>
  </features>

  <devices>
    <!-- FPGA passthrough -->
    <hostdev mode='subsystem' type='pci' managed='yes'>
      <driver name='vfio'/>
      <source>
        <address domain='0x0000' bus='0x01' slot='0x00' function='0x0'/>
      </source>
    </hostdev>

    <!-- Isolated network -->
    <interface type='network'>
      <mac address='52:54:00:02:00:01'/>
      <source network='r4w-secret-isolated'/>
      <model type='virtio'/>
    </interface>

    <!-- Disk -->
    <disk type='file' device='disk'>
      <driver name='qemu' type='qcow2'/>
      <source file='/var/lib/r4w/vm-sincgars.qcow2'/>
      <target dev='vda' bus='virtio'/>
    </disk>
  </devices>

  <seclabel type='static' model='selinux' relabel='yes'>
    <label>system_u:system_r:svirt_t:s0:c123,c456</label>
  </seclabel>
</domain>
```

---

## Level 7: Hardware Isolation

### CPU and Memory Isolation

```rust
use libc::{cpu_set_t, sched_setaffinity, CPU_SET, CPU_ZERO};
use std::mem::MaybeUninit;

/// Pin waveform to specific CPU cores
pub fn pin_to_cores(cores: &[usize]) -> Result<()> {
    let mut cpuset = MaybeUninit::<cpu_set_t>::uninit();
    unsafe {
        CPU_ZERO(cpuset.as_mut_ptr());
        for &core in cores {
            CPU_SET(core, cpuset.as_mut_ptr());
        }

        let result = sched_setaffinity(0, std::mem::size_of::<cpu_set_t>(), cpuset.as_ptr());
        if result != 0 {
            return Err(Error::AffinityFailed(std::io::Error::last_os_error()));
        }
    }
    Ok(())
}

/// Configure NUMA memory policy
pub fn set_numa_policy(node: usize) -> Result<()> {
    use libc::{set_mempolicy, MPOL_BIND};

    let mut nodemask: u64 = 1 << node;
    unsafe {
        let result = set_mempolicy(
            MPOL_BIND,
            &nodemask as *const u64,
            std::mem::size_of::<u64>() * 8,
        );
        if result != 0 {
            return Err(Error::NumaFailed(std::io::Error::last_os_error()));
        }
    }
    Ok(())
}

/// Hardware isolation configuration
pub struct HardwareIsolation {
    /// Dedicated CPU cores for this waveform
    pub cpu_cores: Vec<usize>,
    /// NUMA node for memory allocation
    pub numa_node: usize,
    /// IOMMU group for FPGA isolation
    pub iommu_group: Option<String>,
    /// Enable Intel CAT (Cache Allocation Technology)
    pub cache_allocation: Option<CacheConfig>,
    /// Enable AMD SEV (Secure Encrypted Virtualization)
    pub memory_encryption: bool,
}

impl HardwareIsolation {
    pub fn apply(&self) -> Result<()> {
        // Pin to dedicated cores
        pin_to_cores(&self.cpu_cores)?;

        // Set NUMA policy
        set_numa_policy(self.numa_node)?;

        // Configure cache allocation if available
        if let Some(ref cache) = self.cache_allocation {
            configure_intel_cat(cache)?;
        }

        // IOMMU isolation is configured at boot/VM level

        Ok(())
    }
}
```

### Intel Cache Allocation Technology (CAT)

```bash
#!/bin/bash
# configure-cat.sh - Configure Intel CAT for waveform isolation

# Check CAT support
if [ ! -d /sys/fs/resctrl ]; then
    mount -t resctrl resctrl /sys/fs/resctrl
fi

# Create resource groups for each security level
mkdir -p /sys/fs/resctrl/r4w-unclass
mkdir -p /sys/fs/resctrl/r4w-secret
mkdir -p /sys/fs/resctrl/r4w-topsecret

# Allocate cache ways (assuming 11 ways available, 0-10)
# Unclass: 2 ways (0-1)
echo "L3:0=003" > /sys/fs/resctrl/r4w-unclass/schemata

# Secret: 4 ways (2-5)
echo "L3:0=03c" > /sys/fs/resctrl/r4w-secret/schemata

# Top Secret: 5 ways (6-10)
echo "L3:0=7c0" > /sys/fs/resctrl/r4w-topsecret/schemata

# Assign PIDs to resource groups
# (done dynamically when waveforms start)
```

### IOMMU Configuration for FPGA

```bash
# /etc/default/grub
GRUB_CMDLINE_LINUX="intel_iommu=on iommu=pt"

# Verify IOMMU groups
for g in /sys/kernel/iommu_groups/*/devices/*; do
    echo "IOMMU Group $(basename $(dirname $(dirname $g))):"
    echo "    $(lspci -nns $(basename $g))"
done

# Bind FPGA to VFIO for passthrough
echo "10ee 9034" > /sys/bus/pci/drivers/vfio-pci/new_id
echo "0000:01:00.0" > /sys/bus/pci/drivers/xilinx_dma/unbind
echo "0000:01:00.0" > /sys/bus/pci/drivers/vfio-pci/bind
```

---

## Level 8: Air-Gap Isolation

### Data Diode Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      Air-Gap with Data Diode                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────────┐         ┌─────────────────────┐                   │
│  │   High-Side System  │         │   Low-Side System   │                   │
│  │   (TOP SECRET)      │         │   (UNCLASS)         │                   │
│  │                     │         │                     │                   │
│  │  ┌───────────────┐  │         │  ┌───────────────┐  │                   │
│  │  │ Link-16       │  │         │  │ ADS-B         │  │                   │
│  │  │ SINCGARS      │  │         │  │ AIS           │  │                   │
│  │  │ HAVEQUICK     │  │         │  │ Commercial FM │  │                   │
│  │  └───────────────┘  │         │  └───────────────┘  │                   │
│  │         │           │         │         ▲           │                   │
│  │         ▼           │         │         │           │                   │
│  │  ┌───────────────┐  │         │  ┌───────────────┐  │                   │
│  │  │ Data Diode TX │──┼────────►│  │ Data Diode RX │  │                   │
│  │  │ (fiber optic) │  │  ONE    │  │ (fiber optic) │  │                   │
│  │  └───────────────┘  │  WAY    │  └───────────────┘  │                   │
│  │                     │         │                     │                   │
│  └─────────────────────┘         └─────────────────────┘                   │
│                                                                            │
│  • Physical fiber ensures unidirectional flow                              │
│  • No electrical connection between systems                                │
│  • High-to-low only (sanitized data)                                       │
│  • Certified for cross-domain transfer                                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## FPGA Isolation

### Bitstream Partitioning

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    FPGA Partial Reconfiguration                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         FPGA Fabric                                 │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │   Static    │  │  Partition  │  │  Partition  │  │  Partition │  │   │
│  │  │   Region    │  │     A       │  │     B       │  │     C      │  │   │
│  │  │             │  │             │  │             │  │            │  │   │
│  │  │ • AXI Bus   │  │ • UNCLASS   │  │ • SECRET    │  │ • TOPSECRET│  │   │
│  │  │ • Clocking  │  │ • Waveform  │  │ • Waveform  │  │ • Waveform │  │   │
│  │  │ • Reset     │  │             │  │             │  │            │  │   │
│  │  │ • Firewall  │◄─┤             │◄─┤             │◄─┤            │  │   │
│  │  │             │  │             │  │             │  │            │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  │                         ▲                ▲                ▲         │   │
│  │                         │                │                │         │   │
│  │         ┌───────────────┴────────────────┴────────────────┴──────┐  │   │ 
│  │         │              AXI Firewall                              │  │   │
│  │         │   • Address filtering per partition                    │  │   │
│  │         │   • Transaction logging                                │  │   │
│  │         │   • Illegal access blocking                            │  │   │
│  │         └────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### AXI Firewall Configuration

```rust
/// FPGA partition isolation using AXI Firewall
pub struct FpgaPartition {
    /// Partition identifier
    pub id: u8,
    /// Base address in FPGA memory map
    pub base_addr: u64,
    /// Size of partition
    pub size: u64,
    /// Security level
    pub security_level: SecurityLevel,
    /// Allowed AXI masters
    pub allowed_masters: Vec<AxiMasterId>,
}

impl FpgaPartition {
    /// Configure AXI firewall rules for this partition
    pub fn configure_firewall(&self, fpga: &mut FpgaHandle) -> Result<()> {
        // Set address range
        fpga.write_reg(
            FIREWALL_BASE + self.id as u64 * FIREWALL_STRIDE + ADDR_LOW,
            self.base_addr,
        )?;
        fpga.write_reg(
            FIREWALL_BASE + self.id as u64 * FIREWALL_STRIDE + ADDR_HIGH,
            self.base_addr + self.size - 1,
        )?;

        // Configure allowed masters
        let mut master_mask: u32 = 0;
        for master in &self.allowed_masters {
            master_mask |= 1 << master.0;
        }
        fpga.write_reg(
            FIREWALL_BASE + self.id as u64 * FIREWALL_STRIDE + MASTER_ALLOW,
            master_mask as u64,
        )?;

        // Enable firewall
        fpga.write_reg(
            FIREWALL_BASE + self.id as u64 * FIREWALL_STRIDE + CTRL,
            FIREWALL_ENABLE | FIREWALL_LOG_VIOLATIONS,
        )?;

        Ok(())
    }
}
```

---

## Memory Protection

### Encrypted Memory Regions

```rust
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Secure buffer that is always encrypted in memory
#[derive(ZeroizeOnDrop)]
pub struct SecureBuffer {
    /// Encrypted data
    data: Vec<u8>,
    /// Encryption key (derived from hardware key)
    key: [u8; 32],
    /// Nonce for encryption
    nonce: [u8; 12],
}

impl SecureBuffer {
    /// Create new secure buffer
    pub fn new(size: usize) -> Result<Self> {
        // Derive key from hardware (TPM or CPU key)
        let key = derive_hardware_key()?;
        let nonce = generate_nonce();

        // Allocate and lock memory
        let mut data = vec![0u8; size + 16]; // +16 for auth tag
        mlock(&data)?;

        Ok(Self { data, key, nonce })
    }

    /// Write data (encrypts automatically)
    pub fn write(&mut self, plaintext: &[u8]) -> Result<()> {
        use aes_gcm::{Aes256Gcm, Key, Nonce};
        use aes_gcm::aead::{Aead, NewAead};

        let cipher = Aes256Gcm::new(Key::from_slice(&self.key));
        let nonce = Nonce::from_slice(&self.nonce);

        let ciphertext = cipher.encrypt(nonce, plaintext)
            .map_err(|_| Error::EncryptionFailed)?;

        self.data[..ciphertext.len()].copy_from_slice(&ciphertext);

        // Increment nonce
        increment_nonce(&mut self.nonce);

        Ok(())
    }

    /// Read data (decrypts automatically)
    pub fn read(&self, len: usize) -> Result<Vec<u8>> {
        use aes_gcm::{Aes256Gcm, Key, Nonce};
        use aes_gcm::aead::{Aead, NewAead};

        let cipher = Aes256Gcm::new(Key::from_slice(&self.key));
        // Use previous nonce for decryption
        let prev_nonce = decrement_nonce(&self.nonce);
        let nonce = Nonce::from_slice(&prev_nonce);

        let plaintext = cipher.decrypt(nonce, &self.data[..len + 16])
            .map_err(|_| Error::DecryptionFailed)?;

        Ok(plaintext)
    }
}

impl Drop for SecureBuffer {
    fn drop(&mut self) {
        // Zeroize is handled by derive macro, but also unlock
        munlock(&self.data).ok();
    }
}
```

### Guard Pages

```rust
/// Allocate buffer with guard pages
pub fn allocate_guarded(size: usize) -> Result<GuardedBuffer> {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;

    // Round up to page boundary
    let aligned_size = (size + page_size - 1) & !(page_size - 1);

    // Allocate: guard + data + guard
    let total_size = aligned_size + 2 * page_size;

    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            total_size,
            libc::PROT_NONE,  // Start with no permissions
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };

    if ptr == libc::MAP_FAILED {
        return Err(Error::MmapFailed(std::io::Error::last_os_error()));
    }

    // Make data region readable/writable
    let data_ptr = unsafe { ptr.add(page_size) };
    unsafe {
        libc::mprotect(
            data_ptr,
            aligned_size,
            libc::PROT_READ | libc::PROT_WRITE,
        );
    }

    // Guard pages remain PROT_NONE - any access triggers SIGSEGV

    Ok(GuardedBuffer {
        ptr: data_ptr,
        size: aligned_size,
        total_ptr: ptr,
        total_size,
    })
}
```

---

## Cross-Domain Solutions

### Multi-Level Security Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Cross-Domain Solution (CDS)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                      ┌─────────────────────┐                                │
│                      │    Guard Server     │                                │
│                      │                     │                                │
│                      │ • Policy Engine     │                                │
│                      │ • Content Filter    │                                │
│                      │ • Audit Logger      │                                │
│                      │ • Crypto Services   │                                │
│                      └──────────┬──────────┘                                │
│                                 │                                           │
│         ┌───────────────────────┼───────────────────────┐                   │
│         │                       │                       │                   │
│         ▼                       ▼                       ▼                   │
│  ┌──────────────┐       ┌──────────────┐       ┌──────────────┐             │
│  │  UNCLASS     │       │   SECRET     │       │  TOP SECRET  │             │
│  │  Domain      │       │   Domain     │       │   Domain     │             │
│  │              │       │              │       │              │             │
│  │ • ADS-B      │       │ • SINCGARS   │       │ • Link-16    │             │
│  │ • AIS        │       │ • P25        │       │ • HAVEQUICK  │             │
│  │ • Commercial │       │ • MIL-STD    │       │ • JTRS       │             │
│  └──────────────┘       └──────────────┘       └──────────────┘             │
│                                                                             │
│  Data Flow Rules:                                                           │
│  • UNCLASS ──► SECRET ──► TOP SECRET  (upward allowed)                      │
│  • TOP SECRET ──► SECRET ──► UNCLASS  (downward: guard filtered only)       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Guide

### r4w-sandbox Crate Structure

```
crates/r4w-sandbox/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── levels/
│   │   ├── mod.rs
│   │   ├── process.rs      # L1-L2: Process/namespace isolation
│   │   ├── lsm.rs          # L3: seccomp/SELinux/AppArmor
│   │   ├── container.rs    # L4: Docker/Podman integration
│   │   ├── microvm.rs      # L5: Firecracker/gVisor
│   │   ├── vm.rs           # L6: KVM/QEMU
│   │   └── hardware.rs     # L7: CPU pinning, NUMA, IOMMU
│   ├── memory/
│   │   ├── mod.rs
│   │   ├── secure.rs       # Encrypted buffers
│   │   ├── guarded.rs      # Guard pages
│   │   └── zeroize.rs      # Secure cleanup
│   ├── ipc/
│   │   ├── mod.rs
│   │   ├── channel.rs      # Secure IPC channels
│   │   └── shm.rs          # Isolated shared memory
│   ├── fpga/
│   │   ├── mod.rs
│   │   ├── partition.rs    # FPGA partitioning
│   │   └── firewall.rs     # AXI firewall config
│   └── policy/
│       ├── mod.rs
│       ├── seccomp.rs      # Seccomp profiles
│       ├── selinux.rs      # SELinux policy generation
│       └── apparmor.rs     # AppArmor profile generation
├── profiles/
│   ├── seccomp-dsp.json
│   ├── seccomp-crypto.json
│   └── apparmor-dsp.profile
└── tests/
    ├── isolation_tests.rs
    └── escape_tests.rs
```

### Basic Usage

```rust
use r4w_sandbox::{Sandbox, IsolationLevel};
use r4w_core::waveform::bpsk::Bpsk;

fn main() -> Result<()> {
    // Create isolated sandbox for BPSK waveform
    let sandbox = Sandbox::new(IsolationLevel::L3_LSM)?
        .name("waveform-bpsk")
        .memory_limit(256 * 1024 * 1024)
        .cpu_cores(&[2, 3])
        .seccomp_profile("dsp")
        .build()?;

    // Spawn waveform in sandbox
    let handle = sandbox.spawn(|| {
        let waveform = Bpsk::new(48000.0);
        // ... process samples
    })?;

    // Communicate via secure IPC
    let (tx, rx) = sandbox.create_channel::<WaveformCommand>()?;
    tx.send(WaveformCommand::SetFrequency(915_000_000))?;

    handle.wait()?;
    Ok(())
}
```

---

## Deployment Configurations

### Development (L1-L2)

```bash
# Minimal isolation for development
r4w sandbox --level process --waveform BPSK
```

### Production Single-Tenant (L3)

```bash
# Production with LSM enforcement
r4w sandbox --level lsm \
    --seccomp-profile dsp \
    --selinux-context r4w_production_t \
    --waveform SINCGARS
```

### Production Multi-Tenant (L4)

```bash
# Container-based multi-waveform
docker-compose -f docker-compose.isolation.yml up
```

### High-Security Multi-Tenant (L5)

```bash
# Firecracker microVM for each waveform
# Provides VM-level isolation with container-like startup times

# Create microVM for SECRET waveform
r4w sandbox --level microvm \
    --kernel /var/lib/r4w/vmlinux-dsp \
    --rootfs /var/lib/r4w/rootfs-sincgars.ext4 \
    --vcpus 2 \
    --memory 512M \
    --network tap-sincgars \
    --waveform SINCGARS

# Alternative: gVisor for OCI container isolation with syscall interception
docker run --runtime=runsc \
    --security-opt seccomp=seccomp-dsp.json \
    --cpus=2 --memory=512m \
    r4w/waveform:latest --waveform SINCGARS
```

```yaml
# firecracker-waveform.yaml - Firecracker configuration
machine-config:
  vcpu_count: 2
  mem_size_mib: 512
  smt: false
boot-source:
  kernel_image_path: /var/lib/r4w/vmlinux-dsp
  boot_args: "console=ttyS0 quiet r4w.waveform=SINCGARS r4w.level=SECRET"
drives:
  - drive_id: rootfs
    path_on_host: /var/lib/r4w/rootfs-sincgars.ext4
    is_root_device: true
    is_read_only: true
network-interfaces:
  - iface_id: eth0
    host_dev_name: tap-sincgars
    guest_mac: "06:00:AC:10:00:02"
```

### High-Security (L6-L7)

```bash
# VM with hardware isolation
r4w sandbox --level vm \
    --cpu-pinning 4,5,6,7 \
    --numa-node 1 \
    --iommu-group 12 \
    --waveform LINK16
```

---

## References

- [Linux Namespaces](https://man7.org/linux/man-pages/man7/namespaces.7.html)
- [Seccomp BPF](https://www.kernel.org/doc/html/latest/userspace-api/seccomp_filter.html)
- [SELinux](https://selinuxproject.org/)
- [AppArmor](https://apparmor.net/)
- [Firecracker MicroVMs](https://firecracker-microvm.github.io/)
- [Intel CAT](https://www.intel.com/content/www/us/en/developer/articles/technical/introduction-to-cache-allocation-technology.html)
- [AMD SEV](https://developer.amd.com/sev/)
- [VFIO](https://www.kernel.org/doc/html/latest/driver-api/vfio.html)
- [Cross Domain Solutions](https://www.nsa.gov/What-We-Do/Cybersecurity/Cross-Domain/)
