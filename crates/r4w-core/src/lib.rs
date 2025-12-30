//! # LoRa Core DSP Library
//!
//! This crate provides core Digital Signal Processing (DSP) algorithms for
//! implementing LoRa (Long Range) modulation and demodulation in software.
//!
//! ## Overview
//!
//! LoRa uses Chirp Spread Spectrum (CSS) modulation, which provides excellent
//! noise immunity and long-range communication at low data rates. This library
//! implements the full LoRa PHY layer including:
//!
//! - **Chirp Generation**: Create up-chirps and down-chirps for modulation
//! - **CSS Modulation**: Convert symbols to chirped waveforms
//! - **CSS Demodulation**: Extract symbols from received I/Q samples
//! - **Synchronization**: Preamble detection, CFO estimation, timing recovery
//! - **Coding**: Whitening, interleaving, Hamming FEC, Gray coding
//!
//! ## Signal Flow
//!
//! ```text
//! TX: Data → Whitening → Hamming FEC → Interleave → Gray Code → CSS Mod → I/Q
//! RX: I/Q → Sync → CSS Demod → Gray Decode → De-interleave → FEC → De-whiten → Data
//! ```
//!
//! ## Example
//!
//! ```rust,no_run
//! use r4w_core::{LoRaParams, Modulator, Demodulator};
//!
//! // Configure LoRa parameters using the builder pattern
//! let params = LoRaParams::builder()
//!     .spreading_factor(7)
//!     .bandwidth(125_000)
//!     .coding_rate(1)
//!     .build();
//!
//! // Create modulator and generate a packet
//! let mut modulator = Modulator::new(params.clone());
//! let payload = b"Hello LoRa!";
//! let samples = modulator.modulate(payload);
//!
//! // Demodulate (in real use, these would be received I/Q samples)
//! let mut demodulator = Demodulator::new(params);
//! let decoded = demodulator.demodulate(&samples);
//! ```

pub mod agent;
pub mod anti_jam;
pub mod benchmark;
pub mod chirp;
pub mod coding;
pub mod config;
pub mod demodulation;
pub mod fft_utils;
pub mod filters;
pub mod gps_time;
pub mod lpi_metrics;
pub mod modulation;
pub mod observe;
pub mod packet;
pub mod params;
pub mod plugin;
pub mod rt;
pub mod simd_utils;
pub mod spreading;
pub mod scheduler;
pub mod rt_scheduler;
pub mod sync;
pub mod synthesizer;
pub mod time_sync;
pub mod timing;
pub mod types;
pub mod waveform;
pub mod whitening;

// Mesh networking support
pub mod mesh;

// Parallel processing (requires `parallel` feature)
#[cfg(feature = "parallel")]
pub mod parallel;

// FPGA hardware acceleration (requires `fpga` feature)
#[cfg(feature = "fpga")]
pub mod fpga_accel;

// Re-export main types
pub use chirp::{ChirpGenerator, ChirpType};
pub use coding::{GrayCode, HammingCode, Interleaver};
pub use demodulation::Demodulator;
pub use modulation::Modulator;
pub use packet::{LoRaPacket, PacketHeader};
pub use params::{CodingRate, LoRaParams, SpreadingFactor};
pub use sync::{PreambleDetector, Synchronizer};
pub use types::{Complex, IQSample, Sample};
pub use waveform::{CommonParams, DemodResult, Waveform, WaveformFactory, WaveformInfo, VisualizationData};
pub use whitening::Whitening;

// Mesh networking re-exports
pub use mesh::{
    MeshNetwork, MeshPhy, MeshPacket, NodeId, FloodRouter, MacLayer,
    LoRaMesh, LoRaMeshPhy, LoRaMeshConfig, MeshtasticNode, ModemPreset, Region,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::chirp::{ChirpGenerator, ChirpType};
    pub use crate::demodulation::Demodulator;
    pub use crate::modulation::Modulator;
    pub use crate::params::{CodingRate, LoRaParams, SpreadingFactor};
    pub use crate::types::{Complex, IQSample};
    // Mesh networking
    pub use crate::mesh::{MeshNetwork, MeshPhy, MeshPacket, NodeId, LoRaMesh, LoRaMeshConfig};
}
