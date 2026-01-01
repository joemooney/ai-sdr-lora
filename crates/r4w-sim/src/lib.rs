//! # R4W SDR Hardware Abstraction
//!
//! This crate provides a hardware abstraction layer for Software Defined Radio
//! devices, enabling waveform transmission and reception on various platforms.
//!
//! ## Supported Devices
//!
//! - **Simulator**: Pure software simulation for testing and learning
//! - **USRP** (via UHD): Ettus Research radios (B200, B210, X310, etc.)
//! - **SoapySDR**: Generic interface supporting HackRF, RTL-SDR, LimeSDR, etc.
//! - **File I/O**: SigMF file reading and writing
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    Application Layer                    │
//! │               (r4w-gui, r4w-cli, custom)                │
//! └─────────────────────────────────────────────────────────┘
//!                            │
//!                            ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │                  HAL Interface (Rust traits)            │
//! │   SdrDevice, StreamHandle, TunerControl, ClockControl   │
//! └─────────────────────────────────────────────────────────┘
//!         │          │          │          │
//!         ▼          ▼          ▼          ▼
//!   ┌──────────┐ ┌──────┐ ┌──────────┐ ┌──────────┐
//!   │Simulator │ │File  │ │   USRP   │ │ SoapySDR │
//!   │          │ │I/O   │ │(via UHD) │ │ (many)   │
//!   └──────────┘ └──────┘ └──────────┘ └──────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use r4w_sim::{SdrDevice, Simulator, SdrConfig};
//!
//! // Create a simulated SDR for testing
//! let config = SdrConfig::default();
//! let mut sdr = Simulator::new(config);
//!
//! // Start receiving
//! sdr.start_rx()?;
//!
//! // Get samples
//! let samples = sdr.read_samples(1024)?;
//! ```

pub mod channel;
pub mod device;
pub mod doppler;
pub mod hal;
pub mod simulator;

// Re-exports
pub use channel::{Channel, ChannelConfig, ChannelModel, TappedDelayLine, TdlProfile, TdlTap, DopplerModelConfig};
pub use device::{SdrConfig, SdrDevice, SdrError, SdrResult};
pub use doppler::{DopplerGenerator, DopplerModel, JakesDoppler, velocity_to_doppler};
pub use hal::{ClockControl, ClockSource, DriverRegistry, SampleFormat, SdrDeviceExt, StreamConfig, StreamDirection, StreamHandle, StreamStatus, TunerControl};
pub use simulator::Simulator;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::channel::{Channel, ChannelConfig, ChannelModel};
    pub use crate::device::{SdrConfig, SdrDevice};
    pub use crate::simulator::Simulator;
}
