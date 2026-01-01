//! Signal Analysis Module
//!
//! This module provides comprehensive signal analysis tools for I/Q signals,
//! including spectrum analysis, waterfall/spectrogram generation, statistics
//! computation, and peak detection.
//!
//! ## Features
//!
//! - **Spectrum Analysis**: FFT-based power spectrum with windowing
//! - **Waterfall Display**: Time-frequency spectrogram with PNG/ASCII output
//! - **Signal Statistics**: Power, SNR, PAPR, DC offset, bandwidth estimation
//! - **Peak Detection**: Find spectral peaks above threshold
//!
//! ## Example
//!
//! ```rust,no_run
//! use r4w_core::analysis::{SpectrumAnalyzer, SignalStats};
//! use r4w_core::types::IQSample;
//!
//! let samples: Vec<IQSample> = vec![]; // Your I/Q samples
//! let sample_rate = 1_000_000.0;
//!
//! // Compute spectrum
//! let analyzer = SpectrumAnalyzer::new(1024);
//! let spectrum = analyzer.compute(&samples);
//!
//! // Compute statistics
//! let stats = SignalStats::compute(&samples, Some(sample_rate));
//! println!("Mean power: {:.2} dBFS", stats.mean_power_dbfs);
//! ```

pub mod peaks;
pub mod spectrum;
pub mod statistics;
pub mod waterfall;

pub use peaks::{PeakFinder, SpectralPeak};
pub use spectrum::{SpectrumAnalyzer, SpectrumResult, WindowFunction};
pub use statistics::{IQImbalance, SignalStats};
pub use waterfall::{Colormap, WaterfallGenerator, WaterfallResult};
