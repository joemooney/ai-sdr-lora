//! Channel Models for SDR Simulation
//!
//! This module provides various channel models to simulate realistic
//! wireless propagation effects in software.
//!
//! ## Channel Effects
//!
//! Real wireless channels introduce several impairments:
//!
//! 1. **AWGN (Additive White Gaussian Noise)**: Thermal noise
//! 2. **Path Loss**: Signal attenuation with distance
//! 3. **Fading**: Time-varying signal strength
//! 4. **Multipath**: Multiple signal copies arriving at different times
//! 5. **Frequency Offset**: Carrier frequency mismatch
//! 6. **Timing Drift**: Clock differences between TX and RX
//!
//! ## Usage
//!
//! ```rust
//! use r4w_sim::channel::{Channel, ChannelConfig, ChannelModel};
//! use r4w_core::types::Complex;
//!
//! let config = ChannelConfig {
//!     model: ChannelModel::Awgn,
//!     snr_db: 10.0,
//!     ..Default::default()
//! };
//!
//! // Some example I/Q samples
//! let clean_samples: Vec<Complex> = vec![Complex::new(1.0, 0.0); 100];
//!
//! let mut channel = Channel::new(config);
//! let noisy = channel.apply(&clean_samples);
//! ```

use crate::doppler::{DopplerGenerator, DopplerModel, JakesDoppler};
use r4w_core::types::{Complex, IQSample};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::f64::consts::PI;

/// Channel model type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelModel {
    /// Perfect channel (no impairments)
    Ideal,
    /// Additive White Gaussian Noise only
    Awgn,
    /// AWGN + frequency offset
    AwgnWithCfo,
    /// AWGN + multipath (2-ray model)
    Multipath,
    /// Rayleigh fading (for mobile scenarios)
    Rayleigh,
    /// Rician fading (line-of-sight + multipath)
    Rician,
    /// Tapped Delay Line with AWGN (frequency-selective fading)
    TdlAwgn,
    /// Time-varying fading with Jake's Doppler spectrum
    JakesFading,
    /// Full TDL with per-tap Doppler (frequency + time selective)
    FrequencySelective,
}

impl Default for ChannelModel {
    fn default() -> Self {
        Self::Awgn
    }
}

/// Standard TDL (Tapped Delay Line) channel profiles
///
/// These profiles are based on 3GPP specifications for modeling
/// different propagation environments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TdlProfile {
    /// Extended Pedestrian A (EPA): Low delay spread, pedestrian speeds
    #[default]
    Epa,
    /// Extended Vehicular A (EVA): Medium delay spread, vehicular speeds
    Eva,
    /// Extended Typical Urban (ETU): High delay spread, urban environment
    Etu,
    /// Custom TDL profile
    Custom,
}

impl TdlProfile {
    /// Get the standard tap configuration for this profile
    ///
    /// Returns (delay_ns, power_db) for each tap
    pub fn tap_config(&self) -> Vec<(f64, f64)> {
        match self {
            // EPA: Low delay spread (~45 ns RMS)
            TdlProfile::Epa => vec![
                (0.0, 0.0),      // Tap 0: Direct path
                (30.0, -1.0),    // Tap 1
                (70.0, -2.0),    // Tap 2
                (90.0, -3.0),    // Tap 3
                (110.0, -8.0),   // Tap 4
                (190.0, -17.2),  // Tap 5
                (410.0, -20.8),  // Tap 6
            ],
            // EVA: Medium delay spread (~357 ns RMS)
            TdlProfile::Eva => vec![
                (0.0, 0.0),       // Tap 0
                (30.0, -1.5),     // Tap 1
                (150.0, -1.4),    // Tap 2
                (310.0, -3.6),    // Tap 3
                (370.0, -0.6),    // Tap 4
                (710.0, -9.1),    // Tap 5
                (1090.0, -7.0),   // Tap 6
                (1730.0, -12.0),  // Tap 7
                (2510.0, -16.9),  // Tap 8
            ],
            // ETU: High delay spread (~991 ns RMS)
            TdlProfile::Etu => vec![
                (0.0, -1.0),       // Tap 0
                (50.0, -1.0),      // Tap 1
                (120.0, -1.0),     // Tap 2
                (200.0, 0.0),      // Tap 3
                (230.0, 0.0),      // Tap 4
                (500.0, 0.0),      // Tap 5
                (1600.0, -3.0),    // Tap 6
                (2300.0, -5.0),    // Tap 7
                (5000.0, -7.0),    // Tap 8
            ],
            TdlProfile::Custom => vec![(0.0, 0.0)], // Default single tap
        }
    }

    /// Get typical maximum Doppler frequency for this profile
    pub fn typical_doppler_hz(&self) -> f64 {
        match self {
            TdlProfile::Epa => 5.0,    // Walking speed
            TdlProfile::Eva => 70.0,   // Vehicular speed
            TdlProfile::Etu => 300.0,  // High-speed train
            TdlProfile::Custom => 0.0,
        }
    }
}

/// Single tap in a tapped delay line
#[derive(Debug, Clone)]
pub struct TdlTap {
    /// Delay in seconds
    pub delay_sec: f64,
    /// Power gain (linear scale, not dB)
    pub power: f64,
    /// Phase offset in radians
    pub phase: f64,
}

impl TdlTap {
    /// Create a new tap
    pub fn new(delay_sec: f64, power_db: f64) -> Self {
        Self {
            delay_sec,
            power: 10.0_f64.powf(power_db / 10.0),
            phase: 0.0,
        }
    }

    /// Create with random phase
    pub fn with_random_phase(delay_sec: f64, power_db: f64, rng: &mut StdRng) -> Self {
        use rand::Rng;
        Self {
            delay_sec,
            power: 10.0_f64.powf(power_db / 10.0),
            phase: rng.gen::<f64>() * 2.0 * PI,
        }
    }
}

/// Tapped Delay Line multipath channel
///
/// Implements a tapped delay line model with optional per-tap
/// Doppler fading for time-varying channels.
#[derive(Debug)]
pub struct TappedDelayLine {
    /// Configuration of each tap
    taps: Vec<TdlTap>,
    /// Delay line buffers (one per tap)
    delay_buffers: Vec<VecDeque<IQSample>>,
    /// Optional Doppler generators (one per tap for time-varying)
    doppler_generators: Vec<Option<JakesDoppler>>,
    /// Sample rate
    sample_rate: f64,
    /// Whether Doppler fading is enabled
    doppler_enabled: bool,
}

impl TappedDelayLine {
    /// Create a new TDL from a standard profile
    pub fn from_profile(profile: TdlProfile, sample_rate: f64) -> Self {
        let tap_config = profile.tap_config();
        let taps: Vec<TdlTap> = tap_config
            .iter()
            .map(|(delay_ns, power_db)| TdlTap::new(delay_ns * 1e-9, *power_db))
            .collect();

        Self::new(taps, sample_rate, false, 0.0)
    }

    /// Create a new TDL from a profile with Doppler
    pub fn from_profile_with_doppler(
        profile: TdlProfile,
        sample_rate: f64,
        max_doppler_hz: f64,
    ) -> Self {
        let tap_config = profile.tap_config();
        let taps: Vec<TdlTap> = tap_config
            .iter()
            .map(|(delay_ns, power_db)| TdlTap::new(delay_ns * 1e-9, *power_db))
            .collect();

        Self::new(taps, sample_rate, true, max_doppler_hz)
    }

    /// Create a new TDL with custom taps
    pub fn new(
        taps: Vec<TdlTap>,
        sample_rate: f64,
        doppler_enabled: bool,
        max_doppler_hz: f64,
    ) -> Self {
        // Create delay buffers sized for maximum delay
        let delay_buffers: Vec<VecDeque<IQSample>> = taps
            .iter()
            .map(|tap| {
                let delay_samples = (tap.delay_sec * sample_rate).ceil() as usize;
                let mut buffer = VecDeque::with_capacity(delay_samples + 1);
                buffer.resize(delay_samples + 1, IQSample::new(0.0, 0.0));
                buffer
            })
            .collect();

        // Create Doppler generators if enabled
        let doppler_generators: Vec<Option<JakesDoppler>> = if doppler_enabled && max_doppler_hz > 0.0 {
            taps.iter()
                .map(|_| Some(JakesDoppler::new(max_doppler_hz, sample_rate, 16)))
                .collect()
        } else {
            taps.iter().map(|_| None).collect()
        };

        Self {
            taps,
            delay_buffers,
            doppler_generators,
            sample_rate,
            doppler_enabled,
        }
    }

    /// Reset all delay buffers
    pub fn reset(&mut self) {
        for buffer in &mut self.delay_buffers {
            buffer.iter_mut().for_each(|s| *s = IQSample::new(0.0, 0.0));
        }
    }

    /// Get number of taps
    pub fn num_taps(&self) -> usize {
        self.taps.len()
    }

    /// Get RMS delay spread in seconds
    pub fn rms_delay_spread(&self) -> f64 {
        if self.taps.is_empty() {
            return 0.0;
        }

        let total_power: f64 = self.taps.iter().map(|t| t.power).sum();
        if total_power == 0.0 {
            return 0.0;
        }

        let mean_delay: f64 = self.taps.iter()
            .map(|t| t.delay_sec * t.power)
            .sum::<f64>() / total_power;

        let mean_sq_delay: f64 = self.taps.iter()
            .map(|t| t.delay_sec * t.delay_sec * t.power)
            .sum::<f64>() / total_power;

        (mean_sq_delay - mean_delay * mean_delay).sqrt()
    }

    /// Get coherence bandwidth in Hz (approximately 1 / (5 * RMS delay))
    pub fn coherence_bandwidth(&self) -> f64 {
        let rms = self.rms_delay_spread();
        if rms > 0.0 {
            1.0 / (5.0 * rms)
        } else {
            f64::INFINITY
        }
    }

    /// Apply the TDL to input samples
    pub fn apply(&mut self, samples: &[IQSample]) -> Vec<IQSample> {
        let mut output = Vec::with_capacity(samples.len());

        for &sample in samples {
            let mut total = IQSample::new(0.0, 0.0);

            for (i, tap) in self.taps.iter().enumerate() {
                // Push sample into delay buffer
                self.delay_buffers[i].push_back(sample);
                let delayed = self.delay_buffers[i].pop_front()
                    .unwrap_or(IQSample::new(0.0, 0.0));

                // Apply tap gain and phase
                let phase_rot = IQSample::new(tap.phase.cos(), tap.phase.sin());
                let mut tap_output = IQSample::new(
                    delayed.re * phase_rot.re - delayed.im * phase_rot.im,
                    delayed.re * phase_rot.im + delayed.im * phase_rot.re,
                ) * tap.power.sqrt();

                // Apply Doppler fading if enabled
                if let Some(ref mut doppler) = self.doppler_generators[i] {
                    let fade = doppler.next_sample();
                    tap_output = IQSample::new(
                        tap_output.re * fade.re - tap_output.im * fade.im,
                        tap_output.re * fade.im + tap_output.im * fade.re,
                    );
                }

                total = total + tap_output;
            }

            output.push(total);
        }

        output
    }
}

/// Channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    /// Channel model to use
    pub model: ChannelModel,
    /// Target SNR in dB
    pub snr_db: f64,
    /// Carrier frequency offset in Hz
    pub cfo_hz: f64,
    /// Clock drift in PPM (parts per million)
    pub clock_drift_ppm: f64,
    /// Multipath delay in samples
    pub multipath_delay: usize,
    /// Multipath relative amplitude (0-1)
    pub multipath_amplitude: f64,
    /// Rician K-factor (ratio of LOS to scattered power)
    pub rician_k: f64,
    /// Sample rate (needed for some calculations)
    pub sample_rate: f64,
    /// Path loss in dB
    pub path_loss_db: f64,

    // === Doppler/TDL Configuration ===
    /// Enable Doppler fading (time-varying channel)
    #[serde(default)]
    pub doppler_enabled: bool,
    /// Maximum Doppler frequency in Hz
    #[serde(default)]
    pub max_doppler_hz: f64,
    /// Velocity in m/s (alternative to max_doppler_hz)
    #[serde(default)]
    pub velocity_mps: f64,
    /// Carrier frequency in Hz (used to calculate Doppler from velocity)
    #[serde(default = "default_carrier_freq")]
    pub carrier_frequency_hz: f64,
    /// Doppler spectrum model
    #[serde(default)]
    pub doppler_model: DopplerModelConfig,
    /// Enable tapped delay line multipath
    #[serde(default)]
    pub tdl_enabled: bool,
    /// TDL channel profile
    #[serde(default)]
    pub tdl_profile: TdlProfile,
}

fn default_carrier_freq() -> f64 {
    915_000_000.0 // Default to 915 MHz ISM band
}

/// Doppler model configuration (serializable wrapper)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DopplerModelConfig {
    /// Jake's/Clarke's classic model
    #[default]
    Jakes,
    /// Flat Doppler spectrum
    Flat,
    /// Gaussian Doppler spectrum
    Gaussian,
    /// No Doppler (static)
    Static,
}

impl From<DopplerModelConfig> for DopplerModel {
    fn from(config: DopplerModelConfig) -> Self {
        match config {
            DopplerModelConfig::Jakes => DopplerModel::Jakes,
            DopplerModelConfig::Flat => DopplerModel::Flat,
            DopplerModelConfig::Gaussian => DopplerModel::Gaussian,
            DopplerModelConfig::Static => DopplerModel::Static,
        }
    }
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            model: ChannelModel::Awgn,
            snr_db: 20.0,
            cfo_hz: 0.0,
            clock_drift_ppm: 0.0,
            multipath_delay: 0,
            multipath_amplitude: 0.0,
            rician_k: 10.0,
            sample_rate: 125_000.0,
            path_loss_db: 0.0,
            // Doppler/TDL defaults
            doppler_enabled: false,
            max_doppler_hz: 0.0,
            velocity_mps: 0.0,
            carrier_frequency_hz: default_carrier_freq(),
            doppler_model: DopplerModelConfig::default(),
            tdl_enabled: false,
            tdl_profile: TdlProfile::default(),
        }
    }
}

impl ChannelConfig {
    /// Create a clean channel with only specified SNR
    pub fn with_snr(snr_db: f64) -> Self {
        Self {
            snr_db,
            ..Default::default()
        }
    }

    /// Create a channel with CFO
    pub fn with_cfo(snr_db: f64, cfo_hz: f64) -> Self {
        Self {
            model: ChannelModel::AwgnWithCfo,
            snr_db,
            cfo_hz,
            ..Default::default()
        }
    }

    /// Create a multipath channel
    pub fn multipath(snr_db: f64, delay_samples: usize, amplitude: f64) -> Self {
        Self {
            model: ChannelModel::Multipath,
            snr_db,
            multipath_delay: delay_samples,
            multipath_amplitude: amplitude,
            ..Default::default()
        }
    }

    /// Create a TDL channel with standard profile
    pub fn tdl(snr_db: f64, profile: TdlProfile, sample_rate: f64) -> Self {
        Self {
            model: ChannelModel::TdlAwgn,
            snr_db,
            sample_rate,
            tdl_enabled: true,
            tdl_profile: profile,
            ..Default::default()
        }
    }

    /// Create a TDL channel with Doppler fading
    pub fn tdl_with_doppler(
        snr_db: f64,
        profile: TdlProfile,
        sample_rate: f64,
        max_doppler_hz: f64,
    ) -> Self {
        Self {
            model: ChannelModel::FrequencySelective,
            snr_db,
            sample_rate,
            tdl_enabled: true,
            tdl_profile: profile,
            doppler_enabled: true,
            max_doppler_hz,
            ..Default::default()
        }
    }

    /// Create a Jake's fading channel (flat fading with Doppler)
    pub fn jakes_fading(snr_db: f64, max_doppler_hz: f64, sample_rate: f64) -> Self {
        Self {
            model: ChannelModel::JakesFading,
            snr_db,
            sample_rate,
            doppler_enabled: true,
            max_doppler_hz,
            doppler_model: DopplerModelConfig::Jakes,
            ..Default::default()
        }
    }

    /// Get effective Doppler frequency (from velocity if specified)
    pub fn effective_doppler_hz(&self) -> f64 {
        if self.max_doppler_hz > 0.0 {
            self.max_doppler_hz
        } else if self.velocity_mps > 0.0 {
            crate::doppler::velocity_to_doppler(self.velocity_mps, self.carrier_frequency_hz)
        } else {
            0.0
        }
    }
}

/// Channel simulator
#[derive(Debug)]
pub struct Channel {
    config: ChannelConfig,
    rng: StdRng,
    /// Phase accumulator for CFO simulation
    cfo_phase: f64,
    /// Sample counter for timing drift
    sample_count: u64,
    /// Multipath history buffer
    multipath_buffer: Vec<IQSample>,
    /// Tapped delay line (optional)
    tdl: Option<TappedDelayLine>,
    /// Doppler generator for flat fading (optional)
    doppler: Option<DopplerGenerator>,
}

impl Channel {
    /// Create a new channel with the given configuration
    pub fn new(config: ChannelConfig) -> Self {
        let multipath_buffer = vec![Complex::new(0.0, 0.0); config.multipath_delay + 1];

        // Create TDL if enabled
        let tdl = if config.tdl_enabled {
            let doppler_hz = config.effective_doppler_hz();
            if config.doppler_enabled && doppler_hz > 0.0 {
                Some(TappedDelayLine::from_profile_with_doppler(
                    config.tdl_profile,
                    config.sample_rate,
                    doppler_hz,
                ))
            } else {
                Some(TappedDelayLine::from_profile(
                    config.tdl_profile,
                    config.sample_rate,
                ))
            }
        } else {
            None
        };

        // Create Doppler generator for flat fading (JakesFading model)
        let doppler = if config.doppler_enabled && !config.tdl_enabled {
            let doppler_hz = config.effective_doppler_hz();
            if doppler_hz > 0.0 {
                Some(DopplerGenerator::new(
                    config.doppler_model.into(),
                    doppler_hz,
                    config.sample_rate,
                    16,
                ))
            } else {
                None
            }
        } else {
            None
        };

        Self {
            config,
            rng: StdRng::from_entropy(),
            cfo_phase: 0.0,
            sample_count: 0,
            multipath_buffer,
            tdl,
            doppler,
        }
    }

    /// Reset channel state
    pub fn reset(&mut self) {
        self.cfo_phase = 0.0;
        self.sample_count = 0;
        self.multipath_buffer.fill(Complex::new(0.0, 0.0));
        if let Some(ref mut tdl) = self.tdl {
            tdl.reset();
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &ChannelConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: ChannelConfig) {
        // Recreate the channel with new config
        *self = Self::new(config);
    }

    /// Apply channel effects to samples
    pub fn apply(&mut self, samples: &[IQSample]) -> Vec<IQSample> {
        match self.config.model {
            ChannelModel::Ideal => samples.to_vec(),
            ChannelModel::Awgn => self.apply_awgn(samples),
            ChannelModel::AwgnWithCfo => {
                let with_cfo = self.apply_cfo(samples);
                self.apply_awgn(&with_cfo)
            }
            ChannelModel::Multipath => {
                let with_multipath = self.apply_multipath(samples);
                self.apply_awgn(&with_multipath)
            }
            ChannelModel::Rayleigh => self.apply_rayleigh(samples),
            ChannelModel::Rician => self.apply_rician(samples),
            ChannelModel::TdlAwgn => {
                let with_tdl = self.apply_tdl(samples);
                self.apply_awgn(&with_tdl)
            }
            ChannelModel::JakesFading => {
                let with_fading = self.apply_doppler_fading(samples);
                self.apply_awgn(&with_fading)
            }
            ChannelModel::FrequencySelective => {
                // TDL with per-tap Doppler + AWGN
                let with_tdl = self.apply_tdl(samples);
                self.apply_awgn(&with_tdl)
            }
        }
    }

    /// Apply TDL multipath
    fn apply_tdl(&mut self, samples: &[IQSample]) -> Vec<IQSample> {
        if let Some(ref mut tdl) = self.tdl {
            tdl.apply(samples)
        } else {
            samples.to_vec()
        }
    }

    /// Apply Doppler fading (flat, no multipath)
    fn apply_doppler_fading(&mut self, samples: &[IQSample]) -> Vec<IQSample> {
        if let Some(ref mut doppler) = self.doppler {
            doppler.apply(samples)
        } else {
            samples.to_vec()
        }
    }

    /// Apply AWGN (Additive White Gaussian Noise)
    fn apply_awgn(&mut self, samples: &[IQSample]) -> Vec<IQSample> {
        // Calculate noise power from SNR
        // SNR = signal_power / noise_power
        // noise_power = signal_power / 10^(SNR_dB/10)

        // Estimate signal power (assume normalized input)
        let signal_power: f64 = samples.iter().map(|s| s.norm_sqr()).sum::<f64>() / samples.len() as f64;

        // Apply path loss
        let path_loss_linear = 10.0_f64.powf(-self.config.path_loss_db / 20.0);

        // Calculate noise standard deviation
        let snr_linear = 10.0_f64.powf(self.config.snr_db / 10.0);
        let noise_power = signal_power / snr_linear;
        let noise_std = (noise_power / 2.0).sqrt(); // Divide by 2 for I and Q

        // Generate noise distribution
        let noise_dist = Normal::new(0.0, noise_std).unwrap();

        samples
            .iter()
            .map(|&s| {
                // Apply path loss and add noise
                let attenuated = s * path_loss_linear;
                let noise = Complex::new(
                    noise_dist.sample(&mut self.rng),
                    noise_dist.sample(&mut self.rng),
                );
                attenuated + noise
            })
            .collect()
    }

    /// Apply carrier frequency offset
    fn apply_cfo(&mut self, samples: &[IQSample]) -> Vec<IQSample> {
        let cfo_rad_per_sample = 2.0 * PI * self.config.cfo_hz / self.config.sample_rate;

        samples
            .iter()
            .map(|&s| {
                let rotation = Complex::new(self.cfo_phase.cos(), self.cfo_phase.sin());
                self.cfo_phase += cfo_rad_per_sample;

                // Keep phase in reasonable range
                if self.cfo_phase > 2.0 * PI {
                    self.cfo_phase -= 2.0 * PI;
                }

                s * rotation
            })
            .collect()
    }

    /// Apply multipath (2-ray model)
    fn apply_multipath(&mut self, samples: &[IQSample]) -> Vec<IQSample> {
        let delay = self.config.multipath_delay;
        let amp = self.config.multipath_amplitude;

        if delay == 0 || amp == 0.0 {
            return samples.to_vec();
        }

        let mut output = Vec::with_capacity(samples.len());

        for &sample in samples {
            // Shift buffer
            self.multipath_buffer.remove(0);
            self.multipath_buffer.push(sample);

            // Sum direct path and delayed path
            let direct = sample;
            let delayed = self.multipath_buffer[0] * amp;
            output.push(direct + delayed);
        }

        output
    }

    /// Apply Rayleigh fading
    fn apply_rayleigh(&mut self, samples: &[IQSample]) -> Vec<IQSample> {
        let noise_dist = Normal::new(0.0, 1.0 / 2.0_f64.sqrt()).unwrap();

        samples
            .iter()
            .map(|&s| {
                // Rayleigh fading coefficient: complex Gaussian
                let h = Complex::new(
                    noise_dist.sample(&mut self.rng),
                    noise_dist.sample(&mut self.rng),
                );
                s * h
            })
            .collect()
    }

    /// Apply Rician fading
    fn apply_rician(&mut self, samples: &[IQSample]) -> Vec<IQSample> {
        let k = self.config.rician_k;

        // LOS component amplitude
        let los_amp = (k / (k + 1.0)).sqrt();
        // Scattered component amplitude
        let scatter_amp = (1.0 / (k + 1.0)).sqrt();

        let noise_dist = Normal::new(0.0, scatter_amp / 2.0_f64.sqrt()).unwrap();

        samples
            .iter()
            .map(|&s| {
                // Rician = LOS + scattered (Rayleigh)
                let los = Complex::new(los_amp, 0.0);
                let scattered = Complex::new(
                    noise_dist.sample(&mut self.rng),
                    noise_dist.sample(&mut self.rng),
                );
                s * (los + scattered)
            })
            .collect()
    }

    /// Calculate theoretical BER for AWGN channel
    pub fn theoretical_ber_awgn(snr_db: f64, spreading_factor: u8) -> f64 {
        // Simplified approximation for LoRa
        let snr_linear = 10.0_f64.powf(snr_db / 10.0);
        let m = 2.0_f64.powi(spreading_factor as i32);

        // Q-function approximation
        let arg = (2.0 * snr_linear * spreading_factor as f64 / m).sqrt();
        0.5 * (-arg * arg / 2.0).exp()
    }
}

/// Generate channel statistics for visualization
#[derive(Debug, Clone)]
pub struct ChannelStats {
    /// Mean signal power
    pub signal_power: f64,
    /// Mean noise power
    pub noise_power: f64,
    /// Measured SNR in dB
    pub measured_snr_db: f64,
    /// Peak to average power ratio
    pub papr_db: f64,
}

impl ChannelStats {
    /// Compute statistics from samples
    pub fn compute(clean: &[IQSample], noisy: &[IQSample]) -> Self {
        let signal_power: f64 = clean.iter().map(|s| s.norm_sqr()).sum::<f64>() / clean.len() as f64;

        // Estimate noise by differencing
        let noise_power: f64 = clean
            .iter()
            .zip(noisy.iter())
            .map(|(c, n)| (n - c).norm_sqr())
            .sum::<f64>()
            / clean.len() as f64;

        let measured_snr_db = 10.0 * (signal_power / noise_power).log10();

        let peak_power = noisy.iter().map(|s| s.norm_sqr()).fold(0.0_f64, f64::max);
        let avg_power: f64 = noisy.iter().map(|s| s.norm_sqr()).sum::<f64>() / noisy.len() as f64;
        let papr_db = 10.0 * (peak_power / avg_power).log10();

        Self {
            signal_power,
            noise_power,
            measured_snr_db,
            papr_db,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_awgn_channel() {
        let config = ChannelConfig::with_snr(20.0);
        let mut channel = Channel::new(config);

        // Create test signal (constant amplitude)
        let samples: Vec<IQSample> = (0..1000)
            .map(|_| Complex::new(1.0, 0.0))
            .collect();

        let noisy = channel.apply(&samples);

        // Output should be same length
        assert_eq!(noisy.len(), samples.len());

        // Check that noise was added (samples shouldn't be identical)
        let diff: f64 = samples
            .iter()
            .zip(noisy.iter())
            .map(|(a, b)| (a - b).norm())
            .sum();
        assert!(diff > 0.0);
    }

    #[test]
    fn test_cfo_channel() {
        let config = ChannelConfig::with_cfo(100.0, 1000.0); // 1 kHz offset
        let mut channel = Channel::new(config);

        let samples: Vec<IQSample> = (0..1000)
            .map(|_| Complex::new(1.0, 0.0))
            .collect();

        let output = channel.apply(&samples);

        // Phase should rotate between consecutive samples
        // With 1kHz CFO and 125kHz sample rate: 2π * 1000 / 125000 = 0.05 rad/sample
        let phase_diff = (output[1].arg() - output[0].arg()).abs();
        assert!(phase_diff > 0.01, "Phase should rotate between samples, got diff = {}", phase_diff);

        // Verify all samples have same magnitude (CFO only rotates phase)
        for sample in &output {
            assert!((sample.norm() - 1.0).abs() < 0.01, "Magnitude should be preserved");
        }
    }

    #[test]
    fn test_ideal_channel() {
        let config = ChannelConfig {
            model: ChannelModel::Ideal,
            ..Default::default()
        };
        let mut channel = Channel::new(config);

        let samples: Vec<IQSample> = (0..100)
            .map(|i| Complex::new(i as f64, 0.0))
            .collect();

        let output = channel.apply(&samples);

        // Ideal channel should pass through unchanged
        assert_eq!(samples, output);
    }
}
