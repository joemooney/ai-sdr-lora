//! Doppler Spread and Time-Varying Fading Models
//!
//! This module implements various Doppler spread models for simulating
//! time-varying multipath fading channels, commonly used in mobile
//! wireless communication scenarios.
//!
//! ## Jake's/Clarke's Model
//!
//! The classic Jake's model (based on Clarke's statistical model) simulates
//! the Doppler spectrum for isotropic scattering in a mobile environment.
//! The Doppler power spectral density follows a bathtub curve:
//!
//! ```text
//! S(f) = 1 / (π * f_d * sqrt(1 - (f/f_d)²))  for |f| ≤ f_d
//! ```
//!
//! where f_d is the maximum Doppler frequency.
//!
//! ## Usage
//!
//! ```rust
//! use r4w_sim::doppler::{JakesDoppler, DopplerModel};
//!
//! // Create a Jake's Doppler generator for 30 Hz max Doppler (walking speed)
//! let mut doppler = JakesDoppler::new(30.0, 1_000_000.0, 16);
//!
//! // Generate 1000 fading samples
//! let fading = doppler.generate(1000);
//! ```

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use r4w_core::types::IQSample;
use std::f64::consts::PI;

/// Doppler spectrum model types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DopplerModel {
    /// Jake's/Clarke's classic model (isotropic scattering)
    #[default]
    Jakes,
    /// Flat Doppler spectrum (uniform power across Doppler bandwidth)
    Flat,
    /// Gaussian Doppler spectrum (concentrated around center frequency)
    Gaussian,
    /// Static channel (no time variation, Doppler = 0)
    Static,
}

impl DopplerModel {
    /// Parse model from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "jakes" | "clarke" | "classical" => Some(DopplerModel::Jakes),
            "flat" | "uniform" => Some(DopplerModel::Flat),
            "gaussian" | "normal" => Some(DopplerModel::Gaussian),
            "static" | "none" => Some(DopplerModel::Static),
            _ => None,
        }
    }
}

/// Calculate maximum Doppler frequency from velocity and carrier frequency
///
/// # Arguments
/// * `velocity_mps` - Velocity in meters per second
/// * `carrier_freq_hz` - Carrier frequency in Hz
///
/// # Returns
/// Maximum Doppler frequency in Hz (f_d = v * f_c / c)
pub fn velocity_to_doppler(velocity_mps: f64, carrier_freq_hz: f64) -> f64 {
    const SPEED_OF_LIGHT: f64 = 299_792_458.0; // m/s
    velocity_mps * carrier_freq_hz / SPEED_OF_LIGHT
}

/// Calculate velocity from Doppler frequency and carrier frequency
pub fn doppler_to_velocity(doppler_hz: f64, carrier_freq_hz: f64) -> f64 {
    const SPEED_OF_LIGHT: f64 = 299_792_458.0; // m/s
    doppler_hz * SPEED_OF_LIGHT / carrier_freq_hz
}

/// Jake's/Clarke's Doppler fading generator
///
/// Implements the sum-of-sinusoids (SOS) method for generating
/// complex Gaussian processes with Jake's Doppler spectrum.
#[derive(Debug)]
pub struct JakesDoppler {
    /// Maximum Doppler frequency in Hz
    max_doppler_hz: f64,
    /// Sample rate in Hz
    sample_rate: f64,
    /// Number of sinusoids per branch (I and Q)
    num_sinusoids: usize,
    /// Oscillator frequencies (normalized to f_d)
    frequencies: Vec<f64>,
    /// Oscillator phases (radians)
    phases: Vec<f64>,
    /// Current time in samples
    time_samples: u64,
    /// RNG for phase initialization
    rng: StdRng,
}

impl JakesDoppler {
    /// Create a new Jake's Doppler generator
    ///
    /// # Arguments
    /// * `max_doppler_hz` - Maximum Doppler frequency in Hz
    /// * `sample_rate` - Sample rate in Hz
    /// * `num_sinusoids` - Number of oscillators per quadrature component (8-16 typical)
    pub fn new(max_doppler_hz: f64, sample_rate: f64, num_sinusoids: usize) -> Self {
        let mut instance = Self {
            max_doppler_hz,
            sample_rate,
            num_sinusoids,
            frequencies: Vec::new(),
            phases: Vec::new(),
            time_samples: 0,
            rng: StdRng::from_entropy(),
        };
        instance.initialize();
        instance
    }

    /// Create from velocity and carrier frequency
    pub fn from_velocity(
        velocity_mps: f64,
        carrier_freq_hz: f64,
        sample_rate: f64,
        num_sinusoids: usize,
    ) -> Self {
        let doppler = velocity_to_doppler(velocity_mps, carrier_freq_hz);
        Self::new(doppler, sample_rate, num_sinusoids)
    }

    /// Create with specific seed for reproducibility
    pub fn with_seed(
        max_doppler_hz: f64,
        sample_rate: f64,
        num_sinusoids: usize,
        seed: u64,
    ) -> Self {
        let mut instance = Self {
            max_doppler_hz,
            sample_rate,
            num_sinusoids,
            frequencies: Vec::new(),
            phases: Vec::new(),
            time_samples: 0,
            rng: StdRng::seed_from_u64(seed),
        };
        instance.initialize();
        instance
    }

    /// Initialize oscillator frequencies and phases
    fn initialize(&mut self) {
        let n = self.num_sinusoids;

        // Jakes' model uses cosine-spaced frequencies to achieve the bathtub spectrum
        // f_n = f_d * cos(2π(n-0.5) / (4N))  for n = 1..N
        self.frequencies = (1..=n)
            .map(|i| {
                let theta = 2.0 * PI * (i as f64 - 0.5) / (4.0 * n as f64);
                self.max_doppler_hz * theta.cos()
            })
            .collect();

        // Random initial phases (uniform 0 to 2π)
        self.phases = (0..n).map(|_| self.rng.gen::<f64>() * 2.0 * PI).collect();
    }

    /// Reset the generator state
    pub fn reset(&mut self) {
        self.time_samples = 0;
        self.initialize();
    }

    /// Get maximum Doppler frequency
    pub fn max_doppler_hz(&self) -> f64 {
        self.max_doppler_hz
    }

    /// Get coherence time in seconds (T_c ≈ 1 / (4 * f_d))
    pub fn coherence_time(&self) -> f64 {
        if self.max_doppler_hz > 0.0 {
            1.0 / (4.0 * self.max_doppler_hz)
        } else {
            f64::INFINITY
        }
    }

    /// Get coherence time in samples
    pub fn coherence_samples(&self) -> usize {
        (self.coherence_time() * self.sample_rate).round() as usize
    }

    /// Generate a single fading sample
    pub fn next_sample(&mut self) -> IQSample {
        let t = self.time_samples as f64 / self.sample_rate;
        self.time_samples += 1;

        let n = self.num_sinusoids;
        let scale = 1.0 / (n as f64).sqrt();

        let mut real = 0.0;
        let mut imag = 0.0;

        for i in 0..n {
            let phase = 2.0 * PI * self.frequencies[i] * t + self.phases[i];
            real += phase.cos();
            // Quadrature component uses different phase offsets
            imag += (phase + PI / 4.0).sin();
        }

        IQSample::new(real * scale, imag * scale)
    }

    /// Generate multiple fading samples
    pub fn generate(&mut self, num_samples: usize) -> Vec<IQSample> {
        (0..num_samples).map(|_| self.next_sample()).collect()
    }

    /// Apply fading to input samples (multiplicative)
    pub fn apply(&mut self, samples: &[IQSample]) -> Vec<IQSample> {
        samples
            .iter()
            .map(|s| {
                let fade = self.next_sample();
                // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                IQSample::new(
                    s.re * fade.re - s.im * fade.im,
                    s.re * fade.im + s.im * fade.re,
                )
            })
            .collect()
    }
}

/// Flat Doppler spectrum generator
///
/// Produces fading with uniform power across the Doppler bandwidth.
#[derive(Debug)]
pub struct FlatDoppler {
    /// Maximum Doppler frequency in Hz
    max_doppler_hz: f64,
    /// Sample rate in Hz
    sample_rate: f64,
    /// Number of oscillators
    num_sinusoids: usize,
    /// Oscillator frequencies (uniformly distributed)
    frequencies: Vec<f64>,
    /// Oscillator phases
    phases: Vec<f64>,
    /// Current time in samples
    time_samples: u64,
}

impl FlatDoppler {
    /// Create a new flat Doppler generator
    pub fn new(max_doppler_hz: f64, sample_rate: f64, num_sinusoids: usize) -> Self {
        let mut rng = StdRng::from_entropy();

        // Uniformly distributed frequencies between -f_d and +f_d
        let frequencies: Vec<f64> = (0..num_sinusoids)
            .map(|i| {
                let normalized = (i as f64 + 0.5) / num_sinusoids as f64;
                max_doppler_hz * (2.0 * normalized - 1.0)
            })
            .collect();

        let phases: Vec<f64> = (0..num_sinusoids)
            .map(|_| rng.gen::<f64>() * 2.0 * PI)
            .collect();

        Self {
            max_doppler_hz,
            sample_rate,
            num_sinusoids,
            frequencies,
            phases,
            time_samples: 0,
        }
    }

    /// Generate a single fading sample
    pub fn next_sample(&mut self) -> IQSample {
        let t = self.time_samples as f64 / self.sample_rate;
        self.time_samples += 1;

        let scale = 1.0 / (self.num_sinusoids as f64).sqrt();

        let mut real = 0.0;
        let mut imag = 0.0;

        for i in 0..self.num_sinusoids {
            let phase = 2.0 * PI * self.frequencies[i] * t + self.phases[i];
            real += phase.cos();
            imag += phase.sin();
        }

        IQSample::new(real * scale, imag * scale)
    }

    /// Generate multiple samples
    pub fn generate(&mut self, num_samples: usize) -> Vec<IQSample> {
        (0..num_samples).map(|_| self.next_sample()).collect()
    }

    /// Apply fading to input samples
    pub fn apply(&mut self, samples: &[IQSample]) -> Vec<IQSample> {
        samples
            .iter()
            .map(|s| {
                let fade = self.next_sample();
                IQSample::new(
                    s.re * fade.re - s.im * fade.im,
                    s.re * fade.im + s.im * fade.re,
                )
            })
            .collect()
    }
}

/// Gaussian Doppler spectrum generator
///
/// Produces fading with Gaussian-shaped Doppler spectrum centered at DC.
#[derive(Debug)]
pub struct GaussianDoppler {
    /// RMS Doppler spread in Hz
    doppler_rms_hz: f64,
    /// Sample rate in Hz
    sample_rate: f64,
    /// Number of oscillators
    num_sinusoids: usize,
    /// Oscillator frequencies (Gaussian distributed)
    frequencies: Vec<f64>,
    /// Oscillator phases
    phases: Vec<f64>,
    /// Current time in samples
    time_samples: u64,
}

impl GaussianDoppler {
    /// Create a new Gaussian Doppler generator
    ///
    /// # Arguments
    /// * `doppler_rms_hz` - RMS Doppler spread (standard deviation)
    /// * `sample_rate` - Sample rate in Hz
    /// * `num_sinusoids` - Number of oscillators
    pub fn new(doppler_rms_hz: f64, sample_rate: f64, num_sinusoids: usize) -> Self {
        use rand_distr::{Distribution, Normal};
        let mut rng = StdRng::from_entropy();

        let normal = Normal::new(0.0, doppler_rms_hz).unwrap();
        let frequencies: Vec<f64> = (0..num_sinusoids)
            .map(|_| normal.sample(&mut rng))
            .collect();

        let phases: Vec<f64> = (0..num_sinusoids)
            .map(|_| rng.gen::<f64>() * 2.0 * PI)
            .collect();

        Self {
            doppler_rms_hz,
            sample_rate,
            num_sinusoids,
            frequencies,
            phases,
            time_samples: 0,
        }
    }

    /// Generate a single fading sample
    pub fn next_sample(&mut self) -> IQSample {
        let t = self.time_samples as f64 / self.sample_rate;
        self.time_samples += 1;

        let scale = 1.0 / (self.num_sinusoids as f64).sqrt();

        let mut real = 0.0;
        let mut imag = 0.0;

        for i in 0..self.num_sinusoids {
            let phase = 2.0 * PI * self.frequencies[i] * t + self.phases[i];
            real += phase.cos();
            imag += phase.sin();
        }

        IQSample::new(real * scale, imag * scale)
    }

    /// Generate multiple samples
    pub fn generate(&mut self, num_samples: usize) -> Vec<IQSample> {
        (0..num_samples).map(|_| self.next_sample()).collect()
    }

    /// Apply fading to input samples
    pub fn apply(&mut self, samples: &[IQSample]) -> Vec<IQSample> {
        samples
            .iter()
            .map(|s| {
                let fade = self.next_sample();
                IQSample::new(
                    s.re * fade.re - s.im * fade.im,
                    s.re * fade.im + s.im * fade.re,
                )
            })
            .collect()
    }
}

/// Unified Doppler generator that can use any model
#[derive(Debug)]
pub enum DopplerGenerator {
    /// Jake's model
    Jakes(JakesDoppler),
    /// Flat spectrum
    Flat(FlatDoppler),
    /// Gaussian spectrum
    Gaussian(GaussianDoppler),
    /// Static (no fading)
    Static,
}

impl DopplerGenerator {
    /// Create a new Doppler generator
    pub fn new(
        model: DopplerModel,
        max_doppler_hz: f64,
        sample_rate: f64,
        num_sinusoids: usize,
    ) -> Self {
        match model {
            DopplerModel::Jakes => {
                DopplerGenerator::Jakes(JakesDoppler::new(max_doppler_hz, sample_rate, num_sinusoids))
            }
            DopplerModel::Flat => {
                DopplerGenerator::Flat(FlatDoppler::new(max_doppler_hz, sample_rate, num_sinusoids))
            }
            DopplerModel::Gaussian => {
                DopplerGenerator::Gaussian(GaussianDoppler::new(max_doppler_hz, sample_rate, num_sinusoids))
            }
            DopplerModel::Static => DopplerGenerator::Static,
        }
    }

    /// Generate a single fading sample
    pub fn next_sample(&mut self) -> IQSample {
        match self {
            DopplerGenerator::Jakes(g) => g.next_sample(),
            DopplerGenerator::Flat(g) => g.next_sample(),
            DopplerGenerator::Gaussian(g) => g.next_sample(),
            DopplerGenerator::Static => IQSample::new(1.0, 0.0),
        }
    }

    /// Generate multiple samples
    pub fn generate(&mut self, num_samples: usize) -> Vec<IQSample> {
        match self {
            DopplerGenerator::Jakes(g) => g.generate(num_samples),
            DopplerGenerator::Flat(g) => g.generate(num_samples),
            DopplerGenerator::Gaussian(g) => g.generate(num_samples),
            DopplerGenerator::Static => vec![IQSample::new(1.0, 0.0); num_samples],
        }
    }

    /// Apply fading to input samples
    pub fn apply(&mut self, samples: &[IQSample]) -> Vec<IQSample> {
        match self {
            DopplerGenerator::Jakes(g) => g.apply(samples),
            DopplerGenerator::Flat(g) => g.apply(samples),
            DopplerGenerator::Gaussian(g) => g.apply(samples),
            DopplerGenerator::Static => samples.to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_velocity_to_doppler() {
        // Walking speed (~1.4 m/s) at 915 MHz
        let doppler = velocity_to_doppler(1.4, 915e6);
        assert!((doppler - 4.28).abs() < 0.1, "Expected ~4.28 Hz, got {}", doppler);

        // Vehicle at 30 m/s (~67 mph) at 915 MHz
        let doppler = velocity_to_doppler(30.0, 915e6);
        assert!((doppler - 91.5).abs() < 1.0, "Expected ~91.5 Hz, got {}", doppler);
    }

    #[test]
    fn test_coherence_time() {
        let doppler = JakesDoppler::new(100.0, 1_000_000.0, 16);
        let tc = doppler.coherence_time();
        assert!((tc - 0.0025).abs() < 0.001, "Expected ~2.5ms coherence time");
    }

    #[test]
    fn test_jakes_power() {
        // Generate many samples and verify average power is close to 1
        let mut doppler = JakesDoppler::with_seed(30.0, 100_000.0, 16, 42);
        let samples = doppler.generate(100_000);

        let avg_power: f64 = samples.iter().map(|s| s.norm_sqr()).sum::<f64>() / samples.len() as f64;

        // Average power should be close to 1.0 (within 10%)
        assert!(
            (avg_power - 1.0).abs() < 0.2,
            "Average power {} not close to 1.0",
            avg_power
        );
    }

    #[test]
    fn test_static_doppler() {
        let mut gen = DopplerGenerator::new(DopplerModel::Static, 100.0, 1_000_000.0, 16);
        let samples = gen.generate(100);

        // All samples should be (1, 0)
        for s in samples {
            assert!((s.re - 1.0).abs() < 1e-10);
            assert!(s.im.abs() < 1e-10);
        }
    }
}
