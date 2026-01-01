//! Spectrum Analysis
//!
//! FFT-based power spectrum computation with windowing and averaging.

use crate::fft_utils::FftProcessor;
use crate::types::IQSample;
use rustfft::num_complex::Complex64;
use std::f64::consts::PI;

/// Window functions for spectral analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowFunction {
    /// No windowing (rectangular)
    None,
    /// Hann window (default) - good general purpose
    #[default]
    Hann,
    /// Hamming window - slightly less sidelobe suppression than Hann
    Hamming,
    /// Blackman window - excellent sidelobe suppression
    Blackman,
    /// Blackman-Harris window - very low sidelobes
    BlackmanHarris,
    /// Flat-top window - accurate amplitude measurement
    FlatTop,
}

impl WindowFunction {
    /// Generate window coefficients for the given size
    pub fn generate(&self, size: usize) -> Vec<f64> {
        match self {
            WindowFunction::None => vec![1.0; size],
            WindowFunction::Hann => (0..size)
                .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / size as f64).cos()))
                .collect(),
            WindowFunction::Hamming => (0..size)
                .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / size as f64).cos())
                .collect(),
            WindowFunction::Blackman => (0..size)
                .map(|i| {
                    let n = i as f64 / size as f64;
                    0.42 - 0.5 * (2.0 * PI * n).cos() + 0.08 * (4.0 * PI * n).cos()
                })
                .collect(),
            WindowFunction::BlackmanHarris => (0..size)
                .map(|i| {
                    let n = i as f64 / size as f64;
                    0.35875 - 0.48829 * (2.0 * PI * n).cos()
                        + 0.14128 * (4.0 * PI * n).cos()
                        - 0.01168 * (6.0 * PI * n).cos()
                })
                .collect(),
            WindowFunction::FlatTop => (0..size)
                .map(|i| {
                    let n = i as f64 / size as f64;
                    0.21557895 - 0.41663158 * (2.0 * PI * n).cos()
                        + 0.277263158 * (4.0 * PI * n).cos()
                        - 0.083578947 * (6.0 * PI * n).cos()
                        + 0.006947368 * (8.0 * PI * n).cos()
                })
                .collect(),
        }
    }

    /// Parse window function from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" | "rectangular" | "rect" => Some(WindowFunction::None),
            "hann" | "hanning" => Some(WindowFunction::Hann),
            "hamming" => Some(WindowFunction::Hamming),
            "blackman" => Some(WindowFunction::Blackman),
            "blackman-harris" | "blackmanharris" => Some(WindowFunction::BlackmanHarris),
            "flat-top" | "flattop" => Some(WindowFunction::FlatTop),
            _ => None,
        }
    }
}

/// Result of spectrum analysis
#[derive(Debug, Clone)]
pub struct SpectrumResult {
    /// Power spectrum in dB (FFT-shifted, DC at center)
    pub power_db: Vec<f64>,
    /// Frequency bins in Hz (FFT-shifted)
    pub frequencies: Vec<f64>,
    /// FFT size used
    pub fft_size: usize,
    /// Sample rate in Hz
    pub sample_rate: f64,
    /// Frequency resolution in Hz
    pub freq_resolution: f64,
    /// Number of frames averaged
    pub num_averages: usize,
}

impl SpectrumResult {
    /// Get the peak frequency and power
    pub fn find_peak(&self) -> (f64, f64) {
        let mut max_idx = 0;
        let mut max_power = f64::NEG_INFINITY;

        for (i, &power) in self.power_db.iter().enumerate() {
            if power > max_power {
                max_power = power;
                max_idx = i;
            }
        }

        (self.frequencies[max_idx], max_power)
    }

    /// Get 3dB bandwidth around the peak
    pub fn bandwidth_3db(&self) -> Option<f64> {
        let (peak_freq, peak_power) = self.find_peak();
        let threshold = peak_power - 3.0;

        let peak_idx = self
            .frequencies
            .iter()
            .position(|&f| (f - peak_freq).abs() < self.freq_resolution / 2.0)?;

        // Find lower edge
        let mut lower_idx = peak_idx;
        while lower_idx > 0 && self.power_db[lower_idx] > threshold {
            lower_idx -= 1;
        }

        // Find upper edge
        let mut upper_idx = peak_idx;
        while upper_idx < self.power_db.len() - 1 && self.power_db[upper_idx] > threshold {
            upper_idx += 1;
        }

        Some(self.frequencies[upper_idx] - self.frequencies[lower_idx])
    }

    /// Format spectrum as text table
    pub fn to_text(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!(
            "Spectrum Analysis (FFT size: {}, averages: {})\n",
            self.fft_size, self.num_averages
        ));
        output.push_str(&format!(
            "Sample rate: {:.0} Hz, Resolution: {:.2} Hz\n",
            self.sample_rate, self.freq_resolution
        ));
        output.push_str("─".repeat(50).as_str());
        output.push('\n');
        output.push_str("  Frequency (Hz)    Power (dB)\n");
        output.push_str("─".repeat(50).as_str());
        output.push('\n');

        // Show top 20 bins by power
        let mut indices: Vec<usize> = (0..self.power_db.len()).collect();
        indices.sort_by(|&a, &b| {
            self.power_db[b]
                .partial_cmp(&self.power_db[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for &i in indices.iter().take(20) {
            output.push_str(&format!(
                "{:>14.2}    {:>10.2}\n",
                self.frequencies[i], self.power_db[i]
            ));
        }

        output
    }

    /// Format spectrum as JSON
    pub fn to_json(&self) -> String {
        let (peak_freq, peak_power) = self.find_peak();
        let bandwidth = self.bandwidth_3db();

        format!(
            r#"{{
  "fft_size": {},
  "sample_rate": {},
  "freq_resolution": {},
  "num_averages": {},
  "peak_frequency_hz": {},
  "peak_power_db": {},
  "bandwidth_3db_hz": {},
  "frequencies": {:?},
  "power_db": {:?}
}}"#,
            self.fft_size,
            self.sample_rate,
            self.freq_resolution,
            self.num_averages,
            peak_freq,
            peak_power,
            bandwidth.map(|b| format!("{:.2}", b)).unwrap_or_else(|| "null".to_string()),
            self.frequencies,
            self.power_db
        )
    }

    /// Format spectrum as CSV
    pub fn to_csv(&self) -> String {
        let mut output = String::from("frequency_hz,power_db\n");
        for (freq, power) in self.frequencies.iter().zip(self.power_db.iter()) {
            output.push_str(&format!("{},{}\n", freq, power));
        }
        output
    }

    /// Format spectrum as ASCII art
    pub fn to_ascii(&self, width: usize, height: usize) -> String {
        let mut output = String::new();
        let n = self.power_db.len();

        // Find power range
        let max_power = self
            .power_db
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_power = (max_power - 60.0).max(
            self.power_db
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min),
        );
        let power_range = max_power - min_power;

        // Bin the spectrum to fit width
        let bins_per_col = n / width;
        let mut binned: Vec<f64> = Vec::with_capacity(width);
        for col in 0..width {
            let start = col * bins_per_col;
            let end = ((col + 1) * bins_per_col).min(n);
            let max_in_bin = self.power_db[start..end]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            binned.push(max_in_bin);
        }

        // Draw from top to bottom
        let chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        for row in 0..height {
            let threshold = max_power - (row as f64 + 1.0) * power_range / height as f64;
            for &power in &binned {
                if power >= threshold {
                    // Determine which character based on how far above threshold
                    let frac = (power - threshold) / (power_range / height as f64);
                    let char_idx = ((frac * 8.0) as usize).min(7);
                    output.push(chars[char_idx]);
                } else {
                    output.push(' ');
                }
            }
            output.push('\n');
        }

        // Add frequency axis
        output.push_str(&"─".repeat(width));
        output.push('\n');

        let min_freq = self.frequencies.first().unwrap_or(&0.0);
        let max_freq = self.frequencies.last().unwrap_or(&0.0);
        output.push_str(&format!(
            "{:<width$}",
            format!("{:.0}", min_freq),
            width = width / 3
        ));
        output.push_str(&format!(
            "{:^width$}",
            format!("{:.0} Hz", (min_freq + max_freq) / 2.0),
            width = width / 3
        ));
        output.push_str(&format!(
            "{:>width$}",
            format!("{:.0}", max_freq),
            width = width / 3
        ));
        output.push('\n');

        output
    }
}

/// Spectrum analyzer with configurable FFT size and windowing
pub struct SpectrumAnalyzer {
    fft_size: usize,
    window: WindowFunction,
    window_coeffs: Vec<f64>,
    processor: FftProcessor,
}

impl SpectrumAnalyzer {
    /// Create a new spectrum analyzer with the given FFT size
    pub fn new(fft_size: usize) -> Self {
        Self::with_window(fft_size, WindowFunction::Hann)
    }

    /// Create a new spectrum analyzer with custom window function
    pub fn with_window(fft_size: usize, window: WindowFunction) -> Self {
        let window_coeffs = window.generate(fft_size);
        let processor = FftProcessor::new(fft_size);

        Self {
            fft_size,
            window,
            window_coeffs,
            processor,
        }
    }

    /// Get the FFT size
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Get the window function
    pub fn window(&self) -> WindowFunction {
        self.window
    }

    /// Compute power spectrum of a signal
    pub fn compute(&mut self, samples: &[IQSample], sample_rate: f64) -> SpectrumResult {
        self.compute_averaged(samples, sample_rate, 1)
    }

    /// Compute averaged power spectrum
    pub fn compute_averaged(
        &mut self,
        samples: &[IQSample],
        sample_rate: f64,
        num_averages: usize,
    ) -> SpectrumResult {
        let hop_size = self.fft_size;
        let mut accumulated = vec![0.0f64; self.fft_size];
        let mut frame_count = 0;

        let mut pos = 0;
        while pos + self.fft_size <= samples.len() && frame_count < num_averages {
            // Apply window and compute FFT
            let mut frame: Vec<Complex64> = samples[pos..pos + self.fft_size]
                .iter()
                .enumerate()
                .map(|(i, &s)| s * self.window_coeffs[i])
                .collect();

            self.processor.fft_inplace(&mut frame);

            // Accumulate power spectrum
            for (i, sample) in frame.iter().enumerate() {
                accumulated[i] += sample.norm_sqr();
            }

            frame_count += 1;
            pos += hop_size;
        }

        // Average and convert to dB
        let power_db: Vec<f64> = accumulated
            .iter()
            .map(|&p| {
                let avg_power = p / frame_count.max(1) as f64;
                if avg_power > 1e-20 {
                    10.0 * avg_power.log10()
                } else {
                    -200.0
                }
            })
            .collect();

        // FFT shift for display
        let power_db = FftProcessor::fft_shift(&power_db);

        // Compute frequency axis
        let freq_resolution = sample_rate / self.fft_size as f64;
        let frequencies: Vec<f64> = (0..self.fft_size)
            .map(|i| {
                let idx = if i < self.fft_size / 2 {
                    i as f64
                } else {
                    (i as i64 - self.fft_size as i64) as f64
                };
                idx * freq_resolution
            })
            .collect();
        let frequencies = FftProcessor::fft_shift(&frequencies);

        SpectrumResult {
            power_db,
            frequencies,
            fft_size: self.fft_size,
            sample_rate,
            freq_resolution,
            num_averages: frame_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_generation() {
        let size = 64;

        // Hann window should be 0 at edges, 1 at center
        let hann = WindowFunction::Hann.generate(size);
        assert!(hann[0] < 0.01);
        assert!(hann[size / 2] > 0.99);

        // Hamming should be ~0.08 at edges
        let hamming = WindowFunction::Hamming.generate(size);
        assert!((hamming[0] - 0.08).abs() < 0.01);
    }

    #[test]
    fn test_spectrum_single_tone() {
        let fft_size = 1024;
        let sample_rate = 48000.0;
        let freq = 1000.0; // 1 kHz tone

        // Generate test signal
        let samples: Vec<IQSample> = (0..fft_size)
            .map(|i| {
                let t = i as f64 / sample_rate;
                let phase = 2.0 * PI * freq * t;
                Complex64::new(phase.cos(), phase.sin())
            })
            .collect();

        let mut analyzer = SpectrumAnalyzer::new(fft_size);
        let result = analyzer.compute(&samples, sample_rate);

        let (peak_freq, _) = result.find_peak();
        assert!(
            (peak_freq - freq).abs() < result.freq_resolution,
            "Peak at {} Hz, expected {} Hz",
            peak_freq,
            freq
        );
    }

    #[test]
    fn test_window_from_str() {
        assert_eq!(
            WindowFunction::from_str("hann"),
            Some(WindowFunction::Hann)
        );
        assert_eq!(
            WindowFunction::from_str("HAMMING"),
            Some(WindowFunction::Hamming)
        );
        assert_eq!(WindowFunction::from_str("invalid"), None);
    }
}
