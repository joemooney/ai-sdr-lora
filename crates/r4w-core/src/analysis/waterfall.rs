//! Waterfall/Spectrogram Generation
//!
//! Time-frequency visualization with PNG and ASCII output support.

use crate::fft_utils::FftProcessor;
use crate::types::IQSample;
use rustfft::num_complex::Complex64;
use std::f64::consts::PI;

/// Colormap for waterfall visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Colormap {
    /// Viridis (perceptually uniform, colorblind-friendly)
    #[default]
    Viridis,
    /// Plasma (perceptually uniform, high contrast)
    Plasma,
    /// Magma (dark to light, good for low-light)
    Magma,
    /// Inferno (dark to bright yellow)
    Inferno,
    /// Turbo (rainbow-like, high contrast)
    Turbo,
    /// Grayscale (simple black to white)
    Grayscale,
}

impl Colormap {
    /// Map normalized value (0-1) to RGB color
    pub fn map(&self, value: f64) -> [u8; 3] {
        let t = value.clamp(0.0, 1.0);

        match self {
            Colormap::Viridis => Self::viridis(t),
            Colormap::Plasma => Self::plasma(t),
            Colormap::Magma => Self::magma(t),
            Colormap::Inferno => Self::inferno(t),
            Colormap::Turbo => Self::turbo(t),
            Colormap::Grayscale => {
                let v = (t * 255.0) as u8;
                [v, v, v]
            }
        }
    }

    /// Parse colormap from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "viridis" => Some(Colormap::Viridis),
            "plasma" => Some(Colormap::Plasma),
            "magma" => Some(Colormap::Magma),
            "inferno" => Some(Colormap::Inferno),
            "turbo" => Some(Colormap::Turbo),
            "grayscale" | "gray" | "grey" => Some(Colormap::Grayscale),
            _ => None,
        }
    }

    // Polynomial approximations of matplotlib colormaps
    fn viridis(t: f64) -> [u8; 3] {
        let r = (0.267 + t * (0.329 + t * (1.451 + t * (-1.808 + t * 0.758)))).clamp(0.0, 1.0);
        let g = (0.004 + t * (1.513 + t * (-0.838 + t * (0.731 - t * 0.466)))).clamp(0.0, 1.0);
        let b = (0.329 + t * (1.442 + t * (-2.642 + t * (1.963 - t * 0.440)))).clamp(0.0, 1.0);
        [
            (r * 255.0) as u8,
            (g * 255.0) as u8,
            (b * 255.0) as u8,
        ]
    }

    fn plasma(t: f64) -> [u8; 3] {
        let r = (0.050 + t * (2.735 + t * (-2.811 + t * (1.327 - t * 0.259)))).clamp(0.0, 1.0);
        let g = (0.030 + t * (0.259 + t * (2.042 + t * (-2.802 + t * 1.429)))).clamp(0.0, 1.0);
        let b = (0.528 + t * (1.502 + t * (-3.489 + t * (3.003 - t * 0.985)))).clamp(0.0, 1.0);
        [
            (r * 255.0) as u8,
            (g * 255.0) as u8,
            (b * 255.0) as u8,
        ]
    }

    fn magma(t: f64) -> [u8; 3] {
        let r = (0.001 + t * (0.912 + t * (1.287 + t * (-1.466 + t * 0.532)))).clamp(0.0, 1.0);
        let g = (0.000 + t * (0.188 + t * (1.612 + t * (-1.681 + t * 0.859)))).clamp(0.0, 1.0);
        let b = (0.014 + t * (1.937 + t * (-2.578 + t * (2.079 - t * 0.570)))).clamp(0.0, 1.0);
        [
            (r * 255.0) as u8,
            (g * 255.0) as u8,
            (b * 255.0) as u8,
        ]
    }

    fn inferno(t: f64) -> [u8; 3] {
        let r = (0.000 + t * (1.132 + t * (0.737 + t * (-0.972 + t * 0.441)))).clamp(0.0, 1.0);
        let g = (0.000 + t * (0.142 + t * (1.746 + t * (-1.834 + t * 0.926)))).clamp(0.0, 1.0);
        let b = (0.016 + t * (1.980 + t * (-2.897 + t * (2.182 - t * 0.565)))).clamp(0.0, 1.0);
        [
            (r * 255.0) as u8,
            (g * 255.0) as u8,
            (b * 255.0) as u8,
        ]
    }

    fn turbo(t: f64) -> [u8; 3] {
        // Simplified turbo colormap
        let r = (0.13572 + t * (4.6153 + t * (-42.660 + t * (132.13 + t * (-152.95 + t * 56.14)))))
            .clamp(0.0, 1.0);
        let g = (0.09140 + t * (2.9243 + t * (1.5424 + t * (-26.155 + t * (38.792 - t * 16.29)))))
            .clamp(0.0, 1.0);
        let b = (0.10667 + t * (12.750 + t * (-60.582 + t * (132.33 + t * (-134.87 + t * 50.36)))))
            .clamp(0.0, 1.0);
        [
            (r * 255.0) as u8,
            (g * 255.0) as u8,
            (b * 255.0) as u8,
        ]
    }
}

/// Result of waterfall generation
#[derive(Debug, Clone)]
pub struct WaterfallResult {
    /// Power values in dB [time][frequency]
    pub power_db: Vec<Vec<f64>>,
    /// Time axis (seconds)
    pub times: Vec<f64>,
    /// Frequency axis (Hz, FFT-shifted)
    pub frequencies: Vec<f64>,
    /// FFT size used
    pub fft_size: usize,
    /// Hop size in samples
    pub hop_size: usize,
    /// Sample rate in Hz
    pub sample_rate: f64,
}

impl WaterfallResult {
    /// Get dimensions (width, height) = (freq bins, time bins)
    pub fn dimensions(&self) -> (usize, usize) {
        (self.fft_size, self.power_db.len())
    }

    /// Get power range (min_db, max_db)
    pub fn power_range(&self) -> (f64, f64) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for row in &self.power_db {
            for &val in row {
                if val > max {
                    max = val;
                }
                if val < min && val > -200.0 {
                    min = val;
                }
            }
        }

        (min, max)
    }

    /// Generate PNG image data
    #[cfg(feature = "image")]
    pub fn to_png(&self, colormap: Colormap, min_db: f64, max_db: f64) -> Vec<u8> {
        use image::{ImageBuffer, Rgb, ImageEncoder, codecs::png::PngEncoder};

        let width = self.fft_size;
        let height = self.power_db.len();
        let range = max_db - min_db;

        let mut img = ImageBuffer::new(width as u32, height as u32);

        for (y, row) in self.power_db.iter().enumerate() {
            for (x, &power) in row.iter().enumerate() {
                let normalized = ((power - min_db) / range).clamp(0.0, 1.0);
                let [r, g, b] = colormap.map(normalized);
                img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
            }
        }

        let mut buffer = Vec::new();
        let encoder = PngEncoder::new(&mut buffer);
        encoder.write_image(&img, width as u32, height as u32, image::ColorType::Rgb8.into())
            .expect("Failed to encode PNG");
        buffer
    }

    /// Generate raw RGB pixel data (for non-image feature builds)
    pub fn to_rgb_pixels(&self, colormap: Colormap, min_db: f64, max_db: f64) -> Vec<u8> {
        let width = self.fft_size;
        let height = self.power_db.len();
        let range = max_db - min_db;
        let mut pixels = Vec::with_capacity(width * height * 3);

        for row in &self.power_db {
            for &power in row {
                let normalized = ((power - min_db) / range).clamp(0.0, 1.0);
                let [r, g, b] = colormap.map(normalized);
                pixels.push(r);
                pixels.push(g);
                pixels.push(b);
            }
        }

        pixels
    }

    /// Generate ASCII art waterfall for terminal display
    pub fn to_ascii(&self, width: usize, height: usize) -> String {
        let (auto_min, auto_max) = self.power_range();
        self.to_ascii_with_range(width, height, auto_min, auto_max)
    }

    /// Generate ASCII art with specified dB range
    pub fn to_ascii_with_range(
        &self,
        width: usize,
        height: usize,
        min_db: f64,
        max_db: f64,
    ) -> String {
        let mut output = String::new();
        let range = max_db - min_db;

        // Unicode block characters for grayscale
        let chars = [' ', '░', '▒', '▓', '█'];
        let num_chars = chars.len();

        // Calculate downsampling factors
        let time_bins = self.power_db.len();
        let freq_bins = self.fft_size;
        let rows_per_char = (time_bins + height - 1) / height;
        let cols_per_char = (freq_bins + width - 1) / width;

        // Header
        output.push_str(&format!(
            "Waterfall: {}x{} bins, {:.0} Hz to {:.0} Hz\n",
            freq_bins,
            time_bins,
            self.frequencies.first().unwrap_or(&0.0),
            self.frequencies.last().unwrap_or(&0.0)
        ));
        output.push_str(&format!(
            "Time: {:.3}s to {:.3}s\n",
            self.times.first().unwrap_or(&0.0),
            self.times.last().unwrap_or(&0.0)
        ));
        output.push_str(&"─".repeat(width));
        output.push('\n');

        for row_idx in 0..height {
            let time_start = row_idx * rows_per_char;
            let time_end = ((row_idx + 1) * rows_per_char).min(time_bins);

            if time_start >= time_bins {
                break;
            }

            for col_idx in 0..width {
                let freq_start = col_idx * cols_per_char;
                let freq_end = ((col_idx + 1) * cols_per_char).min(freq_bins);

                if freq_start >= freq_bins {
                    break;
                }

                // Find max power in this cell
                let mut max_power = f64::NEG_INFINITY;
                for t in time_start..time_end {
                    for f in freq_start..freq_end {
                        if self.power_db[t][f] > max_power {
                            max_power = self.power_db[t][f];
                        }
                    }
                }

                let normalized = ((max_power - min_db) / range).clamp(0.0, 1.0);
                let char_idx = ((normalized * (num_chars - 1) as f64) as usize).min(num_chars - 1);
                output.push(chars[char_idx]);
            }
            output.push('\n');
        }

        output.push_str(&"─".repeat(width));
        output.push('\n');

        // Frequency axis
        let min_freq = self.frequencies.first().unwrap_or(&0.0);
        let max_freq = self.frequencies.last().unwrap_or(&0.0);
        output.push_str(&format!(
            "{:<width$}",
            format!("{:.0}", min_freq),
            width = width / 3
        ));
        output.push_str(&format!(
            "{:^width$}",
            "Hz",
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

/// Waterfall spectrogram generator
pub struct WaterfallGenerator {
    fft_size: usize,
    hop_size: usize,
    window_coeffs: Vec<f64>,
    processor: FftProcessor,
}

impl WaterfallGenerator {
    /// Create a new waterfall generator
    pub fn new(fft_size: usize) -> Self {
        Self::with_hop(fft_size, fft_size / 2)
    }

    /// Create a waterfall generator with custom hop size
    pub fn with_hop(fft_size: usize, hop_size: usize) -> Self {
        // Generate Hann window
        let window_coeffs: Vec<f64> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / fft_size as f64).cos()))
            .collect();

        let processor = FftProcessor::new(fft_size);

        Self {
            fft_size,
            hop_size,
            window_coeffs,
            processor,
        }
    }

    /// Get the FFT size
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Get the hop size
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Compute waterfall spectrogram
    pub fn compute(&mut self, samples: &[IQSample], sample_rate: f64) -> WaterfallResult {
        self.compute_with_limit(samples, sample_rate, None)
    }

    /// Compute waterfall with maximum number of rows
    pub fn compute_with_limit(
        &mut self,
        samples: &[IQSample],
        sample_rate: f64,
        max_rows: Option<usize>,
    ) -> WaterfallResult {
        let mut power_db = Vec::new();
        let mut times = Vec::new();

        let mut pos = 0;
        while pos + self.fft_size <= samples.len() {
            if let Some(max) = max_rows {
                if power_db.len() >= max {
                    break;
                }
            }

            // Apply window and compute FFT
            let mut frame: Vec<Complex64> = samples[pos..pos + self.fft_size]
                .iter()
                .enumerate()
                .map(|(i, &s)| s * self.window_coeffs[i])
                .collect();

            self.processor.fft_inplace(&mut frame);

            // Compute power spectrum in dB
            let power: Vec<f64> = frame
                .iter()
                .map(|c| {
                    let p = c.norm_sqr();
                    if p > 1e-20 {
                        10.0 * p.log10()
                    } else {
                        -200.0
                    }
                })
                .collect();

            // FFT shift for display
            power_db.push(FftProcessor::fft_shift(&power));
            times.push((pos + self.fft_size / 2) as f64 / sample_rate);

            pos += self.hop_size;
        }

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

        WaterfallResult {
            power_db,
            times,
            frequencies,
            fft_size: self.fft_size,
            hop_size: self.hop_size,
            sample_rate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colormap_range() {
        for colormap in [
            Colormap::Viridis,
            Colormap::Plasma,
            Colormap::Magma,
            Colormap::Inferno,
            Colormap::Turbo,
            Colormap::Grayscale,
        ] {
            // Test edges
            let _ = colormap.map(0.0);
            let _ = colormap.map(1.0);

            // Test mid-range
            for i in 0..=10 {
                let t = i as f64 / 10.0;
                let [r, g, b] = colormap.map(t);
                // All values should be valid u8
                assert!(r <= 255);
                assert!(g <= 255);
                assert!(b <= 255);
            }
        }
    }

    #[test]
    fn test_waterfall_basic() {
        let fft_size = 256;
        let sample_rate = 48000.0;

        // Generate test chirp
        let duration = 0.1; // 100ms
        let num_samples = (sample_rate * duration) as usize;
        let samples: Vec<IQSample> = (0..num_samples)
            .map(|i| {
                let t = i as f64 / sample_rate;
                let freq = 1000.0 + 5000.0 * t; // Chirp from 1kHz to 6kHz
                let phase = 2.0 * PI * freq * t;
                Complex64::new(phase.cos(), phase.sin())
            })
            .collect();

        let mut generator = WaterfallGenerator::new(fft_size);
        let result = generator.compute(&samples, sample_rate);

        assert!(!result.power_db.is_empty());
        assert_eq!(result.fft_size, fft_size);
        assert_eq!(result.frequencies.len(), fft_size);
    }

    #[test]
    fn test_colormap_from_str() {
        assert_eq!(Colormap::from_str("viridis"), Some(Colormap::Viridis));
        assert_eq!(Colormap::from_str("PLASMA"), Some(Colormap::Plasma));
        assert_eq!(Colormap::from_str("gray"), Some(Colormap::Grayscale));
        assert_eq!(Colormap::from_str("invalid"), None);
    }
}
