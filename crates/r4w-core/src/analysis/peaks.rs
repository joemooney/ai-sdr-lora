//! Spectral Peak Detection
//!
//! Find peaks in power spectrum above a threshold.

use crate::analysis::spectrum::SpectrumResult;

/// A detected spectral peak
#[derive(Debug, Clone, Copy)]
pub struct SpectralPeak {
    /// Frequency in Hz
    pub frequency: f64,
    /// Power in dB
    pub power_db: f64,
    /// Bin index in spectrum
    pub bin_index: usize,
    /// Distance from noise floor in dB
    pub above_noise_db: f64,
}

/// Peak detection configuration and results
pub struct PeakFinder {
    /// Threshold above noise floor in dB
    threshold_db: f64,
    /// Maximum number of peaks to find
    max_peaks: usize,
    /// Minimum distance between peaks in bins
    min_distance: usize,
}

impl Default for PeakFinder {
    fn default() -> Self {
        Self {
            threshold_db: 10.0,
            max_peaks: 10,
            min_distance: 3,
        }
    }
}

impl PeakFinder {
    /// Create a new peak finder with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the threshold above noise floor
    pub fn with_threshold(mut self, threshold_db: f64) -> Self {
        self.threshold_db = threshold_db;
        self
    }

    /// Set the maximum number of peaks to find
    pub fn with_max_peaks(mut self, max_peaks: usize) -> Self {
        self.max_peaks = max_peaks;
        self
    }

    /// Set minimum distance between peaks
    pub fn with_min_distance(mut self, min_distance: usize) -> Self {
        self.min_distance = min_distance;
        self
    }

    /// Find peaks in a spectrum result
    pub fn find_peaks(&self, spectrum: &SpectrumResult) -> Vec<SpectralPeak> {
        self.find_peaks_in_power(&spectrum.power_db, &spectrum.frequencies)
    }

    /// Find peaks in raw power spectrum
    pub fn find_peaks_in_power(&self, power_db: &[f64], frequencies: &[f64]) -> Vec<SpectralPeak> {
        let n = power_db.len();
        if n == 0 {
            return Vec::new();
        }

        // Estimate noise floor using median
        let mut sorted_power: Vec<f64> = power_db.iter().cloned().collect();
        sorted_power.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let noise_floor = sorted_power[n / 4]; // Lower quartile

        let threshold = noise_floor + self.threshold_db;

        // Find all local maxima above threshold
        let mut candidates: Vec<(usize, f64)> = Vec::new();

        for i in 1..n - 1 {
            let power = power_db[i];
            if power > threshold && power > power_db[i - 1] && power > power_db[i + 1] {
                candidates.push((i, power));
            }
        }

        // Sort by power (descending)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select peaks with minimum distance constraint
        let mut peaks = Vec::new();
        let mut used = vec![false; n];

        for (idx, power) in candidates {
            if peaks.len() >= self.max_peaks {
                break;
            }

            // Check if too close to an existing peak
            let mut too_close = false;
            for i in idx.saturating_sub(self.min_distance)..=(idx + self.min_distance).min(n - 1) {
                if used[i] {
                    too_close = true;
                    break;
                }
            }

            if !too_close {
                used[idx] = true;
                peaks.push(SpectralPeak {
                    frequency: frequencies[idx],
                    power_db: power,
                    bin_index: idx,
                    above_noise_db: power - noise_floor,
                });
            }
        }

        // Sort by frequency for consistent output
        peaks.sort_by(|a, b| {
            a.frequency
                .partial_cmp(&b.frequency)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        peaks
    }

    /// Format peaks as text table
    pub fn format_text(peaks: &[SpectralPeak]) -> String {
        let mut output = String::new();
        output.push_str("Spectral Peaks\n");
        output.push_str(&"═".repeat(60));
        output.push('\n');
        output.push_str(&format!(
            "{:>4}  {:>14}  {:>10}  {:>12}\n",
            "#", "Frequency (Hz)", "Power (dB)", "Above Noise"
        ));
        output.push_str(&"─".repeat(60));
        output.push('\n');

        for (i, peak) in peaks.iter().enumerate() {
            output.push_str(&format!(
                "{:>4}  {:>14.2}  {:>10.2}  {:>12.2} dB\n",
                i + 1,
                peak.frequency,
                peak.power_db,
                peak.above_noise_db
            ));
        }

        if peaks.is_empty() {
            output.push_str("  No peaks found above threshold\n");
        }

        output
    }

    /// Format peaks as JSON
    pub fn format_json(peaks: &[SpectralPeak]) -> String {
        let peaks_json: Vec<String> = peaks
            .iter()
            .map(|p| {
                format!(
                    r#"    {{
      "frequency_hz": {:.6},
      "power_db": {:.6},
      "bin_index": {},
      "above_noise_db": {:.6}
    }}"#,
                    p.frequency, p.power_db, p.bin_index, p.above_noise_db
                )
            })
            .collect();

        format!(
            r#"{{
  "num_peaks": {},
  "peaks": [
{}
  ]
}}"#,
            peaks.len(),
            peaks_json.join(",\n")
        )
    }

    /// Format peaks as CSV
    pub fn format_csv(peaks: &[SpectralPeak]) -> String {
        let mut output = String::from("frequency_hz,power_db,bin_index,above_noise_db\n");
        for peak in peaks {
            output.push_str(&format!(
                "{},{},{},{}\n",
                peak.frequency, peak.power_db, peak.bin_index, peak.above_noise_db
            ));
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::SpectrumAnalyzer;
    use crate::types::IQSample;
    use std::f64::consts::PI;

    #[test]
    fn test_find_single_peak() {
        let fft_size = 1024;
        let sample_rate = 48000.0;
        let freq = 5000.0;

        // Generate single tone
        let samples: Vec<IQSample> = (0..fft_size)
            .map(|i| {
                let t = i as f64 / sample_rate;
                let phase = 2.0 * PI * freq * t;
                IQSample::new(phase.cos(), phase.sin())
            })
            .collect();

        let mut analyzer = SpectrumAnalyzer::new(fft_size);
        let spectrum = analyzer.compute(&samples, sample_rate);

        let finder = PeakFinder::new().with_threshold(20.0);
        let peaks = finder.find_peaks(&spectrum);

        assert!(!peaks.is_empty(), "Should find at least one peak");

        // The peak should be near the tone frequency
        let main_peak = peaks
            .iter()
            .max_by(|a, b| a.power_db.partial_cmp(&b.power_db).unwrap())
            .unwrap();

        let freq_resolution = sample_rate / fft_size as f64;
        assert!(
            (main_peak.frequency - freq).abs() < freq_resolution * 2.0,
            "Peak at {} Hz, expected {} Hz",
            main_peak.frequency,
            freq
        );
    }

    #[test]
    fn test_find_multiple_peaks() {
        let fft_size = 1024;
        let sample_rate = 48000.0;
        let freqs = [1000.0, 5000.0, 10000.0];

        // Generate multi-tone signal
        let samples: Vec<IQSample> = (0..fft_size)
            .map(|i| {
                let t = i as f64 / sample_rate;
                let mut sample = IQSample::new(0.0, 0.0);
                for &freq in &freqs {
                    let phase = 2.0 * PI * freq * t;
                    sample += IQSample::new(phase.cos(), phase.sin());
                }
                sample
            })
            .collect();

        let mut analyzer = SpectrumAnalyzer::new(fft_size);
        let spectrum = analyzer.compute(&samples, sample_rate);

        let finder = PeakFinder::new()
            .with_threshold(10.0)
            .with_max_peaks(5);
        let peaks = finder.find_peaks(&spectrum);

        assert!(
            peaks.len() >= 3,
            "Should find at least 3 peaks, found {}",
            peaks.len()
        );
    }

    #[test]
    fn test_no_peaks_in_noise() {
        let n = 1024;

        // Generate flat noise (all same value = no peaks)
        let power_db = vec![-60.0; n];
        let frequencies: Vec<f64> = (0..n).map(|i| i as f64 * 100.0).collect();

        let finder = PeakFinder::new().with_threshold(10.0);
        let peaks = finder.find_peaks_in_power(&power_db, &frequencies);

        assert!(peaks.is_empty(), "Should find no peaks in flat spectrum");
    }
}
