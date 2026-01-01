//! Signal Statistics
//!
//! Compute comprehensive statistics for I/Q signals including power, SNR, PAPR,
//! DC offset, and I/Q imbalance.

use crate::types::IQSample;

/// I/Q imbalance measurements
#[derive(Debug, Clone, Copy, Default)]
pub struct IQImbalance {
    /// Amplitude imbalance in dB (I/Q power ratio)
    pub amplitude_db: f64,
    /// Phase imbalance in degrees
    pub phase_deg: f64,
    /// DC offset for I component
    pub dc_offset_i: f64,
    /// DC offset for Q component
    pub dc_offset_q: f64,
}

/// Comprehensive signal statistics
#[derive(Debug, Clone)]
pub struct SignalStats {
    /// Number of samples analyzed
    pub num_samples: usize,
    /// Signal duration in seconds (if sample rate provided)
    pub duration_sec: Option<f64>,
    /// Mean power in dBFS (dB relative to full scale)
    pub mean_power_dbfs: f64,
    /// Peak power in dBFS
    pub peak_power_dbfs: f64,
    /// Peak-to-Average Power Ratio in dB
    pub papr_db: f64,
    /// RMS amplitude
    pub rms_amplitude: f64,
    /// Peak amplitude
    pub peak_amplitude: f64,
    /// DC offset (complex)
    pub dc_offset: IQSample,
    /// Estimated SNR in dB (using noise floor estimation)
    pub estimated_snr_db: f64,
    /// Crest factor in dB
    pub crest_factor_db: f64,
    /// I/Q imbalance measurements
    pub iq_imbalance: IQImbalance,
    /// Sample rate (if provided)
    pub sample_rate: Option<f64>,
    /// Minimum I value
    pub min_i: f64,
    /// Maximum I value
    pub max_i: f64,
    /// Minimum Q value
    pub min_q: f64,
    /// Maximum Q value
    pub max_q: f64,
}

impl SignalStats {
    /// Compute statistics for the given samples
    pub fn compute(samples: &[IQSample], sample_rate: Option<f64>) -> Self {
        if samples.is_empty() {
            return Self::empty(sample_rate);
        }

        let num_samples = samples.len();
        let duration_sec = sample_rate.map(|sr| num_samples as f64 / sr);

        // Compute DC offset (mean)
        let sum: IQSample = samples.iter().sum();
        let dc_offset = sum / num_samples as f64;

        // Remove DC for further calculations
        let dc_removed: Vec<IQSample> = samples.iter().map(|s| s - dc_offset).collect();

        // Compute power statistics
        let powers: Vec<f64> = dc_removed.iter().map(|s| s.norm_sqr()).collect();
        let mean_power: f64 = powers.iter().sum::<f64>() / num_samples as f64;
        let peak_power: f64 = powers.iter().cloned().fold(0.0, f64::max);

        // Convert to dBFS (assuming full scale = 1.0)
        let mean_power_dbfs = if mean_power > 1e-20 {
            10.0 * mean_power.log10()
        } else {
            -200.0
        };
        let peak_power_dbfs = if peak_power > 1e-20 {
            10.0 * peak_power.log10()
        } else {
            -200.0
        };

        // PAPR
        let papr_db = peak_power_dbfs - mean_power_dbfs;

        // Amplitudes
        let rms_amplitude = mean_power.sqrt();
        let peak_amplitude = peak_power.sqrt();
        let crest_factor_db = 20.0 * (peak_amplitude / rms_amplitude.max(1e-20)).log10();

        // I/Q statistics
        let mut sum_i2 = 0.0;
        let mut sum_q2 = 0.0;
        let mut min_i = f64::INFINITY;
        let mut max_i = f64::NEG_INFINITY;
        let mut min_q = f64::INFINITY;
        let mut max_q = f64::NEG_INFINITY;

        for sample in samples {
            let i = sample.re;
            let q = sample.im;
            sum_i2 += i * i;
            sum_q2 += q * q;
            min_i = min_i.min(i);
            max_i = max_i.max(i);
            min_q = min_q.min(q);
            max_q = max_q.max(q);
        }

        let n = num_samples as f64;
        let power_i = sum_i2 / n;
        let power_q = sum_q2 / n;

        // I/Q imbalance
        let amplitude_db = if power_q > 1e-20 {
            10.0 * (power_i / power_q).log10()
        } else {
            0.0
        };

        // Estimate phase imbalance using correlation
        let mut sum_iq = 0.0;
        for sample in samples {
            sum_iq += sample.re * sample.im;
        }
        let correlation = sum_iq / n;
        let phase_rad = if power_i > 1e-20 && power_q > 1e-20 {
            (correlation / (power_i * power_q).sqrt()).asin()
        } else {
            0.0
        };
        let phase_deg = phase_rad.to_degrees();

        let iq_imbalance = IQImbalance {
            amplitude_db,
            phase_deg,
            dc_offset_i: dc_offset.re,
            dc_offset_q: dc_offset.im,
        };

        // Estimate SNR using noise floor estimation
        // Simple approach: sort power values, estimate noise from lower quartile
        let mut sorted_powers = powers.clone();
        sorted_powers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let noise_floor_idx = num_samples / 4;
        let noise_power = sorted_powers[..noise_floor_idx]
            .iter()
            .sum::<f64>()
            / noise_floor_idx.max(1) as f64;

        let signal_power = mean_power - noise_power;
        let estimated_snr_db = if noise_power > 1e-20 && signal_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            60.0 // High SNR default
        };

        Self {
            num_samples,
            duration_sec,
            mean_power_dbfs,
            peak_power_dbfs,
            papr_db,
            rms_amplitude,
            peak_amplitude,
            dc_offset,
            estimated_snr_db,
            crest_factor_db,
            iq_imbalance,
            sample_rate,
            min_i,
            max_i,
            min_q,
            max_q,
        }
    }

    /// Create empty statistics
    fn empty(sample_rate: Option<f64>) -> Self {
        Self {
            num_samples: 0,
            duration_sec: Some(0.0),
            mean_power_dbfs: -200.0,
            peak_power_dbfs: -200.0,
            papr_db: 0.0,
            rms_amplitude: 0.0,
            peak_amplitude: 0.0,
            dc_offset: IQSample::new(0.0, 0.0),
            estimated_snr_db: 0.0,
            crest_factor_db: 0.0,
            iq_imbalance: IQImbalance::default(),
            sample_rate,
            min_i: 0.0,
            max_i: 0.0,
            min_q: 0.0,
            max_q: 0.0,
        }
    }

    /// Format as text report
    pub fn to_text(&self) -> String {
        let mut output = String::new();
        output.push_str("Signal Statistics\n");
        output.push_str(&"═".repeat(50));
        output.push('\n');

        output.push_str(&format!("Samples:           {}\n", self.num_samples));
        if let Some(dur) = self.duration_sec {
            output.push_str(&format!("Duration:          {:.6} s\n", dur));
        }
        if let Some(sr) = self.sample_rate {
            output.push_str(&format!("Sample Rate:       {:.0} Hz\n", sr));
        }

        output.push_str("\nPower Measurements\n");
        output.push_str(&"─".repeat(50));
        output.push('\n');
        output.push_str(&format!("Mean Power:        {:.2} dBFS\n", self.mean_power_dbfs));
        output.push_str(&format!("Peak Power:        {:.2} dBFS\n", self.peak_power_dbfs));
        output.push_str(&format!("PAPR:              {:.2} dB\n", self.papr_db));
        output.push_str(&format!("Crest Factor:      {:.2} dB\n", self.crest_factor_db));
        output.push_str(&format!("Estimated SNR:     {:.1} dB\n", self.estimated_snr_db));

        output.push_str("\nAmplitude\n");
        output.push_str(&"─".repeat(50));
        output.push('\n');
        output.push_str(&format!("RMS Amplitude:     {:.6}\n", self.rms_amplitude));
        output.push_str(&format!("Peak Amplitude:    {:.6}\n", self.peak_amplitude));
        output.push_str(&format!("I Range:           [{:.6}, {:.6}]\n", self.min_i, self.max_i));
        output.push_str(&format!("Q Range:           [{:.6}, {:.6}]\n", self.min_q, self.max_q));

        output.push_str("\nI/Q Analysis\n");
        output.push_str(&"─".repeat(50));
        output.push('\n');
        output.push_str(&format!(
            "DC Offset:         ({:.6}, {:.6})\n",
            self.dc_offset.re, self.dc_offset.im
        ));
        output.push_str(&format!(
            "Amplitude Imbalance: {:.3} dB\n",
            self.iq_imbalance.amplitude_db
        ));
        output.push_str(&format!(
            "Phase Imbalance:     {:.3}°\n",
            self.iq_imbalance.phase_deg
        ));

        output
    }

    /// Format as JSON
    pub fn to_json(&self) -> String {
        format!(
            r#"{{
  "num_samples": {},
  "duration_sec": {},
  "sample_rate": {},
  "power": {{
    "mean_dbfs": {:.6},
    "peak_dbfs": {:.6},
    "papr_db": {:.6},
    "crest_factor_db": {:.6},
    "estimated_snr_db": {:.6}
  }},
  "amplitude": {{
    "rms": {:.6},
    "peak": {:.6},
    "i_range": [{:.6}, {:.6}],
    "q_range": [{:.6}, {:.6}]
  }},
  "iq_analysis": {{
    "dc_offset_i": {:.6},
    "dc_offset_q": {:.6},
    "amplitude_imbalance_db": {:.6},
    "phase_imbalance_deg": {:.6}
  }}
}}"#,
            self.num_samples,
            self.duration_sec.map(|d| format!("{:.6}", d)).unwrap_or_else(|| "null".to_string()),
            self.sample_rate.map(|sr| format!("{:.0}", sr)).unwrap_or_else(|| "null".to_string()),
            self.mean_power_dbfs,
            self.peak_power_dbfs,
            self.papr_db,
            self.crest_factor_db,
            self.estimated_snr_db,
            self.rms_amplitude,
            self.peak_amplitude,
            self.min_i,
            self.max_i,
            self.min_q,
            self.max_q,
            self.dc_offset.re,
            self.dc_offset.im,
            self.iq_imbalance.amplitude_db,
            self.iq_imbalance.phase_deg
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_stats_empty() {
        let stats = SignalStats::compute(&[], Some(48000.0));
        assert_eq!(stats.num_samples, 0);
    }

    #[test]
    fn test_stats_single_tone() {
        let sample_rate = 48000.0;
        let freq = 1000.0;
        let amplitude = 0.5;

        let samples: Vec<IQSample> = (0..1000)
            .map(|i| {
                let t = i as f64 / sample_rate;
                let phase = 2.0 * PI * freq * t;
                IQSample::new(amplitude * phase.cos(), amplitude * phase.sin())
            })
            .collect();

        let stats = SignalStats::compute(&samples, Some(sample_rate));

        // RMS amplitude should be close to amplitude (for a complex exponential)
        assert!(
            (stats.rms_amplitude - amplitude).abs() < 0.01,
            "RMS {} != expected {}",
            stats.rms_amplitude,
            amplitude
        );

        // DC offset should be near zero
        assert!(stats.dc_offset.norm() < 0.01);
    }

    #[test]
    fn test_stats_dc_offset() {
        let dc_i = 0.1;
        let dc_q = -0.2;

        let samples: Vec<IQSample> = (0..1000)
            .map(|_| IQSample::new(dc_i, dc_q))
            .collect();

        let stats = SignalStats::compute(&samples, Some(48000.0));

        assert!((stats.dc_offset.re - dc_i).abs() < 0.001);
        assert!((stats.dc_offset.im - dc_q).abs() < 0.001);
    }
}
