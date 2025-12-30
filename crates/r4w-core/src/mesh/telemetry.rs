//! Meshtastic Telemetry Support
//!
//! This module provides data structures for Meshtastic telemetry messages,
//! including device metrics, environment sensors, and power monitoring.
//!
//! ## Telemetry Types
//!
//! - **DeviceMetrics**: Battery level, voltage, channel utilization, uptime
//! - **EnvironmentMetrics**: Temperature, humidity, pressure, gas resistance
//! - **PowerMetrics**: Multi-channel power measurements (INA219/INA3221)
//!
//! ## Example
//!
//! ```rust,ignore
//! use r4w_core::mesh::telemetry::{DeviceMetrics, Telemetry, TelemetryVariant};
//!
//! let metrics = DeviceMetrics {
//!     battery_level: Some(85),
//!     voltage: Some(4.1),
//!     channel_utilization: Some(0.05),
//!     air_util_tx: Some(0.02),
//!     uptime_seconds: Some(3600),
//! };
//!
//! let telemetry = Telemetry::new(TelemetryVariant::Device(metrics));
//! ```

use std::time::{SystemTime, UNIX_EPOCH};

/// Device metrics telemetry
///
/// Reports battery status, radio utilization, and uptime.
/// Sent periodically by nodes to their neighbors.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DeviceMetrics {
    /// Battery level (0-100%)
    /// None if device is powered externally
    pub battery_level: Option<u8>,

    /// Battery/supply voltage in volts
    pub voltage: Option<f32>,

    /// Channel utilization (0.0-1.0)
    /// Fraction of time the channel was busy during measurement period
    pub channel_utilization: Option<f32>,

    /// TX airtime utilization (0.0-1.0)
    /// Fraction of time this node was transmitting
    pub air_util_tx: Option<f32>,

    /// Uptime in seconds since boot
    pub uptime_seconds: Option<u32>,
}

impl DeviceMetrics {
    /// Create new device metrics with all fields populated
    pub fn new(
        battery_level: Option<u8>,
        voltage: Option<f32>,
        channel_utilization: Option<f32>,
        air_util_tx: Option<f32>,
        uptime_seconds: Option<u32>,
    ) -> Self {
        Self {
            battery_level,
            voltage,
            channel_utilization,
            air_util_tx,
            uptime_seconds,
        }
    }

    /// Check if battery is low (below threshold)
    pub fn is_battery_low(&self, threshold: u8) -> bool {
        self.battery_level.map(|l| l < threshold).unwrap_or(false)
    }

    /// Check if channel is congested (utilization above threshold)
    pub fn is_channel_congested(&self, threshold: f32) -> bool {
        self.channel_utilization.map(|u| u > threshold).unwrap_or(false)
    }
}

/// Environment sensor metrics
///
/// Data from environmental sensors (BME280, BME680, SHT31, etc.)
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EnvironmentMetrics {
    /// Temperature in Celsius
    pub temperature: Option<f32>,

    /// Relative humidity (0-100%)
    pub relative_humidity: Option<f32>,

    /// Barometric pressure in hPa (hectopascals)
    pub barometric_pressure: Option<f32>,

    /// Gas resistance in ohms (BME680)
    /// Higher values indicate cleaner air
    pub gas_resistance: Option<f32>,

    /// Indoor Air Quality index (BME680)
    /// 0-50 = Good, 51-100 = Moderate, 101-150 = Unhealthy for sensitive groups
    /// 151-200 = Unhealthy, 201-300 = Very unhealthy, 301-500 = Hazardous
    pub iaq: Option<u16>,

    /// Distance measurement in meters (ultrasonic/lidar sensors)
    pub distance: Option<f32>,

    /// Illuminance in lux
    pub lux: Option<f32>,

    /// UV index
    pub uv_index: Option<f32>,

    /// Wind speed in m/s
    pub wind_speed: Option<f32>,

    /// Wind direction in degrees (0-360)
    pub wind_direction: Option<u16>,

    /// Weight measurement in kg
    pub weight: Option<f32>,
}

impl EnvironmentMetrics {
    /// Create new environment metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with temperature only
    pub fn with_temperature(temp: f32) -> Self {
        Self {
            temperature: Some(temp),
            ..Default::default()
        }
    }

    /// Create with basic weather data (temp, humidity, pressure)
    pub fn with_weather(temp: f32, humidity: f32, pressure: f32) -> Self {
        Self {
            temperature: Some(temp),
            relative_humidity: Some(humidity),
            barometric_pressure: Some(pressure),
            ..Default::default()
        }
    }

    /// Calculate heat index (apparent temperature) in Celsius
    /// Only valid for temperatures > 27C and humidity > 40%
    pub fn heat_index(&self) -> Option<f32> {
        let t = self.temperature?;
        let rh = self.relative_humidity?;

        if t < 27.0 || rh < 40.0 {
            return Some(t); // Heat index not applicable
        }

        // Steadman's formula adapted for Celsius
        let t_f = t * 9.0 / 5.0 + 32.0; // Convert to Fahrenheit
        let hi_f = -42.379
            + 2.04901523 * t_f
            + 10.14333127 * rh
            - 0.22475541 * t_f * rh
            - 0.00683783 * t_f * t_f
            - 0.05481717 * rh * rh
            + 0.00122874 * t_f * t_f * rh
            + 0.00085282 * t_f * rh * rh
            - 0.00000199 * t_f * t_f * rh * rh;

        Some((hi_f - 32.0) * 5.0 / 9.0) // Convert back to Celsius
    }
}

/// Power channel measurement
///
/// Single channel measurement from a power monitoring IC (INA219/INA3221)
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PowerChannel {
    /// Channel number (0-indexed)
    pub channel: u8,

    /// Voltage in volts
    pub voltage: Option<f32>,

    /// Current in milliamps
    pub current: Option<f32>,
}

impl PowerChannel {
    /// Create a new power channel measurement
    pub fn new(channel: u8, voltage: Option<f32>, current: Option<f32>) -> Self {
        Self {
            channel,
            voltage,
            current,
        }
    }

    /// Calculate power in milliwatts
    pub fn power_mw(&self) -> Option<f32> {
        let v = self.voltage?;
        let i = self.current?;
        Some(v * i)
    }
}

/// Power metrics telemetry
///
/// Multi-channel power measurements for monitoring battery,
/// solar panels, and other power sources.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PowerMetrics {
    /// Power channel measurements
    pub channels: Vec<PowerChannel>,
}

impl PowerMetrics {
    /// Create new power metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a single channel
    pub fn with_channel(channel: PowerChannel) -> Self {
        Self {
            channels: vec![channel],
        }
    }

    /// Add a channel measurement
    pub fn add_channel(&mut self, channel: PowerChannel) {
        self.channels.push(channel);
    }

    /// Get total power consumption in milliwatts
    pub fn total_power_mw(&self) -> f32 {
        self.channels
            .iter()
            .filter_map(|c| c.power_mw())
            .sum()
    }
}

/// Telemetry variant
///
/// Enumeration of all supported telemetry message types.
#[derive(Debug, Clone, PartialEq)]
pub enum TelemetryVariant {
    /// Device metrics (battery, utilization, uptime)
    Device(DeviceMetrics),

    /// Environment metrics (temperature, humidity, etc.)
    Environment(EnvironmentMetrics),

    /// Power metrics (voltage, current measurements)
    Power(PowerMetrics),
}

/// Telemetry message
///
/// Container for telemetry data with timestamp.
#[derive(Debug, Clone, PartialEq)]
pub struct Telemetry {
    /// Timestamp (seconds since Unix epoch)
    pub time: u32,

    /// Telemetry data variant
    pub variant: TelemetryVariant,
}

impl Telemetry {
    /// Create a new telemetry message with current timestamp
    pub fn new(variant: TelemetryVariant) -> Self {
        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as u32)
            .unwrap_or(0);

        Self { time, variant }
    }

    /// Create a telemetry message with specific timestamp
    pub fn with_time(time: u32, variant: TelemetryVariant) -> Self {
        Self { time, variant }
    }

    /// Create device telemetry
    pub fn device(metrics: DeviceMetrics) -> Self {
        Self::new(TelemetryVariant::Device(metrics))
    }

    /// Create environment telemetry
    pub fn environment(metrics: EnvironmentMetrics) -> Self {
        Self::new(TelemetryVariant::Environment(metrics))
    }

    /// Create power telemetry
    pub fn power(metrics: PowerMetrics) -> Self {
        Self::new(TelemetryVariant::Power(metrics))
    }

    /// Get device metrics if this is a device telemetry message
    pub fn as_device(&self) -> Option<&DeviceMetrics> {
        match &self.variant {
            TelemetryVariant::Device(m) => Some(m),
            _ => None,
        }
    }

    /// Get environment metrics if this is an environment telemetry message
    pub fn as_environment(&self) -> Option<&EnvironmentMetrics> {
        match &self.variant {
            TelemetryVariant::Environment(m) => Some(m),
            _ => None,
        }
    }

    /// Get power metrics if this is a power telemetry message
    pub fn as_power(&self) -> Option<&PowerMetrics> {
        match &self.variant {
            TelemetryVariant::Power(m) => Some(m),
            _ => None,
        }
    }
}

/// Telemetry configuration
///
/// Controls how often different telemetry types are sent.
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// Send device metrics every N seconds (0 = disabled)
    pub device_update_interval: u32,

    /// Send environment metrics every N seconds (0 = disabled)
    pub environment_update_interval: u32,

    /// Send power metrics every N seconds (0 = disabled)
    pub power_update_interval: u32,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            device_update_interval: 900,       // 15 minutes
            environment_update_interval: 900,  // 15 minutes
            power_update_interval: 0,          // Disabled by default
        }
    }
}

impl TelemetryConfig {
    /// Create a config with all telemetry enabled at specified interval
    pub fn all_enabled(interval_secs: u32) -> Self {
        Self {
            device_update_interval: interval_secs,
            environment_update_interval: interval_secs,
            power_update_interval: interval_secs,
        }
    }

    /// Create a config with only device telemetry enabled
    pub fn device_only(interval_secs: u32) -> Self {
        Self {
            device_update_interval: interval_secs,
            environment_update_interval: 0,
            power_update_interval: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_metrics() {
        let metrics = DeviceMetrics::new(
            Some(85),
            Some(4.1),
            Some(0.05),
            Some(0.02),
            Some(3600),
        );

        assert_eq!(metrics.battery_level, Some(85));
        assert!(!metrics.is_battery_low(20));
        assert!(!metrics.is_channel_congested(0.1));
    }

    #[test]
    fn test_device_metrics_low_battery() {
        let metrics = DeviceMetrics::new(Some(15), None, None, None, None);
        assert!(metrics.is_battery_low(20));
    }

    #[test]
    fn test_environment_metrics() {
        let metrics = EnvironmentMetrics::with_weather(25.0, 60.0, 1013.25);

        assert_eq!(metrics.temperature, Some(25.0));
        assert_eq!(metrics.relative_humidity, Some(60.0));
        assert_eq!(metrics.barometric_pressure, Some(1013.25));
    }

    #[test]
    fn test_heat_index() {
        // Heat index only applies at high temps and humidity
        let cool = EnvironmentMetrics::with_weather(20.0, 50.0, 1013.0);
        assert_eq!(cool.heat_index(), Some(20.0)); // Returns actual temp

        let hot_humid = EnvironmentMetrics::with_weather(35.0, 70.0, 1013.0);
        let hi = hot_humid.heat_index().unwrap();
        assert!(hi > 35.0); // Heat index should be higher than actual temp
    }

    #[test]
    fn test_power_channel() {
        let channel = PowerChannel::new(0, Some(5.0), Some(100.0));
        assert_eq!(channel.power_mw(), Some(500.0));
    }

    #[test]
    fn test_power_metrics() {
        let mut power = PowerMetrics::new();
        power.add_channel(PowerChannel::new(0, Some(5.0), Some(100.0)));
        power.add_channel(PowerChannel::new(1, Some(3.3), Some(50.0)));

        assert_eq!(power.channels.len(), 2);
        assert_eq!(power.total_power_mw(), 665.0);
    }

    #[test]
    fn test_telemetry_creation() {
        let device = DeviceMetrics::new(Some(90), Some(4.2), None, None, Some(1000));
        let telemetry = Telemetry::device(device.clone());

        assert!(telemetry.time > 0);
        assert_eq!(telemetry.as_device(), Some(&device));
        assert!(telemetry.as_environment().is_none());
    }

    #[test]
    fn test_telemetry_config() {
        let config = TelemetryConfig::default();
        assert_eq!(config.device_update_interval, 900);

        let fast = TelemetryConfig::all_enabled(60);
        assert_eq!(fast.device_update_interval, 60);
        assert_eq!(fast.environment_update_interval, 60);
        assert_eq!(fast.power_update_interval, 60);
    }

    #[test]
    fn test_telemetry_variants() {
        let env = Telemetry::environment(EnvironmentMetrics::with_temperature(22.5));
        assert!(env.as_environment().is_some());

        let power = Telemetry::power(PowerMetrics::with_channel(
            PowerChannel::new(0, Some(12.0), Some(200.0)),
        ));
        assert!(power.as_power().is_some());
    }
}
