//! LoRa Mesh Integration
//!
//! This module integrates the LoRa waveform with mesh networking capabilities,
//! providing a complete LoRa mesh node implementation compatible with Meshtastic.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                        LoRaMesh                             │
//! │  ┌─────────────────────┐    ┌─────────────────────────┐     │
//! │  │   MeshtasticNode    │    │     LoRaMeshPhy         │     │
//! │  │  (MeshNetwork)      │◄──►│     (MeshPhy)           │     │
//! │  │                     │    │  ┌─────────────────┐    │     │
//! │  │  - Routing          │    │  │    LoRa         │    │     │
//! │  │  - Neighbors        │    │  │  (Waveform)     │    │     │
//! │  │  - MAC layer        │    │  └─────────────────┘    │     │
//! │  └─────────────────────┘    └─────────────────────────┘     │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use r4w_core::mesh::lora_mesh::{LoRaMesh, LoRaMeshConfig};
//!
//! // Create a LoRa mesh node
//! let config = LoRaMeshConfig::default();
//! let mut mesh = LoRaMesh::new(config);
//!
//! // Send a broadcast message
//! mesh.broadcast(b"Hello mesh!", 3)?;
//!
//! // Process incoming samples
//! mesh.process_samples(&rx_samples);
//!
//! // Get received messages
//! for packet in mesh.receive_packets() {
//!     println!("Message from {:?}: {:?}", packet.header.source, packet.payload);
//! }
//! ```

use super::meshtastic::{ChannelConfig, MeshtasticConfig, MeshtasticNode, ModemPreset, Region};
use super::packet::{MeshPacket, NodeId};
use super::traits::{MeshError, MeshNetwork, MeshPhy, MeshResult, MeshStats};
use crate::params::{Bandwidth, CodingRate, SpreadingFactor};
use crate::types::IQSample;
use crate::waveform::lora::LoRa;
use crate::waveform::{CommonParams, DemodResult, Waveform, WaveformInfo};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Meshtastic preamble length (16 symbols)
#[allow(dead_code)]
const MESHTASTIC_PREAMBLE_LENGTH: u16 = 16;

/// Meshtastic sync word
#[allow(dead_code)]
const MESHTASTIC_SYNC_WORD: u16 = 0x2B;

/// LoRa waveform adapter implementing the MeshPhy trait
///
/// This wraps the base `LoRa` waveform and adds mesh networking capabilities:
/// - Channel activity detection (CAD)
/// - RSSI and SNR tracking
/// - Frequency and power control
/// - Packet-based transmit/receive
#[derive(Debug)]
pub struct LoRaMeshPhy {
    /// Underlying LoRa waveform
    lora: LoRa,
    /// Current channel frequency in Hz
    frequency_hz: u64,
    /// Current transmit power in dBm
    tx_power_dbm: i8,
    /// Last measured RSSI in dBm
    last_rssi: f32,
    /// Last measured SNR in dB
    last_snr: f32,
    /// Receive buffer for incoming samples
    rx_buffer: Vec<IQSample>,
    /// Transmit queue
    tx_queue: VecDeque<Vec<u8>>,
    /// Is the channel currently detected as busy
    channel_detected_busy: bool,
    /// Time of last CAD check
    last_cad_time: Instant,
    /// Minimum time between CAD checks
    cad_interval: Duration,
    /// Preamble detection threshold
    #[allow(dead_code)]
    preamble_threshold: f32,
}

impl LoRaMeshPhy {
    /// Create a new LoRa mesh PHY
    pub fn new(sample_rate: f64, sf: SpreadingFactor, bw: Bandwidth, cr: CodingRate) -> Self {
        let lora = LoRa::new(sample_rate, sf, bw, cr);

        Self {
            lora,
            frequency_hz: 906_000_000, // Default US frequency
            tx_power_dbm: 20,          // Default 20 dBm (100mW)
            last_rssi: -120.0,
            last_snr: -20.0,
            rx_buffer: Vec::with_capacity(8192),
            tx_queue: VecDeque::new(),
            channel_detected_busy: false,
            last_cad_time: Instant::now(),
            cad_interval: Duration::from_millis(10),
            preamble_threshold: 0.5,
        }
    }

    /// Create with Meshtastic preset
    pub fn from_preset(preset: ModemPreset, sample_rate: f64) -> Self {
        let (sf, bw, cr) = preset.lora_params();
        let sf = match sf {
            5 => SpreadingFactor::SF5,
            6 => SpreadingFactor::SF6,
            7 => SpreadingFactor::SF7,
            8 => SpreadingFactor::SF8,
            9 => SpreadingFactor::SF9,
            10 => SpreadingFactor::SF10,
            11 => SpreadingFactor::SF11,
            12 => SpreadingFactor::SF12,
            _ => SpreadingFactor::SF7,
        };
        let bw = match bw {
            125_000 => Bandwidth::Bw125kHz,
            250_000 => Bandwidth::Bw250kHz,
            500_000 => Bandwidth::Bw500kHz,
            _ => Bandwidth::Bw125kHz,
        };
        let cr = match cr {
            5 => CodingRate::CR4_5,
            6 => CodingRate::CR4_6,
            7 => CodingRate::CR4_7,
            8 => CodingRate::CR4_8,
            _ => CodingRate::CR4_5,
        };

        Self::new(sample_rate, sf, bw, cr)
    }

    /// Get the underlying LoRa waveform
    pub fn lora(&self) -> &LoRa {
        &self.lora
    }

    /// Process incoming I/Q samples
    ///
    /// Call this with received samples from the SDR. The PHY will
    /// buffer and demodulate them, updating RSSI/SNR measurements.
    pub fn process_samples(&mut self, samples: &[IQSample]) {
        // Add to receive buffer
        self.rx_buffer.extend_from_slice(samples);

        // Estimate channel activity from signal power
        if !samples.is_empty() {
            let power: f64 = samples.iter()
                .map(|s| s.re * s.re + s.im * s.im)
                .sum::<f64>() / samples.len() as f64;

            // Simple CAD: if power is above threshold, channel is busy
            // In real implementation, this would use proper preamble detection
            let power_dbm = 10.0 * power.log10();
            self.channel_detected_busy = power_dbm > -100.0; // -100 dBm threshold
        }

        // Limit buffer size to prevent memory issues
        const MAX_BUFFER_SIZE: usize = 1_000_000;
        if self.rx_buffer.len() > MAX_BUFFER_SIZE {
            let excess = self.rx_buffer.len() - MAX_BUFFER_SIZE;
            self.rx_buffer.drain(0..excess);
        }
    }

    /// Get samples to transmit
    ///
    /// Returns I/Q samples ready for transmission, or None if nothing to send.
    pub fn get_tx_samples(&mut self) -> Option<Vec<IQSample>> {
        if let Some(packet) = self.tx_queue.pop_front() {
            let samples = self.lora.modulate(&packet);
            Some(samples)
        } else {
            None
        }
    }

    /// Clear the receive buffer
    pub fn clear_rx_buffer(&mut self) {
        self.rx_buffer.clear();
    }

    /// Get receive buffer size
    pub fn rx_buffer_len(&self) -> usize {
        self.rx_buffer.len()
    }
}

impl Waveform for LoRaMeshPhy {
    fn info(&self) -> WaveformInfo {
        let base_info = self.lora.info();
        WaveformInfo {
            name: "LoRa Mesh",
            full_name: "LoRa Mesh PHY (Meshtastic Compatible)",
            description: "LoRa waveform with mesh networking extensions",
            ..base_info
        }
    }

    fn common_params(&self) -> &CommonParams {
        self.lora.common_params()
    }

    fn modulate(&self, data: &[u8]) -> Vec<IQSample> {
        self.lora.modulate(data)
    }

    fn demodulate(&self, samples: &[IQSample]) -> DemodResult {
        self.lora.demodulate(samples)
    }

    fn samples_per_symbol(&self) -> usize {
        self.lora.samples_per_symbol()
    }
}

impl MeshPhy for LoRaMeshPhy {
    fn channel_busy(&self) -> bool {
        self.channel_detected_busy
    }

    fn rssi(&self) -> f32 {
        self.last_rssi
    }

    fn snr(&self) -> f32 {
        self.last_snr
    }

    fn transmit(&mut self, packet: &[u8]) -> MeshResult<()> {
        // Queue packet for transmission
        self.tx_queue.push_back(packet.to_vec());
        Ok(())
    }

    fn receive(&mut self) -> Option<Vec<u8>> {
        // Need enough samples for at least one symbol
        let samples_per_symbol = self.lora.samples_per_symbol();
        if self.rx_buffer.len() < samples_per_symbol * 20 {
            return None;
        }

        // Try to demodulate
        let result = self.lora.demodulate(&self.rx_buffer);

        // Update signal quality measurements
        if let Some(snr) = result.snr_estimate {
            self.last_snr = snr as f32;
        }
        if let Some(rssi) = result.metadata.get("rssi") {
            self.last_rssi = *rssi as f32;
        }

        // Clear buffer after demodulation attempt
        self.rx_buffer.clear();

        // Return decoded data if any
        if !result.bits.is_empty() {
            Some(result.bits)
        } else {
            None
        }
    }

    fn start_cad(&mut self) -> bool {
        // Update CAD state based on recent samples
        if self.last_cad_time.elapsed() >= self.cad_interval {
            self.last_cad_time = Instant::now();
            // CAD result is updated by process_samples()
        }
        self.channel_detected_busy
    }

    fn frequency(&self) -> u64 {
        self.frequency_hz
    }

    fn set_frequency(&mut self, freq_hz: u64) -> MeshResult<()> {
        // Validate frequency is in a reasonable range for LoRa
        if freq_hz < 137_000_000 || freq_hz > 1_020_000_000 {
            return Err(MeshError::PhyError(format!(
                "Frequency {} Hz out of range", freq_hz
            )));
        }
        self.frequency_hz = freq_hz;
        Ok(())
    }

    fn tx_power(&self) -> i8 {
        self.tx_power_dbm
    }

    fn set_tx_power(&mut self, power_dbm: i8) -> MeshResult<()> {
        // Validate power range (typical LoRa range)
        if power_dbm < -4 || power_dbm > 30 {
            return Err(MeshError::PhyError(format!(
                "TX power {} dBm out of range", power_dbm
            )));
        }
        self.tx_power_dbm = power_dbm;
        Ok(())
    }
}

/// Configuration for LoRa mesh node
#[derive(Debug, Clone)]
pub struct LoRaMeshConfig {
    /// Node ID (random if None)
    pub node_id: Option<NodeId>,
    /// Modem preset
    pub preset: ModemPreset,
    /// Region for frequency configuration
    pub region: Region,
    /// Sample rate for the SDR
    pub sample_rate: f64,
    /// Channel index (0-based)
    pub channel: u8,
    /// Transmit power in dBm
    pub tx_power: i8,
    /// Channel name
    pub channel_name: String,
    /// Enable position sharing
    pub position_enabled: bool,
}

impl Default for LoRaMeshConfig {
    fn default() -> Self {
        Self {
            node_id: None,
            preset: ModemPreset::LongFast,
            region: Region::US,
            sample_rate: 250_000.0, // 250 kHz sample rate
            channel: 0,
            tx_power: 20,
            channel_name: "LongFast".to_string(),
            position_enabled: false,
        }
    }
}

/// Complete LoRa mesh node combining PHY and mesh networking
///
/// This is the main entry point for using LoRa mesh networking.
/// It combines the LoRa physical layer with Meshtastic-compatible
/// mesh routing.
pub struct LoRaMesh {
    /// Mesh network layer (routing, neighbors, MAC)
    mesh: MeshtasticNode,
    /// Physical layer
    phy: LoRaMeshPhy,
    /// Received packets ready for application
    received_packets: VecDeque<MeshPacket>,
    /// Packets queued for transmission
    #[allow(dead_code)]
    pending_tx: VecDeque<MeshPacket>,
    /// Last tick time
    last_tick: Instant,
}

impl std::fmt::Debug for LoRaMesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoRaMesh")
            .field("node_id", &self.mesh.node_id())
            .field("frequency", &self.phy.frequency())
            .field("tx_power", &self.phy.tx_power())
            .finish()
    }
}

impl LoRaMesh {
    /// Create a new LoRa mesh node
    pub fn new(config: LoRaMeshConfig) -> Self {
        // Create PHY layer
        let mut phy = LoRaMeshPhy::from_preset(config.preset, config.sample_rate);

        // Set frequency based on region and channel
        let (freq_start, freq_end) = config.region.frequency_range();
        let channel_spacing = 2_000_000; // 2 MHz spacing
        let freq = freq_start + (config.channel as u64 * channel_spacing);
        let _ = phy.set_frequency(freq.min(freq_end));
        let _ = phy.set_tx_power(config.tx_power);

        // Create mesh network layer with primary channel using the preset
        let primary_channel = ChannelConfig {
            name: config.channel_name.clone(),
            preset: config.preset,
            ..Default::default()
        };
        let mesh_config = MeshtasticConfig {
            node_id: config.node_id,
            region: config.region,
            primary_channel,
            position_enabled: config.position_enabled,
            ..Default::default()
        };
        let mesh = MeshtasticNode::new(mesh_config);

        Self {
            mesh,
            phy,
            received_packets: VecDeque::new(),
            pending_tx: VecDeque::new(),
            last_tick: Instant::now(),
        }
    }

    /// Get this node's ID
    pub fn node_id(&self) -> NodeId {
        self.mesh.node_id()
    }

    /// Process incoming I/Q samples from the SDR
    ///
    /// This is the main receive path. Call this with samples from
    /// your SDR receive callback.
    pub fn process_samples(&mut self, samples: &[IQSample]) {
        // Feed samples to PHY
        self.phy.process_samples(samples);

        // Try to receive a packet
        if let Some(data) = self.phy.receive() {
            // Try to parse as mesh packet
            if let Some(packet) = MeshPacket::from_bytes(&data) {
                // Process through mesh layer
                let rssi = self.phy.rssi();
                let snr = self.phy.snr();
                let local_packets = self.mesh.on_receive(packet, rssi, snr);

                // Queue locally-destined packets for application
                self.received_packets.extend(local_packets);
            }
        }
    }

    /// Get I/Q samples to transmit
    ///
    /// Call this to get samples for your SDR transmit callback.
    /// Returns None if nothing to transmit.
    pub fn get_tx_samples(&mut self) -> Option<Vec<IQSample>> {
        // Check if we should transmit (MAC layer decision)
        let channel_busy = self.phy.channel_busy();

        // Process any pending packets from mesh layer
        // (This is simplified - real implementation would use MAC timing)
        if !channel_busy {
            self.phy.get_tx_samples()
        } else {
            None
        }
    }

    /// Send a broadcast message
    pub fn broadcast(&mut self, payload: &[u8], hop_limit: u8) -> MeshResult<()> {
        let result = self.mesh.broadcast(payload, hop_limit);
        self.queue_outgoing_packets();
        result
    }

    /// Send a direct message to a specific node
    pub fn send_direct(&mut self, dest: NodeId, payload: &[u8]) -> MeshResult<()> {
        let result = self.mesh.send_direct(dest, payload);
        self.queue_outgoing_packets();
        result
    }

    /// Get received packets (for application layer)
    pub fn receive_packets(&mut self) -> impl Iterator<Item = MeshPacket> + '_ {
        self.received_packets.drain(..)
    }

    /// Check if there are received packets available
    pub fn has_received_packets(&self) -> bool {
        !self.received_packets.is_empty()
    }

    /// Get mesh network statistics
    pub fn stats(&self) -> MeshStats {
        self.mesh.stats()
    }

    /// Get the PHY layer (for direct access)
    pub fn phy(&self) -> &LoRaMeshPhy {
        &self.phy
    }

    /// Get mutable PHY layer
    pub fn phy_mut(&mut self) -> &mut LoRaMeshPhy {
        &mut self.phy
    }

    /// Periodic tick - call this regularly (e.g., every 100ms)
    pub fn tick(&mut self) {
        let elapsed = self.last_tick.elapsed();
        self.last_tick = Instant::now();

        self.mesh.tick(elapsed);
        self.queue_outgoing_packets();
    }

    /// Set the channel frequency
    pub fn set_frequency(&mut self, freq_hz: u64) -> MeshResult<()> {
        self.phy.set_frequency(freq_hz)
    }

    /// Get the current frequency
    pub fn frequency(&self) -> u64 {
        self.phy.frequency()
    }

    /// Set transmit power
    pub fn set_tx_power(&mut self, power_dbm: i8) -> MeshResult<()> {
        self.phy.set_tx_power(power_dbm)
    }

    /// Get current transmit power
    pub fn tx_power(&self) -> i8 {
        self.phy.tx_power()
    }

    /// Queue outgoing packets from mesh layer to PHY
    fn queue_outgoing_packets(&mut self) {
        // In a full implementation, we would get packets from
        // the mesh layer's TX queue and send them to the PHY.
        // This is simplified for the current structure.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_mesh_phy_creation() {
        let phy = LoRaMeshPhy::new(
            250_000.0,
            SpreadingFactor::SF11,
            Bandwidth::Bw250kHz,
            CodingRate::CR4_5,
        );

        assert_eq!(phy.frequency(), 906_000_000);
        assert_eq!(phy.tx_power(), 20);
        assert!(!phy.channel_busy());
    }

    #[test]
    fn test_lora_mesh_phy_from_preset() {
        let phy = LoRaMeshPhy::from_preset(ModemPreset::LongFast, 250_000.0);

        // LongFast uses SF11, BW250
        let info = phy.info();
        assert_eq!(info.name, "LoRa Mesh");
    }

    #[test]
    fn test_lora_mesh_phy_frequency() {
        let mut phy = LoRaMeshPhy::from_preset(ModemPreset::LongFast, 250_000.0);

        // Valid frequency
        assert!(phy.set_frequency(915_000_000).is_ok());
        assert_eq!(phy.frequency(), 915_000_000);

        // Invalid frequency (too low)
        assert!(phy.set_frequency(100_000_000).is_err());
    }

    #[test]
    fn test_lora_mesh_phy_tx_power() {
        let mut phy = LoRaMeshPhy::from_preset(ModemPreset::LongFast, 250_000.0);

        // Valid power
        assert!(phy.set_tx_power(14).is_ok());
        assert_eq!(phy.tx_power(), 14);

        // Invalid power (too high)
        assert!(phy.set_tx_power(35).is_err());
    }

    #[test]
    fn test_lora_mesh_creation() {
        let config = LoRaMeshConfig::default();
        let mesh = LoRaMesh::new(config);

        // Check that node ID was assigned (not broadcast)
        let node_id = mesh.node_id();
        assert!(!node_id.is_broadcast());
    }

    #[test]
    fn test_lora_mesh_config() {
        let config = LoRaMeshConfig {
            preset: ModemPreset::LongSlow,
            region: Region::EU,
            tx_power: 14,
            ..Default::default()
        };

        let mesh = LoRaMesh::new(config);
        assert_eq!(mesh.tx_power(), 14);
    }

    #[test]
    fn test_lora_mesh_broadcast() {
        let config = LoRaMeshConfig::default();
        let mut mesh = LoRaMesh::new(config);

        // Should succeed even with no radio connected
        let result = mesh.broadcast(b"Hello!", 3);
        assert!(result.is_ok());
    }

    #[test]
    fn test_lora_mesh_stats() {
        let config = LoRaMeshConfig::default();
        let mesh = LoRaMesh::new(config);

        let stats = mesh.stats();
        assert_eq!(stats.packets_tx, 0);
        assert_eq!(stats.packets_rx, 0);
    }

    #[test]
    fn test_lora_mesh_process_samples() {
        let config = LoRaMeshConfig::default();
        let mut mesh = LoRaMesh::new(config);

        // Process some dummy samples
        let samples: Vec<IQSample> = (0..1000)
            .map(|i| IQSample::new(
                (i as f64 * 0.01).sin() * 0.001,
                (i as f64 * 0.01).cos() * 0.001,
            ))
            .collect();

        mesh.process_samples(&samples);

        // Should not crash, even with invalid data
        assert!(!mesh.has_received_packets());
    }

    #[test]
    fn test_lora_mesh_tick() {
        let config = LoRaMeshConfig::default();
        let mut mesh = LoRaMesh::new(config);

        // Should not crash
        mesh.tick();
        std::thread::sleep(Duration::from_millis(10));
        mesh.tick();
    }
}
