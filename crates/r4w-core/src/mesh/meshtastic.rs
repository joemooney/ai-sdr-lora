//! Meshtastic Protocol Implementation
//!
//! This module implements the Meshtastic mesh networking protocol, enabling
//! R4W to participate in the global Meshtastic network with 40,000+ nodes.
//!
//! ## Protocol Overview
//!
//! - Physical Layer: LoRa CSS modulation with 16-symbol preamble, sync word 0x2B
//! - MAC Layer: CSMA/CA with SNR-based contention windows
//! - Routing: Managed flood for broadcasts, next-hop for direct messages
//! - Messages: Protobuf-encoded (Position, Text, NodeInfo, Telemetry, etc.)
//!
//! ## Modem Presets
//!
//! | Preset | SF | BW | CR | Description |
//! |--------|----|----|-----|-------------|
//! | LongFast | 11 | 250 | 4/5 | Long range, higher throughput |
//! | LongSlow | 12 | 125 | 4/8 | Maximum range, lowest throughput |
//! | LongModerate | 11 | 125 | 4/8 | Long range, moderate throughput |
//! | MediumFast | 9 | 250 | 4/5 | Medium range |
//! | MediumSlow | 10 | 250 | 4/5 | Medium range, lower throughput |
//! | ShortFast | 7 | 250 | 4/5 | Short range, highest throughput |
//! | ShortSlow | 8 | 250 | 4/5 | Short range |

#[cfg(feature = "crypto")]
use super::crypto::{ChannelKey, CryptoContext};
use super::mac::{CsmaConfig, MacLayer, TransmitDecision};
use super::neighbor::{Neighbor, NeighborTable, NodeInfo};
use super::packet::{MeshPacket, NodeId, PacketType};
#[cfg(feature = "meshtastic-interop")]
use super::proto::{Data, PortNum, Position as ProtoPosition, User as ProtoUser};
#[cfg(feature = "meshtastic-interop")]
use super::proto::{DeviceMetrics as ProtoDeviceMetrics, Telemetry as ProtoTelemetry, TelemetryVariant as ProtoTelemetryVariant};
#[cfg(feature = "meshtastic-interop")]
use prost::Message;
use super::routing::{FloodRouter, NextHopRouter, Route};
use super::telemetry::{DeviceMetrics, EnvironmentMetrics, Telemetry, TelemetryConfig, TelemetryVariant};
use super::traits::{MeshError, MeshNetwork, MeshResult, MeshStats};
use super::wire::{WireHeader, WIRE_HEADER_SIZE};
use std::time::{Duration, Instant};

/// Meshtastic modem presets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModemPreset {
    /// Long range, fast: SF11, BW250, CR4/5
    LongFast,
    /// Long range, slow (maximum range): SF12, BW125, CR4/8
    LongSlow,
    /// Long range, moderate: SF11, BW125, CR4/8
    LongModerate,
    /// Medium range, fast: SF9, BW250, CR4/5
    MediumFast,
    /// Medium range, slow: SF10, BW250, CR4/5
    MediumSlow,
    /// Short range, fast (highest throughput): SF7, BW250, CR4/5
    ShortFast,
    /// Short range, slow: SF8, BW250, CR4/5
    ShortSlow,
}

impl ModemPreset {
    /// Get LoRa parameters for this preset
    pub fn lora_params(&self) -> (u8, u32, u8) {
        // Returns (spreading_factor, bandwidth_hz, coding_rate)
        match self {
            ModemPreset::LongFast => (11, 250_000, 5),
            ModemPreset::LongSlow => (12, 125_000, 8),
            ModemPreset::LongModerate => (11, 125_000, 8),
            ModemPreset::MediumFast => (9, 250_000, 5),
            ModemPreset::MediumSlow => (10, 250_000, 5),
            ModemPreset::ShortFast => (7, 250_000, 5),
            ModemPreset::ShortSlow => (8, 250_000, 5),
        }
    }

    /// Get the sync word for Meshtastic (0x2B)
    pub fn sync_word() -> u8 {
        0x2B
    }

    /// Get preamble length (16 symbols)
    pub fn preamble_length() -> u8 {
        16
    }
}

impl Default for ModemPreset {
    fn default() -> Self {
        ModemPreset::LongFast
    }
}

/// Regional frequency configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    /// US (902-928 MHz)
    US,
    /// EU (863-870 MHz)
    EU,
    /// China (470-510 MHz)
    CN,
    /// Japan (920-925 MHz)
    JP,
    /// Australia/New Zealand (915-928 MHz)
    ANZ,
    /// Korea (920-923 MHz)
    KR,
    /// Taiwan (920-925 MHz)
    TW,
    /// India (865-867 MHz)
    IN,
    /// Unset (use EU defaults)
    Unset,
}

impl Region {
    /// Get frequency range for region
    pub fn frequency_range(&self) -> (u64, u64) {
        match self {
            Region::US => (902_000_000, 928_000_000),
            Region::EU | Region::Unset => (863_000_000, 870_000_000),
            Region::CN => (470_000_000, 510_000_000),
            Region::JP => (920_000_000, 925_000_000),
            Region::ANZ => (915_000_000, 928_000_000),
            Region::KR => (920_000_000, 923_000_000),
            Region::TW => (920_000_000, 925_000_000),
            Region::IN => (865_000_000, 867_000_000),
        }
    }

    /// Get default primary frequency
    pub fn primary_frequency(&self) -> u64 {
        let (start, end) = self.frequency_range();
        (start + end) / 2
    }

    /// Get duty cycle limit (fraction)
    pub fn duty_cycle_limit(&self) -> f32 {
        match self {
            Region::EU => 0.01, // 1% in EU
            _ => 1.0,          // No limit (check local regulations)
        }
    }
}

impl Default for Region {
    fn default() -> Self {
        Region::US
    }
}

/// Channel configuration
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    /// Channel name
    pub name: String,
    /// Pre-shared key (32 bytes, None for unencrypted)
    pub psk: Option<[u8; 32]>,
    /// Modem preset
    pub preset: ModemPreset,
    /// Uplink enabled (MQTT gateway)
    pub uplink_enabled: bool,
    /// Downlink enabled (receive from MQTT)
    pub downlink_enabled: bool,
}

impl ChannelConfig {
    /// Create a new channel configuration
    pub fn new(name: &str, preset: ModemPreset) -> Self {
        Self {
            name: name.to_string(),
            psk: None,
            preset,
            uplink_enabled: false,
            downlink_enabled: false,
        }
    }

    /// Create an encrypted channel with the given PSK
    pub fn with_psk(name: &str, psk: [u8; 32], preset: ModemPreset) -> Self {
        Self {
            name: name.to_string(),
            psk: Some(psk),
            preset,
            uplink_enabled: false,
            downlink_enabled: false,
        }
    }

    /// Create an encrypted channel using the default Meshtastic PSK
    #[cfg(feature = "crypto")]
    pub fn with_default_psk(name: &str, preset: ModemPreset) -> Self {
        use super::crypto::DEFAULT_PSK;
        let mut psk = [0u8; 32];
        // Extend default PSK to 32 bytes
        psk[..DEFAULT_PSK.len()].copy_from_slice(DEFAULT_PSK);
        Self::with_psk(name, psk, preset)
    }

    /// Check if this channel uses encryption
    pub fn is_encrypted(&self) -> bool {
        self.psk.is_some()
    }

    /// Create a CryptoContext for this channel (if encrypted)
    #[cfg(feature = "crypto")]
    pub fn crypto_context(&self) -> Option<CryptoContext> {
        self.psk.map(|psk| {
            let key = ChannelKey::from_raw(psk, &self.name);
            CryptoContext::from_key(key)
        })
    }

    /// Get channel hash for wire format
    pub fn channel_hash(&self) -> u8 {
        #[cfg(feature = "crypto")]
        if let Some(ref ctx) = self.crypto_context() {
            return ctx.channel_hash();
        }
        // Simple hash without crypto feature
        self.name.bytes().fold(0u8, |acc, b| acc.wrapping_add(b))
    }
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            name: "LongFast".to_string(),
            psk: None, // Default public channel
            preset: ModemPreset::LongFast,
            uplink_enabled: false,
            downlink_enabled: false,
        }
    }
}

/// Meshtastic node configuration
#[derive(Debug, Clone)]
pub struct MeshtasticConfig {
    /// Node ID (random if None)
    pub node_id: Option<NodeId>,
    /// Short name (up to 4 chars)
    pub short_name: String,
    /// Long name (up to 40 chars)
    pub long_name: String,
    /// Hardware model ID
    pub hardware_model: u8,
    /// Region for frequency selection
    pub region: Region,
    /// Primary channel configuration
    pub primary_channel: ChannelConfig,
    /// Secondary channels (up to 7)
    pub secondary_channels: Vec<ChannelConfig>,
    /// Position sharing enabled
    pub position_enabled: bool,
    /// Position update interval (seconds)
    pub position_interval: u64,
    /// Node is a router (longer range settings)
    pub is_router: bool,
    /// Hop limit for broadcasts
    pub hop_limit: u8,
    /// Telemetry configuration
    pub telemetry: TelemetryConfig,
    /// Enable encryption on primary channel
    pub encryption_enabled: bool,
}

impl Default for MeshtasticConfig {
    fn default() -> Self {
        Self {
            node_id: None,
            short_name: "NODE".to_string(),
            long_name: "R4W Node".to_string(),
            hardware_model: 0xFF, // Unknown/custom
            region: Region::default(),
            primary_channel: ChannelConfig::default(),
            secondary_channels: Vec::new(),
            position_enabled: false,
            position_interval: 900, // 15 minutes
            is_router: false,
            hop_limit: 3,
            telemetry: TelemetryConfig::default(),
            encryption_enabled: false,
        }
    }
}

/// Meshtastic mesh node implementation
#[derive(Debug)]
pub struct MeshtasticNode {
    /// Node configuration
    config: MeshtasticConfig,
    /// Our node ID
    node_id: NodeId,
    /// Node information
    node_info: NodeInfo,
    /// Neighbor table
    neighbors: NeighborTable,
    /// Flood router
    flood_router: FloodRouter,
    /// Next-hop router
    next_hop_router: NextHopRouter,
    /// MAC layer
    mac: MacLayer,
    /// Statistics
    stats: MeshStats,
    /// Application-layer receive queue
    rx_queue: Vec<MeshPacket>,
    /// Last position broadcast time
    last_position_broadcast: Option<Instant>,
    /// Last node info broadcast time
    last_nodeinfo_broadcast: Option<Instant>,
    /// Crypto context for primary channel (when encryption enabled)
    #[cfg(feature = "crypto")]
    crypto: Option<CryptoContext>,
    /// Current device metrics
    device_metrics: DeviceMetrics,
    /// Current environment metrics (if sensors available)
    environment_metrics: Option<EnvironmentMetrics>,
    /// Last telemetry broadcast time
    last_telemetry_broadcast: Option<Instant>,
    /// Node start time for uptime calculation
    start_time: Instant,
}

impl MeshtasticNode {
    /// Create a new Meshtastic node
    pub fn new(config: MeshtasticConfig) -> Self {
        let node_id = config.node_id.unwrap_or_else(NodeId::random);

        let node_info = NodeInfo {
            node_id,
            short_name: config.short_name.clone(),
            long_name: config.long_name.clone(),
            hardware_model: config.hardware_model,
            firmware_version: env!("CARGO_PKG_VERSION").to_string(),
            position: None,
            battery_level: None,
            is_router: config.is_router,
        };

        // Configure MAC for Meshtastic
        let mac_config = CsmaConfig {
            cw_min: 16,
            cw_max: 256,
            slot_time_ms: 10,
            max_backoff_attempts: 7,
            difs_ms: 50,
            target_utilization: config.region.duty_cycle_limit(),
            cad_threshold: -115.0,
        };

        // Initialize crypto context if encryption is enabled
        #[cfg(feature = "crypto")]
        let crypto = if config.encryption_enabled {
            config.primary_channel.crypto_context()
        } else {
            None
        };

        Self {
            node_id,
            flood_router: FloodRouter::new(node_id),
            next_hop_router: NextHopRouter::new(node_id),
            mac: MacLayer::new(mac_config),
            neighbors: NeighborTable::new(7200, 256),
            node_info,
            config,
            stats: MeshStats::default(),
            rx_queue: Vec::new(),
            last_position_broadcast: None,
            last_nodeinfo_broadcast: None,
            #[cfg(feature = "crypto")]
            crypto,
            device_metrics: DeviceMetrics::default(),
            environment_metrics: None,
            last_telemetry_broadcast: None,
            start_time: Instant::now(),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(MeshtasticConfig::default())
    }

    /// Get the node's configuration
    pub fn config(&self) -> &MeshtasticConfig {
        &self.config
    }

    /// Set the node's position
    pub fn set_position(&mut self, lat: f64, lon: f64, alt: f32) {
        self.node_info.position = Some((lat, lon, alt));
    }

    /// Set battery level
    pub fn set_battery_level(&mut self, level: u8) {
        self.node_info.battery_level = Some(level.min(100));
        self.device_metrics.battery_level = Some(level.min(100));
    }

    /// Update device metrics
    pub fn update_device_metrics(&mut self) {
        // Update uptime
        self.device_metrics.uptime_seconds = Some(self.start_time.elapsed().as_secs() as u32);

        // Update channel utilization from MAC layer
        self.device_metrics.channel_utilization = Some(self.mac.channel_utilization_cached());
    }

    /// Set voltage reading
    pub fn set_voltage(&mut self, voltage: f32) {
        self.device_metrics.voltage = Some(voltage);
    }

    /// Set air utilization TX percentage
    pub fn set_air_util_tx(&mut self, util: f32) {
        self.device_metrics.air_util_tx = Some(util);
    }

    /// Set environment metrics (for nodes with sensors)
    pub fn set_environment_metrics(&mut self, metrics: EnvironmentMetrics) {
        self.environment_metrics = Some(metrics);
    }

    /// Get current device metrics
    pub fn device_metrics(&self) -> &DeviceMetrics {
        &self.device_metrics
    }

    /// Get current environment metrics
    pub fn environment_metrics(&self) -> Option<&EnvironmentMetrics> {
        self.environment_metrics.as_ref()
    }

    /// Check if encryption is enabled
    pub fn is_encrypted(&self) -> bool {
        #[cfg(feature = "crypto")]
        return self.crypto.is_some();
        #[cfg(not(feature = "crypto"))]
        return false;
    }

    /// Get packets ready for application layer
    pub fn receive(&mut self) -> Vec<MeshPacket> {
        std::mem::take(&mut self.rx_queue)
    }

    /// Get LoRa parameters for current channel
    pub fn lora_params(&self) -> (u8, u32, u8) {
        self.config.primary_channel.preset.lora_params()
    }

    /// Get the primary frequency for current region
    pub fn frequency(&self) -> u64 {
        self.config.region.primary_frequency()
    }

    /// Process channel state and get packet to transmit (if any)
    pub fn process_tx(&mut self, channel_busy: bool) -> Option<Vec<u8>> {
        // Check for pending rebroadcasts
        if let Some(packet) = self.flood_router.get_pending_rebroadcast() {
            if self.queue_packet(&packet).is_ok() {
                self.stats.packets_forwarded += 1;
            }
        }

        // Check MAC layer
        match self.mac.can_transmit(channel_busy) {
            TransmitDecision::TransmitNow => {
                if let Some(packet) = self.mac.start_tx() {
                    self.stats.packets_tx += 1;
                    self.stats.bytes_tx += packet.len() as u64;
                    Some(packet)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Called when transmission is complete
    pub fn tx_complete(&mut self, duration: Duration) {
        self.mac.tx_complete(duration);
    }

    /// Queue a packet for transmission using Meshtastic wire format
    ///
    /// Wire format: WireHeader (16 bytes) + payload + optional MIC (4 bytes)
    fn queue_packet(&mut self, packet: &MeshPacket) -> MeshResult<()> {
        // Get channel hash for wire header
        let channel_hash = self.config.primary_channel.channel_hash();

        // Convert internal header to wire format
        let wire_header = WireHeader::from_packet_header(&packet.header, channel_hash);
        let header_bytes = wire_header.to_bytes();

        // Build the wire packet
        let mut bytes = Vec::with_capacity(WIRE_HEADER_SIZE + packet.payload.len() + 4);

        // Add wire header (16 bytes, little-endian)
        bytes.extend_from_slice(&header_bytes);

        // Add payload (encrypted if crypto enabled)
        #[cfg(feature = "crypto")]
        if let Some(ref crypto) = self.crypto {
            // Encrypt payload and append MIC (MIC is computed over header + ciphertext)
            match crypto.encrypt(
                &packet.payload,
                packet.header.source,
                packet.header.packet_id as u32,
                &header_bytes,
            ) {
                Ok((ciphertext, mic)) => {
                    bytes.extend_from_slice(&ciphertext);
                    bytes.extend_from_slice(&mic);
                }
                Err(_) => {
                    // Fall back to unencrypted on error
                    bytes.extend_from_slice(&packet.payload);
                }
            }
        } else {
            bytes.extend_from_slice(&packet.payload);
        }

        #[cfg(not(feature = "crypto"))]
        bytes.extend_from_slice(&packet.payload);

        self.mac.queue_tx(bytes).map_err(|_| MeshError::QueueFull)
    }

    /// Process raw bytes received from radio using Meshtastic wire format
    ///
    /// Wire format: WireHeader (16 bytes) + payload + optional MIC (4 bytes)
    ///
    /// Returns parsed packets ready for application layer, or empty if packet
    /// was invalid, duplicate, or not for us.
    pub fn receive_bytes(&mut self, data: &[u8], rssi: f32, snr: f32) -> Vec<MeshPacket> {
        // Need at least header + 1 byte payload
        if data.len() < WIRE_HEADER_SIZE + 1 {
            return Vec::new();
        }

        // Parse wire header
        let wire_header = match WireHeader::from_bytes(&data[..WIRE_HEADER_SIZE]) {
            Some(h) => h,
            None => return Vec::new(),
        };

        // Validate channel hash (quick rejection for wrong channel)
        let expected_hash = self.config.primary_channel.channel_hash();
        if wire_header.channel_hash != expected_hash {
            // Check secondary channels
            let matches_secondary = self.config.secondary_channels.iter()
                .any(|ch| ch.channel_hash() == wire_header.channel_hash);
            if !matches_secondary {
                // Packet is for a different channel
                return Vec::new();
            }
        }

        // Extract payload (and MIC if encrypted)
        let header_bytes = &data[..WIRE_HEADER_SIZE];
        let payload_and_mic = &data[WIRE_HEADER_SIZE..];

        // Try to decrypt/parse payload
        let payload = self.extract_payload(
            payload_and_mic,
            wire_header.source_node_id(),
            wire_header.id,
            header_bytes,
        );

        let payload = match payload {
            Some(p) => p,
            None => return Vec::new(),
        };

        // Convert wire header to internal format and create packet
        let header = wire_header.to_packet_header();

        // Decode protobuf Data envelope if meshtastic-interop is enabled
        #[cfg(feature = "meshtastic-interop")]
        let (packet_type, inner_payload) = self.decode_proto_payload(&payload);

        #[cfg(not(feature = "meshtastic-interop"))]
        let (packet_type, inner_payload) = (PacketType::Text, payload);

        let packet = MeshPacket {
            header,
            packet_type,
            payload: inner_payload,
            mic: None,
            rx_rssi: None,
            rx_snr: None,
            rx_time: None,
        };

        // Process through normal receive path
        self.on_receive(packet, rssi, snr)
    }

    /// Extract and optionally decrypt payload from wire format
    #[cfg(feature = "crypto")]
    fn extract_payload(
        &self,
        payload_and_mic: &[u8],
        source: NodeId,
        packet_id: u32,
        header_bytes: &[u8],
    ) -> Option<Vec<u8>> {
        if let Some(ref crypto) = self.crypto {
            // Encrypted: payload + 4-byte MIC
            if payload_and_mic.len() < 5 {
                return None; // Need at least 1 byte payload + 4 byte MIC
            }

            let mic_start = payload_and_mic.len() - 4;
            let ciphertext = &payload_and_mic[..mic_start];
            let mic: [u8; 4] = payload_and_mic[mic_start..].try_into().ok()?;

            // Decrypt
            crypto.decrypt(ciphertext, source, packet_id, header_bytes, &mic).ok()
        } else {
            // Unencrypted: just return payload as-is
            Some(payload_and_mic.to_vec())
        }
    }

    /// Extract payload (no crypto support)
    #[cfg(not(feature = "crypto"))]
    fn extract_payload(
        &self,
        payload_and_mic: &[u8],
        _source: NodeId,
        _packet_id: u32,
        _header_bytes: &[u8],
    ) -> Option<Vec<u8>> {
        Some(payload_and_mic.to_vec())
    }

    // ========================================================================
    // Protobuf RX decoding (meshtastic-interop feature)
    // ========================================================================

    /// Decode protobuf Data message and return (packet_type, inner_payload)
    ///
    /// If decoding fails, returns the original payload as Text type.
    #[cfg(feature = "meshtastic-interop")]
    fn decode_proto_payload(&self, payload: &[u8]) -> (PacketType, Vec<u8>) {
        match Data::decode(payload) {
            Ok(data) => {
                let packet_type = self.portnum_to_packet_type(PortNum::from_u32(data.portnum as u32));
                // For text messages, return the inner payload bytes directly
                // For other types, the payload is already the encoded inner message
                (packet_type, data.payload)
            }
            Err(_) => {
                // Not a valid protobuf message, treat as raw text
                (PacketType::Text, payload.to_vec())
            }
        }
    }

    /// Convert PortNum to PacketType
    #[cfg(feature = "meshtastic-interop")]
    fn portnum_to_packet_type(&self, portnum: PortNum) -> PacketType {
        match portnum {
            PortNum::Text | PortNum::TextMessageCompressed => PacketType::Text,
            PortNum::Position => PacketType::Position,
            PortNum::NodeInfo => PacketType::NodeInfo,
            PortNum::Routing => PacketType::Routing,
            PortNum::Telemetry => PacketType::Telemetry,
            PortNum::Admin => PacketType::Admin,
            _ => PacketType::Custom,
        }
    }

    /// Broadcast node info
    fn broadcast_node_info(&mut self) {
        let packet = MeshPacket::node_info(
            self.node_id,
            &self.node_info.short_name,
            &self.node_info.long_name,
        );
        let _ = self.queue_packet(&packet);
        self.last_nodeinfo_broadcast = Some(Instant::now());
    }

    /// Broadcast position
    fn broadcast_position(&mut self) {
        if let Some((lat, lon, alt)) = self.node_info.position {
            let packet = MeshPacket::position(self.node_id, lat, lon, alt);
            let _ = self.queue_packet(&packet);
            self.last_position_broadcast = Some(Instant::now());
        }
    }

    /// Broadcast telemetry
    fn broadcast_telemetry(&mut self) {
        // Update metrics before broadcasting
        self.update_device_metrics();

        let telemetry = Telemetry {
            time: self.start_time.elapsed().as_secs() as u32,
            variant: TelemetryVariant::Device(self.device_metrics.clone()),
        };

        let packet = MeshPacket::telemetry(self.node_id, &telemetry);
        let _ = self.queue_packet(&packet);
        self.last_telemetry_broadcast = Some(Instant::now());
    }

    // ========================================================================
    // Protobuf-enabled TX methods (meshtastic-interop feature)
    // ========================================================================

    /// Send a text message using Meshtastic protobuf encoding
    ///
    /// The message is wrapped in a proto::Data message with PortNum::Text.
    #[cfg(feature = "meshtastic-interop")]
    pub fn send_text(&mut self, message: &str, hop_limit: u8) -> MeshResult<()> {
        let data = Data::text(message);
        let payload = data.encode_to_vec();
        self.send_proto_packet(payload, PacketType::Text, hop_limit, None)
    }

    /// Send a text message to a specific node using Meshtastic protobuf encoding
    #[cfg(feature = "meshtastic-interop")]
    pub fn send_text_to(&mut self, message: &str, destination: NodeId) -> MeshResult<()> {
        let data = Data::text(message);
        let payload = data.encode_to_vec();
        self.send_proto_packet(payload, PacketType::Text, 3, Some(destination))
    }

    /// Send position using Meshtastic protobuf encoding
    #[cfg(feature = "meshtastic-interop")]
    pub fn send_position(&mut self, lat: f64, lon: f64, alt: i32) -> MeshResult<()> {
        let pos = ProtoPosition::from_coords(lat, lon, alt);
        let data = Data::position(pos);
        let payload = data.encode_to_vec();
        self.send_proto_packet(payload, PacketType::Position, 3, None)
    }

    /// Send node info (User message) using Meshtastic protobuf encoding
    #[cfg(feature = "meshtastic-interop")]
    pub fn send_node_info(&mut self) -> MeshResult<()> {
        let user = ProtoUser::new(
            &format!("!{:08x}", self.node_id.to_u32()),
            &self.node_info.short_name,
            &self.node_info.long_name,
        );
        let data = Data::user(user);
        let payload = data.encode_to_vec();
        self.send_proto_packet(payload, PacketType::NodeInfo, 3, None)
    }

    /// Send telemetry using Meshtastic protobuf encoding
    #[cfg(feature = "meshtastic-interop")]
    pub fn send_telemetry_proto(&mut self) -> MeshResult<()> {
        self.update_device_metrics();

        let dm = ProtoDeviceMetrics::from_internal(&self.device_metrics);
        let telemetry = ProtoTelemetry {
            time: self.start_time.elapsed().as_secs() as u32,
            variant: Some(ProtoTelemetryVariant::DeviceMetrics(dm)),
        };
        let data = Data::telemetry(telemetry);
        let payload = data.encode_to_vec();
        self.send_proto_packet(payload, PacketType::Telemetry, 3, None)
    }

    /// Internal helper to send a protobuf-encoded packet
    #[cfg(feature = "meshtastic-interop")]
    fn send_proto_packet(
        &mut self,
        payload: Vec<u8>,
        packet_type: PacketType,
        hop_limit: u8,
        destination: Option<NodeId>,
    ) -> MeshResult<()> {
        let header = if let Some(dest) = destination {
            super::packet::PacketHeader::direct(self.node_id, dest)
        } else {
            super::packet::PacketHeader::broadcast(self.node_id, hop_limit)
        };

        let packet = MeshPacket {
            header,
            packet_type,
            payload,
            mic: None,
            rx_rssi: None,
            rx_snr: None,
            rx_time: None,
        };

        self.queue_packet(&packet)
    }
}

impl MeshNetwork for MeshtasticNode {
    fn node_id(&self) -> NodeId {
        self.node_id
    }

    fn discover_neighbors(&mut self) -> Vec<Neighbor> {
        // Broadcast node info to announce ourselves
        let should_broadcast = self.last_nodeinfo_broadcast
            .map(|t| t.elapsed() > Duration::from_secs(900))
            .unwrap_or(true);

        if should_broadcast {
            self.broadcast_node_info();
        }

        // Return current neighbors
        self.neighbors.active().into_iter().cloned().collect()
    }

    fn neighbors(&self) -> &[Neighbor] {
        // Note: This returns empty slice due to ownership constraints
        // Use discover_neighbors() for actual neighbor list
        &[]
    }

    fn route(&self, dest: NodeId) -> Option<Route> {
        // Check if direct neighbor
        if self.neighbors.get(&dest).is_some() {
            return Some(Route::direct(dest));
        }

        // Check routing table
        self.next_hop_router.get_route(&dest).cloned()
    }

    fn forward(&mut self, packet: MeshPacket) -> MeshResult<()> {
        self.queue_packet(&packet)
    }

    fn on_receive(&mut self, mut packet: MeshPacket, rssi: f32, snr: f32) -> Vec<MeshPacket> {
        // Set reception metadata
        packet.set_rx_metadata(rssi, snr);

        // Update neighbor table
        self.neighbors.update(packet.header.source, rssi, snr);

        // Update routing table from overheard packet
        if let Some(neighbor) = self.neighbors.get(&packet.header.source) {
            self.next_hop_router.learn_route(
                &packet,
                packet.header.source,
                neighbor.link_quality.quality_score(),
            );
        }

        // Process with flood router
        let (local, _rebroadcast) = self.flood_router.process_incoming(packet.clone(), rssi, snr);

        // Update stats
        self.stats.packets_rx += 1;
        self.stats.bytes_rx += packet.to_bytes().len() as u64;

        // Handle specific packet types
        if let Some(ref local_packet) = local {
            match local_packet.packet_type {
                PacketType::NodeInfo => {
                    // Update node info in neighbor table
                    if let Some(info) = self.parse_node_info(&local_packet.payload) {
                        self.neighbors.update_info(local_packet.header.source, info);
                    }
                }
                PacketType::Telemetry => {
                    // Parse and store telemetry from neighbor
                    if let Some(telemetry) = self.parse_telemetry(&local_packet.payload) {
                        self.neighbors.update_telemetry(local_packet.header.source, telemetry);
                    }
                }
                PacketType::Ack => {
                    self.stats.acks_received += 1;
                }
                _ => {}
            }
        }

        // Return packets for application layer
        local.into_iter().collect()
    }

    fn stats(&self) -> MeshStats {
        let mut stats = self.stats.clone();
        stats.channel_utilization = self.mac.channel_utilization_cached();
        stats.neighbor_count = self.neighbors.len();
        stats.route_count = self.next_hop_router.route_count();
        stats
    }

    fn tick(&mut self, _elapsed: Duration) {
        // Prune stale neighbors
        self.neighbors.prune_stale();

        // Prune expired routes
        self.next_hop_router.prune();

        // Check for periodic broadcasts
        if self.config.position_enabled {
            let should_broadcast = self.last_position_broadcast
                .map(|t| t.elapsed() > Duration::from_secs(self.config.position_interval))
                .unwrap_or(true);

            if should_broadcast {
                self.broadcast_position();
            }
        }

        // Check for telemetry broadcast
        if self.config.telemetry.device_update_interval > 0 {
            let interval = Duration::from_secs(self.config.telemetry.device_update_interval as u64);
            let should_broadcast = self.last_telemetry_broadcast
                .map(|t| t.elapsed() > interval)
                .unwrap_or(true);

            if should_broadcast {
                self.broadcast_telemetry();
            }
        }
    }
}

impl MeshtasticNode {
    /// Parse node info from payload
    fn parse_node_info(&self, payload: &[u8]) -> Option<NodeInfo> {
        if payload.len() < 2 {
            return None;
        }

        let short_len = payload[0] as usize;
        if payload.len() < 2 + short_len {
            return None;
        }

        let short_name = String::from_utf8_lossy(&payload[1..1 + short_len]).to_string();

        let long_start = 1 + short_len;
        if payload.len() <= long_start {
            return Some(NodeInfo::with_names(NodeId::UNKNOWN, &short_name, ""));
        }

        let long_len = payload[long_start] as usize;
        let long_name = if payload.len() >= long_start + 1 + long_len {
            String::from_utf8_lossy(&payload[long_start + 1..long_start + 1 + long_len]).to_string()
        } else {
            String::new()
        };

        Some(NodeInfo::with_names(NodeId::UNKNOWN, &short_name, &long_name))
    }

    /// Parse telemetry from payload
    fn parse_telemetry(&self, payload: &[u8]) -> Option<Telemetry> {
        Telemetry::from_bytes(payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modem_presets() {
        let (sf, bw, _cr) = ModemPreset::LongFast.lora_params();
        assert_eq!(sf, 11);
        assert_eq!(bw, 250_000);

        let (sf, bw, _cr) = ModemPreset::LongSlow.lora_params();
        assert_eq!(sf, 12);
        assert_eq!(bw, 125_000);
    }

    #[test]
    fn test_region_frequencies() {
        let (start, end) = Region::US.frequency_range();
        assert!(start < end);
        assert!(start >= 902_000_000);

        let freq = Region::EU.primary_frequency();
        assert!(freq >= 863_000_000 && freq <= 870_000_000);
    }

    #[test]
    fn test_meshtastic_node_creation() {
        let node = MeshtasticNode::with_defaults();
        assert!(!node.node_id().is_unknown());
    }

    #[test]
    fn test_meshtastic_config() {
        let config = MeshtasticConfig {
            short_name: "TEST".to_string(),
            long_name: "Test Node".to_string(),
            region: Region::EU,
            hop_limit: 5,
            ..Default::default()
        };

        let node = MeshtasticNode::new(config);
        assert_eq!(node.config().hop_limit, 5);
        assert_eq!(node.config().region, Region::EU);
    }

    #[test]
    fn test_packet_receive() {
        let mut node = MeshtasticNode::with_defaults();
        let source = NodeId::random();

        let packet = MeshPacket::broadcast(source, b"Hello", 3);
        let delivered = node.on_receive(packet, -80.0, 10.0);

        // Should be delivered (it's a broadcast to all)
        assert_eq!(delivered.len(), 1);

        // Source should now be a neighbor
        assert!(node.neighbors.get(&source).is_some());
    }

    #[test]
    fn test_duplicate_detection() {
        let mut node = MeshtasticNode::with_defaults();
        let source = NodeId::random();

        let packet = MeshPacket::broadcast(source, b"Hello", 3);

        // First receive
        let delivered = node.on_receive(packet.clone(), -80.0, 10.0);
        assert_eq!(delivered.len(), 1);

        // Duplicate should not be delivered
        let delivered = node.on_receive(packet, -80.0, 10.0);
        assert_eq!(delivered.len(), 0);
    }

    #[test]
    fn test_send_message() {
        let mut node = MeshtasticNode::with_defaults();

        // Send broadcast
        let result = node.broadcast(b"Hello mesh!", 3);
        assert!(result.is_ok());

        // Check queue
        assert_eq!(node.mac.queue_depth(), 1);
    }

    #[test]
    fn test_stats() {
        let mut node = MeshtasticNode::with_defaults();
        let source = NodeId::random();

        let packet = MeshPacket::broadcast(source, b"Test", 3);
        node.on_receive(packet, -80.0, 10.0);

        let stats = node.stats();
        assert_eq!(stats.packets_rx, 1);
    }

    #[test]
    fn test_receive_bytes_wire_format() {
        use crate::mesh::wire::WireHeader;

        let mut node = MeshtasticNode::with_defaults();
        let source = NodeId::random();

        // Build a wire format packet
        let channel_hash = node.config().primary_channel.channel_hash();
        let wire_header = WireHeader::broadcast(source.to_u32(), 0x12345678, 3, channel_hash);
        let payload = b"Hello wire!";

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&wire_header.to_bytes());
        bytes.extend_from_slice(payload);

        // Receive the wire format packet
        let delivered = node.receive_bytes(&bytes, -75.0, 12.0);

        // Should be delivered (it's a broadcast)
        assert_eq!(delivered.len(), 1);
        assert_eq!(delivered[0].payload, payload);

        // Source should be a neighbor
        assert!(node.neighbors.get(&source).is_some());
    }

    #[test]
    fn test_receive_bytes_wrong_channel() {
        use crate::mesh::wire::WireHeader;

        let mut node = MeshtasticNode::with_defaults();
        let source = NodeId::random();

        // Build a wire format packet with wrong channel hash
        let wrong_channel_hash = 0xFF;
        let wire_header = WireHeader::broadcast(source.to_u32(), 0x12345678, 3, wrong_channel_hash);
        let payload = b"Hello wrong channel!";

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&wire_header.to_bytes());
        bytes.extend_from_slice(payload);

        // Should be rejected due to wrong channel
        let delivered = node.receive_bytes(&bytes, -75.0, 12.0);
        assert!(delivered.is_empty());
    }

    #[test]
    fn test_receive_bytes_too_short() {
        let mut node = MeshtasticNode::with_defaults();

        // Too short (less than header + 1 byte payload)
        let short_bytes = [0u8; 16]; // Just header, no payload
        let delivered = node.receive_bytes(&short_bytes, -75.0, 12.0);
        assert!(delivered.is_empty());
    }

    #[test]
    fn test_wire_format_roundtrip() {
        // Create two nodes with same channel configuration
        let config = MeshtasticConfig::default();
        let mut sender = MeshtasticNode::new(config.clone());
        let mut receiver = MeshtasticNode::new(config);

        let payload = b"Hello roundtrip test!";

        // Sender broadcasts a message (goes through queue_packet with wire format)
        sender.broadcast(payload, 3).unwrap();

        // Wait for DIFS period (50ms) before MAC allows transmission
        std::thread::sleep(std::time::Duration::from_millis(60));

        // Extract the wire format bytes from sender's TX queue
        let wire_bytes = sender.process_tx(false).expect("should have packet to send");

        // Verify wire format structure: 16-byte header + payload
        assert!(wire_bytes.len() >= WIRE_HEADER_SIZE + payload.len());

        // Receiver processes the wire format bytes
        let delivered = receiver.receive_bytes(&wire_bytes, -70.0, 15.0);

        // Should receive exactly one packet
        assert_eq!(delivered.len(), 1);

        // Payload should match
        assert_eq!(delivered[0].payload, payload);

        // Source should be sender's node ID
        assert_eq!(delivered[0].header.source, sender.node_id());
    }

    #[cfg(feature = "crypto")]
    #[test]
    fn test_wire_format_roundtrip_encrypted() {
        use crate::mesh::crypto::DEFAULT_PSK;

        // Create encrypted channel configuration
        let mut config = MeshtasticConfig::default();
        config.encryption_enabled = true;
        let mut psk = [0u8; 32];
        psk[..DEFAULT_PSK.len()].copy_from_slice(DEFAULT_PSK);
        config.primary_channel = ChannelConfig::with_psk("Encrypted", psk, ModemPreset::LongFast);

        let mut sender = MeshtasticNode::new(config.clone());
        let mut receiver = MeshtasticNode::new(config);

        let payload = b"Secret message!";

        // Sender broadcasts encrypted message
        sender.broadcast(payload, 3).unwrap();

        // Wait for DIFS period (50ms) before MAC allows transmission
        std::thread::sleep(std::time::Duration::from_millis(60));

        // Extract wire format (header + ciphertext + MIC)
        let wire_bytes = sender.process_tx(false).expect("should have packet to send");

        // Wire format should be larger due to 4-byte MIC
        assert!(wire_bytes.len() >= WIRE_HEADER_SIZE + payload.len() + 4);

        // Verify ciphertext is different from plaintext
        let ciphertext = &wire_bytes[WIRE_HEADER_SIZE..wire_bytes.len() - 4];
        assert_ne!(ciphertext, payload);

        // Receiver decrypts and processes
        let delivered = receiver.receive_bytes(&wire_bytes, -70.0, 15.0);

        // Should receive exactly one packet
        assert_eq!(delivered.len(), 1);

        // Decrypted payload should match original
        assert_eq!(delivered[0].payload, payload);
    }

    // ========================================================================
    // Protobuf integration tests (meshtastic-interop feature)
    // ========================================================================

    #[cfg(feature = "meshtastic-interop")]
    #[test]
    fn test_protobuf_text_roundtrip() {
        let config = MeshtasticConfig::default();
        let mut sender = MeshtasticNode::new(config.clone());
        let mut receiver = MeshtasticNode::new(config);

        let message = "Hello Meshtastic protobuf!";

        // Send text using protobuf encoding
        sender.send_text(message, 3).unwrap();

        // Wait for DIFS period
        std::thread::sleep(std::time::Duration::from_millis(60));

        // Get wire bytes
        let wire_bytes = sender.process_tx(false).expect("should have packet");

        // Receiver processes wire format
        let delivered = receiver.receive_bytes(&wire_bytes, -70.0, 15.0);

        assert_eq!(delivered.len(), 1);
        assert_eq!(delivered[0].packet_type, PacketType::Text);

        // Payload should be the raw text bytes (extracted from proto::Data)
        assert_eq!(String::from_utf8_lossy(&delivered[0].payload), message);
    }

    #[cfg(feature = "meshtastic-interop")]
    #[test]
    fn test_protobuf_position_roundtrip() {
        use crate::mesh::proto::Position as ProtoPosition;
        use prost::Message;

        let config = MeshtasticConfig::default();
        let mut sender = MeshtasticNode::new(config.clone());
        let mut receiver = MeshtasticNode::new(config);

        let lat = 37.422;
        let lon = -122.084;
        let alt = 10;

        // Send position using protobuf encoding
        sender.send_position(lat, lon, alt).unwrap();

        // Wait for DIFS period
        std::thread::sleep(std::time::Duration::from_millis(60));

        let wire_bytes = sender.process_tx(false).expect("should have packet");
        let delivered = receiver.receive_bytes(&wire_bytes, -70.0, 15.0);

        assert_eq!(delivered.len(), 1);
        assert_eq!(delivered[0].packet_type, PacketType::Position);

        // Decode inner position from payload
        let pos = ProtoPosition::decode(delivered[0].payload.as_slice()).unwrap();
        assert!((pos.latitude() - lat).abs() < 0.0001);
        assert!((pos.longitude() - lon).abs() < 0.0001);
        assert_eq!(pos.altitude, alt);
    }

    #[cfg(feature = "meshtastic-interop")]
    #[test]
    fn test_protobuf_telemetry_roundtrip() {
        use crate::mesh::proto::{Telemetry as ProtoTelemetry, TelemetryVariant as ProtoTelemetryVariant};
        use prost::Message;

        let config = MeshtasticConfig::default();
        let mut sender = MeshtasticNode::new(config.clone());
        let mut receiver = MeshtasticNode::new(config);

        // Set some metrics
        sender.set_battery_level(85);
        sender.set_voltage(4.1);

        // Send telemetry using protobuf encoding
        sender.send_telemetry_proto().unwrap();

        // Wait for DIFS period
        std::thread::sleep(std::time::Duration::from_millis(60));

        let wire_bytes = sender.process_tx(false).expect("should have packet");
        let delivered = receiver.receive_bytes(&wire_bytes, -70.0, 15.0);

        assert_eq!(delivered.len(), 1);
        assert_eq!(delivered[0].packet_type, PacketType::Telemetry);

        // Decode inner telemetry from payload
        let telemetry = ProtoTelemetry::decode(delivered[0].payload.as_slice()).unwrap();
        if let Some(ProtoTelemetryVariant::DeviceMetrics(dm)) = telemetry.variant {
            assert_eq!(dm.battery_level, 85);
            assert!((dm.voltage - 4.1).abs() < 0.01);
        } else {
            panic!("Expected DeviceMetrics variant");
        }
    }

    #[cfg(feature = "meshtastic-interop")]
    #[test]
    fn test_protobuf_node_info_roundtrip() {
        use crate::mesh::proto::User as ProtoUser;
        use prost::Message;

        let mut config = MeshtasticConfig::default();
        config.short_name = "TEST".to_string();
        config.long_name = "Test Proto Node".to_string();

        let mut sender = MeshtasticNode::new(config.clone());
        let mut receiver = MeshtasticNode::new(MeshtasticConfig::default());

        // Send node info using protobuf encoding
        sender.send_node_info().unwrap();

        // Wait for DIFS period
        std::thread::sleep(std::time::Duration::from_millis(60));

        let wire_bytes = sender.process_tx(false).expect("should have packet");
        let delivered = receiver.receive_bytes(&wire_bytes, -70.0, 15.0);

        assert_eq!(delivered.len(), 1);
        assert_eq!(delivered[0].packet_type, PacketType::NodeInfo);

        // Decode inner user from payload
        let user = ProtoUser::decode(delivered[0].payload.as_slice()).unwrap();
        assert_eq!(user.short_name, "TEST");
        assert_eq!(user.long_name, "Test Proto Node");
    }

    #[cfg(all(feature = "meshtastic-interop", feature = "crypto"))]
    #[test]
    fn test_protobuf_encrypted_roundtrip() {
        use crate::mesh::crypto::DEFAULT_PSK;

        // Create encrypted channel configuration
        let mut config = MeshtasticConfig::default();
        config.encryption_enabled = true;
        let mut psk = [0u8; 32];
        psk[..DEFAULT_PSK.len()].copy_from_slice(DEFAULT_PSK);
        config.primary_channel = ChannelConfig::with_psk("Encrypted", psk, ModemPreset::LongFast);

        let mut sender = MeshtasticNode::new(config.clone());
        let mut receiver = MeshtasticNode::new(config);

        let message = "Secret protobuf message!";

        // Send text using protobuf encoding (will be encrypted)
        sender.send_text(message, 3).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(60));

        let wire_bytes = sender.process_tx(false).expect("should have packet");

        // Verify it's encrypted (has MIC)
        assert!(wire_bytes.len() > WIRE_HEADER_SIZE + message.len() + 4);

        // Receiver decrypts and decodes protobuf
        let delivered = receiver.receive_bytes(&wire_bytes, -70.0, 15.0);

        assert_eq!(delivered.len(), 1);
        assert_eq!(delivered[0].packet_type, PacketType::Text);
        assert_eq!(String::from_utf8_lossy(&delivered[0].payload), message);
    }
}
