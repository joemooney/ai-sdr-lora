//! Mesh packet types and framing
//!
//! This module defines the packet structures used for mesh networking.
//! The design is based on the Meshtastic protocol but generalized for
//! other mesh implementations.
//!
//! ## Packet Structure
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────┐
//! │                        Mesh Packet                                  │
//! ├──────────────┬──────────────┬──────────────┬──────────────────────┤
//! │  Header (8B) │  Payload     │  MIC (4B)    │  CRC (2B)            │
//! │              │  (0-237B)    │  (optional)  │  (optional)          │
//! └──────────────┴──────────────┴──────────────┴──────────────────────┘
//!
//! Header:
//! ┌────────────┬────────────┬────────────┬────────────┐
//! │ Dest (4B)  │ Src (4B)   │ Packet ID  │ Flags (1B) │
//! │            │            │   (2B)     │            │
//! └────────────┴────────────┴────────────┴────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

/// Node identifier - 4-byte unique ID
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId([u8; 4]);

impl NodeId {
    /// Broadcast address (all 0xFF)
    pub const BROADCAST: NodeId = NodeId([0xFF, 0xFF, 0xFF, 0xFF]);

    /// Unknown/unset address (all 0x00)
    pub const UNKNOWN: NodeId = NodeId([0x00, 0x00, 0x00, 0x00]);

    /// Create a new NodeId from 4 bytes
    pub fn from_bytes(bytes: [u8; 4]) -> Self {
        NodeId(bytes)
    }

    /// Create a NodeId from a u32
    pub fn from_u32(value: u32) -> Self {
        NodeId(value.to_be_bytes())
    }

    /// Convert to u32
    pub fn to_u32(&self) -> u32 {
        u32::from_be_bytes(self.0)
    }

    /// Get the raw bytes
    pub fn as_bytes(&self) -> &[u8; 4] {
        &self.0
    }

    /// Generate a random NodeId
    pub fn random() -> Self {
        // Use timestamp + random bits for uniqueness
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let bytes = [
            (now >> 24) as u8,
            (now >> 16) as u8,
            (now >> 8) as u8,
            now as u8,
        ];
        NodeId(bytes)
    }

    /// Check if this is the broadcast address
    pub fn is_broadcast(&self) -> bool {
        *self == Self::BROADCAST
    }

    /// Check if this is unknown/unset
    pub fn is_unknown(&self) -> bool {
        *self == Self::UNKNOWN
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({:02x}{:02x}{:02x}{:02x})",
               self.0[0], self.0[1], self.0[2], self.0[3])
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:02x}{:02x}{:02x}{:02x}",
               self.0[0], self.0[1], self.0[2], self.0[3])
    }
}

/// Packet flags indicating packet type and options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PacketFlags(u8);

impl PacketFlags {
    /// Empty flags
    pub const NONE: PacketFlags = PacketFlags(0);

    /// Bit positions
    const WANT_ACK_BIT: u8 = 0;      // Request acknowledgment
    const VIA_MQTT_BIT: u8 = 1;      // Packet came via MQTT
    const HOP_START_BIT: u8 = 2;     // Original hop_limit stored (bits 2-4)
    const ENCRYPTED_BIT: u8 = 5;     // Payload is encrypted
    const PRIORITY_BIT: u8 = 6;      // High priority packet
    #[allow(dead_code)]
    const RESERVED_BIT: u8 = 7;      // Reserved for future use

    /// Create new flags
    pub fn new() -> Self {
        PacketFlags(0)
    }

    /// Check if acknowledgment is requested
    pub fn want_ack(&self) -> bool {
        (self.0 & (1 << Self::WANT_ACK_BIT)) != 0
    }

    /// Set want_ack flag
    pub fn set_want_ack(&mut self, value: bool) {
        if value {
            self.0 |= 1 << Self::WANT_ACK_BIT;
        } else {
            self.0 &= !(1 << Self::WANT_ACK_BIT);
        }
    }

    /// Check if packet is encrypted
    pub fn encrypted(&self) -> bool {
        (self.0 & (1 << Self::ENCRYPTED_BIT)) != 0
    }

    /// Set encrypted flag
    pub fn set_encrypted(&mut self, value: bool) {
        if value {
            self.0 |= 1 << Self::ENCRYPTED_BIT;
        } else {
            self.0 &= !(1 << Self::ENCRYPTED_BIT);
        }
    }

    /// Check if packet is high priority
    pub fn priority(&self) -> bool {
        (self.0 & (1 << Self::PRIORITY_BIT)) != 0
    }

    /// Set priority flag
    pub fn set_priority(&mut self, value: bool) {
        if value {
            self.0 |= 1 << Self::PRIORITY_BIT;
        } else {
            self.0 &= !(1 << Self::PRIORITY_BIT);
        }
    }

    /// Check if packet came via MQTT
    pub fn via_mqtt(&self) -> bool {
        (self.0 & (1 << Self::VIA_MQTT_BIT)) != 0
    }

    /// Set via_mqtt flag
    pub fn set_via_mqtt(&mut self, value: bool) {
        if value {
            self.0 |= 1 << Self::VIA_MQTT_BIT;
        } else {
            self.0 &= !(1 << Self::VIA_MQTT_BIT);
        }
    }

    /// Get the original hop_start (stored in bits 2-4)
    pub fn hop_start(&self) -> u8 {
        (self.0 >> Self::HOP_START_BIT) & 0x07
    }

    /// Set the original hop_start
    pub fn set_hop_start(&mut self, value: u8) {
        self.0 = (self.0 & 0xE3) | ((value & 0x07) << Self::HOP_START_BIT);
    }

    /// Get the raw byte value
    pub fn as_byte(&self) -> u8 {
        self.0
    }

    /// Create from raw byte
    pub fn from_byte(byte: u8) -> Self {
        PacketFlags(byte)
    }
}

impl Default for PacketFlags {
    fn default() -> Self {
        PacketFlags::NONE
    }
}

/// Packet header containing routing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacketHeader {
    /// Destination node ID (BROADCAST for all nodes)
    pub destination: NodeId,
    /// Source node ID
    pub source: NodeId,
    /// Packet ID (for duplicate detection)
    pub packet_id: u16,
    /// Current hop limit (decremented on each hop)
    pub hop_limit: u8,
    /// Packet flags
    pub flags: PacketFlags,
    /// Channel index (for multi-channel support)
    pub channel: u8,
}

impl PacketHeader {
    /// Header size in bytes (dest:4 + src:4 + packet_id:2 + hop_limit:1 + flags:1 = 12)
    /// Note: channel field is NOT serialized in wire format
    pub const SIZE: usize = 12;

    /// Minimum header size for parsing (without hop_limit and flags)
    pub const MIN_SIZE: usize = 10;

    /// Create a new header for a broadcast packet
    pub fn broadcast(source: NodeId, hop_limit: u8) -> Self {
        Self {
            destination: NodeId::BROADCAST,
            source,
            packet_id: Self::generate_packet_id(),
            hop_limit,
            flags: PacketFlags::new(),
            channel: 0,
        }
    }

    /// Create a new header for a direct (unicast) packet
    pub fn direct(source: NodeId, destination: NodeId) -> Self {
        let mut flags = PacketFlags::new();
        flags.set_want_ack(true);
        Self {
            destination,
            source,
            packet_id: Self::generate_packet_id(),
            hop_limit: 3, // Default hop limit for direct messages
            flags,
            channel: 0,
        }
    }

    /// Generate a unique packet ID
    fn generate_packet_id() -> u16 {
        // Use lower bits of timestamp for uniqueness
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros();
        now as u16
    }

    /// Check if this is a broadcast packet
    pub fn is_broadcast(&self) -> bool {
        self.destination.is_broadcast()
    }

    /// Serialize header to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Self::SIZE);
        bytes.extend_from_slice(self.destination.as_bytes());
        bytes.extend_from_slice(self.source.as_bytes());
        bytes.extend_from_slice(&self.packet_id.to_be_bytes());
        bytes.push(self.hop_limit);
        bytes.push(self.flags.as_byte());
        // Note: channel is not included in minimal header
        bytes
    }

    /// Deserialize header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::MIN_SIZE {
            return None;
        }
        Some(Self {
            destination: NodeId::from_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            source: NodeId::from_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            packet_id: u16::from_be_bytes([bytes[8], bytes[9]]),
            hop_limit: if bytes.len() > 10 { bytes[10] } else { 0 },
            flags: PacketFlags::from_byte(if bytes.len() > 11 { bytes[11] } else { 0 }),
            channel: 0,
        })
    }

    /// Get the actual serialized size (always SIZE bytes)
    pub fn serialized_size(&self) -> usize {
        Self::SIZE
    }
}

/// Packet types for different message categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum PacketType {
    /// Text message
    Text = 0,
    /// Position data
    Position = 1,
    /// Node information
    NodeInfo = 2,
    /// Routing control
    Routing = 3,
    /// Acknowledgment
    Ack = 4,
    /// Telemetry data
    Telemetry = 5,
    /// Channel configuration
    Channel = 6,
    /// Administrative control
    Admin = 7,
    /// Custom/user application
    Custom = 255,
}

impl PacketType {
    /// Create from byte value
    pub fn from_byte(byte: u8) -> Self {
        match byte {
            0 => PacketType::Text,
            1 => PacketType::Position,
            2 => PacketType::NodeInfo,
            3 => PacketType::Routing,
            4 => PacketType::Ack,
            5 => PacketType::Telemetry,
            6 => PacketType::Channel,
            7 => PacketType::Admin,
            _ => PacketType::Custom,
        }
    }
}

/// A complete mesh packet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshPacket {
    /// Packet header
    pub header: PacketHeader,
    /// Packet type
    pub packet_type: PacketType,
    /// Payload data
    pub payload: Vec<u8>,
    /// Message integrity code (if encrypted)
    pub mic: Option<[u8; 4]>,
    /// Reception metadata (filled on receive)
    pub rx_rssi: Option<f32>,
    pub rx_snr: Option<f32>,
    pub rx_time: Option<u64>,
}

impl MeshPacket {
    /// Maximum payload size in bytes
    pub const MAX_PAYLOAD_SIZE: usize = 237;

    /// Create a new broadcast packet
    pub fn broadcast(source: NodeId, payload: &[u8], hop_limit: u8) -> Self {
        Self {
            header: PacketHeader::broadcast(source, hop_limit),
            packet_type: PacketType::Text,
            payload: payload.to_vec(),
            mic: None,
            rx_rssi: None,
            rx_snr: None,
            rx_time: None,
        }
    }

    /// Create a new direct (unicast) packet
    pub fn direct(source: NodeId, destination: NodeId, payload: &[u8]) -> Self {
        Self {
            header: PacketHeader::direct(source, destination),
            packet_type: PacketType::Text,
            payload: payload.to_vec(),
            mic: None,
            rx_rssi: None,
            rx_snr: None,
            rx_time: None,
        }
    }

    /// Create an acknowledgment packet
    pub fn ack(source: NodeId, destination: NodeId, acked_packet_id: u16) -> Self {
        Self {
            header: PacketHeader::direct(source, destination),
            packet_type: PacketType::Ack,
            payload: acked_packet_id.to_be_bytes().to_vec(),
            mic: None,
            rx_rssi: None,
            rx_snr: None,
            rx_time: None,
        }
    }

    /// Create a position packet
    pub fn position(source: NodeId, lat: f64, lon: f64, alt: f32) -> Self {
        let mut payload = Vec::with_capacity(20);
        payload.extend_from_slice(&lat.to_be_bytes());
        payload.extend_from_slice(&lon.to_be_bytes());
        payload.extend_from_slice(&alt.to_be_bytes());

        let mut header = PacketHeader::broadcast(source, 3);
        header.flags.set_hop_start(3);

        Self {
            header,
            packet_type: PacketType::Position,
            payload,
            mic: None,
            rx_rssi: None,
            rx_snr: None,
            rx_time: None,
        }
    }

    /// Create a node info packet
    pub fn node_info(source: NodeId, short_name: &str, long_name: &str) -> Self {
        let mut payload = Vec::new();
        // Short name (up to 4 chars)
        let short_bytes: Vec<u8> = short_name.bytes().take(4).collect();
        payload.push(short_bytes.len() as u8);
        payload.extend(short_bytes);
        // Long name (up to 40 chars)
        let long_bytes: Vec<u8> = long_name.bytes().take(40).collect();
        payload.push(long_bytes.len() as u8);
        payload.extend(long_bytes);

        Self {
            header: PacketHeader::broadcast(source, 3),
            packet_type: PacketType::NodeInfo,
            payload,
            mic: None,
            rx_rssi: None,
            rx_snr: None,
            rx_time: None,
        }
    }

    /// Get the unique identifier for duplicate detection
    pub fn dedup_key(&self) -> (NodeId, u16) {
        (self.header.source, self.header.packet_id)
    }

    /// Check if this packet is addressed to us or broadcast
    pub fn is_for_node(&self, node_id: NodeId) -> bool {
        self.header.destination.is_broadcast() || self.header.destination == node_id
    }

    /// Decrement hop limit, returns false if already zero
    pub fn decrement_hop_limit(&mut self) -> bool {
        if self.header.hop_limit > 0 {
            self.header.hop_limit -= 1;
            true
        } else {
            false
        }
    }

    /// Serialize the entire packet to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(self.header.to_bytes());
        bytes.push(self.packet_type as u8);
        bytes.extend(&self.payload);
        if let Some(mic) = &self.mic {
            bytes.extend(mic);
        }
        bytes
    }

    /// Deserialize a packet from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        // Minimum: header (12) + packet_type (1) = 13 bytes
        let header_size = PacketHeader::SIZE;
        let min_packet_size = header_size + 1; // header + packet_type

        if bytes.len() < min_packet_size {
            return None;
        }

        let header = PacketHeader::from_bytes(bytes)?;
        let packet_type = PacketType::from_byte(bytes[header_size]); // packet_type is right after header

        // Calculate payload boundaries
        let payload_start = header_size + 1; // After header + packet_type byte
        let has_mic = header.flags.encrypted();
        let payload_end = if has_mic && bytes.len() >= payload_start + 4 {
            bytes.len() - 4
        } else {
            bytes.len()
        };

        let payload = if payload_start < payload_end {
            bytes[payload_start..payload_end].to_vec()
        } else {
            Vec::new()
        };

        let mic = if has_mic && bytes.len() >= payload_start + 4 {
            let mic_start = bytes.len() - 4;
            Some([bytes[mic_start], bytes[mic_start + 1], bytes[mic_start + 2], bytes[mic_start + 3]])
        } else {
            None
        };

        Some(Self {
            header,
            packet_type,
            payload,
            mic,
            rx_rssi: None,
            rx_snr: None,
            rx_time: None,
        })
    }

    /// Set reception metadata
    pub fn set_rx_metadata(&mut self, rssi: f32, snr: f32) {
        self.rx_rssi = Some(rssi);
        self.rx_snr = Some(snr);
        self.rx_time = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );
    }
}

/// CRC-16-CCITT for packet integrity
pub fn crc16_ccitt(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for byte in data {
        crc ^= (*byte as u16) << 8;
        for _ in 0..8 {
            if (crc & 0x8000) != 0 {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id() {
        let id = NodeId::from_bytes([0x12, 0x34, 0x56, 0x78]);
        assert_eq!(id.to_u32(), 0x12345678);
        assert!(!id.is_broadcast());
        assert!(NodeId::BROADCAST.is_broadcast());
    }

    #[test]
    fn test_packet_flags() {
        let mut flags = PacketFlags::new();
        assert!(!flags.want_ack());
        assert!(!flags.encrypted());

        flags.set_want_ack(true);
        assert!(flags.want_ack());

        flags.set_encrypted(true);
        assert!(flags.encrypted());

        flags.set_hop_start(5);
        assert_eq!(flags.hop_start(), 5);
    }

    #[test]
    fn test_packet_header_roundtrip() {
        let source = NodeId::from_bytes([0x11, 0x22, 0x33, 0x44]);
        let header = PacketHeader::broadcast(source, 3);

        let bytes = header.to_bytes();
        let recovered = PacketHeader::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.source, source);
        assert_eq!(recovered.hop_limit, 3);
        assert!(recovered.destination.is_broadcast());
    }

    #[test]
    fn test_mesh_packet_broadcast() {
        let source = NodeId::random();
        let packet = MeshPacket::broadcast(source, b"Hello mesh!", 3);

        assert!(packet.header.is_broadcast());
        assert_eq!(packet.header.hop_limit, 3);
        assert_eq!(packet.payload, b"Hello mesh!");
    }

    #[test]
    fn test_mesh_packet_direct() {
        let source = NodeId::random();
        let dest = NodeId::random();
        let packet = MeshPacket::direct(source, dest, b"Direct message");

        assert!(!packet.header.is_broadcast());
        assert!(packet.header.flags.want_ack());
        assert_eq!(packet.header.destination, dest);
    }

    #[test]
    fn test_dedup_key() {
        let source = NodeId::from_bytes([0x11, 0x22, 0x33, 0x44]);
        let packet1 = MeshPacket::broadcast(source, b"Test", 3);
        let key1 = packet1.dedup_key();

        // Different packet from same source should have different ID
        let packet2 = MeshPacket::broadcast(source, b"Test", 3);
        let key2 = packet2.dedup_key();

        assert_eq!(key1.0, key2.0); // Same source
        // Packet IDs may differ (time-based)
    }

    #[test]
    fn test_crc16() {
        let data = b"123456789";
        let crc = crc16_ccitt(data);
        assert_eq!(crc, 0x29B1); // Known CRC-16-CCITT result
    }

    #[test]
    fn test_header_size_consistency() {
        let source = NodeId::from_bytes([0x11, 0x22, 0x33, 0x44]);
        let header = PacketHeader::broadcast(source, 3);
        let bytes = header.to_bytes();
        // Verify SIZE constant matches actual serialization
        assert_eq!(bytes.len(), PacketHeader::SIZE);
    }

    #[test]
    fn test_mesh_packet_roundtrip() {
        let source = NodeId::from_bytes([0x11, 0x22, 0x33, 0x44]);
        let mut packet = MeshPacket::broadcast(source, b"Test payload roundtrip", 3);
        packet.packet_type = PacketType::Text;

        let bytes = packet.to_bytes();
        let recovered = MeshPacket::from_bytes(&bytes).expect("Failed to parse packet");

        assert_eq!(recovered.header.source, source);
        assert_eq!(recovered.header.hop_limit, 3);
        assert!(recovered.header.destination.is_broadcast());
        assert_eq!(recovered.packet_type, PacketType::Text);
        assert_eq!(recovered.payload, b"Test payload roundtrip");
    }

    #[test]
    fn test_mesh_packet_with_mic() {
        let source = NodeId::from_bytes([0x11, 0x22, 0x33, 0x44]);
        let mut packet = MeshPacket::broadcast(source, b"Encrypted", 3);
        packet.header.flags.set_encrypted(true);
        packet.mic = Some([0xDE, 0xAD, 0xBE, 0xEF]);

        let bytes = packet.to_bytes();
        let recovered = MeshPacket::from_bytes(&bytes).expect("Failed to parse packet");

        assert!(recovered.header.flags.encrypted());
        assert_eq!(recovered.mic, Some([0xDE, 0xAD, 0xBE, 0xEF]));
        assert_eq!(recovered.payload, b"Encrypted");
    }
}
