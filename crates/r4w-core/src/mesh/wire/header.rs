//! Meshtastic Wire Format Header
//!
//! Implements the 16-byte radio header format used by Meshtastic devices.
//! All multi-byte fields are little-endian.
//!
//! ## Header Format (16 bytes)
//!
//! ```text
//! Offset  Size  Field
//! ------  ----  -----
//! 0x00    4B    to (destination node ID)
//! 0x04    4B    from (source node ID)
//! 0x08    4B    id (32-bit packet ID)
//! 0x0C    1B    flags (hop_limit:3 | want_ack:1 | via_mqtt:1 | hop_start:3)
//! 0x0D    1B    channel_hash (first byte of SHA256(channel_name))
//! 0x0E    1B    next_hop (for routed packets)
//! 0x0F    1B    relay_node (last relay, for traceroute)
//! ```
//!
//! ## Flags Byte Layout
//!
//! ```text
//! Bits 0-2:  hop_limit (remaining hops, 0-7)
//! Bit 3:     want_ack (1 = request ACK)
//! Bit 4:     via_mqtt (1 = received via MQTT gateway)
//! Bits 5-7:  hop_start (initial hop count, 0-7)
//! ```

use super::super::packet::NodeId;

/// Meshtastic wire format header size
pub const WIRE_HEADER_SIZE: usize = 16;

/// Wire format header flags
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WireFlags(u8);

impl WireFlags {
    /// Create new flags with default values
    pub fn new() -> Self {
        Self(0)
    }

    /// Create from raw byte
    pub fn from_byte(byte: u8) -> Self {
        Self(byte)
    }

    /// Get raw byte value
    pub fn as_byte(&self) -> u8 {
        self.0
    }

    /// Get hop limit (bits 0-2, 0-7)
    pub fn hop_limit(&self) -> u8 {
        self.0 & 0x07
    }

    /// Set hop limit (0-7)
    pub fn set_hop_limit(&mut self, limit: u8) {
        self.0 = (self.0 & 0xF8) | (limit & 0x07);
    }

    /// Get want_ack flag (bit 3)
    pub fn want_ack(&self) -> bool {
        (self.0 & 0x08) != 0
    }

    /// Set want_ack flag
    pub fn set_want_ack(&mut self, want: bool) {
        if want {
            self.0 |= 0x08;
        } else {
            self.0 &= !0x08;
        }
    }

    /// Get via_mqtt flag (bit 4)
    pub fn via_mqtt(&self) -> bool {
        (self.0 & 0x10) != 0
    }

    /// Set via_mqtt flag
    pub fn set_via_mqtt(&mut self, via: bool) {
        if via {
            self.0 |= 0x10;
        } else {
            self.0 &= !0x10;
        }
    }

    /// Get hop start (bits 5-7, 0-7)
    pub fn hop_start(&self) -> u8 {
        (self.0 >> 5) & 0x07
    }

    /// Set hop start (0-7)
    pub fn set_hop_start(&mut self, start: u8) {
        self.0 = (self.0 & 0x1F) | ((start & 0x07) << 5);
    }
}

/// Meshtastic wire format header (16 bytes)
///
/// This is the on-air packet format used by Meshtastic devices.
/// All fields are little-endian.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireHeader {
    /// Destination node ID (broadcast = 0xFFFFFFFF)
    pub to: u32,

    /// Source node ID
    pub from: u32,

    /// 32-bit packet ID (unique per sender)
    pub id: u32,

    /// Flags byte (hop_limit, want_ack, via_mqtt, hop_start)
    pub flags: WireFlags,

    /// Channel hash (first byte of SHA256(channel_name))
    pub channel_hash: u8,

    /// Next hop for routed packets (0 = broadcast/direct)
    pub next_hop: u8,

    /// Last relay node (for traceroute, 0 = not set)
    pub relay_node: u8,
}

impl WireHeader {
    /// Header size in bytes
    pub const SIZE: usize = WIRE_HEADER_SIZE;

    /// Broadcast destination
    pub const BROADCAST: u32 = 0xFFFFFFFF;

    /// Create a new broadcast header
    pub fn broadcast(from: u32, id: u32, hop_limit: u8, channel_hash: u8) -> Self {
        let mut flags = WireFlags::new();
        flags.set_hop_limit(hop_limit);
        flags.set_hop_start(hop_limit);

        Self {
            to: Self::BROADCAST,
            from,
            id,
            flags,
            channel_hash,
            next_hop: 0,
            relay_node: 0,
        }
    }

    /// Create a new direct (unicast) header
    pub fn direct(to: u32, from: u32, id: u32, hop_limit: u8, channel_hash: u8) -> Self {
        let mut flags = WireFlags::new();
        flags.set_hop_limit(hop_limit);
        flags.set_hop_start(hop_limit);

        Self {
            to,
            from,
            id,
            flags,
            channel_hash,
            next_hop: 0,
            relay_node: 0,
        }
    }

    /// Check if this is a broadcast packet
    pub fn is_broadcast(&self) -> bool {
        self.to == Self::BROADCAST
    }

    /// Serialize to bytes (little-endian)
    pub fn to_bytes(&self) -> [u8; WIRE_HEADER_SIZE] {
        let mut buf = [0u8; WIRE_HEADER_SIZE];

        buf[0..4].copy_from_slice(&self.to.to_le_bytes());
        buf[4..8].copy_from_slice(&self.from.to_le_bytes());
        buf[8..12].copy_from_slice(&self.id.to_le_bytes());
        buf[12] = self.flags.as_byte();
        buf[13] = self.channel_hash;
        buf[14] = self.next_hop;
        buf[15] = self.relay_node;

        buf
    }

    /// Parse from bytes (little-endian)
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < WIRE_HEADER_SIZE {
            return None;
        }

        let to = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let from = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let id = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let flags = WireFlags::from_byte(bytes[12]);
        let channel_hash = bytes[13];
        let next_hop = bytes[14];
        let relay_node = bytes[15];

        Some(Self {
            to,
            from,
            id,
            flags,
            channel_hash,
            next_hop,
            relay_node,
        })
    }

    /// Convert from our internal PacketHeader format
    pub fn from_packet_header(
        header: &super::super::packet::PacketHeader,
        channel_hash: u8,
    ) -> Self {
        let mut flags = WireFlags::new();
        flags.set_hop_limit(header.hop_limit.min(7));
        flags.set_hop_start(header.flags.hop_start().min(7));
        flags.set_want_ack(header.flags.want_ack());

        Self {
            to: header.destination.to_u32(),
            from: header.source.to_u32(),
            id: header.packet_id as u32,  // Extend 16-bit to 32-bit
            flags,
            channel_hash,
            next_hop: 0,
            relay_node: 0,
        }
    }

    /// Convert to our internal PacketHeader format
    pub fn to_packet_header(&self) -> super::super::packet::PacketHeader {
        use super::super::packet::{PacketFlags, PacketHeader};

        let mut flags = PacketFlags::new();
        flags.set_hop_start(self.flags.hop_start());
        flags.set_want_ack(self.flags.want_ack());

        PacketHeader {
            destination: NodeId::from_u32(self.to),
            source: NodeId::from_u32(self.from),
            packet_id: (self.id & 0xFFFF) as u16,  // Truncate to 16-bit
            hop_limit: self.flags.hop_limit(),
            flags,
            channel: 0,  // Default channel
        }
    }

    /// Decrement hop limit for rebroadcast
    pub fn decrement_hop_limit(&mut self) {
        let current = self.flags.hop_limit();
        if current > 0 {
            self.flags.set_hop_limit(current - 1);
        }
    }

    /// Get source as NodeId
    pub fn source_node_id(&self) -> NodeId {
        NodeId::from_u32(self.from)
    }

    /// Get destination as NodeId
    pub fn destination_node_id(&self) -> NodeId {
        NodeId::from_u32(self.to)
    }
}

impl Default for WireHeader {
    fn default() -> Self {
        Self {
            to: Self::BROADCAST,
            from: 0,
            id: 0,
            flags: WireFlags::new(),
            channel_hash: 0,
            next_hop: 0,
            relay_node: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wire_flags() {
        let mut flags = WireFlags::new();

        // Test hop limit
        flags.set_hop_limit(5);
        assert_eq!(flags.hop_limit(), 5);

        // Test want_ack
        assert!(!flags.want_ack());
        flags.set_want_ack(true);
        assert!(flags.want_ack());

        // Test via_mqtt
        assert!(!flags.via_mqtt());
        flags.set_via_mqtt(true);
        assert!(flags.via_mqtt());

        // Test hop_start
        flags.set_hop_start(3);
        assert_eq!(flags.hop_start(), 3);

        // Verify all flags are preserved
        assert_eq!(flags.hop_limit(), 5);
        assert!(flags.want_ack());
        assert!(flags.via_mqtt());
    }

    #[test]
    fn test_wire_header_broadcast() {
        let header = WireHeader::broadcast(0x12345678, 0xABCDEF01, 3, 0x42);

        assert!(header.is_broadcast());
        assert_eq!(header.to, WireHeader::BROADCAST);
        assert_eq!(header.from, 0x12345678);
        assert_eq!(header.id, 0xABCDEF01);
        assert_eq!(header.flags.hop_limit(), 3);
        assert_eq!(header.flags.hop_start(), 3);
        assert_eq!(header.channel_hash, 0x42);
    }

    #[test]
    fn test_wire_header_direct() {
        let header = WireHeader::direct(0xDEADBEEF, 0x12345678, 100, 2, 0x55);

        assert!(!header.is_broadcast());
        assert_eq!(header.to, 0xDEADBEEF);
        assert_eq!(header.from, 0x12345678);
        assert_eq!(header.id, 100);
        assert_eq!(header.flags.hop_limit(), 2);
    }

    #[test]
    fn test_wire_header_roundtrip() {
        let header = WireHeader::broadcast(0xAABBCCDD, 0x11223344, 7, 0xEE);
        let bytes = header.to_bytes();
        let recovered = WireHeader::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.to, header.to);
        assert_eq!(recovered.from, header.from);
        assert_eq!(recovered.id, header.id);
        assert_eq!(recovered.flags.as_byte(), header.flags.as_byte());
        assert_eq!(recovered.channel_hash, header.channel_hash);
    }

    #[test]
    fn test_wire_header_size() {
        let header = WireHeader::default();
        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), WIRE_HEADER_SIZE);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn test_wire_header_little_endian() {
        let header = WireHeader {
            to: 0x44332211,
            from: 0x88776655,
            id: 0xCCBBAA99,
            flags: WireFlags::new(),
            channel_hash: 0xEE,
            next_hop: 0xDD,
            relay_node: 0xFF,
        };

        let bytes = header.to_bytes();

        // Verify little-endian encoding
        assert_eq!(bytes[0..4], [0x11, 0x22, 0x33, 0x44]); // to
        assert_eq!(bytes[4..8], [0x55, 0x66, 0x77, 0x88]); // from
        assert_eq!(bytes[8..12], [0x99, 0xAA, 0xBB, 0xCC]); // id
    }

    #[test]
    fn test_decrement_hop_limit() {
        let mut header = WireHeader::broadcast(0x12345678, 100, 3, 0);

        assert_eq!(header.flags.hop_limit(), 3);
        header.decrement_hop_limit();
        assert_eq!(header.flags.hop_limit(), 2);
        header.decrement_hop_limit();
        assert_eq!(header.flags.hop_limit(), 1);
        header.decrement_hop_limit();
        assert_eq!(header.flags.hop_limit(), 0);
        header.decrement_hop_limit(); // Should not go negative
        assert_eq!(header.flags.hop_limit(), 0);
    }

    #[test]
    fn test_packet_header_conversion() {
        use super::super::super::packet::{PacketHeader, NodeId};

        let source = NodeId::from_bytes([0x11, 0x22, 0x33, 0x44]);
        let internal_header = PacketHeader::broadcast(source, 3);

        // Convert to wire format
        let wire = WireHeader::from_packet_header(&internal_header, 0xAB);
        assert!(wire.is_broadcast());
        assert_eq!(wire.from, source.to_u32());
        assert_eq!(wire.channel_hash, 0xAB);

        // Convert back
        let recovered = wire.to_packet_header();
        assert_eq!(recovered.source, internal_header.source);
        assert!(recovered.destination.is_broadcast());
    }
}
