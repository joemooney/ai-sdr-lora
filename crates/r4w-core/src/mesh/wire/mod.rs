//! Meshtastic Wire Format
//!
//! This module implements the on-air packet format used by Meshtastic devices.
//! It provides interoperability with the global Meshtastic network.
//!
//! ## Wire Format Overview
//!
//! Meshtastic packets consist of:
//! - 16-byte header (little-endian)
//! - Variable-length payload (usually protobuf-encoded)
//! - Optional 4-byte MIC (when encrypted)
//!
//! ## Key Differences from Internal Format
//!
//! | Aspect | Internal (packet.rs) | Wire (Meshtastic) |
//! |--------|---------------------|-------------------|
//! | Endianness | Big | Little |
//! | Header size | 12 bytes | 16 bytes |
//! | Packet ID | 16-bit | 32-bit |
//! | Channel | Numeric index | Hash byte |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use r4w_core::mesh::wire::{WireHeader, WIRE_HEADER_SIZE};
//!
//! // Create a broadcast packet
//! let header = WireHeader::broadcast(my_node_id, packet_id, 3, channel_hash);
//! let bytes = header.to_bytes();
//!
//! // Parse received packet
//! let header = WireHeader::from_bytes(&received_data)?;
//! if header.is_broadcast() {
//!     // Process broadcast
//! }
//! ```

pub mod header;

pub use header::{WireHeader, WireFlags, WIRE_HEADER_SIZE};
