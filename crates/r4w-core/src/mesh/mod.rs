//! Mesh Networking Framework
//!
//! This module provides a generic framework for implementing mesh networking protocols
//! over various physical layer waveforms. The architecture separates concerns into:
//!
//! - **Traits**: Generic interfaces for mesh nodes, routing, and MAC layers
//! - **Packet**: Common packet structures and framing
//! - **Routing**: Flood and next-hop routing algorithms
//! - **Neighbor**: Neighbor discovery and table management
//! - **MAC**: Medium Access Control (CSMA/CA)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                      Application Layer                                   │
//! │              (Text Messaging, Position Sharing, etc.)                    │
//! └─────────────────────────────────────────────────────────────────────────┘
//!                                  │
//!                                  ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                      MeshNetwork Trait                                   │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
//! │  │  discover   │  │   route     │  │   forward   │  │ on_receive  │     │
//! │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │
//! └─────────────────────────────────────────────────────────────────────────┘
//!                                  │
//!                                  ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        Routing Layer                                     │
//! │  ┌───────────────────────────┐  ┌─────────────────────────────────────┐  │
//! │  │    FloodRouter            │  │      NextHopRouter                  │  │
//! │  │  (broadcasts, SNR-based)  │  │    (unicast, cached routes)         │  │
//! │  └───────────────────────────┘  └─────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────────────┘
//!                                  │
//!                                  ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                          MAC Layer                                       │
//! │  ┌───────────────────────────┐  ┌─────────────────────────────────────┐  │
//! │  │       CSMA/CA             │  │       PacketFramer                  │  │
//! │  │  (channel access control) │  │    (header, payload, CRC)           │  │
//! │  └───────────────────────────┘  └─────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────────────┘
//!                                  │
//!                                  ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                      MeshPhy Trait                                       │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
//! │  │channel_busy │  │    rssi     │  │     snr     │  │   transmit  │     │
//! │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use r4w_core::mesh::{MeshNetwork, MeshNode, NodeId, MeshPacket};
//!
//! // Create a mesh node with Meshtastic protocol
//! let mut node = MeshtasticNode::new(NodeId::random());
//!
//! // Discover neighbors
//! let neighbors = node.discover_neighbors();
//!
//! // Send a broadcast message
//! let packet = MeshPacket::broadcast(b"Hello mesh!", 3);
//! node.forward(packet)?;
//!
//! // Handle incoming packet
//! node.on_receive(incoming_packet, -80.0, 10.0);
//! ```

pub mod traits;
pub mod packet;
pub mod routing;
pub mod neighbor;
pub mod mac;
pub mod meshtastic;
pub mod lora_mesh;

// Re-export main types
pub use traits::{MeshNetwork, MeshPhy, MeshError, MeshResult, MeshStats, MeshConfig};
pub use packet::{MeshPacket, PacketHeader, PacketFlags, NodeId};
pub use routing::{Route, NextHop, FloodRouter, NextHopRouter, RoutingTable};
pub use neighbor::{Neighbor, NeighborTable, NodeInfo};
pub use mac::{MacLayer, CsmaConfig, ChannelState};
pub use meshtastic::{MeshtasticNode, MeshtasticConfig, ModemPreset, Region};
pub use lora_mesh::{LoRaMesh, LoRaMeshPhy, LoRaMeshConfig};
