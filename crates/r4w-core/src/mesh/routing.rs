//! Mesh routing algorithms
//!
//! This module implements routing strategies for mesh networks:
//!
//! - **Flood Routing**: Broadcast-based routing where packets are rebroadcast
//!   by all receiving nodes (with managed flooding to reduce redundancy)
//! - **Next-Hop Routing**: Unicast routing using cached routes
//!
//! The Meshtastic protocol uses a hybrid approach:
//! - Broadcasts use managed flood routing with SNR-based delays
//! - Direct messages use next-hop routing with flood fallback

// Neighbor and NeighborTable used in FloodRouter for broadcast filtering
use super::packet::{MeshPacket, NodeId};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

/// Simple pseudo-random number generator for routing delays.
/// Uses a Linear Congruential Generator (LCG) with state seeded from system time.
#[derive(Debug)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Create a new RNG seeded from system time
    fn new() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let seed = now.as_secs().wrapping_mul(1_000_000_000) ^ now.subsec_nanos() as u64;
        Self { state: seed | 1 }
    }

    /// Generate next random u64
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    /// Generate random u64 in range [0, max)
    fn next_u64_max(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        self.next_u64() % max
    }
}

/// A route to a destination node
#[derive(Debug, Clone)]
pub struct Route {
    /// Destination node ID
    pub destination: NodeId,
    /// Next hop node ID (may be same as destination if direct)
    pub next_hop: NodeId,
    /// Total hop count to destination
    pub hop_count: u8,
    /// Route quality score (0.0 - 1.0)
    pub quality: f32,
    /// Time route was last updated
    pub last_updated: Instant,
}

impl Route {
    /// Create a direct route (single hop)
    pub fn direct(destination: NodeId) -> Self {
        Self {
            destination,
            next_hop: destination,
            hop_count: 1,
            quality: 1.0,
            last_updated: Instant::now(),
        }
    }

    /// Create an indirect route
    pub fn via(destination: NodeId, next_hop: NodeId, hop_count: u8, quality: f32) -> Self {
        Self {
            destination,
            next_hop,
            hop_count,
            quality,
            last_updated: Instant::now(),
        }
    }

    /// Check if route is expired
    pub fn is_expired(&self, timeout: Duration) -> bool {
        self.last_updated.elapsed() > timeout
    }

    /// Touch the route to update last_updated
    pub fn touch(&mut self) {
        self.last_updated = Instant::now();
    }
}

/// Next hop information for routing decisions
#[derive(Debug, Clone)]
pub struct NextHop {
    /// Node to forward to
    pub node_id: NodeId,
    /// Interface/channel to use
    pub channel: u8,
    /// Expected quality of this hop
    pub quality: f32,
}

/// Routing table for storing known routes
#[derive(Debug)]
pub struct RoutingTable {
    /// Routes indexed by destination
    routes: HashMap<NodeId, Route>,
    /// Route timeout duration
    timeout: Duration,
    /// Maximum number of routes
    max_routes: usize,
}

impl RoutingTable {
    /// Create a new routing table
    pub fn new(timeout_secs: u64, max_routes: usize) -> Self {
        Self {
            routes: HashMap::new(),
            timeout: Duration::from_secs(timeout_secs),
            max_routes,
        }
    }

    /// Add or update a route
    pub fn update(&mut self, route: Route) {
        let dest = route.destination;

        // Only update if better or newer
        if let Some(existing) = self.routes.get(&dest) {
            if route.hop_count < existing.hop_count
                || (route.hop_count == existing.hop_count && route.quality > existing.quality)
                || existing.is_expired(self.timeout)
            {
                self.routes.insert(dest, route);
            }
        } else {
            // Check capacity
            if self.routes.len() >= self.max_routes {
                self.evict_worst();
            }
            self.routes.insert(dest, route);
        }
    }

    /// Get route to destination
    pub fn get(&self, destination: &NodeId) -> Option<&Route> {
        self.routes
            .get(destination)
            .filter(|r| !r.is_expired(self.timeout))
    }

    /// Get mutable route
    pub fn get_mut(&mut self, destination: &NodeId) -> Option<&mut Route> {
        let timeout = self.timeout;
        match self.routes.get_mut(destination) {
            Some(route) if !route.is_expired(timeout) => Some(route),
            _ => None,
        }
    }

    /// Remove a route
    pub fn remove(&mut self, destination: &NodeId) -> Option<Route> {
        self.routes.remove(destination)
    }

    /// Prune expired routes
    pub fn prune(&mut self) -> usize {
        let timeout = self.timeout;
        let before = self.routes.len();
        self.routes.retain(|_, r| !r.is_expired(timeout));
        before - self.routes.len()
    }

    /// Get all active routes
    pub fn all(&self) -> Vec<&Route> {
        self.routes
            .values()
            .filter(|r| !r.is_expired(self.timeout))
            .collect()
    }

    /// Number of routes
    pub fn len(&self) -> usize {
        self.routes.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }

    /// Evict the worst route (highest hop count, lowest quality)
    fn evict_worst(&mut self) {
        if let Some(worst_id) = self
            .routes
            .iter()
            .min_by(|(_, a), (_, b)| {
                // Prefer to evict: expired > high hop count > low quality
                let a_score = if a.is_expired(self.timeout) {
                    -1.0
                } else {
                    a.quality / (a.hop_count as f32)
                };
                let b_score = if b.is_expired(self.timeout) {
                    -1.0
                } else {
                    b.quality / (b.hop_count as f32)
                };
                a_score
                    .partial_cmp(&b_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, _)| *id)
        {
            self.routes.remove(&worst_id);
        }
    }
}

impl Default for RoutingTable {
    fn default() -> Self {
        Self::new(3600, 256) // 1 hour timeout, 256 routes
    }
}

/// Duplicate packet detection cache
#[derive(Debug)]
pub struct DuplicateCache {
    /// Seen packet keys: (source_id, packet_id) -> expiry time
    seen: HashMap<(NodeId, u16), Instant>,
    /// TTL for cache entries
    ttl: Duration,
    /// Maximum cache size
    max_size: usize,
    /// Cleanup interval
    last_cleanup: Instant,
}

impl DuplicateCache {
    /// Create a new duplicate cache
    pub fn new(ttl_secs: u64, max_size: usize) -> Self {
        Self {
            seen: HashMap::new(),
            ttl: Duration::from_secs(ttl_secs),
            max_size,
            last_cleanup: Instant::now(),
        }
    }

    /// Check if packet is a duplicate, and add to cache if not
    /// Returns true if this is a NEW packet (not a duplicate)
    pub fn check_and_add(&mut self, source: NodeId, packet_id: u16) -> bool {
        self.maybe_cleanup();

        let key = (source, packet_id);
        if let Some(expiry) = self.seen.get(&key) {
            if expiry.elapsed() < self.ttl {
                return false; // Duplicate
            }
        }

        // Not a duplicate, add to cache
        if self.seen.len() >= self.max_size {
            self.cleanup();
        }
        self.seen.insert(key, Instant::now());
        true
    }

    /// Check if packet is a duplicate without adding
    pub fn is_duplicate(&self, source: NodeId, packet_id: u16) -> bool {
        let key = (source, packet_id);
        self.seen
            .get(&key)
            .map(|t| t.elapsed() < self.ttl)
            .unwrap_or(false)
    }

    /// Remove expired entries
    pub fn cleanup(&mut self) {
        let ttl = self.ttl;
        self.seen.retain(|_, t| t.elapsed() < ttl);
        self.last_cleanup = Instant::now();
    }

    fn maybe_cleanup(&mut self) {
        // Cleanup every 30 seconds
        if self.last_cleanup.elapsed() > Duration::from_secs(30) {
            self.cleanup();
        }
    }

    /// Number of entries in cache
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

impl Default for DuplicateCache {
    fn default() -> Self {
        Self::new(300, 256) // 5 minute TTL, 256 entries
    }
}

/// Managed flood routing with SNR-based delays
///
/// Implements Meshtastic-style managed flooding:
/// 1. Nodes wait a random time before rebroadcasting
/// 2. Wait time is inversely proportional to SNR (distant nodes flood first)
/// 3. If a rebroadcast is heard during wait, cancel own rebroadcast
#[derive(Debug)]
pub struct FloodRouter {
    /// Our node ID
    node_id: NodeId,
    /// Duplicate detection cache
    dedup: DuplicateCache,
    /// Pending rebroadcasts
    pending: VecDeque<(MeshPacket, Instant)>,
    /// Packets we've heard rebroadcasted (don't rebroadcast again)
    heard_rebroadcast: HashSet<(NodeId, u16)>,
    /// Default hop limit for broadcasts
    default_hop_limit: u8,
    /// Base delay for rebroadcast (milliseconds)
    base_delay_ms: u64,
    /// Random number generator for delay jitter
    rng: SimpleRng,
}

impl FloodRouter {
    /// Create a new flood router
    pub fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            dedup: DuplicateCache::default(),
            pending: VecDeque::new(),
            heard_rebroadcast: HashSet::new(),
            default_hop_limit: 3,
            base_delay_ms: 200,
            rng: SimpleRng::new(),
        }
    }

    /// Calculate rebroadcast delay based on SNR
    /// Lower SNR (distant node) = shorter delay = flood first
    fn rebroadcast_delay(&mut self, snr: f32) -> Duration {
        // SNR typically ranges from -20 to +30 dB
        // Map to 0-1 range (inverted: low SNR = low value = short delay)
        let snr_factor = ((snr + 20.0) / 50.0).clamp(0.0, 1.0);

        // Random component (0-100ms) + SNR-based component (0-base_delay)
        let random_ms = self.rng.next_u64_max(100);
        let snr_ms = (snr_factor * self.base_delay_ms as f32) as u64;

        Duration::from_millis(random_ms + snr_ms)
    }

    /// Process an incoming packet for flooding
    /// Returns packets to deliver locally and packets to rebroadcast
    pub fn process_incoming(
        &mut self,
        packet: MeshPacket,
        _rssi: f32,
        snr: f32,
    ) -> (Option<MeshPacket>, Option<MeshPacket>) {
        let source = packet.header.source;
        let packet_id = packet.header.packet_id;
        let key = (source, packet_id);

        // Check for duplicate
        if !self.dedup.check_and_add(source, packet_id) {
            // This is a duplicate - someone else rebroadcasted
            // Cancel any pending rebroadcast
            self.heard_rebroadcast.insert(key);
            return (None, None);
        }

        // Check if addressed to us or broadcast
        let for_us = packet.header.destination.is_broadcast()
            || packet.header.destination == self.node_id;

        let local_delivery = if for_us {
            Some(packet.clone())
        } else {
            None
        };

        // Determine if we should rebroadcast
        let should_rebroadcast = packet.header.is_broadcast()
            && packet.header.hop_limit > 0
            && packet.header.source != self.node_id;

        let rebroadcast = if should_rebroadcast {
            let mut rebroad = packet.clone();
            rebroad.decrement_hop_limit();

            // Schedule rebroadcast with delay
            let delay = self.rebroadcast_delay(snr);
            let fire_at = Instant::now() + delay;
            self.pending.push_back((rebroad.clone(), fire_at));

            None // Will be returned by get_pending_rebroadcast
        } else {
            None
        };

        (local_delivery, rebroadcast)
    }

    /// Get any pending rebroadcasts that are ready
    pub fn get_pending_rebroadcast(&mut self) -> Option<MeshPacket> {
        let now = Instant::now();

        // Check if front of queue is ready
        if let Some((_, fire_at)) = self.pending.front() {
            if *fire_at <= now {
                let (packet, _) = self.pending.pop_front().unwrap();
                let key = (packet.header.source, packet.header.packet_id);

                // Check if we heard a rebroadcast while waiting
                if self.heard_rebroadcast.contains(&key) {
                    self.heard_rebroadcast.remove(&key);
                    return None; // Don't rebroadcast
                }

                return Some(packet);
            }
        }
        None
    }

    /// Check if there are pending rebroadcasts
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Create a broadcast packet from this node
    pub fn create_broadcast(&self, payload: &[u8]) -> MeshPacket {
        MeshPacket::broadcast(self.node_id, payload, self.default_hop_limit)
    }

    /// Clear all pending rebroadcasts
    pub fn clear_pending(&mut self) {
        self.pending.clear();
        self.heard_rebroadcast.clear();
    }
}

/// Next-hop router for direct messages
///
/// Uses cached routes when available, falls back to flood routing
/// when no route is known.
#[derive(Debug)]
pub struct NextHopRouter {
    /// Our node ID
    node_id: NodeId,
    /// Routing table
    routes: RoutingTable,
    // Note: In practice, neighbor table would be shared via Arc<RwLock<>>
}

impl NextHopRouter {
    /// Create a new next-hop router
    pub fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            routes: RoutingTable::default(),
        }
    }

    /// Learn a route from an overheard packet
    pub fn learn_route(&mut self, packet: &MeshPacket, from_neighbor: NodeId, quality: f32) {
        let source = packet.header.source;
        if source == self.node_id {
            return; // Don't route to ourselves
        }

        // Calculate hop count (original hop_start - current hop_limit + 1)
        let hop_start = packet.header.flags.hop_start().max(packet.header.hop_limit);
        let hops_traveled = hop_start.saturating_sub(packet.header.hop_limit) + 1;

        let route = Route::via(source, from_neighbor, hops_traveled, quality);
        self.routes.update(route);
    }

    /// Get next hop for a destination
    pub fn next_hop(&self, destination: NodeId) -> Option<NextHop> {
        self.routes.get(&destination).map(|r| NextHop {
            node_id: r.next_hop,
            channel: 0,
            quality: r.quality,
        })
    }

    /// Route a direct packet
    /// Returns the packet with updated hop limit, or None if should flood
    pub fn route_direct(&mut self, mut packet: MeshPacket) -> Option<MeshPacket> {
        // Look up route
        if let Some(route) = self.routes.get_mut(&packet.header.destination) {
            route.touch();
            // Packet is already addressed to destination
            // We're just providing the next hop info
            Some(packet)
        } else {
            // No route known - should fall back to flooding
            packet.header.flags.set_want_ack(true);
            None
        }
    }

    /// Get number of known routes
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }

    /// Get a route to a destination
    pub fn get_route(&self, destination: &NodeId) -> Option<&Route> {
        self.routes.get(destination)
    }

    /// Prune expired routes
    pub fn prune(&mut self) -> usize {
        self.routes.prune()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route() {
        let dest = NodeId::random();
        let next = NodeId::random();

        let route = Route::via(dest, next, 2, 0.8);
        assert_eq!(route.hop_count, 2);
        assert!(!route.is_expired(Duration::from_secs(60)));
    }

    #[test]
    fn test_routing_table() {
        let mut table = RoutingTable::new(3600, 10);

        let dest1 = NodeId::from_bytes([1, 0, 0, 0]);
        let dest2 = NodeId::from_bytes([2, 0, 0, 0]);
        let next = NodeId::from_bytes([3, 0, 0, 0]);

        table.update(Route::direct(dest1));
        table.update(Route::via(dest2, next, 2, 0.7));

        assert_eq!(table.len(), 2);
        assert!(table.get(&dest1).is_some());
        assert!(table.get(&dest2).is_some());
    }

    #[test]
    fn test_duplicate_cache() {
        let mut cache = DuplicateCache::new(300, 100);

        let source = NodeId::random();

        // First packet - not duplicate
        assert!(cache.check_and_add(source, 1));

        // Same packet - duplicate
        assert!(!cache.check_and_add(source, 1));

        // Different packet ID - not duplicate
        assert!(cache.check_and_add(source, 2));
    }

    #[test]
    fn test_flood_router() {
        let node_id = NodeId::random();
        let mut router = FloodRouter::new(node_id);

        let source = NodeId::random();
        let packet = MeshPacket::broadcast(source, b"Hello", 3);

        // Process incoming broadcast
        let (local, _) = router.process_incoming(packet.clone(), -80.0, 10.0);
        assert!(local.is_some());

        // Same packet again should be duplicate
        let (local2, _) = router.process_incoming(packet, -80.0, 10.0);
        assert!(local2.is_none());
    }

    #[test]
    fn test_next_hop_router() {
        let node_id = NodeId::random();
        let mut router = NextHopRouter::new(node_id);

        let source = NodeId::random();
        let via = NodeId::random();

        // No route initially
        assert!(router.next_hop(source).is_none());

        // Learn route from packet
        let mut packet = MeshPacket::broadcast(source, b"Test", 3);
        packet.header.flags.set_hop_start(3);
        packet.header.hop_limit = 2;

        router.learn_route(&packet, via, 0.9);

        // Now should have route
        let hop = router.next_hop(source);
        assert!(hop.is_some());
        assert_eq!(hop.unwrap().node_id, via);
    }
}
