//! Medium Access Control (MAC) layer for mesh networking
//!
//! Implements CSMA/CA (Carrier Sense Multiple Access with Collision Avoidance)
//! for coordinating channel access in the mesh network.
//!
//! ## CSMA/CA Algorithm
//!
//! 1. Check if channel is busy (CAD for LoRa)
//! 2. If busy, enter backoff with random delay from contention window
//! 3. Contention window size scales with channel utilization
//! 4. After backoff, check channel again before transmitting
//!
//! ## Channel Utilization
//!
//! Track airtime usage to:
//! - Adjust contention window size
//! - Implement fair access (duty cycle limits)
//! - Report network congestion

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Simple pseudo-random number generator for CSMA backoff.
/// Uses a Linear Congruential Generator (LCG) with state seeded from system time.
#[derive(Debug)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Create a new RNG seeded from system time and a counter
    fn new() -> Self {
        // Seed from system time - use both seconds and nanos for entropy
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let seed = now.as_secs().wrapping_mul(1_000_000_000) ^ now.subsec_nanos() as u64;
        Self { state: seed | 1 } // Ensure odd for LCG
    }

    /// Generate next random u32
    fn next_u32(&mut self) -> u32 {
        // LCG parameters from Numerical Recipes
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    /// Generate random u32 in range [0, max)
    fn next_u32_max(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        self.next_u32() % max
    }
}

/// Channel state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelState {
    /// Channel is idle, can transmit
    Idle,
    /// Channel is busy (transmission detected)
    Busy,
    /// In backoff waiting period
    Backoff,
    /// Currently transmitting
    Transmitting,
    /// Currently receiving
    Receiving,
}

/// CSMA/CA configuration
#[derive(Debug, Clone)]
pub struct CsmaConfig {
    /// Minimum contention window size (slots)
    pub cw_min: u32,
    /// Maximum contention window size (slots)
    pub cw_max: u32,
    /// Slot time in milliseconds
    pub slot_time_ms: u64,
    /// Maximum number of backoff attempts
    pub max_backoff_attempts: u8,
    /// DIFS (Distributed Inter-Frame Space) in milliseconds
    pub difs_ms: u64,
    /// Target maximum channel utilization (0.0 - 1.0)
    pub target_utilization: f32,
    /// Channel activity detection threshold (dBm)
    pub cad_threshold: f32,
}

impl Default for CsmaConfig {
    fn default() -> Self {
        Self {
            cw_min: 16,
            cw_max: 256,
            slot_time_ms: 10,
            max_backoff_attempts: 7,
            difs_ms: 50,
            target_utilization: 0.1, // 10% max utilization
            cad_threshold: -115.0,
        }
    }
}

/// Channel utilization tracker
#[derive(Debug)]
pub struct ChannelUtilization {
    /// Recent transmissions: (start_time, duration)
    history: VecDeque<(Instant, Duration)>,
    /// Window size for utilization calculation
    window: Duration,
    /// Last calculated utilization
    cached_utilization: f32,
    /// Time of last cache update
    last_update: Instant,
}

impl ChannelUtilization {
    /// Create a new utilization tracker
    pub fn new(window_secs: u64) -> Self {
        Self {
            history: VecDeque::new(),
            window: Duration::from_secs(window_secs),
            cached_utilization: 0.0,
            last_update: Instant::now(),
        }
    }

    /// Record a transmission
    pub fn record_tx(&mut self, duration: Duration) {
        self.history.push_back((Instant::now(), duration));
        self.prune_old();
        self.update_cache();
    }

    /// Record a reception (channel was busy)
    pub fn record_rx(&mut self, duration: Duration) {
        self.record_tx(duration); // Same tracking
    }

    /// Get current channel utilization (0.0 - 1.0)
    pub fn utilization(&mut self) -> f32 {
        if self.last_update.elapsed() > Duration::from_millis(100) {
            self.prune_old();
            self.update_cache();
        }
        self.cached_utilization
    }

    /// Get cached channel utilization without updating (non-mutating)
    pub fn utilization_cached(&self) -> f32 {
        self.cached_utilization
    }

    fn prune_old(&mut self) {
        let cutoff = Instant::now() - self.window;
        while let Some((time, _)) = self.history.front() {
            if *time < cutoff {
                self.history.pop_front();
            } else {
                break;
            }
        }
    }

    fn update_cache(&mut self) {
        let total_airtime: Duration = self.history.iter().map(|(_, d)| *d).sum();
        self.cached_utilization = total_airtime.as_secs_f32() / self.window.as_secs_f32();
        self.last_update = Instant::now();
    }
}

impl Default for ChannelUtilization {
    fn default() -> Self {
        Self::new(60) // 1 minute window
    }
}

/// Backoff state machine
#[derive(Debug)]
struct BackoffState {
    /// Current contention window size
    contention_window: u32,
    /// Remaining backoff slots
    remaining_slots: u32,
    /// Number of backoff attempts
    attempts: u8,
    /// Time when backoff started
    start_time: Option<Instant>,
    /// Random number generator for backoff
    rng: SimpleRng,
}

impl BackoffState {
    fn new(cw_min: u32) -> Self {
        Self {
            contention_window: cw_min,
            remaining_slots: 0,
            attempts: 0,
            start_time: None,
            rng: SimpleRng::new(),
        }
    }

    fn start(&mut self, utilization: f32, cw_min: u32, cw_max: u32) {
        // Scale contention window with utilization
        // Use .max() to scale UP with utilization (higher utilization = larger window)
        let base_cw = cw_min.max((cw_min as f32 * (1.0 + utilization * 10.0)) as u32);
        self.contention_window = base_cw.min(cw_max);

        // Pick random backoff using proper RNG
        self.remaining_slots = self.rng.next_u32_max(self.contention_window);
        self.start_time = Some(Instant::now());
        self.attempts += 1;
    }

    fn double_window(&mut self, cw_max: u32) {
        self.contention_window = (self.contention_window * 2).min(cw_max);
    }

    fn reset(&mut self, cw_min: u32) {
        self.contention_window = cw_min;
        self.remaining_slots = 0;
        self.attempts = 0;
        self.start_time = None;
    }
}

/// MAC layer implementation
#[derive(Debug)]
pub struct MacLayer {
    /// Configuration
    config: CsmaConfig,
    /// Current channel state
    state: ChannelState,
    /// Channel utilization tracker
    utilization: ChannelUtilization,
    /// Backoff state
    backoff: BackoffState,
    /// Transmit queue
    tx_queue: VecDeque<Vec<u8>>,
    /// Maximum queue size
    max_queue_size: usize,
    /// Time of last state change
    state_changed: Instant,
}

impl MacLayer {
    /// Create a new MAC layer
    pub fn new(config: CsmaConfig) -> Self {
        Self {
            backoff: BackoffState::new(config.cw_min),
            config,
            state: ChannelState::Idle,
            utilization: ChannelUtilization::default(),
            tx_queue: VecDeque::new(),
            max_queue_size: 16,
            state_changed: Instant::now(),
        }
    }

    /// Queue a packet for transmission
    pub fn queue_tx(&mut self, packet: Vec<u8>) -> Result<(), MacError> {
        if self.tx_queue.len() >= self.max_queue_size {
            return Err(MacError::QueueFull);
        }
        self.tx_queue.push_back(packet);
        Ok(())
    }

    /// Check if we can transmit
    pub fn can_transmit(&mut self, channel_busy: bool) -> TransmitDecision {
        // Update state based on channel
        if channel_busy && self.state != ChannelState::Transmitting {
            self.set_state(ChannelState::Busy);
        }

        match self.state {
            ChannelState::Idle => {
                // Check duty cycle limit
                let util = self.utilization.utilization();
                if util >= self.config.target_utilization {
                    return TransmitDecision::DutyCycleLimit;
                }

                // Channel must be idle for DIFS
                let difs = Duration::from_millis(self.config.difs_ms);
                if self.state_changed.elapsed() >= difs && !channel_busy {
                    if self.tx_queue.is_empty() {
                        TransmitDecision::NothingToSend
                    } else {
                        TransmitDecision::TransmitNow
                    }
                } else {
                    TransmitDecision::WaitDifs
                }
            }

            ChannelState::Busy | ChannelState::Receiving => {
                // Start/continue backoff
                if self.backoff.start_time.is_none() {
                    let util = self.utilization.utilization();
                    self.backoff.start(util, self.config.cw_min, self.config.cw_max);
                    self.set_state(ChannelState::Backoff);
                }
                TransmitDecision::Backoff(self.backoff.remaining_slots)
            }

            ChannelState::Backoff => {
                if channel_busy {
                    // Freeze backoff
                    TransmitDecision::Backoff(self.backoff.remaining_slots)
                } else {
                    // Decrement backoff
                    let elapsed_slots = self.state_changed.elapsed().as_millis()
                        / self.config.slot_time_ms as u128;
                    if elapsed_slots as u32 >= self.backoff.remaining_slots {
                        // Backoff complete
                        if self.backoff.attempts >= self.config.max_backoff_attempts {
                            self.backoff.reset(self.config.cw_min);
                            return TransmitDecision::MaxBackoffExceeded;
                        }
                        self.set_state(ChannelState::Idle);
                        TransmitDecision::TransmitNow
                    } else {
                        self.backoff.remaining_slots -= elapsed_slots as u32;
                        TransmitDecision::Backoff(self.backoff.remaining_slots)
                    }
                }
            }

            ChannelState::Transmitting => TransmitDecision::AlreadyTransmitting,
        }
    }

    /// Start transmitting (call after TransmitNow)
    pub fn start_tx(&mut self) -> Option<Vec<u8>> {
        if let Some(packet) = self.tx_queue.pop_front() {
            self.set_state(ChannelState::Transmitting);
            self.backoff.reset(self.config.cw_min);
            Some(packet)
        } else {
            None
        }
    }

    /// Transmission complete
    pub fn tx_complete(&mut self, duration: Duration) {
        self.utilization.record_tx(duration);
        self.set_state(ChannelState::Idle);
    }

    /// Reception started
    pub fn rx_start(&mut self) {
        self.set_state(ChannelState::Receiving);
    }

    /// Reception complete
    pub fn rx_complete(&mut self, duration: Duration) {
        self.utilization.record_rx(duration);
        self.set_state(ChannelState::Idle);
    }

    /// Collision detected (transmission failed)
    pub fn collision(&mut self) {
        self.backoff.double_window(self.config.cw_max);
        self.set_state(ChannelState::Backoff);
    }

    /// Get current channel utilization (updates cache)
    pub fn channel_utilization(&mut self) -> f32 {
        self.utilization.utilization()
    }

    /// Get cached channel utilization (non-mutating)
    pub fn channel_utilization_cached(&self) -> f32 {
        self.utilization.utilization_cached()
    }

    /// Get current state
    pub fn state(&self) -> ChannelState {
        self.state
    }

    /// Get queue depth
    pub fn queue_depth(&self) -> usize {
        self.tx_queue.len()
    }

    /// Clear the transmit queue
    pub fn clear_queue(&mut self) {
        self.tx_queue.clear();
    }

    fn set_state(&mut self, state: ChannelState) {
        if self.state != state {
            self.state = state;
            self.state_changed = Instant::now();
        }
    }
}

impl Default for MacLayer {
    fn default() -> Self {
        Self::new(CsmaConfig::default())
    }
}

/// Transmission decision from MAC layer
#[derive(Debug, Clone, PartialEq)]
pub enum TransmitDecision {
    /// Transmit immediately
    TransmitNow,
    /// Wait for DIFS period
    WaitDifs,
    /// In backoff, wait N slots
    Backoff(u32),
    /// Already transmitting
    AlreadyTransmitting,
    /// Nothing in queue to send
    NothingToSend,
    /// Duty cycle limit reached
    DutyCycleLimit,
    /// Maximum backoff attempts exceeded
    MaxBackoffExceeded,
}

/// MAC layer errors
#[derive(Debug, Clone, PartialEq)]
pub enum MacError {
    /// Transmit queue is full
    QueueFull,
    /// Channel access timeout
    AccessTimeout,
    /// Collision detected
    Collision,
}

impl std::fmt::Display for MacError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MacError::QueueFull => write!(f, "Transmit queue full"),
            MacError::AccessTimeout => write!(f, "Channel access timeout"),
            MacError::Collision => write!(f, "Collision detected"),
        }
    }
}

impl std::error::Error for MacError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_utilization() {
        let mut util = ChannelUtilization::new(60);

        // Record some transmissions
        util.record_tx(Duration::from_millis(100));
        util.record_tx(Duration::from_millis(100));

        // Should have some utilization
        let u = util.utilization();
        assert!(u > 0.0);
        assert!(u < 1.0);
    }

    #[test]
    fn test_mac_idle_channel() {
        let mut mac = MacLayer::default();

        // Queue a packet
        mac.queue_tx(vec![1, 2, 3]).unwrap();

        // Wait for DIFS
        std::thread::sleep(Duration::from_millis(60));

        // Should be able to transmit
        let decision = mac.can_transmit(false);
        assert_eq!(decision, TransmitDecision::TransmitNow);
    }

    #[test]
    fn test_mac_busy_channel() {
        let mut mac = MacLayer::default();
        mac.queue_tx(vec![1, 2, 3]).unwrap();

        // Channel is busy
        let decision = mac.can_transmit(true);
        match decision {
            TransmitDecision::Backoff(_) => {}
            _ => panic!("Expected backoff"),
        }
    }

    #[test]
    fn test_mac_queue_full() {
        let mut mac = MacLayer::default();
        mac.max_queue_size = 2;

        mac.queue_tx(vec![1]).unwrap();
        mac.queue_tx(vec![2]).unwrap();

        let result = mac.queue_tx(vec![3]);
        assert!(matches!(result, Err(MacError::QueueFull)));
    }

    #[test]
    fn test_csma_config() {
        let config = CsmaConfig::default();
        assert!(config.cw_min < config.cw_max);
        assert!(config.target_utilization > 0.0 && config.target_utilization < 1.0);
    }
}
