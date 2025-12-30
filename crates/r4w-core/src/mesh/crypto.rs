//! Meshtastic-compatible AES-256-CTR encryption
//!
//! Implements channel encryption compatible with the Meshtastic protocol:
//! - AES-256-CTR mode encryption
//! - Key derivation from channel name + PSK
//! - HMAC-SHA256 for Message Integrity Code (MIC)
//!
//! ## Key Derivation
//!
//! The channel key is derived from:
//! ```text
//! key = SHA256(channel_name || PSK || "Meshtastic")
//! ```
//!
//! ## Nonce Construction
//!
//! The 16-byte nonce (IV) for AES-256-CTR is constructed as:
//! ```text
//! Bytes 0-3:   source_node_id (little-endian)
//! Bytes 4-7:   packet_id (little-endian)
//! Bytes 8-15:  nonce_base XOR with fixed pattern
//! ```
//!
//! ## MIC (Message Integrity Code)
//!
//! The 4-byte MIC is the first 4 bytes of:
//! ```text
//! HMAC-SHA256(key, header || ciphertext)
//! ```

use super::packet::{MeshPacket, NodeId};

#[cfg(feature = "crypto")]
use aes::Aes256;
#[cfg(feature = "crypto")]
use ctr::cipher::{KeyIvInit, StreamCipher};
#[cfg(feature = "crypto")]
use ctr::Ctr128BE;
#[cfg(feature = "crypto")]
use hmac::{Hmac, Mac};
#[cfg(feature = "crypto")]
use sha2::{Digest, Sha256};

/// Default Pre-Shared Key (PSK) for the default channel
/// This is the well-known key used by default Meshtastic channels
pub const DEFAULT_PSK: &[u8] = &[
    0xd4, 0xf1, 0xbb, 0x3a, 0x20, 0x29, 0x07, 0x59,
    0xf0, 0xbc, 0xff, 0xab, 0xcf, 0x4e, 0x69, 0x01,
];

/// Magic string for key derivation
const KEY_DERIVATION_MAGIC: &[u8] = b"Meshtastic";

/// Nonce base pattern for XOR
const NONCE_BASE: [u8; 8] = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];

/// Crypto error types
#[derive(Debug, Clone, PartialEq)]
pub enum CryptoError {
    /// Invalid key length
    InvalidKeyLength,
    /// Invalid data length
    InvalidDataLength,
    /// MIC verification failed
    MicMismatch,
    /// Encryption failed
    EncryptionFailed,
    /// Decryption failed
    DecryptionFailed,
    /// Crypto feature not enabled
    FeatureNotEnabled,
}

impl std::fmt::Display for CryptoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CryptoError::InvalidKeyLength => write!(f, "Invalid key length"),
            CryptoError::InvalidDataLength => write!(f, "Invalid data length"),
            CryptoError::MicMismatch => write!(f, "MIC verification failed"),
            CryptoError::EncryptionFailed => write!(f, "Encryption failed"),
            CryptoError::DecryptionFailed => write!(f, "Decryption failed"),
            CryptoError::FeatureNotEnabled => {
                write!(f, "Crypto feature not enabled, compile with --features crypto")
            }
        }
    }
}

impl std::error::Error for CryptoError {}

/// Result type for crypto operations
pub type CryptoResult<T> = Result<T, CryptoError>;

/// Channel encryption key
#[derive(Clone)]
pub struct ChannelKey {
    /// 32-byte AES-256 key
    key: [u8; 32],
    /// Channel name for identification
    channel_name: String,
}

impl ChannelKey {
    /// Create a channel key from name and PSK
    #[cfg(feature = "crypto")]
    pub fn new(channel_name: &str, psk: &[u8]) -> Self {
        let key = Self::derive_key(channel_name, psk);
        Self {
            key,
            channel_name: channel_name.to_string(),
        }
    }

    /// Create a channel key with the default PSK
    #[cfg(feature = "crypto")]
    pub fn with_default_psk(channel_name: &str) -> Self {
        Self::new(channel_name, DEFAULT_PSK)
    }

    /// Create from raw 32-byte key
    pub fn from_raw(key: [u8; 32], channel_name: &str) -> Self {
        Self {
            key,
            channel_name: channel_name.to_string(),
        }
    }

    /// Derive key from channel name and PSK using SHA-256
    #[cfg(feature = "crypto")]
    fn derive_key(channel_name: &str, psk: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(channel_name.as_bytes());
        hasher.update(psk);
        hasher.update(KEY_DERIVATION_MAGIC);
        let result = hasher.finalize();
        let mut key = [0u8; 32];
        key.copy_from_slice(&result);
        key
    }

    /// Get the raw key bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.key
    }

    /// Get the channel name
    pub fn channel_name(&self) -> &str {
        &self.channel_name
    }

    /// Compute channel hash (first byte of SHA-256(channel_name))
    /// Used for quick channel identification in packet headers
    #[cfg(feature = "crypto")]
    pub fn channel_hash(&self) -> u8 {
        let mut hasher = Sha256::new();
        hasher.update(self.channel_name.as_bytes());
        hasher.finalize()[0]
    }

    /// Compute channel hash (stub when crypto disabled)
    #[cfg(not(feature = "crypto"))]
    pub fn channel_hash(&self) -> u8 {
        // Simple hash without SHA-256
        self.channel_name.bytes().fold(0u8, |acc, b| acc.wrapping_add(b))
    }
}

impl std::fmt::Debug for ChannelKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelKey")
            .field("channel_name", &self.channel_name)
            .field("key", &"[REDACTED]")
            .finish()
    }
}

/// Crypto context for a channel
#[derive(Debug)]
pub struct CryptoContext {
    /// Channel key
    key: ChannelKey,
}

impl CryptoContext {
    /// Create a new crypto context for a channel
    #[cfg(feature = "crypto")]
    pub fn new(channel_name: &str, psk: &[u8]) -> Self {
        Self {
            key: ChannelKey::new(channel_name, psk),
        }
    }

    /// Create with default PSK
    #[cfg(feature = "crypto")]
    pub fn with_default_psk(channel_name: &str) -> Self {
        Self {
            key: ChannelKey::with_default_psk(channel_name),
        }
    }

    /// Create from an existing channel key
    pub fn from_key(key: ChannelKey) -> Self {
        Self { key }
    }

    /// Get channel hash
    pub fn channel_hash(&self) -> u8 {
        self.key.channel_hash()
    }

    /// Construct the 16-byte nonce/IV for AES-CTR
    fn make_nonce(source: NodeId, packet_id: u32) -> [u8; 16] {
        let mut nonce = [0u8; 16];

        // Bytes 0-3: source node ID (little-endian)
        nonce[0..4].copy_from_slice(source.as_bytes());

        // Bytes 4-7: packet_id (little-endian)
        nonce[4..8].copy_from_slice(&packet_id.to_le_bytes());

        // Bytes 8-15: nonce_base XOR pattern
        for (i, &b) in NONCE_BASE.iter().enumerate() {
            nonce[8 + i] = b ^ (i as u8);
        }

        nonce
    }

    /// Encrypt payload and compute MIC
    ///
    /// Returns (ciphertext, mic)
    #[cfg(feature = "crypto")]
    pub fn encrypt(
        &self,
        plaintext: &[u8],
        source: NodeId,
        packet_id: u32,
        header_bytes: &[u8],
    ) -> CryptoResult<(Vec<u8>, [u8; 4])> {
        // Create nonce
        let nonce = Self::make_nonce(source, packet_id);

        // Encrypt with AES-256-CTR
        let mut ciphertext = plaintext.to_vec();
        let mut cipher = Ctr128BE::<Aes256>::new(self.key.as_bytes().into(), &nonce.into());
        cipher.apply_keystream(&mut ciphertext);

        // Compute MIC = first 4 bytes of HMAC-SHA256(key, header || ciphertext)
        let mic = self.compute_mic(header_bytes, &ciphertext)?;

        Ok((ciphertext, mic))
    }

    /// Decrypt payload and verify MIC
    #[cfg(feature = "crypto")]
    pub fn decrypt(
        &self,
        ciphertext: &[u8],
        source: NodeId,
        packet_id: u32,
        header_bytes: &[u8],
        mic: &[u8; 4],
    ) -> CryptoResult<Vec<u8>> {
        // Verify MIC first
        let expected_mic = self.compute_mic(header_bytes, ciphertext)?;
        if expected_mic != *mic {
            return Err(CryptoError::MicMismatch);
        }

        // Create nonce
        let nonce = Self::make_nonce(source, packet_id);

        // Decrypt with AES-256-CTR (same as encrypt - XOR operation)
        let mut plaintext = ciphertext.to_vec();
        let mut cipher = Ctr128BE::<Aes256>::new(self.key.as_bytes().into(), &nonce.into());
        cipher.apply_keystream(&mut plaintext);

        Ok(plaintext)
    }

    /// Compute MIC (first 4 bytes of HMAC-SHA256)
    #[cfg(feature = "crypto")]
    fn compute_mic(&self, header: &[u8], payload: &[u8]) -> CryptoResult<[u8; 4]> {
        type HmacSha256 = Hmac<Sha256>;

        let mut mac = HmacSha256::new_from_slice(self.key.as_bytes())
            .map_err(|_| CryptoError::InvalidKeyLength)?;
        mac.update(header);
        mac.update(payload);

        let result = mac.finalize().into_bytes();
        let mut mic = [0u8; 4];
        mic.copy_from_slice(&result[..4]);
        Ok(mic)
    }

    /// Stub encrypt when crypto feature is disabled
    #[cfg(not(feature = "crypto"))]
    pub fn encrypt(
        &self,
        _plaintext: &[u8],
        _source: NodeId,
        _packet_id: u32,
        _header_bytes: &[u8],
    ) -> CryptoResult<(Vec<u8>, [u8; 4])> {
        Err(CryptoError::FeatureNotEnabled)
    }

    /// Stub decrypt when crypto feature is disabled
    #[cfg(not(feature = "crypto"))]
    pub fn decrypt(
        &self,
        _ciphertext: &[u8],
        _source: NodeId,
        _packet_id: u32,
        _header_bytes: &[u8],
        _mic: &[u8; 4],
    ) -> CryptoResult<Vec<u8>> {
        Err(CryptoError::FeatureNotEnabled)
    }
}

/// Extension trait for encrypting/decrypting MeshPacket
pub trait PacketCrypto {
    /// Encrypt the packet payload
    fn encrypt(&mut self, ctx: &CryptoContext) -> CryptoResult<()>;

    /// Decrypt the packet payload
    fn decrypt(&mut self, ctx: &CryptoContext) -> CryptoResult<()>;
}

impl PacketCrypto for MeshPacket {
    #[cfg(feature = "crypto")]
    fn encrypt(&mut self, ctx: &CryptoContext) -> CryptoResult<()> {
        if self.header.flags.encrypted() {
            // Already encrypted
            return Ok(());
        }

        // Set encrypted flag BEFORE calculating MIC so header is consistent
        self.header.flags.set_encrypted(true);

        // Get header bytes for MIC calculation (with encrypted flag set)
        let header_bytes = self.header.to_bytes();

        // Use 32-bit packet_id for encryption (extend 16-bit ID)
        let packet_id_32 = self.header.packet_id as u32;

        // Encrypt payload
        let (ciphertext, mic) =
            ctx.encrypt(&self.payload, self.header.source, packet_id_32, &header_bytes)?;

        // Update packet
        self.payload = ciphertext;
        self.mic = Some(mic);

        Ok(())
    }

    #[cfg(feature = "crypto")]
    fn decrypt(&mut self, ctx: &CryptoContext) -> CryptoResult<()> {
        if !self.header.flags.encrypted() {
            // Not encrypted
            return Ok(());
        }

        let mic = self.mic.ok_or(CryptoError::MicMismatch)?;

        // Get header bytes for MIC verification (encrypted flag is still set)
        let header_bytes = self.header.to_bytes();

        // Use 32-bit packet_id
        let packet_id_32 = self.header.packet_id as u32;

        // Decrypt payload
        let plaintext = ctx.decrypt(
            &self.payload,
            self.header.source,
            packet_id_32,
            &header_bytes,
            &mic,
        )?;

        // Update packet after successful decryption
        self.payload = plaintext;
        self.mic = None;
        self.header.flags.set_encrypted(false);

        Ok(())
    }

    #[cfg(not(feature = "crypto"))]
    fn encrypt(&mut self, _ctx: &CryptoContext) -> CryptoResult<()> {
        Err(CryptoError::FeatureNotEnabled)
    }

    #[cfg(not(feature = "crypto"))]
    fn decrypt(&mut self, _ctx: &CryptoContext) -> CryptoResult<()> {
        Err(CryptoError::FeatureNotEnabled)
    }
}

// Stub implementations when crypto feature is disabled
#[cfg(not(feature = "crypto"))]
impl ChannelKey {
    /// Create a channel key (stub - returns zeroed key)
    pub fn new(channel_name: &str, _psk: &[u8]) -> Self {
        Self {
            key: [0u8; 32],
            channel_name: channel_name.to_string(),
        }
    }

    /// Create with default PSK (stub)
    pub fn with_default_psk(channel_name: &str) -> Self {
        Self::new(channel_name, DEFAULT_PSK)
    }
}

#[cfg(not(feature = "crypto"))]
impl CryptoContext {
    /// Create a new crypto context (stub)
    pub fn new(channel_name: &str, psk: &[u8]) -> Self {
        Self {
            key: ChannelKey::new(channel_name, psk),
        }
    }

    /// Create with default PSK (stub)
    pub fn with_default_psk(channel_name: &str) -> Self {
        Self {
            key: ChannelKey::with_default_psk(channel_name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_key_creation() {
        let key = ChannelKey::new("LongFast", DEFAULT_PSK);
        assert_eq!(key.channel_name(), "LongFast");
        assert_ne!(key.as_bytes(), &[0u8; 32]);
    }

    #[test]
    fn test_channel_hash() {
        let key = ChannelKey::new("LongFast", DEFAULT_PSK);
        let hash = key.channel_hash();
        // Hash should be consistent
        assert_eq!(hash, key.channel_hash());
    }

    #[test]
    fn test_crypto_context_creation() {
        let ctx = CryptoContext::with_default_psk("LongFast");
        let hash = ctx.channel_hash();
        // Should return same hash as key
        let key = ChannelKey::with_default_psk("LongFast");
        assert_eq!(hash, key.channel_hash());
    }

    #[cfg(feature = "crypto")]
    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let ctx = CryptoContext::with_default_psk("TestChannel");
        let source = NodeId::from_bytes([0x11, 0x22, 0x33, 0x44]);
        let packet_id = 12345u32;
        let header_bytes = [0u8; 16]; // Dummy header

        let plaintext = b"Hello, Meshtastic!";

        // Encrypt
        let (ciphertext, mic) = ctx
            .encrypt(plaintext, source, packet_id, &header_bytes)
            .expect("Encryption failed");

        // Ciphertext should be different from plaintext
        assert_ne!(ciphertext, plaintext);
        assert_eq!(ciphertext.len(), plaintext.len());

        // Decrypt
        let decrypted = ctx
            .decrypt(&ciphertext, source, packet_id, &header_bytes, &mic)
            .expect("Decryption failed");

        assert_eq!(decrypted, plaintext);
    }

    #[cfg(feature = "crypto")]
    #[test]
    fn test_mic_verification_fails_on_tamper() {
        let ctx = CryptoContext::with_default_psk("TestChannel");
        let source = NodeId::from_bytes([0x11, 0x22, 0x33, 0x44]);
        let packet_id = 12345u32;
        let header_bytes = [0u8; 16];

        let plaintext = b"Secret message";

        // Encrypt
        let (mut ciphertext, mic) = ctx
            .encrypt(plaintext, source, packet_id, &header_bytes)
            .expect("Encryption failed");

        // Tamper with ciphertext
        ciphertext[0] ^= 0xFF;

        // Decrypt should fail MIC verification
        let result = ctx.decrypt(&ciphertext, source, packet_id, &header_bytes, &mic);
        assert!(matches!(result, Err(CryptoError::MicMismatch)));
    }

    #[cfg(feature = "crypto")]
    #[test]
    fn test_packet_encrypt_decrypt() {
        use super::super::packet::PacketType;

        let ctx = CryptoContext::with_default_psk("TestChannel");
        let source = NodeId::from_bytes([0xAA, 0xBB, 0xCC, 0xDD]);

        let mut packet = MeshPacket::broadcast(source, b"Test payload", 3);
        packet.packet_type = PacketType::Text;

        // Initially not encrypted
        assert!(!packet.header.flags.encrypted());
        assert!(packet.mic.is_none());

        // Encrypt
        packet.encrypt(&ctx).expect("Encryption failed");
        assert!(packet.header.flags.encrypted());
        assert!(packet.mic.is_some());

        // Payload is now ciphertext
        assert_ne!(packet.payload, b"Test payload");

        // Decrypt
        packet.decrypt(&ctx).expect("Decryption failed");
        assert!(!packet.header.flags.encrypted());
        assert!(packet.mic.is_none());
        assert_eq!(packet.payload, b"Test payload");
    }

    #[cfg(feature = "crypto")]
    #[test]
    fn test_different_keys_produce_different_ciphertext() {
        let ctx1 = CryptoContext::with_default_psk("Channel1");
        let ctx2 = CryptoContext::with_default_psk("Channel2");
        let source = NodeId::from_bytes([0x11, 0x22, 0x33, 0x44]);
        let packet_id = 100u32;
        let header_bytes = [0u8; 16];

        let plaintext = b"Same message";

        let (ciphertext1, _) = ctx1
            .encrypt(plaintext, source, packet_id, &header_bytes)
            .unwrap();
        let (ciphertext2, _) = ctx2
            .encrypt(plaintext, source, packet_id, &header_bytes)
            .unwrap();

        // Different keys should produce different ciphertexts
        assert_ne!(ciphertext1, ciphertext2);
    }

    #[cfg(feature = "crypto")]
    #[test]
    fn test_nonce_uniqueness() {
        // Different source/packet_id should produce different nonces
        let nonce1 = CryptoContext::make_nonce(
            NodeId::from_bytes([1, 2, 3, 4]),
            100,
        );
        let nonce2 = CryptoContext::make_nonce(
            NodeId::from_bytes([1, 2, 3, 4]),
            101,
        );
        let nonce3 = CryptoContext::make_nonce(
            NodeId::from_bytes([5, 6, 7, 8]),
            100,
        );

        assert_ne!(nonce1, nonce2);
        assert_ne!(nonce1, nonce3);
        assert_ne!(nonce2, nonce3);
    }
}
