//! SDR Waveform Explorer Command-Line Interface
//!
//! This CLI provides tools for:
//! - Simulating SDR transmissions (including LoRa/CSS)
//! - Generating I/Q sample files
//! - Demodulating captured samples
//! - Testing channel conditions
//!
//! For real hardware operations, enable the `hardware` feature.

use anyhow::{Context, Result};
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{generate, Shell};
use r4w_core::agent::{AgentClient, AgentServer, DEFAULT_AGENT_PORT};
use r4w_core::benchmark::{BenchmarkMetrics, BenchmarkReceiver, BenchmarkReport, SampleFormat, WaveformRunner};
use r4w_core::demodulation::Demodulator;
use r4w_core::mesh::{LoRaMesh, LoRaMeshConfig, MeshPhy, ModemPreset, NodeId, Region};
use r4w_core::modulation::Modulator;
use r4w_core::waveform::adsb::{AdsbMessage, CprDecoder};
use r4w_core::waveform::ppm::PPM;
use r4w_core::params::LoRaParams;
use r4w_core::types::IQSample;
use r4w_core::waveform::{CommonParams, WaveformFactory};
use r4w_sim::{Channel, ChannelConfig, ChannelModel};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};

#[derive(Parser)]
#[command(name = "r4w")]
#[command(author, version, about = "SDR Waveform Explorer CLI", long_about = None)]
struct Cli {
    /// Enable verbose output
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Transmit (modulate) a message to I/Q samples
    Tx {
        /// Message to transmit
        #[arg(short, long)]
        message: String,

        /// Output file for I/Q samples (or - for stdout)
        #[arg(short, long, default_value = "tx_samples.iq")]
        output: PathBuf,

        /// Spreading factor (7-12)
        #[arg(long, default_value = "7")]
        sf: u8,

        /// Bandwidth in kHz (125, 250, 500)
        #[arg(long, default_value = "125")]
        bw: u32,

        /// Coding rate (5-8 for 4/5 to 4/8)
        #[arg(long, default_value = "5")]
        cr: u8,

        /// Output format (f32, f64, i16)
        #[arg(long, default_value = "f32")]
        format: String,
    },

    /// Receive (demodulate) I/Q samples to a message
    Rx {
        /// Input file with I/Q samples
        #[arg(short, long)]
        input: PathBuf,

        /// Spreading factor (7-12)
        #[arg(long, default_value = "7")]
        sf: u8,

        /// Bandwidth in kHz
        #[arg(long, default_value = "125")]
        bw: u32,

        /// Coding rate
        #[arg(long, default_value = "5")]
        cr: u8,

        /// Input format (f32, f64, i16)
        #[arg(long, default_value = "f32")]
        format: String,
    },

    /// Simulate a complete TX -> Channel -> RX pipeline
    Simulate {
        /// Message to transmit
        #[arg(short, long)]
        message: String,

        /// Signal-to-noise ratio in dB
        #[arg(long, default_value = "10.0")]
        snr: f64,

        /// Carrier frequency offset in Hz
        #[arg(long, default_value = "0.0")]
        cfo: f64,

        /// Channel model (awgn, rayleigh, rician)
        #[arg(long, default_value = "awgn")]
        channel: String,

        /// Spreading factor
        #[arg(long, default_value = "7")]
        sf: u8,

        /// Bandwidth in kHz
        #[arg(long, default_value = "125")]
        bw: u32,

        /// Coding rate
        #[arg(long, default_value = "5")]
        cr: u8,

        /// Save intermediate files
        #[arg(long)]
        save_samples: bool,
    },

    /// Generate test chirps for analysis
    Chirp {
        /// Output file
        #[arg(short, long, default_value = "chirp.iq")]
        output: PathBuf,

        /// Chirp type (up, down, symbol)
        #[arg(long, default_value = "up")]
        chirp_type: String,

        /// Symbol value (0 to 2^SF - 1)
        #[arg(long, default_value = "0")]
        symbol: u16,

        /// Spreading factor
        #[arg(long, default_value = "7")]
        sf: u8,

        /// Bandwidth in kHz
        #[arg(long, default_value = "125")]
        bw: u32,
    },

    /// Show LoRa parameter calculations
    Info {
        /// Spreading factor
        #[arg(long, default_value = "7")]
        sf: u8,

        /// Bandwidth in kHz
        #[arg(long, default_value = "125")]
        bw: u32,

        /// Coding rate
        #[arg(long, default_value = "5")]
        cr: u8,

        /// Payload length in bytes
        #[arg(long, default_value = "10")]
        payload_len: usize,
    },

    /// Analyze I/Q samples (spectrum, waterfall, statistics, peaks)
    Analyze {
        /// Input file with I/Q samples or SigMF metadata
        #[arg(short, long)]
        input: PathBuf,

        /// Input format (f32, f64, i16, sigmf)
        #[arg(long, default_value = "f32")]
        format: String,

        /// Sample rate in Hz (required for frequency calculations)
        #[arg(long, default_value = "1000000")]
        sample_rate: f64,

        /// Number of samples to analyze (0 = all)
        #[arg(long, default_value = "0")]
        samples: usize,

        /// Analysis mode: basic, spectrum, waterfall, stats, peaks
        #[arg(long, default_value = "basic")]
        mode: String,

        /// FFT size for spectrum/waterfall analysis
        #[arg(long, default_value = "1024")]
        fft_size: usize,

        /// Number of frames to average (for spectrum)
        #[arg(long, default_value = "1")]
        average: usize,

        /// Output format: text, json, csv, ascii
        #[arg(long, short = 'o', default_value = "text")]
        output_format: String,

        /// Output file (stdout if not specified)
        #[arg(long)]
        output: Option<PathBuf>,

        /// Window function: hann, hamming, blackman, none
        #[arg(long, default_value = "hann")]
        window: String,

        /// Minimum dB for waterfall color scale
        #[arg(long, default_value = "-60")]
        min_db: f64,

        /// Maximum dB for waterfall color scale
        #[arg(long, default_value = "0")]
        max_db: f64,

        /// Colormap for waterfall: viridis, plasma, turbo, grayscale
        #[arg(long, default_value = "viridis")]
        colormap: String,

        /// Peak detection threshold in dB above noise floor
        #[arg(long, default_value = "10")]
        threshold: f64,

        /// Maximum number of peaks to find
        #[arg(long, default_value = "10")]
        max_peaks: usize,

        /// Waterfall height (rows) for PNG output
        #[arg(long, default_value = "512")]
        height: usize,
    },

    /// Simulate a waveform (AM, FM, OOK, FSK, PSK, QAM)
    Waveform {
        /// Waveform type (CW, AM, FM, OOK, BFSK, BPSK, QPSK, 16QAM)
        #[arg(short, long, default_value = "")]
        waveform: String,

        /// Test data as string (e.g., "Hello")
        #[arg(short, long, default_value = "Hello")]
        data: String,

        /// Signal-to-noise ratio in dB
        #[arg(long, default_value = "10.0")]
        snr: f64,

        /// Sample rate in Hz
        #[arg(long, default_value = "10000")]
        sample_rate: u32,

        /// Symbol rate in symbols/second
        #[arg(long, default_value = "1000")]
        symbol_rate: u32,

        /// List available waveforms
        #[arg(long)]
        list: bool,
    },

    /// Run waveform benchmark with UDP input
    Benchmark {
        /// UDP port to listen on
        #[arg(short, long, default_value = "5000")]
        port: u16,

        /// Sample format (f32 or i16)
        #[arg(short, long, default_value = "f32")]
        format: String,

        /// Waveform to benchmark (BPSK, QPSK, LoRa, etc.)
        #[arg(short, long, default_value = "")]
        waveform: String,

        /// Sample rate in Hz
        #[arg(short, long, default_value = "48000")]
        sample_rate: f64,

        /// Batch size (samples per processing cycle)
        #[arg(short, long, default_value = "1024")]
        batch_size: usize,

        /// Duration in seconds (0 = run until Ctrl+C)
        #[arg(short, long, default_value = "0")]
        duration: u64,

        /// Output format (json, text, csv)
        #[arg(short, long, default_value = "text")]
        output: String,

        /// Output file (stdout if not specified)
        #[arg(long)]
        output_file: Option<PathBuf>,

        /// Print live stats every N seconds (0 = disabled)
        #[arg(long, default_value = "1")]
        stats_interval: u64,

        /// List available waveforms
        #[arg(long)]
        list: bool,
    },

    /// Generate and send test I/Q samples via UDP
    UdpSend {
        /// Target address (host:port)
        #[arg(short, long, default_value = "127.0.0.1:5000")]
        target: String,

        /// Waveform to generate
        #[arg(short, long, default_value = "")]
        waveform: String,

        /// Sample rate in Hz
        #[arg(short, long, default_value = "48000")]
        sample_rate: f64,

        /// Sample format (f32 or i16)
        #[arg(short, long, default_value = "f32")]
        format: String,

        /// Test message to modulate
        #[arg(short, long, default_value = "Hello SDR Benchmark!")]
        message: String,

        /// Packets per second (0 = as fast as possible)
        #[arg(long, default_value = "100")]
        pps: u32,

        /// Samples per packet
        #[arg(long, default_value = "1024")]
        samples_per_packet: usize,

        /// Duration in seconds (0 = infinite)
        #[arg(short, long, default_value = "0")]
        duration: u64,

        /// Add noise (SNR in dB, negative = no noise)
        #[arg(long, default_value = "-1")]
        snr: f64,

        /// List available waveforms
        #[arg(long)]
        list: bool,

        /// Repeat message continuously
        #[arg(long)]
        repeat: bool,
    },

    /// Run as a remote agent daemon (for Raspberry Pi deployment)
    Agent {
        /// Port to listen on for control connections
        #[arg(short, long, default_value_t = DEFAULT_AGENT_PORT)]
        port: u16,

        /// Run in foreground (don't daemonize)
        #[arg(long)]
        foreground: bool,
    },

    /// Connect to a remote agent
    Remote {
        /// Remote agent address (host:port or just host for default port)
        #[arg(short, long)]
        address: String,

        /// Command to send
        #[command(subcommand)]
        command: RemoteCommand,
    },

    /// LoRa mesh networking commands
    Mesh {
        #[command(subcommand)]
        command: MeshCommand,
    },

    /// ADS-B aircraft tracking commands
    Adsb {
        #[command(subcommand)]
        command: AdsbCommand,
    },

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },

    /// Compare waveforms across SNR range
    #[command(allow_negative_numbers = true)]
    Compare {
        /// Waveforms to compare (comma-separated, e.g., "BPSK,QPSK,8PSK")
        #[arg(short, long, default_value = "")]
        waveforms: String,

        /// Minimum SNR in dB
        #[arg(long, default_value = "-5")]
        snr_min: f64,

        /// Maximum SNR in dB
        #[arg(long, default_value = "20")]
        snr_max: f64,

        /// SNR step in dB
        #[arg(long, default_value = "2.5")]
        snr_step: f64,

        /// Number of bits per test point
        #[arg(long, default_value = "10000")]
        bits: usize,

        /// Sample rate in Hz
        #[arg(long, default_value = "48000")]
        sample_rate: f64,

        /// Output format (text, json, csv)
        #[arg(short, long, default_value = "text")]
        output: String,

        /// Output file (stdout if not specified)
        #[arg(long)]
        output_file: Option<PathBuf>,

        /// List available waveforms
        #[arg(long)]
        list: bool,
    },

    /// Record I/Q samples to SigMF file
    Record {
        /// Output file path (without extension, creates .sigmf-meta and .sigmf-data)
        #[arg(short, long)]
        output: PathBuf,

        /// Sample rate in Hz
        #[arg(short, long, default_value = "1000000")]
        sample_rate: f64,

        /// Center frequency in Hz
        #[arg(short, long, default_value = "915000000")]
        frequency: f64,

        /// Duration in seconds (0 = until Ctrl+C)
        #[arg(short, long, default_value = "0")]
        duration: f64,

        /// Description for the recording
        #[arg(long)]
        description: Option<String>,

        /// Waveform name (for metadata)
        #[arg(long)]
        waveform: Option<String>,

        /// Generate test signal instead of recording from device
        #[arg(long)]
        generate: Option<String>,
    },

    /// Playback I/Q samples from SigMF file
    Playback {
        /// Input SigMF file (.sigmf-meta)
        #[arg(short, long)]
        input: PathBuf,

        /// Show file info only (don't play)
        #[arg(long)]
        info: bool,

        /// Output format for playback (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Number of samples to read (0 = all)
        #[arg(short, long, default_value = "0")]
        samples: usize,

        /// Start offset in samples
        #[arg(long, default_value = "0")]
        offset: usize,

        /// Apply waveform demodulation
        #[arg(long)]
        demodulate: Option<String>,
    },

    /// Convert between signal file formats
    Convert {
        /// Input file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Input format (auto-detect if not specified)
        #[arg(long)]
        from: Option<String>,

        /// Output format (sigmf, raw-cf32, raw-ci16)
        #[arg(long, default_value = "sigmf")]
        to: String,

        /// Sample rate (required for raw input)
        #[arg(long)]
        sample_rate: Option<f64>,

        /// Center frequency (optional)
        #[arg(long)]
        frequency: Option<f64>,
    },

    /// Display or serve Prometheus metrics
    Metrics {
        /// Output format (text, json, prometheus)
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Start HTTP server for Prometheus scraping
        #[arg(long)]
        serve: bool,

        /// Port for metrics server
        #[arg(short, long, default_value = "9090")]
        port: u16,
    },
}

#[derive(Subcommand)]
enum RemoteCommand {
    /// Get agent status
    Status,

    /// Ping the agent
    Ping,

    /// Start transmitting
    StartTx {
        /// Target address for UDP samples (host:port)
        #[arg(short, long)]
        target: String,

        /// Waveform to transmit
        #[arg(short, long)]
        waveform: String,

        /// Sample rate in Hz
        #[arg(short, long, default_value = "48000")]
        sample_rate: f64,

        /// Message to transmit
        #[arg(short, long, default_value = "Hello SDR!")]
        message: String,

        /// SNR to add (-1 for none)
        #[arg(long, default_value = "-1")]
        snr: f64,

        /// Packets per second
        #[arg(long, default_value = "100")]
        pps: u32,

        /// Repeat continuously
        #[arg(long)]
        repeat: bool,
    },

    /// Stop transmitting
    StopTx,

    /// Start receiving/benchmarking
    StartRx {
        /// UDP port to listen on
        #[arg(short, long, default_value = "5000")]
        port: u16,

        /// Waveform to demodulate
        #[arg(short, long)]
        waveform: String,

        /// Sample rate in Hz
        #[arg(short, long, default_value = "48000")]
        sample_rate: f64,
    },

    /// Stop receiving
    StopRx,

    /// List available waveforms
    ListWaveforms,

    /// Shutdown the remote agent
    Shutdown,
}

#[derive(Subcommand)]
enum MeshCommand {
    /// Show mesh node status and statistics
    Status {
        /// Node ID (hex, e.g., "a1b2c3d4")
        #[arg(short, long)]
        node_id: Option<String>,

        /// Modem preset (LongFast, LongSlow, MedFast, MedSlow, ShortFast, ShortSlow)
        #[arg(short, long, default_value = "LongFast")]
        preset: String,

        /// Region (US, EU868, CN, JP, ANZ, KR, TW, RU, IN, NZ, UK, AU)
        #[arg(short, long, default_value = "US")]
        region: String,
    },

    /// Send a mesh message
    Send {
        /// Message to send
        #[arg(short, long)]
        message: String,

        /// Destination node ID (hex) for direct message, or "broadcast"
        #[arg(short, long, default_value = "broadcast")]
        dest: String,

        /// Hop limit for message propagation
        #[arg(long, default_value = "3")]
        hop_limit: u8,

        /// Node ID (hex, e.g., "a1b2c3d4")
        #[arg(short, long)]
        node_id: Option<String>,

        /// Modem preset
        #[arg(short, long, default_value = "LongFast")]
        preset: String,

        /// Region
        #[arg(short, long, default_value = "US")]
        region: String,
    },

    /// List discovered neighbors
    Neighbors {
        /// Node ID (hex)
        #[arg(short, long)]
        node_id: Option<String>,

        /// Modem preset
        #[arg(short, long, default_value = "LongFast")]
        preset: String,

        /// Region
        #[arg(short, long, default_value = "US")]
        region: String,
    },

    /// Simulate a mesh network with multiple nodes
    Simulate {
        /// Number of nodes to simulate
        #[arg(short, long, default_value = "4")]
        nodes: usize,

        /// Number of messages to exchange
        #[arg(short, long, default_value = "10")]
        messages: usize,

        /// SNR in dB for channel simulation
        #[arg(long, default_value = "10.0")]
        snr: f64,

        /// Modem preset
        #[arg(short, long, default_value = "LongFast")]
        preset: String,

        /// Region
        #[arg(short, long, default_value = "US")]
        region: String,

        /// Verbose output showing packet flow
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show available presets and regions
    Info,

    /// Analyze a Meshtastic packet (hex dump with annotations)
    Analyze {
        /// Packet data as hex string (space-separated or continuous)
        #[arg(long)]
        hex: String,

        /// Show compact one-line summary only
        #[arg(short, long)]
        summary: bool,
    },

    /// Interactive mesh REPL for testing and debugging
    Repl {
        /// Node ID (hex, e.g., "a1b2c3d4")
        #[arg(short, long)]
        node_id: Option<String>,

        /// Modem preset
        #[arg(short, long, default_value = "LongFast")]
        preset: String,

        /// Region
        #[arg(short, long, default_value = "US")]
        region: String,

        /// Number of simulated nodes for multi-node mode
        #[arg(long, default_value = "1")]
        sim_nodes: usize,
    },
}

#[derive(Subcommand)]
enum AdsbCommand {
    /// Decode a raw ADS-B message (hex format)
    Decode {
        /// Raw message in hex (e.g., "8D4840D6202CC371C32CE0576098")
        #[arg(short, long)]
        message: String,

        /// Show raw bit fields
        #[arg(short, long)]
        verbose: bool,
    },

    /// Decode ADS-B messages from I/Q sample file
    File {
        /// Input file with I/Q samples
        #[arg(short, long)]
        input: PathBuf,

        /// Sample rate in Hz
        #[arg(short, long, default_value = "2000000")]
        sample_rate: f64,

        /// Show all messages (including CRC failures)
        #[arg(long)]
        all: bool,
    },

    /// Show ADS-B protocol information
    Info,

    /// Generate a test ADS-B signal
    Generate {
        /// Output file for I/Q samples
        #[arg(short, long, default_value = "adsb_test.iq")]
        output: PathBuf,

        /// ICAO address (hex)
        #[arg(long, default_value = "AABBCC")]
        icao: String,

        /// Callsign (8 chars max)
        #[arg(long, default_value = "TEST1234")]
        callsign: String,

        /// Altitude in feet
        #[arg(long, default_value = "35000")]
        altitude: i32,

        /// Sample rate in Hz
        #[arg(short, long, default_value = "2000000")]
        sample_rate: f64,
    },
}

fn validate_sf(sf: u8) -> Result<u8> {
    if (5..=12).contains(&sf) {
        Ok(sf)
    } else {
        anyhow::bail!("Invalid spreading factor: {}. Must be 5-12", sf)
    }
}

fn validate_bw(bw: u32) -> Result<u32> {
    match bw {
        125 => Ok(125_000), // Convert kHz to Hz
        250 => Ok(250_000),
        500 => Ok(500_000),
        _ => anyhow::bail!("Invalid bandwidth: {}kHz. Must be 125, 250, or 500", bw),
    }
}

fn validate_cr(cr: u8) -> Result<u8> {
    if (5..=8).contains(&cr) {
        Ok(cr)
    } else {
        anyhow::bail!("Invalid coding rate: 4/{}. Must be 4/5 to 4/8", cr)
    }
}

fn parse_channel_model(model: &str) -> Result<ChannelModel> {
    match model.to_lowercase().as_str() {
        "awgn" => Ok(ChannelModel::Awgn),
        "rayleigh" => Ok(ChannelModel::Rayleigh),
        "rician" => Ok(ChannelModel::Rician),
        _ => anyhow::bail!("Unknown channel model: {}. Use awgn, rayleigh, or rician", model),
    }
}

fn parse_preset(preset: &str) -> Result<ModemPreset> {
    match preset.to_lowercase().as_str() {
        "longfast" | "long_fast" => Ok(ModemPreset::LongFast),
        "longslow" | "long_slow" => Ok(ModemPreset::LongSlow),
        "longmoderate" | "long_moderate" => Ok(ModemPreset::LongModerate),
        "medfast" | "med_fast" | "mediumfast" | "medium_fast" => Ok(ModemPreset::MediumFast),
        "medslow" | "med_slow" | "mediumslow" | "medium_slow" => Ok(ModemPreset::MediumSlow),
        "shortfast" | "short_fast" => Ok(ModemPreset::ShortFast),
        "shortslow" | "short_slow" => Ok(ModemPreset::ShortSlow),
        _ => anyhow::bail!(
            "Unknown preset: {}. Use LongFast, LongSlow, LongModerate, MediumFast, MediumSlow, ShortFast, ShortSlow",
            preset
        ),
    }
}

fn parse_region(region: &str) -> Result<Region> {
    match region.to_uppercase().as_str() {
        "US" => Ok(Region::US),
        "EU868" | "EU" => Ok(Region::EU),
        "CN" | "CHINA" => Ok(Region::CN),
        "JP" | "JAPAN" => Ok(Region::JP),
        "ANZ" | "AU" | "NZ" | "AUSTRALIA" => Ok(Region::ANZ),
        "KR" | "KOREA" => Ok(Region::KR),
        "TW" | "TAIWAN" => Ok(Region::TW),
        "IN" | "INDIA" => Ok(Region::IN),
        _ => anyhow::bail!(
            "Unknown region: {}. Use US, EU, CN, JP, ANZ, KR, TW, IN",
            region
        ),
    }
}

fn parse_node_id(id: &Option<String>) -> Result<Option<NodeId>> {
    match id {
        Some(s) => {
            let value = u32::from_str_radix(s.trim_start_matches("0x"), 16)
                .with_context(|| format!("Invalid node ID hex: {}", s))?;
            Ok(Some(NodeId::from_u32(value)))
        }
        None => Ok(None),
    }
}

fn create_mesh_config(
    node_id: Option<String>,
    preset: String,
    region: String,
) -> Result<LoRaMeshConfig> {
    Ok(LoRaMeshConfig {
        node_id: parse_node_id(&node_id)?,
        preset: parse_preset(&preset)?,
        region: parse_region(&region)?,
        ..Default::default()
    })
}

fn write_samples_f32(samples: &[IQSample], path: &PathBuf) -> Result<()> {
    use byteorder::{LittleEndian, WriteBytesExt};

    let file = File::create(path).context("Failed to create output file")?;
    let mut writer = BufWriter::new(file);

    for sample in samples {
        writer.write_f32::<LittleEndian>(sample.re as f32)?;
        writer.write_f32::<LittleEndian>(sample.im as f32)?;
    }

    writer.flush()?;
    Ok(())
}

fn read_samples_f32(path: &PathBuf) -> Result<Vec<IQSample>> {
    use byteorder::{LittleEndian, ReadBytesExt};

    let file = File::open(path).context("Failed to open input file")?;
    let metadata = file.metadata()?;
    let num_samples = metadata.len() as usize / 8; // 2 x f32 per sample

    let mut reader = BufReader::new(file);
    let mut samples = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let re = reader.read_f32::<LittleEndian>()? as f64;
        let im = reader.read_f32::<LittleEndian>()? as f64;
        samples.push(IQSample::new(re, im));
    }

    Ok(samples)
}

fn cmd_tx(
    message: String,
    output: PathBuf,
    sf: u8,
    bw: u32,
    cr: u8,
    _format: String,
) -> Result<()> {
    validate_sf(sf)?;
    validate_cr(cr)?;
    let bw_hz = validate_bw(bw)?;

    let params = LoRaParams::builder()
        .spreading_factor(sf)
        .bandwidth(bw_hz)
        .coding_rate(cr)
        .build();

    info!("Transmitting message: '{}'", message);
    info!("Parameters: SF{}, BW {}kHz, CR 4/{}", sf, bw, cr);

    let mut modulator = Modulator::new(params.clone());
    let samples = modulator.modulate(message.as_bytes());

    info!("Generated {} I/Q samples", samples.len());
    info!(
        "Duration: {:.3} ms",
        samples.len() as f64 / params.sample_rate * 1000.0
    );

    write_samples_f32(&samples, &output)?;
    info!("Wrote samples to {:?}", output);

    Ok(())
}

fn cmd_rx(input: PathBuf, sf: u8, bw: u32, cr: u8, _format: String) -> Result<()> {
    validate_sf(sf)?;
    validate_cr(cr)?;
    let bw_hz = validate_bw(bw)?;

    let params = LoRaParams::builder()
        .spreading_factor(sf)
        .bandwidth(bw_hz)
        .coding_rate(cr)
        .build();

    info!("Reading samples from {:?}", input);
    let samples = read_samples_f32(&input)?;
    info!("Read {} I/Q samples", samples.len());

    let mut demodulator = Demodulator::new(params.clone());

    // Skip preamble (simplified - in real use, we'd do proper sync)
    let n = params.samples_per_symbol();
    let preamble_len = (params.preamble_length + 4) * n + n / 4;

    if samples.len() <= preamble_len {
        warn!("Sample file too short to contain payload");
        return Ok(());
    }

    let payload_samples = &samples[preamble_len..];

    match demodulator.demodulate(payload_samples) {
        Ok(result) => {
            info!("Demodulated {} symbols", result.symbols.len());

            match String::from_utf8(result.payload.clone()) {
                Ok(text) => {
                    println!("Received message: {}", text);
                }
                Err(_) => {
                    println!("Received bytes: {:02X?}", result.payload);
                }
            }

            info!("RSSI: {:.1} dB", result.rssi);
            info!("CFO estimate: {:.1} Hz", result.cfo);
        }
        Err(e) => {
            warn!("Demodulation failed: {}", e);
        }
    }

    Ok(())
}

fn cmd_simulate(
    message: String,
    snr: f64,
    cfo: f64,
    channel_model: String,
    sf: u8,
    bw: u32,
    cr: u8,
    save_samples: bool,
) -> Result<()> {
    validate_sf(sf)?;
    validate_cr(cr)?;
    let bw_hz = validate_bw(bw)?;

    let params = LoRaParams::builder()
        .spreading_factor(sf)
        .bandwidth(bw_hz)
        .coding_rate(cr)
        .build();

    let channel_config = ChannelConfig {
        model: parse_channel_model(&channel_model)?,
        snr_db: snr,
        cfo_hz: cfo,
        ..Default::default()
    };

    println!("=== LoRa Simulation ===");
    println!("Message: '{}'", message);
    println!("SF{}, BW {}kHz, CR 4/{}", sf, bw, cr);
    println!("Channel: {:?}, SNR: {:.1} dB, CFO: {:.1} Hz", channel_config.model, snr, cfo);
    println!();

    // Transmit - prepend length byte to payload for proper decoding
    let mut modulator = Modulator::new(params.clone());
    let payload_with_length = {
        let mut data = Vec::with_capacity(message.len() + 1);
        data.push(message.len() as u8); // Length byte
        data.extend_from_slice(message.as_bytes());
        data
    };
    let tx_samples = modulator.modulate(&payload_with_length);
    println!("TX: {} samples generated", tx_samples.len());

    if save_samples {
        write_samples_f32(&tx_samples, &PathBuf::from("sim_tx.iq"))?;
        println!("  Saved to sim_tx.iq");
    }

    // Channel
    let mut channel = Channel::new(channel_config);
    let rx_samples = channel.apply(&tx_samples);
    println!("Channel: Applied {} model", channel_model);

    if save_samples {
        write_samples_f32(&rx_samples, &PathBuf::from("sim_rx.iq"))?;
        println!("  Saved to sim_rx.iq");
    }

    // Receive
    let mut demodulator = Demodulator::new(params.clone());
    let n = params.samples_per_symbol();
    let preamble_len = (params.preamble_length + 4) * n + n / 4;

    if rx_samples.len() <= preamble_len {
        println!("RX: Sample count too short for payload");
        return Ok(());
    }

    let payload_samples = &rx_samples[preamble_len..];

    match demodulator.demodulate(payload_samples) {
        Ok(result) => {
            println!("RX: {} symbols demodulated", result.symbols.len());

            // Compare TX and RX symbols
            let tx_symbols = modulator.get_symbols(&payload_with_length);
            let matching = tx_symbols
                .iter()
                .zip(result.symbols.iter())
                .filter(|(a, b)| a == b)
                .count();
            let ser = 1.0 - (matching as f64 / tx_symbols.len() as f64);
            println!(
                "SER: {:.2}% ({}/{} symbols correct)",
                ser * 100.0,
                matching,
                tx_symbols.len()
            );

            // Extract length byte and truncate payload
            if result.payload.is_empty() {
                println!("RX: Empty payload");
                println!("Result: DECODE FAILURE");
                return Ok(());
            }

            let decoded_length = result.payload[0] as usize;
            let decoded_payload = if decoded_length + 1 <= result.payload.len() {
                &result.payload[1..decoded_length + 1]
            } else {
                // Length field corrupted, show raw data
                println!("RX: Length field corrupted ({} > {})", decoded_length, result.payload.len() - 1);
                println!("RX raw: {:02X?}", result.payload);
                println!("Result: DECODE FAILURE");
                return Ok(());
            };

            match String::from_utf8(decoded_payload.to_vec()) {
                Ok(text) => {
                    let ber = message
                        .bytes()
                        .zip(text.bytes())
                        .filter(|(a, b)| a != b)
                        .count() as f64
                        / message.len().max(1) as f64;

                    println!();
                    println!("TX: '{}'", message);
                    println!("RX: '{}'", text);
                    println!("BER: {:.2}%", ber * 100.0);

                    if text == message {
                        println!("Result: SUCCESS");
                    } else {
                        println!("Result: ERRORS DETECTED");
                    }
                }
                Err(_) => {
                    println!("RX: {:02X?}", decoded_payload);
                    println!("Result: DECODE FAILURE (not valid UTF-8)");
                }
            }
        }
        Err(e) => {
            println!("RX: Demodulation failed - {}", e);
            println!("Result: FAILURE");
        }
    }

    Ok(())
}

fn cmd_chirp(output: PathBuf, chirp_type: String, symbol: u16, sf: u8, bw: u32) -> Result<()> {
    use r4w_core::chirp::ChirpGenerator;

    validate_sf(sf)?;
    let bw_hz = validate_bw(bw)?;

    let params = LoRaParams::builder()
        .spreading_factor(sf)
        .bandwidth(bw_hz)
        .build();

    let chirp_gen = ChirpGenerator::new(params.clone());

    let samples = match chirp_type.as_str() {
        "up" => chirp_gen.base_upchirp().to_vec(),
        "down" => chirp_gen.base_downchirp().to_vec(),
        "symbol" => {
            let max_symbol = (1u16 << sf) - 1;
            if symbol > max_symbol {
                anyhow::bail!("Symbol {} exceeds max {} for SF{}", symbol, max_symbol, sf);
            }
            chirp_gen.generate_symbol_chirp(symbol)
        }
        _ => anyhow::bail!("Unknown chirp type: {}. Use up, down, or symbol", chirp_type),
    };

    write_samples_f32(&samples, &output)?;

    println!("Generated {} chirp (symbol {})", chirp_type, symbol);
    println!("SF{}, BW {}kHz", sf, bw);
    println!("{} samples written to {:?}", samples.len(), output);

    Ok(())
}

fn cmd_info(sf: u8, bw: u32, cr: u8, payload_len: usize) -> Result<()> {
    validate_sf(sf)?;
    validate_cr(cr)?;
    let bw_hz = validate_bw(bw)?;

    let params = LoRaParams::builder()
        .spreading_factor(sf)
        .bandwidth(bw_hz)
        .coding_rate(cr)
        .build();

    // Symbol rate = 1 / symbol_duration
    let symbol_rate = 1.0 / params.symbol_duration();

    println!("=== LoRa Parameter Calculator ===");
    println!();
    println!("Configuration:");
    println!("  Spreading Factor:  SF{}", sf);
    println!("  Bandwidth:         {} kHz", bw);
    println!("  Coding Rate:       4/{}", cr);
    println!("  Payload Length:    {} bytes", payload_len);
    println!();
    println!("Derived Values:");
    println!("  Chips per symbol:  {}", 1 << sf);
    println!("  Sample rate:       {} Hz", params.sample_rate as u32);
    println!("  Samples/symbol:    {}", params.samples_per_symbol());
    println!("  Symbol rate:       {:.2} symbols/s", symbol_rate);
    println!("  Bit rate:          {:.2} bits/s", params.bit_rate());
    println!();
    println!("Timing:");
    println!(
        "  Symbol duration:   {:.3} ms",
        params.symbol_duration() * 1000.0
    );
    println!(
        "  Time on air:       {:.2} ms",
        params.time_on_air(payload_len) * 1000.0
    );
    println!();
    println!("Performance:");
    println!("  Sensitivity:       {:.1} dBm", params.sensitivity());
    println!(
        "  SNR threshold:     {:.1} dB",
        params.sf.snr_threshold()
    );
    println!();
    println!("Preamble:");
    println!("  Upchirps:          {}", params.preamble_length);
    println!("  Sync words:        2");
    println!("  Downchirps:        2.25");

    Ok(())
}

/// Arguments for the analyze command
struct AnalyzeArgs {
    input: PathBuf,
    #[allow(dead_code)]
    format: String,
    sample_rate: f64,
    samples: usize,
    mode: String,
    fft_size: usize,
    average: usize,
    output_format: String,
    output: Option<PathBuf>,
    window: String,
    min_db: f64,
    max_db: f64,
    colormap: String,
    threshold: f64,
    max_peaks: usize,
    height: usize,
}

fn cmd_analyze(args: AnalyzeArgs) -> Result<()> {
    use r4w_core::analysis::{
        Colormap, PeakFinder, SignalStats, SpectrumAnalyzer, WaterfallGenerator, WindowFunction,
    };
    use r4w_core::fft_utils::FftProcessor;

    // Determine data file path (handle SigMF format)
    let input_ext = args.input.extension().and_then(|e| e.to_str()).unwrap_or("");
    let data_path = if input_ext == "sigmf-meta" {
        args.input.with_extension("sigmf-data")
    } else if input_ext == "sigmf-data" {
        args.input.clone()
    } else {
        args.input.clone()
    };

    // Read samples
    let samples = read_samples_f32(&data_path)?;
    let analyze_count = if args.samples == 0 {
        samples.len()
    } else {
        args.samples.min(samples.len())
    };
    let analyze_samples = &samples[..analyze_count];

    match args.mode.as_str() {
        "basic" => {
            // Original basic analysis
            println!("=== I/Q Sample Analysis ===");
            println!("File: {:?}", args.input);
            println!("Total samples: {}", samples.len());
            println!("Analyzing: {}", analyze_count);
            println!();

            let mean_i: f64 = analyze_samples.iter().map(|s| s.re).sum::<f64>() / analyze_count as f64;
            let mean_q: f64 = analyze_samples.iter().map(|s| s.im).sum::<f64>() / analyze_count as f64;

            let var_i: f64 = analyze_samples
                .iter()
                .map(|s| (s.re - mean_i).powi(2))
                .sum::<f64>()
                / analyze_count as f64;
            let var_q: f64 = analyze_samples
                .iter()
                .map(|s| (s.im - mean_q).powi(2))
                .sum::<f64>()
                / analyze_count as f64;

            let magnitudes: Vec<f64> = analyze_samples.iter().map(|s| s.norm()).collect();
            let mean_mag: f64 = magnitudes.iter().sum::<f64>() / analyze_count as f64;
            let max_mag = magnitudes.iter().fold(0.0_f64, |a, &b| a.max(b));
            let min_mag = magnitudes.iter().fold(f64::MAX, |a, &b| a.min(b));

            println!("Time Domain Statistics:");
            println!("  DC offset (I):     {:.6}", mean_i);
            println!("  DC offset (Q):     {:.6}", mean_q);
            println!("  Variance (I):      {:.6}", var_i);
            println!("  Variance (Q):      {:.6}", var_q);
            println!("  Std dev (I):       {:.6}", var_i.sqrt());
            println!("  Std dev (Q):       {:.6}", var_q.sqrt());
            println!();
            println!("Magnitude Statistics:");
            println!("  Mean:              {:.6}", mean_mag);
            println!("  Min:               {:.6}", min_mag);
            println!("  Max:               {:.6}", max_mag);
            println!("  Peak-to-Average:   {:.2} dB", 20.0 * (max_mag / mean_mag).log10());
            println!();

            let fft_size = 1024.min(analyze_count);
            if analyze_count >= fft_size {
                let mut fft = FftProcessor::new(fft_size);
                let spectrum = fft.fft(&analyze_samples[..fft_size]);
                let power_db = FftProcessor::power_spectrum_db(&spectrum);

                let max_bin = power_db
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                let max_power = power_db[max_bin];
                let noise_floor: f64 = power_db.iter().sum::<f64>() / power_db.len() as f64;

                println!("Frequency Domain (FFT size: {}):", fft_size);
                println!("  Peak bin:          {}", max_bin);
                println!("  Peak power:        {:.1} dB", max_power);
                println!("  Noise floor:       {:.1} dB", noise_floor);
                println!("  SNR estimate:      {:.1} dB", max_power - noise_floor);
            }
        }

        "spectrum" => {
            let window = WindowFunction::from_str(&args.window)
                .unwrap_or(WindowFunction::Hann);
            let mut analyzer = SpectrumAnalyzer::with_window(args.fft_size, window);
            let result = analyzer.compute_averaged(analyze_samples, args.sample_rate, args.average);

            let output_text = match args.output_format.as_str() {
                "json" => result.to_json(),
                "csv" => result.to_csv(),
                "ascii" => result.to_ascii(80, 20),
                _ => result.to_text(),
            };

            if let Some(output_path) = args.output {
                std::fs::write(&output_path, &output_text)?;
                println!("Spectrum written to {:?}", output_path);
            } else {
                println!("{}", output_text);
            }
        }

        "waterfall" => {
            let mut generator = WaterfallGenerator::with_hop(args.fft_size, args.fft_size / 2);
            let result = generator.compute_with_limit(analyze_samples, args.sample_rate, Some(args.height));

            #[allow(unused_variables)]
            let colormap = Colormap::from_str(&args.colormap)
                .unwrap_or(Colormap::Viridis);

            match args.output_format.as_str() {
                "ascii" => {
                    let ascii = result.to_ascii_with_range(80, 40, args.min_db, args.max_db);
                    if let Some(output_path) = args.output {
                        std::fs::write(&output_path, &ascii)?;
                        println!("Waterfall written to {:?}", output_path);
                    } else {
                        println!("{}", ascii);
                    }
                }
                "png" => {
                    #[cfg(feature = "image")]
                    {
                        let png_data = result.to_png(colormap, args.min_db, args.max_db);
                        let output_path = args.output.unwrap_or_else(|| PathBuf::from("waterfall.png"));
                        std::fs::write(&output_path, &png_data)?;
                        println!("Waterfall PNG written to {:?}", output_path);
                        println!("Dimensions: {}x{}", result.fft_size, result.power_db.len());
                    }
                    #[cfg(not(feature = "image"))]
                    {
                        anyhow::bail!("PNG output requires the 'image' feature. Use --output-format ascii instead.");
                    }
                }
                _ => {
                    // Default to ASCII for text output
                    let ascii = result.to_ascii_with_range(80, 40, args.min_db, args.max_db);
                    println!("{}", ascii);
                }
            }
        }

        "stats" => {
            let stats = SignalStats::compute(analyze_samples, Some(args.sample_rate));

            let output_text = match args.output_format.as_str() {
                "json" => stats.to_json(),
                _ => stats.to_text(),
            };

            if let Some(output_path) = args.output {
                std::fs::write(&output_path, &output_text)?;
                println!("Statistics written to {:?}", output_path);
            } else {
                println!("{}", output_text);
            }
        }

        "peaks" => {
            let window = WindowFunction::from_str(&args.window)
                .unwrap_or(WindowFunction::Hann);
            let mut analyzer = SpectrumAnalyzer::with_window(args.fft_size, window);
            let spectrum = analyzer.compute_averaged(analyze_samples, args.sample_rate, args.average);

            let finder = PeakFinder::new()
                .with_threshold(args.threshold)
                .with_max_peaks(args.max_peaks);
            let peaks = finder.find_peaks(&spectrum);

            let output_text = match args.output_format.as_str() {
                "json" => PeakFinder::format_json(&peaks),
                "csv" => PeakFinder::format_csv(&peaks),
                _ => PeakFinder::format_text(&peaks),
            };

            if let Some(output_path) = args.output {
                std::fs::write(&output_path, &output_text)?;
                println!("Peaks written to {:?}", output_path);
            } else {
                println!("{}", output_text);
            }
        }

        _ => {
            anyhow::bail!(
                "Unknown analysis mode: '{}'. Use: basic, spectrum, waterfall, stats, peaks",
                args.mode
            );
        }
    }

    Ok(())
}

fn cmd_waveform(
    waveform: String,
    data: String,
    snr: f64,
    sample_rate: u32,
    symbol_rate: u32,
    list: bool,
) -> Result<()> {
    use r4w_core::waveform::{am, ask, fm, fsk, ook, psk, qam};

    // List available waveforms
    if list || waveform.is_empty() {
        println!("=== Available Waveforms ===");
        println!();
        for name in WaveformFactory::list() {
            if let Some(wf) = WaveformFactory::create(name, 10000.0) {
                let info = wf.info();
                println!("{:8} - {} ({})",
                    info.name,
                    info.full_name,
                    if info.carries_data {
                        format!("{} bits/symbol", info.bits_per_symbol)
                    } else {
                        "no data".to_string()
                    }
                );
            }
        }
        println!();
        println!("Also available: 4-AM, 4-FM");
        if waveform.is_empty() && !list {
            println!();
            println!("Use -w <WAVEFORM> to run a simulation, e.g.:");
            println!("  r4w waveform -w BPSK -d \"Hello\"");
        }
        return Ok(());
    }

    let sample_rate_f = sample_rate as f64;
    let symbol_rate_f = symbol_rate as f64;
    let carrier_freq = symbol_rate_f; // Use symbol rate as carrier for baseband

    let common = CommonParams {
        sample_rate: sample_rate_f,
        carrier_freq: 0.0,
        amplitude: 1.0,
    };

    // Create waveform based on name
    let waveform_upper = waveform.to_uppercase();
    let wf: Box<dyn r4w_core::waveform::Waveform> = match waveform_upper.as_str() {
        // Digital amplitude shift keying
        "ASK" => Box::new(ask::ASK::new_binary(common.clone(), symbol_rate_f, carrier_freq)),
        "4-ASK" | "4ASK" | "PAM4" => Box::new(ask::ASK::new_4ask(common.clone(), symbol_rate_f, carrier_freq)),
        // Analog AM/FM (audio modulation)
        "AM" | "AM-BROADCAST" => Box::new(am::AM::broadcast(sample_rate as f64, carrier_freq)),
        "FM" | "FM-BROADCAST" | "WBFM" => Box::new(fm::FM::broadcast(sample_rate as f64, carrier_freq)),
        "NBFM" => Box::new(fm::FM::narrowband(sample_rate as f64, carrier_freq)),
        "OOK" => Box::new(ook::OOK::new(common.clone(), symbol_rate_f)),
        // deviation = sample_rate/20 keeps frequencies well within Nyquist
        "BFSK" | "FSK" => Box::new(fsk::FSK::new_bfsk(common.clone(), symbol_rate_f, sample_rate as f64 / 20.0)),
        "BPSK" => Box::new(psk::PSK::new_bpsk(common.clone(), symbol_rate_f)),
        "QPSK" => Box::new(psk::PSK::new_qpsk(common.clone(), symbol_rate_f)),
        "16QAM" | "QAM16" | "QAM" => Box::new(qam::QAM::new_16qam(common.clone(), symbol_rate_f)),
        _ => {
            anyhow::bail!("Unknown waveform: {}. Use --list to see available waveforms.", waveform);
        }
    };

    let info = wf.info();

    println!("=== Waveform Simulation ===");
    println!("Waveform: {} ({})", info.name, info.full_name);
    println!("Sample rate: {} Hz, Symbol rate: {} symbols/s", sample_rate, symbol_rate);
    println!("Samples per symbol: {}", wf.samples_per_symbol());
    println!("SNR: {:.1} dB", snr);
    println!();

    if !info.carries_data {
        println!("Note: {} does not carry data. Generating demo signal.", info.name);
        let demo = wf.generate_demo(100.0); // 100ms demo
        println!("Generated {} samples ({:.1} ms)", demo.len(), demo.len() as f64 / sample_rate_f * 1000.0);
        return Ok(());
    }

    // Convert input data to bits
    let data_bytes = data.as_bytes();
    let bits_per_symbol = info.bits_per_symbol as usize;

    // Convert bytes to bits
    let mut tx_bits: Vec<u8> = Vec::new();
    for byte in data_bytes {
        for i in (0..8).rev() {
            tx_bits.push((byte >> i) & 1);
        }
    }

    // Ensure we have complete symbols
    while tx_bits.len() % bits_per_symbol != 0 {
        tx_bits.push(0);
    }

    let num_symbols = tx_bits.len() / bits_per_symbol;
    println!("Data: '{}' ({} bytes -> {} bits -> {} symbols)",
        data, data_bytes.len(), tx_bits.len(), num_symbols);
    println!();

    // Modulate
    let tx_samples = wf.modulate(&tx_bits);
    println!("TX: {} samples generated", tx_samples.len());

    // Apply AWGN channel
    let channel_config = ChannelConfig {
        model: ChannelModel::Awgn,
        snr_db: snr,
        cfo_hz: 0.0,
        ..Default::default()
    };
    let mut channel = Channel::new(channel_config);
    let rx_samples = channel.apply(&tx_samples);
    println!("Channel: Applied AWGN model");

    // Demodulate
    let result = wf.demodulate(&rx_samples);
    let rx_bits = &result.bits;

    // Calculate BER
    let bit_errors: usize = tx_bits.iter()
        .zip(rx_bits.iter())
        .filter(|(a, b)| a != b)
        .count();
    let ber = bit_errors as f64 / tx_bits.len().max(1) as f64;

    // Calculate SER (symbol errors)
    let symbol_errors: usize = (0..num_symbols)
        .filter(|&i| {
            let start = i * bits_per_symbol;
            let end = (start + bits_per_symbol).min(tx_bits.len()).min(rx_bits.len());
            if end <= start { return true; }
            tx_bits[start..end] != rx_bits[start..(end.min(rx_bits.len()))]
        })
        .count();
    let ser = symbol_errors as f64 / num_symbols.max(1) as f64;

    println!("RX: {} bits demodulated", rx_bits.len());
    println!();
    println!("BER: {:.2}% ({}/{} bits incorrect)", ber * 100.0, bit_errors, tx_bits.len());
    println!("SER: {:.2}% ({}/{} symbols incorrect)", ser * 100.0, symbol_errors, num_symbols);

    // Try to decode as text
    if bit_errors == 0 {
        // Convert bits back to bytes
        let mut decoded_bytes = Vec::new();
        for chunk in rx_bits.chunks(8) {
            if chunk.len() == 8 {
                let byte = chunk.iter().enumerate().fold(0u8, |acc, (i, &b)| acc | (b << (7 - i)));
                decoded_bytes.push(byte);
            }
        }
        // Trim padding zeros
        while decoded_bytes.last() == Some(&0) && decoded_bytes.len() > data_bytes.len() {
            decoded_bytes.pop();
        }

        if let Ok(text) = String::from_utf8(decoded_bytes.clone()) {
            println!();
            println!("TX: '{}'", data);
            println!("RX: '{}'", text);
            if text == data {
                println!("Result: SUCCESS");
            } else {
                println!("Result: MISMATCH");
            }
        }
    } else if bit_errors < tx_bits.len() / 2 {
        println!();
        println!("Result: {} bit errors detected", bit_errors);
    } else {
        println!();
        println!("Result: High error rate - signal likely corrupted");
    }

    Ok(())
}

fn cmd_benchmark(
    port: u16,
    format: String,
    waveform: String,
    sample_rate: f64,
    _batch_size: usize,
    duration: u64,
    output: String,
    output_file: Option<PathBuf>,
    stats_interval: u64,
    list: bool,
) -> Result<()> {
    // List available waveforms
    if list || waveform.is_empty() {
        println!("=== Available Waveforms for Benchmarking ===\n");
        for name in WaveformRunner::available_waveforms() {
            println!("  {}", name);
        }
        println!("\nUsage: r4w benchmark -w BPSK -p 5000");
        return Ok(());
    }

    // Parse sample format
    let sample_format = SampleFormat::from_str(&format)
        .ok_or_else(|| anyhow::anyhow!("Invalid format: {}. Use f32 or i16", format))?;

    // Create waveform runner
    let runner = WaveformRunner::new(&waveform, sample_rate)
        .map_err(|e| anyhow::anyhow!("Failed to create waveform: {}", e))?;

    // Create UDP receiver
    let mut receiver = BenchmarkReceiver::bind(port, sample_format)
        .context("Failed to bind UDP socket")?;
    receiver.set_timeout(Some(Duration::from_millis(100)))?;

    // Setup Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    }).context("Failed to set Ctrl+C handler")?;

    let mut metrics = BenchmarkMetrics::new();
    metrics.start();

    let start_time = Instant::now();
    let run_duration = if duration > 0 {
        Some(Duration::from_secs(duration))
    } else {
        None
    };

    let mut last_stats_time = Instant::now();
    let stats_duration = Duration::from_secs(stats_interval);

    println!("SDR Waveform Benchmark");
    println!("======================");
    println!("Waveform:     {}", waveform);
    println!("Sample Rate:  {} Hz", sample_rate);
    println!("UDP Port:     {} ({})", port, sample_format);
    println!();
    println!("Waiting for UDP data... (Press Ctrl+C to stop)\n");

    // Main benchmark loop
    while running.load(Ordering::SeqCst) {
        // Check duration limit
        if let Some(max_dur) = run_duration {
            if start_time.elapsed() >= max_dur {
                break;
            }
        }

        // Receive samples
        match receiver.recv_batch(Duration::from_millis(100)) {
            Ok(samples) if !samples.is_empty() => {
                let bytes = samples.len() * sample_format.bytes_per_sample();
                metrics.record_receive(samples.len(), bytes);

                // Process through waveform
                let result = runner.process(&samples);
                metrics.update(&result);
            }
            Ok(_) => {
                // No data received (timeout)
            }
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut
                || e.kind() == std::io::ErrorKind::WouldBlock => {
                // Timeout, continue
            }
            Err(_e) => {
                metrics.record_receive_error();
            }
        }

        // Print live stats
        if stats_interval > 0 && last_stats_time.elapsed() >= stats_duration {
            last_stats_time = Instant::now();
            let summary = metrics.summary();

            // Clear line and print stats
            print!("\r\x1B[K");
            print!(
                "Throughput: {:.0} Sps | Latency: {:.1} us | Symbols: {} | Duration: {}",
                summary.throughput_samples_per_sec,
                summary.avg_latency_us,
                summary.symbols_detected,
                summary.elapsed_formatted()
            );
            std::io::stdout().flush().ok();
        }
    }

    println!("\n");

    // Generate report
    let summary = metrics.summary();
    let report = BenchmarkReport::new(
        &waveform,
        sample_rate,
        1024, // batch_size not used currently
        port,
        &format,
        &summary,
    );

    // Output report
    let output_text = match output.to_lowercase().as_str() {
        "json" => report.to_json(),
        "csv" => {
            format!("{}\n{}", BenchmarkReport::csv_header(), report.to_csv_row())
        }
        _ => report.to_text(),
    };

    if let Some(path) = output_file {
        std::fs::write(&path, &output_text)
            .context("Failed to write output file")?;
        println!("Report written to {:?}", path);
    } else {
        println!("{}", output_text);
    }

    Ok(())
}

fn cmd_udp_send(
    target: String,
    waveform: String,
    sample_rate: f64,
    format: String,
    message: String,
    pps: u32,
    samples_per_packet: usize,
    duration: u64,
    snr: f64,
    list: bool,
    repeat: bool,
) -> Result<()> {
    use r4w_core::benchmark::BenchmarkSender;

    // List available waveforms
    if list || waveform.is_empty() {
        println!("=== Available Waveforms for UDP Send ===\n");
        for name in WaveformRunner::available_waveforms() {
            println!("  {}", name);
        }
        println!("\nUsage: r4w udp-send -w BPSK -t 127.0.0.1:5000 -m \"Hello\"");
        return Ok(());
    }

    // Parse sample format
    let sample_format = SampleFormat::from_str(&format)
        .ok_or_else(|| anyhow::anyhow!("Invalid format: {}. Use f32 or i16", format))?;

    // Create waveform and generate samples
    let wf = WaveformFactory::create(&waveform, sample_rate)
        .ok_or_else(|| anyhow::anyhow!("Unknown waveform: {}", waveform))?;

    let info = wf.info();

    println!("SDR UDP Sample Sender");
    println!("=====================");
    println!("Waveform:     {} ({})", info.name, info.full_name);
    println!("Sample Rate:  {} Hz", sample_rate);
    println!("Target:       {}", target);
    println!("Format:       {}", sample_format);
    println!("Message:      '{}'", message);
    println!();

    // Convert message to bits
    let data_bytes = message.as_bytes();
    let mut tx_bits: Vec<u8> = Vec::new();
    for byte in data_bytes {
        for i in (0..8).rev() {
            tx_bits.push((byte >> i) & 1);
        }
    }

    // Pad to symbol boundary
    let bits_per_symbol = info.bits_per_symbol.max(1) as usize;
    while tx_bits.len() % bits_per_symbol != 0 {
        tx_bits.push(0);
    }

    // Modulate
    let mut samples = wf.modulate(&tx_bits);
    println!("Generated {} samples", samples.len());

    // Add noise if requested
    if snr >= 0.0 {
        let channel_config = ChannelConfig {
            model: ChannelModel::Awgn,
            snr_db: snr,
            ..Default::default()
        };
        let mut channel = Channel::new(channel_config);
        samples = channel.apply(&samples);
        println!("Added AWGN noise (SNR: {} dB)", snr);
    }

    // Create UDP sender
    let sender = BenchmarkSender::new(&target, sample_format)
        .context("Failed to create UDP sender")?;

    // Setup Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    }).context("Failed to set Ctrl+C handler")?;

    let start_time = Instant::now();
    let run_duration = if duration > 0 {
        Some(Duration::from_secs(duration))
    } else {
        None
    };

    let packet_delay = if pps > 0 {
        Duration::from_micros(1_000_000 / pps as u64)
    } else {
        Duration::ZERO
    };

    let mut packets_sent = 0u64;
    let mut sample_idx = 0usize;

    println!();
    println!("Sending... (Press Ctrl+C to stop)");

    while running.load(Ordering::SeqCst) {
        // Check duration limit
        if let Some(max_dur) = run_duration {
            if start_time.elapsed() >= max_dur {
                break;
            }
        }

        // Get next batch of samples
        let end_idx = (sample_idx + samples_per_packet).min(samples.len());
        let batch = &samples[sample_idx..end_idx];

        if batch.is_empty() {
            if repeat {
                sample_idx = 0;
                continue;
            } else {
                break;
            }
        }

        // Send packet
        match sender.send(batch) {
            Ok(_) => {
                packets_sent += 1;
                sample_idx = end_idx;

                if sample_idx >= samples.len() {
                    if repeat {
                        sample_idx = 0;
                    }
                }
            }
            Err(e) => {
                warn!("Send error: {}", e);
            }
        }

        // Rate limiting
        if !packet_delay.is_zero() {
            std::thread::sleep(packet_delay);
        }
    }

    let elapsed = start_time.elapsed();
    println!();
    println!("Sent {} packets in {:.2}s", packets_sent, elapsed.as_secs_f64());
    println!(
        "Rate: {:.1} packets/sec, {:.1} samples/sec",
        packets_sent as f64 / elapsed.as_secs_f64(),
        (packets_sent * samples_per_packet as u64) as f64 / elapsed.as_secs_f64()
    );

    Ok(())
}

fn cmd_agent(port: u16, _foreground: bool) -> Result<()> {
    println!("SDR Agent Daemon");
    println!("================");
    println!("Port: {}", port);
    println!();

    // Setup Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        eprintln!("\nShutting down agent...");
        r.store(false, Ordering::SeqCst);
    }).context("Failed to set Ctrl+C handler")?;

    let mut server = AgentServer::new(port);

    // Run server (blocking)
    server.run().context("Agent server failed")?;

    Ok(())
}

fn cmd_mesh_status(node_id: Option<String>, preset: String, region: String) -> Result<()> {
    let config = create_mesh_config(node_id, preset.clone(), region.clone())?;
    let mesh = LoRaMesh::new(config);

    let preset_parsed = parse_preset(&preset)?;
    let region_parsed = parse_region(&region)?;

    println!("=== LoRa Mesh Node Status ===");
    println!();
    println!("Node ID:        {:08x}", mesh.node_id().to_u32());
    println!("Preset:         {:?}", preset_parsed);
    println!("Region:         {:?}", region_parsed);
    println!();

    // Get PHY info
    let phy = mesh.phy();
    println!("PHY Configuration:");
    println!("  Frequency:    {} MHz", phy.frequency() as f64 / 1_000_000.0);
    println!("  TX Power:     {} dBm", phy.tx_power());
    println!("  RSSI:         {:.1} dBm", phy.rssi());
    println!("  SNR:          {:.1} dB", phy.snr());
    println!();

    // Get stats
    let stats = mesh.stats();
    println!("Statistics:");
    println!("  TX Packets:   {}", stats.packets_tx);
    println!("  RX Packets:   {}", stats.packets_rx);
    println!("  TX Bytes:     {}", stats.bytes_tx);
    println!("  RX Bytes:     {}", stats.bytes_rx);
    println!("  Forwarded:    {}", stats.packets_forwarded);
    println!("  Dropped:      {}", stats.duplicates_dropped + stats.queue_drops);
    println!("  Neighbors:    {}", stats.neighbor_count);

    Ok(())
}

fn cmd_mesh_send(
    message: String,
    dest: String,
    hop_limit: u8,
    node_id: Option<String>,
    preset: String,
    region: String,
) -> Result<()> {
    let config = create_mesh_config(node_id, preset, region)?;
    let mut mesh = LoRaMesh::new(config);

    println!("=== LoRa Mesh Send ===");
    println!();
    println!("From:     {:08x}", mesh.node_id().to_u32());

    if dest.to_lowercase() == "broadcast" {
        println!("To:       BROADCAST");
        println!("Hop Limit: {}", hop_limit);
        println!("Message:  '{}'", message);
        println!();

        mesh.broadcast(message.as_bytes(), hop_limit)
            .map_err(|e| anyhow::anyhow!("Broadcast failed: {:?}", e))?;

        println!("Broadcast message queued for transmission");
    } else {
        let dest_id = u32::from_str_radix(dest.trim_start_matches("0x"), 16)
            .with_context(|| format!("Invalid destination node ID: {}", dest))?;
        let dest_node = NodeId::from_u32(dest_id);

        println!("To:       {:08x}", dest_id);
        println!("Message:  '{}'", message);
        println!();

        mesh.send_direct(dest_node, message.as_bytes())
            .map_err(|e| anyhow::anyhow!("Send failed: {:?}", e))?;

        println!("Direct message queued for transmission");
    }

    // Show TX samples info
    if let Some(samples) = mesh.get_tx_samples() {
        println!("Generated {} I/Q samples for transmission", samples.len());
    }

    Ok(())
}

fn cmd_mesh_neighbors(node_id: Option<String>, preset: String, region: String) -> Result<()> {
    let config = create_mesh_config(node_id, preset, region)?;
    let mesh = LoRaMesh::new(config);

    println!("=== LoRa Mesh Neighbors ===");
    println!();
    println!("Node ID: {:08x}", mesh.node_id().to_u32());
    println!();

    let neighbors = mesh.neighbors();
    if neighbors.is_empty() {
        println!("No neighbors discovered yet.");
        println!();
        println!("Neighbors are discovered when packets are received from other nodes.");
        println!("In a real deployment, run this command after the node has been");
        println!("active on the mesh for some time.");
    } else {
        println!("{:<12} {:<10} {:<10} {:<20}", "Node ID", "RSSI", "SNR", "Last Seen");
        println!("{}", "-".repeat(52));
        for neighbor in neighbors {
            println!(
                "{:08x}     {:.1} dBm   {:.1} dB    {:?}",
                neighbor.node_id().to_u32(),
                neighbor.link_quality.rssi,
                neighbor.link_quality.snr,
                neighbor.time_since_seen()
            );
        }
    }

    Ok(())
}

fn cmd_mesh_simulate(
    num_nodes: usize,
    num_messages: usize,
    snr: f64,
    preset: String,
    region: String,
    verbose: bool,
) -> Result<()> {
    use rand::Rng;

    let preset_parsed = parse_preset(&preset)?;
    let region_parsed = parse_region(&region)?;

    println!("=== LoRa Mesh Network Simulation ===");
    println!();
    println!("Nodes:    {}", num_nodes);
    println!("Messages: {}", num_messages);
    println!("SNR:      {:.1} dB", snr);
    println!("Preset:   {:?}", preset_parsed);
    println!("Region:   {:?}", region_parsed);
    println!();

    // Create nodes
    let mut nodes: Vec<LoRaMesh> = Vec::new();
    for i in 0..num_nodes {
        let config = LoRaMeshConfig {
            node_id: Some(NodeId::from_u32(0x1000 + i as u32)),
            preset: preset_parsed,
            region: region_parsed,
            ..Default::default()
        };
        nodes.push(LoRaMesh::new(config));
    }

    println!("Created {} nodes:", num_nodes);
    for node in &nodes {
        println!("  - {:08x}", node.node_id().to_u32());
    }
    println!();

    // Channel for simulation
    let channel_config = ChannelConfig {
        model: ChannelModel::Awgn,
        snr_db: snr,
        ..Default::default()
    };
    let mut channel = Channel::new(channel_config);

    let mut rng = rand::thread_rng();
    let mut total_sent = 0;
    let mut total_received = 0;
    let mut total_forwarded = 0;

    println!("Simulating message exchange...");
    if verbose {
        println!();
    }

    for msg_idx in 0..num_messages {
        // Pick random source node
        let src_idx = rng.gen_range(0..num_nodes);
        let message = format!("Message {}", msg_idx);

        if verbose {
            println!(
                "[{}] Node {:08x} broadcasts: '{}'",
                msg_idx,
                nodes[src_idx].node_id().to_u32(),
                message
            );
        }

        // Source node broadcasts
        nodes[src_idx]
            .broadcast(message.as_bytes(), 3)
            .map_err(|e| anyhow::anyhow!("Broadcast failed: {:?}", e))?;
        total_sent += 1;

        // Get TX samples from source
        if let Some(tx_samples) = nodes[src_idx].get_tx_samples() {
            // Apply channel to simulate propagation
            let rx_samples = channel.apply(&tx_samples);

            // All other nodes receive the samples
            for (dest_idx, dest_node) in nodes.iter_mut().enumerate() {
                if dest_idx != src_idx {
                    dest_node.process_samples(&rx_samples);

                    // Check for received packets
                    let received: Vec<_> = dest_node.receive_packets().collect();
                    for packet in &received {
                        total_received += 1;
                        if verbose {
                            println!(
                                "  -> Node {:08x} received: '{}'",
                                dest_node.node_id().to_u32(),
                                String::from_utf8_lossy(&packet.payload)
                            );
                        }
                    }

                    // Check if node forwards
                    if let Some(fwd_samples) = dest_node.get_tx_samples() {
                        total_forwarded += 1;
                        if verbose {
                            println!(
                                "  -> Node {:08x} forwarding ({} samples)",
                                dest_node.node_id().to_u32(),
                                fwd_samples.len()
                            );
                        }
                    }
                }
            }
        }
    }

    println!();
    println!("=== Simulation Results ===");
    println!();
    println!("Messages sent:     {}", total_sent);
    println!("Messages received: {}", total_received);
    println!("Messages forwarded: {}", total_forwarded);
    println!(
        "Delivery rate:     {:.1}%",
        if total_sent > 0 {
            total_received as f64 / (total_sent * (num_nodes - 1)) as f64 * 100.0
        } else {
            0.0
        }
    );
    println!();

    // Show per-node stats
    println!("Per-Node Statistics:");
    println!("{:<12} {:<10} {:<10} {:<10}", "Node ID", "TX", "RX", "Fwd");
    println!("{}", "-".repeat(42));
    for node in &nodes {
        let stats = node.stats();
        println!(
            "{:08x}     {:<10} {:<10} {:<10}",
            node.node_id().to_u32(),
            stats.packets_tx,
            stats.packets_rx,
            stats.packets_forwarded
        );
    }

    Ok(())
}

fn cmd_mesh_info() -> Result<()> {
    println!("=== LoRa Mesh Configuration Options ===");
    println!();
    println!("Modem Presets:");
    println!("  LongFast     - SF11, BW 250kHz - Long range, faster data rate");
    println!("  LongSlow     - SF12, BW 125kHz - Maximum range, slowest data rate");
    println!("  LongModerate - SF11, BW 125kHz - Long range, moderate data rate");
    println!("  MediumFast   - SF9,  BW 250kHz - Medium range, good data rate");
    println!("  MediumSlow   - SF10, BW 250kHz - Medium range, slower data rate");
    println!("  ShortFast    - SF7,  BW 250kHz - Short range, fastest data rate");
    println!("  ShortSlow    - SF8,  BW 250kHz - Short range, moderate data rate");
    println!();
    println!("Regions:");
    println!("  US   - 902-928 MHz (Americas)");
    println!("  EU   - 863-870 MHz (Europe)");
    println!("  CN   - 470-510 MHz (China)");
    println!("  JP   - 920-923 MHz (Japan)");
    println!("  ANZ  - 915-928 MHz (Australia/New Zealand)");
    println!("  KR   - 920-923 MHz (Korea)");
    println!("  TW   - 920-925 MHz (Taiwan)");
    println!("  IN   - 865-867 MHz (India)");
    println!();
    println!("Examples:");
    println!("  r4w mesh status --preset LongFast --region US");
    println!("  r4w mesh send -m \"Hello mesh!\" --dest broadcast");
    println!("  r4w mesh send -m \"Private\" --dest a1b2c3d4");
    println!("  r4w mesh simulate --nodes 8 --messages 20 --snr 15");
    println!("  r4w mesh neighbors");
    println!("  r4w mesh analyze --hex 'ff ff ff ff ...'");

    Ok(())
}

/// Analyze a Meshtastic packet with annotated hex dump
fn cmd_mesh_analyze(hex: &str, summary_only: bool) -> Result<()> {
    use r4w_core::mesh::analyze_hex_string;

    match analyze_hex_string(hex) {
        Ok(analysis) => {
            if summary_only {
                println!("{}", analysis.summary());
            } else {
                println!("{}", analysis.format_hex_dump());
            }
            Ok(())
        }
        Err(e) => {
            anyhow::bail!("Failed to parse hex: {}", e);
        }
    }
}

/// Interactive mesh REPL for testing and debugging
fn cmd_mesh_repl(
    node_id: Option<String>,
    preset: String,
    region: String,
    sim_nodes: usize,
) -> Result<()> {
    use r4w_core::mesh::{
        MeshSimulator, MeshNetwork, MeshtasticConfig, MeshtasticNode, NodeId,
        SimConfig, ModemPreset, Region, Traceroute,
    };
    use std::io::{self, BufRead, Write};

    // Parse configuration
    let modem_preset = match preset.to_lowercase().as_str() {
        "longfast" => ModemPreset::LongFast,
        "longslow" => ModemPreset::LongSlow,
        "longmoderate" => ModemPreset::LongModerate,
        "mediumfast" | "medfast" => ModemPreset::MediumFast,
        "mediumslow" | "medslow" => ModemPreset::MediumSlow,
        "shortfast" => ModemPreset::ShortFast,
        "shortslow" => ModemPreset::ShortSlow,
        _ => {
            warn!("Unknown preset '{}', using LongFast", preset);
            ModemPreset::LongFast
        }
    };

    let region_enum = match region.to_uppercase().as_str() {
        "US" => Region::US,
        "EU" | "EU868" => Region::EU,
        "CN" => Region::CN,
        "JP" => Region::JP,
        "ANZ" => Region::ANZ,
        "KR" => Region::KR,
        "TW" => Region::TW,
        "IN" => Region::IN,
        _ => {
            warn!("Unknown region '{}', using US", region);
            Region::US
        }
    };

    // Create node(s)
    let our_node_id = if let Some(ref id_str) = node_id {
        let id = u32::from_str_radix(id_str.trim_start_matches("0x"), 16)
            .context("Invalid node ID (use hex format like 'a1b2c3d4')")?;
        NodeId::from_u32(id)
    } else {
        NodeId::random()
    };

    println!("=== Mesh Network REPL ===");
    println!("Node ID: {:08x}", our_node_id.to_u32());
    println!("Preset: {:?}, Region: {:?}", modem_preset, region_enum);

    if sim_nodes > 1 {
        // Multi-node simulation mode
        println!("Simulation mode with {} nodes", sim_nodes);

        let config = SimConfig::default()
            .with_node_count(sim_nodes)
            .with_area(5000.0, 5000.0)
            .with_verbose(true);

        let mut sim = MeshSimulator::new(config);

        println!("\nSimulated nodes:");
        for i in 0..sim_nodes {
            if let Some(id) = sim.node_id(i) {
                if let Some(pos) = sim.node_position(i) {
                    println!("  Node {}: {:08x} at ({:.0}, {:.0})", i, id.to_u32(), pos.x, pos.y);
                }
            }
        }

        println!("\nCommands:");
        println!("  send <from_idx> <message>  - Send broadcast from node");
        println!("  step [n]                   - Run n simulation steps (default: 1)");
        println!("  run <steps>                - Run many steps");
        println!("  stats                      - Show statistics");
        println!("  topology                   - Show network topology");
        println!("  help                       - Show this help");
        println!("  quit                       - Exit REPL");
        println!();

        let stdin = io::stdin();
        let mut stdout = io::stdout();

        loop {
            print!("mesh-sim> ");
            stdout.flush()?;

            let mut line = String::new();
            if stdin.lock().read_line(&mut line)? == 0 {
                break; // EOF
            }

            let parts: Vec<&str> = line.trim().split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            match parts[0] {
                "send" | "s" => {
                    if parts.len() < 3 {
                        println!("Usage: send <from_idx> <message>");
                        continue;
                    }
                    let from_idx: usize = parts[1].parse().unwrap_or(0);
                    let message = parts[2..].join(" ");
                    if sim.send_message(from_idx, &message, None) {
                        println!("Message queued from node {}", from_idx);
                    } else {
                        println!("Failed to queue message");
                    }
                }

                "step" => {
                    let n: usize = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
                    for _ in 0..n {
                        sim.step();
                    }
                    println!("Ran {} step(s), total: {}", n, sim.step_count());
                }

                "run" | "r" => {
                    let n: u64 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
                    sim.run(n);
                    println!("Ran {} steps, total: {}", n, sim.step_count());
                }

                "stats" => {
                    let stats = sim.stats();
                    println!("=== Simulation Statistics ===");
                    println!("  Messages sent: {}", stats.messages_sent);
                    println!("  Packets transmitted: {}", stats.packets_transmitted);
                    println!("  Packets received: {}", stats.packets_received);
                    println!("  Packets lost: {}", stats.packets_lost);
                    println!("  Collisions: {}", stats.collisions);
                    if stats.packets_transmitted > 0 {
                        let pdr = stats.packets_received as f64 / stats.packets_transmitted as f64 * 100.0;
                        println!("  Packet delivery rate: {:.1}%", pdr);
                    }
                }

                "topology" | "topo" => {
                    let adj = sim.topology();
                    println!("=== Network Topology ===");
                    for (i, neighbors) in adj.iter().enumerate() {
                        if let Some(id) = sim.node_id(i) {
                            let neighbor_str: Vec<String> = neighbors.iter()
                                .map(|&j| format!("{}", j))
                                .collect();
                            println!("  Node {} ({:08x}): [{}]",
                                i, id.to_u32(), neighbor_str.join(", "));
                        }
                    }
                }

                "help" | "?" => {
                    println!("Commands:");
                    println!("  send <from_idx> <message>  - Send broadcast from node");
                    println!("  step [n]                   - Run n simulation steps");
                    println!("  run <steps>                - Run many steps");
                    println!("  stats                      - Show statistics");
                    println!("  topology                   - Show network topology");
                    println!("  quit                       - Exit REPL");
                }

                "quit" | "exit" | "q" => {
                    println!("Goodbye!");
                    break;
                }

                _ => {
                    println!("Unknown command: {}. Type 'help' for available commands.", parts[0]);
                }
            }
        }
    } else {
        // Single node mode
        let mut config = MeshtasticConfig::default();
        config.node_id = Some(our_node_id);
        config.primary_channel.preset = modem_preset;
        config.region = region_enum;
        config.short_name = "CLI".to_string();
        config.long_name = "CLI REPL Node".to_string();

        let mut node = MeshtasticNode::new(config);
        let mut traceroute = Traceroute::with_defaults(our_node_id);

        println!("\nCommands:");
        println!("  send <message>             - Broadcast a message");
        println!("  sendto <node_id> <message> - Send direct message");
        println!("  status                     - Show node status");
        println!("  neighbors                  - Show neighbor table");
        println!("  trace <node_id>            - Start traceroute");
        println!("  help                       - Show this help");
        println!("  quit                       - Exit REPL");
        println!();

        let stdin = io::stdin();
        let mut stdout = io::stdout();

        loop {
            print!("mesh[{:08x}]> ", our_node_id.to_u32());
            stdout.flush()?;

            let mut line = String::new();
            if stdin.lock().read_line(&mut line)? == 0 {
                break; // EOF
            }

            let parts: Vec<&str> = line.trim().split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            match parts[0] {
                "send" | "s" => {
                    if parts.len() < 2 {
                        println!("Usage: send <message>");
                        continue;
                    }
                    let message = parts[1..].join(" ");
                    match node.broadcast(message.as_bytes(), 3) {
                        Ok(_) => println!("Message queued for broadcast"),
                        Err(e) => println!("Error: {:?}", e),
                    }
                }

                "sendto" => {
                    if parts.len() < 3 {
                        println!("Usage: sendto <node_id> <message>");
                        continue;
                    }
                    let dest_id = match u32::from_str_radix(parts[1].trim_start_matches("0x"), 16) {
                        Ok(id) => NodeId::from_u32(id),
                        Err(_) => {
                            println!("Invalid node ID (use hex format)");
                            continue;
                        }
                    };
                    let message = parts[2..].join(" ");
                    match node.send_direct(dest_id, message.as_bytes()) {
                        Ok(_) => println!("Message queued for {:08x}", dest_id.to_u32()),
                        Err(e) => println!("Error: {:?}", e),
                    }
                }

                "status" | "stat" => {
                    let stats = node.stats();
                    println!("=== Node Status ===");
                    println!("  Node ID: {:08x}", our_node_id.to_u32());
                    println!("  Packets TX: {}", stats.packets_tx);
                    println!("  Packets RX: {}", stats.packets_rx);
                    println!("  Packets forwarded: {}", stats.packets_forwarded);
                    println!("  Neighbors: {}", node.neighbors().len());
                }

                "neighbors" | "neigh" | "n" => {
                    let neighbors = node.neighbors();
                    println!("=== Neighbors ({}) ===", neighbors.len());
                    for n in neighbors {
                        println!("  {:08x} - RSSI: {:.0}dBm, SNR: {:.1}dB, hops: {}",
                            n.info.node_id.to_u32(),
                            n.link_quality.rssi,
                            n.link_quality.snr,
                            n.hop_count);
                    }
                }

                "trace" => {
                    if parts.len() < 2 {
                        println!("Usage: trace <node_id>");
                        continue;
                    }
                    let dest_id = match u32::from_str_radix(parts[1].trim_start_matches("0x"), 16) {
                        Ok(id) => NodeId::from_u32(id),
                        Err(_) => {
                            println!("Invalid node ID (use hex format)");
                            continue;
                        }
                    };
                    let request = traceroute.start_trace(dest_id);
                    println!("Traceroute started to {:08x} (request {})",
                        dest_id.to_u32(), request.request_id);
                    // In a real implementation, this would be sent over the mesh
                    println!("(Note: In simulation mode, traceroute requires multi-node setup)");
                }

                "help" | "?" => {
                    println!("Commands:");
                    println!("  send <message>             - Broadcast a message");
                    println!("  sendto <node_id> <message> - Send direct message");
                    println!("  status                     - Show node status");
                    println!("  neighbors                  - Show neighbor table");
                    println!("  trace <node_id>            - Start traceroute");
                    println!("  quit                       - Exit REPL");
                }

                "quit" | "exit" | "q" => {
                    println!("Goodbye!");
                    break;
                }

                _ => {
                    println!("Unknown command: {}. Type 'help' for available commands.", parts[0]);
                }
            }
        }
    }

    Ok(())
}

fn cmd_adsb_decode(message: String, verbose: bool) -> Result<()> {
    // Parse hex string to bytes
    let hex = message.trim().replace(" ", "").replace("0x", "");
    if hex.len() != 28 {
        anyhow::bail!(
            "Invalid message length: {} chars (expected 28 hex chars = 14 bytes = 112 bits)",
            hex.len()
        );
    }

    let mut bytes = [0u8; 14];
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        let s = std::str::from_utf8(chunk).context("Invalid hex character")?;
        bytes[i] = u8::from_str_radix(s, 16).context("Invalid hex value")?;
    }

    let msg = AdsbMessage::decode(&bytes);

    println!("=== ADS-B Message Decode ===");
    println!();
    println!("Raw:      {}", hex.to_uppercase());
    println!("CRC:      {}", if msg.crc_valid { "VALID" } else { "INVALID" });
    println!();
    println!("Downlink Format: {:?} (DF{})", msg.downlink_format, bytes[0] >> 3);
    println!("Capability:      {}", msg.capability);
    println!("ICAO Address:    {} ({})", msg.icao_hex(), msg.icao_address);
    println!("Type Code:       {:?}", msg.type_code);
    println!();

    match &msg.content {
        r4w_core::waveform::adsb::MessageContent::Identification { callsign, category } => {
            println!("Message Type: Aircraft Identification");
            println!("  Callsign:  {}", callsign);
            println!("  Category:  {:?}", category);
        }
        r4w_core::waveform::adsb::MessageContent::AirbornePosition {
            altitude,
            cpr_lat,
            cpr_lon,
            cpr_odd,
            surveillance_status,
            ..
        } => {
            println!("Message Type: Airborne Position");
            if let Some(alt) = altitude {
                println!("  Altitude:  {} ft", alt);
            } else {
                println!("  Altitude:  Unknown");
            }
            println!("  CPR Frame: {}", if *cpr_odd { "Odd" } else { "Even" });
            println!("  CPR Lat:   {}", cpr_lat);
            println!("  CPR Lon:   {}", cpr_lon);
            println!("  Surv Stat: {}", surveillance_status);
            println!();
            println!("Note: Full position requires both even and odd messages.");
        }
        r4w_core::waveform::adsb::MessageContent::AirborneVelocity {
            subtype,
            heading,
            ground_speed,
            vertical_rate,
            vr_source,
        } => {
            println!("Message Type: Airborne Velocity (subtype {})", subtype);
            if let Some(gs) = ground_speed {
                println!("  Ground Speed: {:.1} kts", gs);
            }
            if let Some(hdg) = heading {
                println!("  Heading:      {:.1}°", hdg);
            }
            if let Some(vr) = vertical_rate {
                println!(
                    "  Vertical Rate: {} ft/min ({})",
                    vr,
                    if *vr_source == 0 { "GNSS" } else { "Baro" }
                );
            }
        }
        r4w_core::waveform::adsb::MessageContent::SurfacePosition {
            ground_speed,
            track,
            cpr_odd,
            ..
        } => {
            println!("Message Type: Surface Position");
            if let Some(gs) = ground_speed {
                println!("  Ground Speed: {:.1} kts", gs);
            }
            if let Some(trk) = track {
                println!("  Track:        {:.1}°", trk);
            }
            println!("  CPR Frame:    {}", if *cpr_odd { "Odd" } else { "Even" });
        }
        r4w_core::waveform::adsb::MessageContent::AircraftStatus { emergency, squawk } => {
            println!("Message Type: Aircraft Status");
            println!("  Squawk:    {:04}", squawk);
            println!("  Emergency: {}", emergency);
        }
        r4w_core::waveform::adsb::MessageContent::OperationalStatus {
            version,
            nic_supplement,
            nac_p,
            sil,
            ..
        } => {
            println!("Message Type: Operational Status");
            println!("  Version:      {}", version);
            println!("  NIC Supp:     {}", nic_supplement);
            println!("  NAC-P:        {}", nac_p);
            println!("  SIL:          {}", sil);
        }
        r4w_core::waveform::adsb::MessageContent::Unknown { me_data } => {
            println!("Message Type: Unknown/Reserved");
            println!("  ME Data:   {:02X?}", me_data);
        }
    }

    if verbose {
        println!();
        println!("Raw Bytes:");
        for (i, byte) in bytes.iter().enumerate() {
            println!("  Byte {:2}: {:02X} ({:08b})", i, byte, byte);
        }
    }

    Ok(())
}

fn cmd_adsb_file(input: PathBuf, sample_rate: f64, show_all: bool) -> Result<()> {
    let samples = read_samples_f32(&input)?;

    println!("=== ADS-B I/Q File Decoder ===");
    println!();
    println!("File:        {:?}", input);
    println!("Samples:     {}", samples.len());
    println!("Sample Rate: {} Hz", sample_rate);
    println!(
        "Duration:    {:.3} s",
        samples.len() as f64 / sample_rate
    );
    println!();

    let ppm = PPM::adsb(sample_rate);
    let messages = ppm.decode_stream(&samples);

    if messages.is_empty() {
        println!("No ADS-B messages found in file.");
        if !show_all {
            println!("Try --all to include messages with CRC errors.");
        }
        return Ok(());
    }

    println!("Found {} valid message(s):", messages.len());
    println!();
    println!(
        "{:<10} {:<12} {:<20} {:<30}",
        "ICAO", "Type", "Callsign/Alt", "Details"
    );
    println!("{}", "-".repeat(72));

    let mut cpr_decoder = CprDecoder::new();

    for msg in &messages {
        if !msg.crc_valid && !show_all {
            continue;
        }

        let type_str = match &msg.content {
            r4w_core::waveform::adsb::MessageContent::Identification { .. } => "ID",
            r4w_core::waveform::adsb::MessageContent::AirbornePosition { .. } => "Position",
            r4w_core::waveform::adsb::MessageContent::AirborneVelocity { .. } => "Velocity",
            r4w_core::waveform::adsb::MessageContent::SurfacePosition { .. } => "Surface",
            r4w_core::waveform::adsb::MessageContent::AircraftStatus { .. } => "Status",
            r4w_core::waveform::adsb::MessageContent::OperationalStatus { .. } => "OpStatus",
            r4w_core::waveform::adsb::MessageContent::Unknown { .. } => "Unknown",
        };

        let info = match &msg.content {
            r4w_core::waveform::adsb::MessageContent::Identification { callsign, .. } => {
                callsign.clone()
            }
            r4w_core::waveform::adsb::MessageContent::AirbornePosition { altitude, .. } => {
                altitude
                    .map(|a| format!("{} ft", a))
                    .unwrap_or_else(|| "N/A".to_string())
            }
            r4w_core::waveform::adsb::MessageContent::AirborneVelocity {
                ground_speed,
                heading,
                ..
            } => {
                let gs = ground_speed
                    .map(|g| format!("{:.0} kts", g))
                    .unwrap_or_default();
                let hdg = heading
                    .map(|h| format!("{:.0}°", h))
                    .unwrap_or_default();
                format!("{} {}", gs, hdg)
            }
            r4w_core::waveform::adsb::MessageContent::AircraftStatus { squawk, .. } => {
                format!("Squawk {:04}", squawk)
            }
            _ => String::new(),
        };

        // Try to decode position
        let pos_str = if let Some(cpr) = msg.cpr_position() {
            if let Some(pos) = cpr_decoder.decode(cpr) {
                format!("({:.4}, {:.4})", pos.latitude, pos.longitude)
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        let crc_marker = if msg.crc_valid { "" } else { " [CRC!]" };

        println!(
            "{:<10} {:<12} {:<20} {}{}",
            msg.icao_hex(),
            type_str,
            info,
            pos_str,
            crc_marker
        );
    }

    Ok(())
}

fn cmd_adsb_info() -> Result<()> {
    println!("=== ADS-B Protocol Information ===");
    println!();
    println!("ADS-B (Automatic Dependent Surveillance-Broadcast)");
    println!("---------------------------------------------------");
    println!();
    println!("Frequency:   1090 MHz (Mode S Extended Squitter)");
    println!("Data Rate:   1 Mbps");
    println!("Modulation:  PPM (Pulse Position Modulation)");
    println!("Message:     112 bits (56 bits short)");
    println!("CRC:         24-bit");
    println!();
    println!("Message Structure (112-bit Extended Squitter):");
    println!("  DF    (5 bits)  - Downlink Format (17 = ADS-B)");
    println!("  CA    (3 bits)  - Capability");
    println!("  ICAO  (24 bits) - Aircraft Address");
    println!("  ME    (56 bits) - Message (Type Code + Data)");
    println!("  PI    (24 bits) - Parity/CRC");
    println!();
    println!("Type Codes:");
    println!("  TC 1-4   - Aircraft Identification (Callsign)");
    println!("  TC 5-8   - Surface Position");
    println!("  TC 9-18  - Airborne Position (Baro Altitude)");
    println!("  TC 19    - Airborne Velocity");
    println!("  TC 20-22 - Airborne Position (GNSS Altitude)");
    println!("  TC 28    - Aircraft Status (Emergency/Squawk)");
    println!("  TC 29    - Target State and Status");
    println!("  TC 31    - Operational Status");
    println!();
    println!("Position Encoding (CPR - Compact Position Reporting):");
    println!("  - Uses alternating even/odd frames");
    println!("  - Global decode: requires both even and odd messages");
    println!("  - Local decode: single message + reference position");
    println!("  - Resolution: ~5 meters");
    println!();
    println!("Examples:");
    println!("  r4w adsb decode -m 8D4840D6202CC371C32CE0576098");
    println!("  r4w adsb file -i capture.iq -s 2000000");
    println!("  r4w adsb generate -o test.iq --callsign N12345");

    Ok(())
}

fn cmd_adsb_generate(
    output: PathBuf,
    icao: String,
    callsign: String,
    altitude: i32,
    sample_rate: f64,
) -> Result<()> {
    use r4w_core::waveform::adsb::crc24;
    use r4w_core::waveform::Waveform;

    // Parse ICAO address
    let icao_hex = icao.trim().replace("0x", "");
    let icao_addr = u32::from_str_radix(&icao_hex, 16)
        .with_context(|| format!("Invalid ICAO address: {}", icao))?;

    if icao_addr > 0xFFFFFF {
        anyhow::bail!("ICAO address must be 24 bits (6 hex digits)");
    }

    println!("=== ADS-B Test Signal Generator ===");
    println!();
    println!("ICAO:     {:06X}", icao_addr);
    println!("Callsign: {}", callsign);
    println!("Altitude: {} ft", altitude);
    println!("Output:   {:?}", output);
    println!();

    // Build aircraft identification message (TC=4, CA=0)
    // DF17 = 0x8D (DF=17, CA=5 for airborne)
    let df_ca = 0x8Du8;

    // Type code 4 for aircraft identification, category 0
    let tc_ca = (4 << 3) | 0; // TC=4, CA=0

    // Encode callsign (8 chars, 6 bits each)
    let callsign_padded = format!("{:<8}", callsign.to_uppercase());
    let chars: Vec<u8> = callsign_padded
        .chars()
        .take(8)
        .map(|c| match c {
            'A'..='Z' => c as u8 - b'A' + 1,
            '0'..='9' => c as u8 - b'0' + 48,
            ' ' => 0,
            _ => 0,
        })
        .collect();

    // Pack callsign into 6 bytes (48 bits)
    let mut me_bytes = [0u8; 7];
    me_bytes[0] = tc_ca;

    // Pack 8 chars * 6 bits = 48 bits into bytes 1-6
    let mut bit_pos = 8; // Start after TC/CA byte
    for &ch in &chars {
        let byte_idx = bit_pos / 8;
        let bit_offset = bit_pos % 8;

        if bit_offset <= 2 {
            me_bytes[byte_idx] |= ch << (2 - bit_offset);
        } else {
            me_bytes[byte_idx] |= ch >> (bit_offset - 2);
            if byte_idx + 1 < 7 {
                me_bytes[byte_idx + 1] |= ch << (10 - bit_offset);
            }
        }
        bit_pos += 6;
    }

    // Build full message
    let mut msg_bytes = [0u8; 14];
    msg_bytes[0] = df_ca;
    msg_bytes[1] = ((icao_addr >> 16) & 0xFF) as u8;
    msg_bytes[2] = ((icao_addr >> 8) & 0xFF) as u8;
    msg_bytes[3] = (icao_addr & 0xFF) as u8;
    msg_bytes[4..11].copy_from_slice(&me_bytes);

    // Calculate CRC
    let crc = crc24(&msg_bytes);
    msg_bytes[11] = ((crc >> 16) & 0xFF) as u8;
    msg_bytes[12] = ((crc >> 8) & 0xFF) as u8;
    msg_bytes[13] = (crc & 0xFF) as u8;

    // Convert to bits
    let mut bits = Vec::with_capacity(112);
    for byte in &msg_bytes {
        for i in (0..8).rev() {
            bits.push((byte >> i) & 1);
        }
    }

    // Modulate
    let ppm = PPM::adsb(sample_rate);
    let samples = ppm.modulate(&bits);

    println!(
        "Generated {} samples ({:.3} ms)",
        samples.len(),
        samples.len() as f64 / sample_rate * 1000.0
    );

    // Verify message
    let msg = AdsbMessage::decode(&msg_bytes);
    println!();
    println!("Message:  {}", msg);
    println!(
        "Hex:      {}",
        msg_bytes
            .iter()
            .map(|b| format!("{:02X}", b))
            .collect::<String>()
    );

    // Write samples
    write_samples_f32(&samples, &output)?;
    println!();
    println!("Wrote samples to {:?}", output);

    Ok(())
}

fn cmd_remote(address: String, command: RemoteCommand) -> Result<()> {
    // Parse address
    let (host, port) = if address.contains(':') {
        let parts: Vec<&str> = address.split(':').collect();
        (parts[0].to_string(), parts[1].parse::<u16>().unwrap_or(DEFAULT_AGENT_PORT))
    } else {
        (address, DEFAULT_AGENT_PORT)
    };

    let mut client = AgentClient::connect((&*host, port))
        .map_err(|e| anyhow::anyhow!("Connection failed: {}", e))?;

    match command {
        RemoteCommand::Status => {
            let status = client.status()
                .map_err(|e| anyhow::anyhow!("Status failed: {}", e))?;

            println!("Agent Status");
            println!("============");
            println!("Version:   {}", status.version);
            println!("Uptime:    {}s", status.uptime_secs);
            println!();
            println!("Device:");
            println!("  Hostname: {}", status.device.hostname);
            println!("  OS:       {}", status.device.os);
            println!("  Arch:     {}", status.device.arch);
            println!("  CPU:      {}", status.device.cpu);
            println!("  Memory:   {} MB", status.device.memory_mb);
            println!("  IPs:      {}", status.device.ip_addresses.join(", "));
            println!();
            println!("TX Task:   {:?}", status.tx_task);
            println!("RX Task:   {:?}", status.rx_task);
        }

        RemoteCommand::Ping => {
            let latency = client.ping()
                .map_err(|e| anyhow::anyhow!("Ping failed: {}", e))?;
            println!("Pong! Latency: {} ms", latency);
        }

        RemoteCommand::StartTx {
            target,
            waveform,
            sample_rate,
            message,
            snr,
            pps,
            repeat,
        } => {
            let result = client.start_tx(&target, &waveform, sample_rate, &message, snr, pps, repeat)
                .map_err(|e| anyhow::anyhow!("StartTx failed: {}", e))?;
            println!("{}", result);
        }

        RemoteCommand::StopTx => {
            client.stop_tx()
                .map_err(|e| anyhow::anyhow!("StopTx failed: {}", e))?;
            println!("TX stopped");
        }

        RemoteCommand::StartRx {
            port,
            waveform,
            sample_rate,
        } => {
            let result = client.start_rx(port, &waveform, sample_rate)
                .map_err(|e| anyhow::anyhow!("StartRx failed: {}", e))?;
            println!("{}", result);
        }

        RemoteCommand::StopRx => {
            client.stop_rx()
                .map_err(|e| anyhow::anyhow!("StopRx failed: {}", e))?;
            println!("RX stopped");
        }

        RemoteCommand::ListWaveforms => {
            let waveforms = client.list_waveforms()
                .map_err(|e| anyhow::anyhow!("ListWaveforms failed: {}", e))?;

            println!("Available Waveforms");
            println!("===================");
            for wf in waveforms {
                println!("  {:8} - {} ({} bits/symbol)",
                    wf.name, wf.full_name, wf.bits_per_symbol);
            }
        }

        RemoteCommand::Shutdown => {
            client.shutdown()
                .map_err(|e| anyhow::anyhow!("Shutdown failed: {}", e))?;
            println!("Agent shutdown requested");
        }
    }

    Ok(())
}

/// Compare waveforms across SNR range
fn cmd_compare(
    waveforms: String,
    snr_min: f64,
    snr_max: f64,
    snr_step: f64,
    bits: usize,
    sample_rate: f64,
    output: String,
    output_file: Option<PathBuf>,
    list: bool,
) -> Result<()> {
    use r4w_core::waveform::{psk, qam, fsk, ook, ask, Waveform, CommonParams};

    // List available waveforms
    if list {
        println!("=== Waveforms Available for Comparison ===");
        println!();
        let available = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "BFSK", "4FSK", "OOK", "ASK", "4ASK"];
        for name in available {
            println!("  {}", name);
        }
        println!();
        println!("Example: r4w compare -w BPSK,QPSK,8PSK --snr-min 0 --snr-max 15");
        return Ok(());
    }

    if waveforms.is_empty() {
        anyhow::bail!("No waveforms specified. Use -w BPSK,QPSK or --list to see options.");
    }

    // Parse waveform list
    let wf_names: Vec<&str> = waveforms.split(',').map(|s| s.trim()).collect();

    // Create waveforms
    let symbol_rate = sample_rate / 10.0; // 10 samples per symbol
    let common = CommonParams {
        sample_rate,
        carrier_freq: 0.0,
        amplitude: 1.0,
    };

    let mut wf_list: Vec<(String, Box<dyn Waveform>)> = Vec::new();

    for name in &wf_names {
        let wf: Box<dyn Waveform> = match name.to_uppercase().as_str() {
            "BPSK" => Box::new(psk::PSK::new_bpsk(common.clone(), symbol_rate)),
            "QPSK" => Box::new(psk::PSK::new_qpsk(common.clone(), symbol_rate)),
            "8PSK" => Box::new(psk::PSK::new_8psk(common.clone(), symbol_rate)),
            "16QAM" | "QAM16" => Box::new(qam::QAM::new_16qam(common.clone(), symbol_rate)),
            "64QAM" | "QAM64" => Box::new(qam::QAM::new_64qam(common.clone(), symbol_rate)),
            "BFSK" | "FSK" => Box::new(fsk::FSK::new_bfsk(common.clone(), symbol_rate, sample_rate / 20.0)),
            "4FSK" => Box::new(fsk::FSK::new_4fsk(common.clone(), symbol_rate, sample_rate / 20.0)),
            "OOK" => Box::new(ook::OOK::new(common.clone(), symbol_rate)),
            "ASK" => Box::new(ask::ASK::new_binary(common.clone(), symbol_rate, symbol_rate)),
            "4ASK" | "4-ASK" => Box::new(ask::ASK::new_4ask(common.clone(), symbol_rate, symbol_rate)),
            _ => {
                anyhow::bail!("Unknown waveform: {}. Use --list to see available waveforms.", name);
            }
        };
        wf_list.push((name.to_string(), wf));
    }

    // Generate random test data (bytes)
    let mut rng = rand::thread_rng();
    use rand::Rng;
    let num_bytes = (bits + 7) / 8;
    let tx_bytes: Vec<u8> = (0..num_bytes).map(|_| rng.gen()).collect();

    // SNR sweep
    let snr_points: Vec<f64> = {
        let mut points = Vec::new();
        let mut snr = snr_min;
        while snr <= snr_max + 0.001 {
            points.push(snr);
            snr += snr_step;
        }
        points
    };

    // Results: waveform -> [(snr, ber)]
    let mut results: Vec<(String, Vec<(f64, f64)>)> = Vec::new();

    // Run BER tests
    for (name, wf) in &wf_list {
        let info = wf.info();
        let mut ber_curve = Vec::new();

        for &snr in &snr_points {
            // Modulate
            let samples = wf.modulate(&tx_bytes);

            // Add AWGN noise
            let noisy_samples = if snr < 100.0 {
                let channel_config = ChannelConfig {
                    model: ChannelModel::Awgn,
                    snr_db: snr,
                    ..Default::default()
                };
                let mut channel = Channel::new(channel_config);
                channel.apply(&samples)
            } else {
                samples
            };

            // Demodulate
            let result = wf.demodulate(&noisy_samples);
            let rx_bytes = result.bits;

            // Calculate BER at bit level
            let min_bytes = tx_bytes.len().min(rx_bytes.len());
            let mut bit_errors = 0usize;
            let mut total_bits = 0usize;
            for i in 0..min_bytes {
                let diff = tx_bytes[i] ^ rx_bytes[i];
                bit_errors += diff.count_ones() as usize;
                total_bits += 8;
            }

            let ber = if total_bits > 0 {
                bit_errors as f64 / total_bits as f64
            } else {
                1.0
            };

            ber_curve.push((snr, ber));
        }

        results.push((format!("{} ({} bits/sym)", name, info.bits_per_symbol), ber_curve));
    }

    // Format output
    let output_text = match output.to_lowercase().as_str() {
        "json" => {
            let mut json_results: Vec<serde_json::Value> = Vec::new();
            for (name, curve) in &results {
                json_results.push(serde_json::json!({
                    "waveform": name,
                    "data": curve.iter().map(|(snr, ber)| {
                        serde_json::json!({"snr": snr, "ber": ber})
                    }).collect::<Vec<_>>()
                }));
            }
            serde_json::to_string_pretty(&json_results)?
        }
        "csv" => {
            let mut csv = String::new();
            // Header
            csv.push_str("SNR");
            for (name, _) in &results {
                csv.push(',');
                csv.push_str(name);
            }
            csv.push('\n');
            // Data rows
            for (i, snr) in snr_points.iter().enumerate() {
                csv.push_str(&format!("{:.1}", snr));
                for (_, curve) in &results {
                    csv.push_str(&format!(",{:.6}", curve[i].1));
                }
                csv.push('\n');
            }
            csv
        }
        _ => {
            // Text format (table)
            let mut text = String::new();
            text.push_str("=== Waveform Comparison (BER vs SNR) ===\n\n");
            text.push_str(&format!("Test: {} bits per SNR point\n", bits));
            text.push_str(&format!("SNR range: {} to {} dB (step {})\n\n", snr_min, snr_max, snr_step));

            // Header
            text.push_str(&format!("{:>8}", "SNR(dB)"));
            for (name, _) in &results {
                text.push_str(&format!("{:>15}", name.split(' ').next().unwrap_or(name)));
            }
            text.push('\n');
            text.push_str(&"-".repeat(8 + results.len() * 15));
            text.push('\n');

            // Data rows
            for (i, snr) in snr_points.iter().enumerate() {
                text.push_str(&format!("{:>8.1}", snr));
                for (_, curve) in &results {
                    let ber = curve[i].1;
                    if ber == 0.0 {
                        text.push_str(&format!("{:>15}", "0"));
                    } else if ber < 0.0001 {
                        text.push_str(&format!("{:>15.2e}", ber));
                    } else {
                        text.push_str(&format!("{:>15.4}", ber));
                    }
                }
                text.push('\n');
            }

            text.push('\n');
            text.push_str("Legend: Lower BER is better. 0 = no errors detected.\n");
            text
        }
    };

    // Write output
    if let Some(path) = output_file {
        let mut file = File::create(&path)?;
        file.write_all(output_text.as_bytes())?;
        println!("Results written to {:?}", path);
    } else {
        print!("{}", output_text);
    }

    Ok(())
}

/// Record I/Q samples to SigMF file
fn cmd_record(
    output: PathBuf,
    sample_rate: f64,
    frequency: f64,
    duration: f64,
    description: Option<String>,
    waveform: Option<String>,
    generate: Option<String>,
) -> Result<()> {
    use r4w_sim::hal::sigmf::SigMfWriter;

    println!("=== SigMF Recording ===");
    println!("Output: {:?}", output);
    println!("Sample rate: {} Hz", sample_rate);
    println!("Frequency: {} Hz", frequency);

    let mut writer = SigMfWriter::create(&output, sample_rate, frequency)
        .map_err(|e| anyhow::anyhow!("Failed to create SigMF writer: {}", e))?;

    // Set optional metadata
    if let Some(desc) = &description {
        writer.set_description(desc);
    }
    if let Some(wf) = &waveform {
        writer.set_waveform(wf);
    }

    // Generate test signal or record from device
    if let Some(signal_type) = generate {
        println!("Generating test signal: {}", signal_type);

        let num_samples = if duration > 0.0 {
            (sample_rate * duration) as usize
        } else {
            (sample_rate * 1.0) as usize // Default 1 second
        };

        let samples: Vec<IQSample> = match signal_type.to_lowercase().as_str() {
            "tone" | "cw" => {
                // Generate a CW tone at 1kHz offset
                let tone_freq = 1000.0;
                (0..num_samples)
                    .map(|i| {
                        let t = i as f64 / sample_rate;
                        let phase = 2.0 * std::f64::consts::PI * tone_freq * t;
                        IQSample::new(phase.cos(), phase.sin())
                    })
                    .collect()
            }
            "chirp" => {
                // Generate a chirp from -BW/2 to +BW/2
                let bw = sample_rate / 4.0;
                (0..num_samples)
                    .map(|i| {
                        let t = i as f64 / sample_rate;
                        let freq = -bw / 2.0 + (bw / (num_samples as f64 / sample_rate)) * t;
                        let phase = 2.0 * std::f64::consts::PI * freq * t;
                        IQSample::new(phase.cos(), phase.sin())
                    })
                    .collect()
            }
            "noise" => {
                // Generate white noise
                use rand::Rng;
                let mut rng = rand::thread_rng();
                (0..num_samples)
                    .map(|_| {
                        IQSample::new(
                            rng.gen_range(-1.0f64..1.0f64),
                            rng.gen_range(-1.0f64..1.0f64),
                        )
                    })
                    .collect()
            }
            _ => {
                anyhow::bail!("Unknown signal type: {}. Use: tone, chirp, noise", signal_type);
            }
        };

        writer
            .write_samples(&samples)
            .map_err(|e| anyhow::anyhow!("Failed to write samples: {}", e))?;

        println!("Wrote {} samples ({:.2} seconds)", samples.len(), samples.len() as f64 / sample_rate);
    } else {
        // TODO: Record from SDR device
        println!("Recording from device not yet implemented.");
        println!("Use --generate <tone|chirp|noise> to create test signals.");
        return Ok(());
    }

    writer.close().map_err(|e| anyhow::anyhow!("Failed to close writer: {}", e))?;

    println!("Recording saved to:");
    println!("  {}.sigmf-meta", output.display());
    println!("  {}.sigmf-data", output.display());

    Ok(())
}

/// Playback I/Q samples from SigMF file
fn cmd_playback(
    input: PathBuf,
    info_only: bool,
    format: String,
    num_samples: usize,
    offset: usize,
    demodulate: Option<String>,
) -> Result<()> {
    use r4w_sim::hal::sigmf::SigMfReader;

    let mut reader = SigMfReader::open(&input)
        .map_err(|e| anyhow::anyhow!("Failed to open SigMF file: {}", e))?;

    let meta = reader.metadata();
    let sample_rate = meta.global.sample_rate;
    let frequency = meta.captures.first().and_then(|c| c.frequency).unwrap_or(0.0);
    let total_samples = reader.total_samples();

    if info_only || format == "json" && num_samples == 0 {
        if format == "json" {
            let info = serde_json::json!({
                "file": input.to_string_lossy(),
                "sample_rate": sample_rate,
                "frequency": frequency,
                "datatype": meta.global.datatype,
                "total_samples": total_samples,
                "duration_seconds": total_samples as f64 / sample_rate,
                "description": meta.global.description,
                "waveform": meta.global.r4w_waveform,
                "datetime": meta.global.datetime,
                "author": meta.global.author,
            });
            println!("{}", serde_json::to_string_pretty(&info)?);
        } else {
            println!("=== SigMF File Info ===");
            println!("File: {:?}", input);
            println!("Sample rate: {} Hz", sample_rate);
            println!("Frequency: {} Hz", frequency);
            println!("Datatype: {}", meta.global.datatype);
            println!("Total samples: {}", total_samples);
            println!("Duration: {:.3} seconds", total_samples as f64 / sample_rate);
            if let Some(desc) = &meta.global.description {
                println!("Description: {}", desc);
            }
            if let Some(wf) = &meta.global.r4w_waveform {
                println!("Waveform: {}", wf);
            }
            if let Some(dt) = &meta.global.datetime {
                println!("Recorded: {}", dt);
            }
        }
        return Ok(());
    }

    // Read samples
    let total_usize = total_samples as usize;
    let samples_to_read = if num_samples == 0 {
        total_usize.saturating_sub(offset)
    } else {
        num_samples.min(total_usize.saturating_sub(offset))
    };

    if offset > 0 {
        reader.seek(offset as u64)
            .map_err(|e| anyhow::anyhow!("Failed to seek: {}", e))?;
    }

    let mut buffer = vec![IQSample::default(); samples_to_read];
    let samples_read = reader.read_samples(&mut buffer)
        .map_err(|e| anyhow::anyhow!("Failed to read samples: {}", e))?;

    buffer.truncate(samples_read);

    println!("=== SigMF Playback ===");
    println!("Read {} samples from offset {}", samples_read, offset);

    // Demodulate if requested
    if let Some(wf_name) = demodulate {
        use r4w_core::waveform::{psk, qam, fsk, ook, Waveform, CommonParams};

        let symbol_rate = sample_rate / 10.0;
        let common = CommonParams {
            sample_rate,
            carrier_freq: 0.0,
            amplitude: 1.0,
        };

        let wf: Box<dyn Waveform> = match wf_name.to_uppercase().as_str() {
            "BPSK" => Box::new(psk::PSK::new_bpsk(common, symbol_rate)),
            "QPSK" => Box::new(psk::PSK::new_qpsk(common, symbol_rate)),
            "8PSK" => Box::new(psk::PSK::new_8psk(common, symbol_rate)),
            "16QAM" => Box::new(qam::QAM::new_16qam(common, symbol_rate)),
            "BFSK" => Box::new(fsk::FSK::new_bfsk(common, symbol_rate, sample_rate / 20.0)),
            "OOK" => Box::new(ook::OOK::new(common, symbol_rate)),
            _ => anyhow::bail!("Unknown waveform for demodulation: {}", wf_name),
        };

        let result = wf.demodulate(&buffer);
        println!("Demodulated {} bytes", result.bits.len());

        // Try to decode as text
        if let Ok(text) = String::from_utf8(result.bits.clone()) {
            if text.chars().all(|c| c.is_ascii_graphic() || c.is_ascii_whitespace()) {
                println!("Decoded text: {}", text);
            } else {
                println!("Decoded bytes: {:02X?}", &result.bits[..result.bits.len().min(64)]);
            }
        } else {
            println!("Decoded bytes: {:02X?}", &result.bits[..result.bits.len().min(64)]);
        }
    } else {
        // Just show sample statistics
        let power: f64 = buffer.iter().map(|s| (s.re * s.re + s.im * s.im) as f64).sum::<f64>() / buffer.len() as f64;
        let power_db = 10.0 * power.log10();

        println!("Average power: {:.2} dB", power_db);
        println!("First 10 samples: {:?}", &buffer[..buffer.len().min(10)]);
    }

    Ok(())
}

/// Convert between signal file formats
fn cmd_convert(
    input: PathBuf,
    output: PathBuf,
    from_format: Option<String>,
    to_format: String,
    sample_rate: Option<f64>,
    frequency: Option<f64>,
) -> Result<()> {
    use r4w_sim::hal::sigmf::{SigMfReader, SigMfWriter};
    use std::io::{BufReader, BufWriter};

    println!("=== Signal Format Conversion ===");
    println!("Input: {:?}", input);
    println!("Output: {:?}", output);

    // Detect input format
    let input_ext = input.extension().and_then(|e| e.to_str()).unwrap_or("");
    let detected_format = from_format.unwrap_or_else(|| {
        match input_ext {
            "sigmf-meta" | "sigmf-data" => "sigmf".to_string(),
            "iq" | "cf32" => "raw-cf32".to_string(),
            "cs16" | "ci16" => "raw-ci16".to_string(),
            "cs8" | "ci8" => "raw-ci8".to_string(),
            _ => "raw-cf32".to_string(),
        }
    });

    println!("Input format: {}", detected_format);
    println!("Output format: {}", to_format);

    // Read samples based on input format
    let (samples, src_sample_rate, src_frequency): (Vec<IQSample>, f64, f64) = match detected_format.as_str() {
        "sigmf" => {
            let meta_path = if input_ext == "sigmf-data" {
                input.with_extension("sigmf-meta")
            } else {
                input.clone()
            };

            let mut reader = SigMfReader::open(&meta_path)
                .map_err(|e| anyhow::anyhow!("Failed to open SigMF: {}", e))?;

            let meta = reader.metadata();
            let sr = meta.global.sample_rate;
            let freq = meta.captures.first().and_then(|c| c.frequency).unwrap_or(0.0);
            let total = reader.total_samples() as usize;

            let mut buffer = vec![IQSample::default(); total];
            let read = reader.read_samples(&mut buffer)
                .map_err(|e| anyhow::anyhow!("Failed to read: {}", e))?;
            buffer.truncate(read);

            (buffer, sr, freq)
        }
        "raw-cf32" => {
            let sr = sample_rate.ok_or_else(|| anyhow::anyhow!("--sample-rate required for raw input"))?;
            let freq = frequency.unwrap_or(0.0);

            let file = File::open(&input)?;
            let mut reader = BufReader::new(file);
            let mut samples = Vec::new();

            loop {
                let mut buf = [0u8; 8];
                match std::io::Read::read_exact(&mut reader, &mut buf) {
                    Ok(()) => {
                        let re = f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as f64;
                        let im = f32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as f64;
                        samples.push(IQSample::new(re, im));
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(anyhow::anyhow!("Read error: {}", e)),
                }
            }

            (samples, sr, freq)
        }
        "raw-ci16" => {
            let sr = sample_rate.ok_or_else(|| anyhow::anyhow!("--sample-rate required for raw input"))?;
            let freq = frequency.unwrap_or(0.0);

            let file = File::open(&input)?;
            let mut reader = BufReader::new(file);
            let mut samples = Vec::new();

            loop {
                let mut buf = [0u8; 4];
                match std::io::Read::read_exact(&mut reader, &mut buf) {
                    Ok(()) => {
                        let re = i16::from_le_bytes([buf[0], buf[1]]) as f64 / 32768.0;
                        let im = i16::from_le_bytes([buf[2], buf[3]]) as f64 / 32768.0;
                        samples.push(IQSample::new(re, im));
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(anyhow::anyhow!("Read error: {}", e)),
                }
            }

            (samples, sr, freq)
        }
        _ => anyhow::bail!("Unsupported input format: {}", detected_format),
    };

    println!("Read {} samples", samples.len());

    // Write samples based on output format
    match to_format.as_str() {
        "sigmf" => {
            let out_sr = sample_rate.unwrap_or(src_sample_rate);
            let out_freq = frequency.unwrap_or(src_frequency);

            let mut writer = SigMfWriter::create(&output, out_sr, out_freq)
                .map_err(|e| anyhow::anyhow!("Failed to create SigMF: {}", e))?;

            writer.write_samples(&samples)
                .map_err(|e| anyhow::anyhow!("Failed to write: {}", e))?;

            writer.close().map_err(|e| anyhow::anyhow!("Failed to close: {}", e))?;

            println!("Wrote SigMF files:");
            println!("  {}.sigmf-meta", output.display());
            println!("  {}.sigmf-data", output.display());
        }
        "raw-cf32" => {
            let file = File::create(&output)?;
            let mut writer = BufWriter::new(file);

            for sample in &samples {
                std::io::Write::write_all(&mut writer, &(sample.re as f32).to_le_bytes())?;
                std::io::Write::write_all(&mut writer, &(sample.im as f32).to_le_bytes())?;
            }

            println!("Wrote {} samples to {:?}", samples.len(), output);
        }
        "raw-ci16" => {
            let file = File::create(&output)?;
            let mut writer = BufWriter::new(file);

            for sample in &samples {
                let re = (sample.re * 32767.0).clamp(-32768.0, 32767.0) as i16;
                let im = (sample.im * 32767.0).clamp(-32768.0, 32767.0) as i16;
                std::io::Write::write_all(&mut writer, &re.to_le_bytes())?;
                std::io::Write::write_all(&mut writer, &im.to_le_bytes())?;
            }

            println!("Wrote {} samples to {:?}", samples.len(), output);
        }
        _ => anyhow::bail!("Unsupported output format: {}", to_format),
    }

    Ok(())
}

/// Display or serve Prometheus metrics
fn cmd_metrics(format: String, serve: bool, port: u16) -> Result<()> {
    use r4w_core::observe::Metrics;

    // Create a demo metrics instance
    let metrics = Metrics::new();

    // Populate with demo data
    metrics.rx_samples.inc_by(1_000_000);
    metrics.tx_samples.inc_by(500_000);
    metrics.rx_buffer_level.set(4096);
    metrics.tx_buffer_level.set(2048);
    metrics.packets_decoded.inc_by(1500);
    metrics.packets_failed.inc_by(12);
    metrics.record_rssi(-85.5);
    metrics.record_snr(15.2);
    metrics.set_waveform("lora");

    if serve {
        println!("Starting metrics server on http://0.0.0.0:{}/metrics", port);
        println!("Press Ctrl+C to stop");
        println!();
        println!("Add to prometheus.yml:");
        println!("  - job_name: 'r4w'");
        println!("    static_configs:");
        println!("      - targets: ['localhost:{}']", port);

        // Simple HTTP server for metrics
        use std::net::TcpListener;
        use std::io::{BufRead, BufReader, Write};

        let listener = TcpListener::bind(format!("0.0.0.0:{}", port))?;

        for stream in listener.incoming() {
            match stream {
                Ok(mut stream) => {
                    let reader = BufReader::new(&stream);
                    let request_line = reader.lines().next();

                    if let Some(Ok(line)) = request_line {
                        if line.contains("GET /metrics") || line.contains("GET / ") {
                            let body = metrics.to_prometheus();
                            let response = format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}",
                                body.len(),
                                body
                            );
                            let _ = stream.write_all(response.as_bytes());
                        } else {
                            let response = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
                            let _ = stream.write_all(response.as_bytes());
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Connection error: {}", e);
                }
            }
        }

        Ok(())
    } else {
        // Just display metrics
        match format.to_lowercase().as_str() {
            "prometheus" => {
                println!("{}", metrics.to_prometheus());
            }
            "json" => {
                let snapshot = metrics.snapshot();
                println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                    "rx_samples": snapshot.rx_samples,
                    "tx_samples": snapshot.tx_samples,
                    "rx_buffer_level": snapshot.rx_buffer_level,
                    "tx_buffer_level": snapshot.tx_buffer_level,
                    "rx_overflows": snapshot.rx_overflows,
                    "tx_underflows": snapshot.tx_underflows,
                    "packets_decoded": snapshot.packets_decoded,
                    "packets_failed": snapshot.packets_failed,
                    "rssi_dbm": snapshot.rssi_dbm,
                    "snr_db": snapshot.snr_db,
                    "active_waveform": snapshot.active_waveform,
                    "decode_success_rate": snapshot.decode_success_rate(),
                }))?);
            }
            _ => {
                println!("=== R4W Metrics ===");
                let snapshot = metrics.snapshot();
                println!();
                println!("Samples:");
                println!("  RX: {}", snapshot.rx_samples);
                println!("  TX: {}", snapshot.tx_samples);
                println!();
                println!("Buffers:");
                println!("  RX level: {}", snapshot.rx_buffer_level);
                println!("  TX level: {}", snapshot.tx_buffer_level);
                println!();
                println!("Packets:");
                println!("  Decoded: {}", snapshot.packets_decoded);
                println!("  Failed: {}", snapshot.packets_failed);
                println!("  Success rate: {:.1}%", snapshot.decode_success_rate() * 100.0);
                println!();
                println!("Signal:");
                println!("  RSSI: {:.1} dBm", snapshot.rssi_dbm);
                println!("  SNR: {:.1} dB", snapshot.snr_db);
                println!();
                println!("Waveform: {}", snapshot.active_waveform);
                println!();
                println!("Use -f prometheus for Prometheus format");
                println!("Use --serve to start HTTP endpoint");
            }
        }

        Ok(())
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup logging
    let log_level = match cli.verbose {
        0 => tracing::Level::WARN,
        1 => tracing::Level::INFO,
        2 => tracing::Level::DEBUG,
        _ => tracing::Level::TRACE,
    };

    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_target(false)
        .init();

    match cli.command {
        Commands::Tx {
            message,
            output,
            sf,
            bw,
            cr,
            format,
        } => cmd_tx(message, output, sf, bw, cr, format),

        Commands::Rx {
            input,
            sf,
            bw,
            cr,
            format,
        } => cmd_rx(input, sf, bw, cr, format),

        Commands::Simulate {
            message,
            snr,
            cfo,
            channel,
            sf,
            bw,
            cr,
            save_samples,
        } => cmd_simulate(message, snr, cfo, channel, sf, bw, cr, save_samples),

        Commands::Chirp {
            output,
            chirp_type,
            symbol,
            sf,
            bw,
        } => cmd_chirp(output, chirp_type, symbol, sf, bw),

        Commands::Info {
            sf,
            bw,
            cr,
            payload_len,
        } => cmd_info(sf, bw, cr, payload_len),

        Commands::Analyze {
            input,
            format,
            sample_rate,
            samples,
            mode,
            fft_size,
            average,
            output_format,
            output,
            window,
            min_db,
            max_db,
            colormap,
            threshold,
            max_peaks,
            height,
        } => cmd_analyze(AnalyzeArgs {
            input,
            format,
            sample_rate,
            samples,
            mode,
            fft_size,
            average,
            output_format,
            output,
            window,
            min_db,
            max_db,
            colormap,
            threshold,
            max_peaks,
            height,
        }),

        Commands::Waveform {
            waveform,
            data,
            snr,
            sample_rate,
            symbol_rate,
            list,
        } => cmd_waveform(waveform, data, snr, sample_rate, symbol_rate, list),

        Commands::Benchmark {
            port,
            format,
            waveform,
            sample_rate,
            batch_size,
            duration,
            output,
            output_file,
            stats_interval,
            list,
        } => cmd_benchmark(port, format, waveform, sample_rate, batch_size, duration, output, output_file, stats_interval, list),

        Commands::UdpSend {
            target,
            waveform,
            sample_rate,
            format,
            message,
            pps,
            samples_per_packet,
            duration,
            snr,
            list,
            repeat,
        } => cmd_udp_send(target, waveform, sample_rate, format, message, pps, samples_per_packet, duration, snr, list, repeat),

        Commands::Agent { port, foreground } => cmd_agent(port, foreground),

        Commands::Remote { address, command } => cmd_remote(address, command),

        Commands::Mesh { command } => match command {
            MeshCommand::Status {
                node_id,
                preset,
                region,
            } => cmd_mesh_status(node_id, preset, region),

            MeshCommand::Send {
                message,
                dest,
                hop_limit,
                node_id,
                preset,
                region,
            } => cmd_mesh_send(message, dest, hop_limit, node_id, preset, region),

            MeshCommand::Neighbors {
                node_id,
                preset,
                region,
            } => cmd_mesh_neighbors(node_id, preset, region),

            MeshCommand::Simulate {
                nodes,
                messages,
                snr,
                preset,
                region,
                verbose,
            } => cmd_mesh_simulate(nodes, messages, snr, preset, region, verbose),

            MeshCommand::Info => cmd_mesh_info(),

            MeshCommand::Analyze { hex, summary } => cmd_mesh_analyze(&hex, summary),

            MeshCommand::Repl {
                node_id,
                preset,
                region,
                sim_nodes,
            } => cmd_mesh_repl(node_id, preset, region, sim_nodes),
        },

        Commands::Adsb { command } => match command {
            AdsbCommand::Decode { message, verbose } => cmd_adsb_decode(message, verbose),

            AdsbCommand::File {
                input,
                sample_rate,
                all,
            } => cmd_adsb_file(input, sample_rate, all),

            AdsbCommand::Info => cmd_adsb_info(),

            AdsbCommand::Generate {
                output,
                icao,
                callsign,
                altitude,
                sample_rate,
            } => cmd_adsb_generate(output, icao, callsign, altitude, sample_rate),
        },

        Commands::Completions { shell } => {
            let mut cmd = Cli::command();
            let bin_name = cmd.get_name().to_string();
            generate(shell, &mut cmd, bin_name, &mut std::io::stdout());
            Ok(())
        }

        Commands::Compare {
            waveforms,
            snr_min,
            snr_max,
            snr_step,
            bits,
            sample_rate,
            output,
            output_file,
            list,
        } => cmd_compare(waveforms, snr_min, snr_max, snr_step, bits, sample_rate, output, output_file, list),
        Commands::Record {
            output,
            sample_rate,
            frequency,
            duration,
            description,
            waveform,
            generate,
        } => cmd_record(output, sample_rate, frequency, duration, description, waveform, generate),
        Commands::Playback {
            input,
            info,
            format,
            samples,
            offset,
            demodulate,
        } => cmd_playback(input, info, format, samples, offset, demodulate),
        Commands::Convert {
            input,
            output,
            from,
            to,
            sample_rate,
            frequency,
        } => cmd_convert(input, output, from, to, sample_rate, frequency),
        Commands::Metrics { format, serve, port } => cmd_metrics(format, serve, port),
    }
}
