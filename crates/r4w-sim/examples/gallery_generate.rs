//! Gallery Image Generator
//!
//! Generates constellation diagrams, spectrum plots, waterfall images,
//! and BER curves for the R4W gallery.
//!
//! Run with: cargo run --example gallery_generate -p r4w-sim --features image

use r4w_core::analysis::{Colormap, SpectrumAnalyzer, WaterfallGenerator, WindowFunction};
use r4w_core::types::IQSample;
use r4w_core::waveform::WaveformFactory;
use r4w_sim::channel::{Channel, ChannelConfig, ChannelModel, TdlProfile};
use std::fs;

const GALLERY_DIR: &str = "gallery";

fn main() {
    println!("R4W Gallery Generator");
    println!("=====================\n");

    // Ensure directories exist
    for subdir in &["waveforms", "channels", "ber_curves"] {
        let path = format!("{}/{}", GALLERY_DIR, subdir);
        fs::create_dir_all(&path).expect("Failed to create directory");
    }

    generate_waveform_gallery();
    generate_channel_gallery();
    generate_ber_gallery();

    println!("\nGallery generation complete!");
    println!("Images saved to: {}/", GALLERY_DIR);
}

fn generate_waveform_gallery() {
    println!("Generating waveform gallery...");

    // PSK waveforms
    for &name in &["BPSK", "QPSK", "8PSK"] {
        if let Some(wf) = WaveformFactory::create(name, 48000.0) {
            let info = wf.info();
            println!("  {} - {}", name, info.full_name);

            // Generate constellation from modulated samples
            let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
            let samples = wf.modulate(&data);

            // Downsample to get constellation points (one per symbol)
            let sps = wf.samples_per_symbol().max(1);
            let constellation: Vec<_> = samples.iter().step_by(sps).cloned().collect();
            generate_constellation_png(&constellation, &format!("{}/waveforms/{}_constellation.png", GALLERY_DIR, name.to_lowercase()));

            // Generate spectrum
            generate_spectrum_png(&samples, 48000.0, &format!("{}/waveforms/{}_spectrum.png", GALLERY_DIR, name.to_lowercase()));
        }
    }

    // QAM waveforms
    for &name in &["16QAM", "64QAM"] {
        if let Some(wf) = WaveformFactory::create(name, 48000.0) {
            let info = wf.info();
            println!("  {} - {}", name, info.full_name);

            let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
            let samples = wf.modulate(&data);
            let sps = wf.samples_per_symbol().max(1);
            let constellation: Vec<_> = samples.iter().step_by(sps).cloned().collect();
            generate_constellation_png(&constellation, &format!("{}/waveforms/{}_constellation.png", GALLERY_DIR, name.to_lowercase()));
        }
    }

    // LoRa chirps (waterfall)
    println!("  LoRa CSS waterfall...");
    generate_lora_waterfall();
}

fn generate_channel_gallery() {
    println!("Generating channel effect gallery...");

    let sample_rate = 100_000.0;

    // Generate test signal (QPSK)
    let wf = WaveformFactory::create("QPSK", sample_rate).expect("QPSK not found");
    let data: Vec<u8> = (0..64).collect();
    let clean_samples = wf.modulate(&data);

    // AWGN at various SNRs
    for snr in &[0.0, 5.0, 10.0, 20.0] {
        println!("  AWGN SNR={}dB", snr);
        let config = ChannelConfig::with_snr(*snr);
        let mut channel = Channel::new(config);
        let noisy = channel.apply(&clean_samples);
        generate_constellation_png(&noisy, &format!("{}/channels/awgn_snr{}db.png", GALLERY_DIR, *snr as i32));
    }

    // Rayleigh fading
    println!("  Rayleigh fading");
    let config = ChannelConfig {
        model: ChannelModel::Rayleigh,
        snr_db: 20.0,
        sample_rate,
        ..Default::default()
    };
    let mut channel = Channel::new(config);
    let faded = channel.apply(&clean_samples);
    generate_constellation_png(&faded, &format!("{}/channels/rayleigh.png", GALLERY_DIR));

    // Rician fading at various K-factors
    for k in &[1.0, 5.0, 10.0] {
        println!("  Rician K={}", k);
        let config = ChannelConfig {
            model: ChannelModel::Rician,
            snr_db: 20.0,
            rician_k: *k,
            sample_rate,
            ..Default::default()
        };
        let mut channel = Channel::new(config);
        let faded = channel.apply(&clean_samples);
        generate_constellation_png(&faded, &format!("{}/channels/rician_k{}.png", GALLERY_DIR, *k as i32));
    }

    // TDL profiles
    for profile in &[TdlProfile::Epa, TdlProfile::Eva, TdlProfile::Etu] {
        let name = match profile {
            TdlProfile::Epa => "EPA",
            TdlProfile::Eva => "EVA",
            TdlProfile::Etu => "ETU",
            _ => "Custom",
        };
        println!("  TDL {}", name);
        let config = ChannelConfig::tdl(20.0, *profile, sample_rate);
        let mut channel = Channel::new(config);
        let multipath = channel.apply(&clean_samples);
        generate_constellation_png(&multipath, &format!("{}/channels/tdl_{}.png", GALLERY_DIR, name.to_lowercase()));
    }

    // Jake's Doppler at various speeds
    for doppler_hz in &[5.0, 30.0, 100.0] {
        println!("  Jake's Doppler {} Hz", doppler_hz);
        let config = ChannelConfig::jakes_fading(20.0, *doppler_hz, sample_rate);
        let mut channel = Channel::new(config);
        let faded = channel.apply(&clean_samples);
        generate_constellation_png(&faded, &format!("{}/channels/jakes_{}hz.png", GALLERY_DIR, *doppler_hz as i32));
    }
}

fn generate_ber_gallery() {
    println!("Generating BER curves...");
    // BER curves require more complex simulation - placeholder for now
    println!("  (BER curve generation not yet implemented)");
}

fn generate_lora_waterfall() {
    // Generate LoRa-like chirps for waterfall visualization
    use std::f64::consts::PI;

    let sample_rate = 125_000.0;
    let sf = 7u8;
    let n = 2usize.pow(sf as u32);
    let num_symbols = 8;

    let mut samples = Vec::with_capacity(n * num_symbols);

    for sym in 0..num_symbols {
        for i in 0..n {
            // Generate chirp with symbol value
            let symbol = (sym * 16) % n;
            let phase = 2.0 * PI * (((i + symbol) % n) as f64 / n as f64) * (i as f64 / 2.0);
            samples.push(IQSample::new(phase.cos(), phase.sin()));
        }
    }

    generate_waterfall_png(&samples, sample_rate, &format!("{}/waveforms/lora_sf7_waterfall.png", GALLERY_DIR));
}

#[cfg(feature = "image")]
fn generate_constellation_png(samples: &[IQSample], path: &str) {
    use image::{ImageBuffer, Rgb, ImageEncoder, codecs::png::PngEncoder, ExtendedColorType};

    let size = 256u32;
    let mut img = ImageBuffer::new(size, size);

    // White background
    for pixel in img.pixels_mut() {
        *pixel = Rgb([255, 255, 255]);
    }

    // Draw axes
    for i in 0..size {
        img.put_pixel(size / 2, i, Rgb([200, 200, 200]));
        img.put_pixel(i, size / 2, Rgb([200, 200, 200]));
    }

    // Draw points
    let scale = (size as f64) / 4.0; // Assume max amplitude ~2
    for sample in samples {
        let x = ((sample.re * scale + (size as f64 / 2.0)) as u32).clamp(0, size - 1);
        let y = ((-(sample.im) * scale + (size as f64 / 2.0)) as u32).clamp(0, size - 1);
        img.put_pixel(x, y, Rgb([0, 100, 200]));
    }

    let mut buffer = Vec::new();
    let encoder = PngEncoder::new(&mut buffer);
    encoder.write_image(&img, size, size, ExtendedColorType::Rgb8)
        .expect("Failed to encode PNG");

    fs::write(path, &buffer).expect("Failed to write PNG");
    println!("    Created: {}", path);
}

#[cfg(not(feature = "image"))]
fn generate_constellation_png(_samples: &[IQSample], path: &str) {
    println!("    (Skipping {} - image feature not enabled)", path);
}

#[cfg(feature = "image")]
fn generate_spectrum_png(samples: &[IQSample], sample_rate: f64, path: &str) {
    use image::{ImageBuffer, Rgb, ImageEncoder, codecs::png::PngEncoder, ExtendedColorType};

    let mut analyzer = SpectrumAnalyzer::with_window(1024, WindowFunction::Hann);
    let result = analyzer.compute_averaged(samples, sample_rate, 4);

    let width = 512u32;
    let height = 256u32;
    let mut img = ImageBuffer::new(width, height);

    // White background
    for pixel in img.pixels_mut() {
        *pixel = Rgb([255, 255, 255]);
    }

    // Calculate power range
    let mut min_db = f64::INFINITY;
    let mut max_db = f64::NEG_INFINITY;
    for &p in &result.power_db {
        if p > max_db {
            max_db = p;
        }
        if p < min_db && p > -200.0 {
            min_db = p;
        }
    }
    let range = max_db - min_db;

    // Draw spectrum
    let step = result.power_db.len() as f64 / width as f64;
    for x in 0..width {
        let idx = (x as f64 * step) as usize;
        if idx < result.power_db.len() {
            let normalized = ((result.power_db[idx] - min_db) / range).clamp(0.0, 1.0);
            let y = ((1.0 - normalized) * (height - 1) as f64) as u32;

            // Draw vertical line from bottom to y
            for py in y..height {
                img.put_pixel(x, py, Rgb([0, 100, 200]));
            }
        }
    }

    let mut buffer = Vec::new();
    let encoder = PngEncoder::new(&mut buffer);
    encoder.write_image(&img, width, height, ExtendedColorType::Rgb8)
        .expect("Failed to encode PNG");

    fs::write(path, &buffer).expect("Failed to write PNG");
    println!("    Created: {}", path);
}

#[cfg(not(feature = "image"))]
fn generate_spectrum_png(_samples: &[IQSample], _sample_rate: f64, path: &str) {
    println!("    (Skipping {} - image feature not enabled)", path);
}

#[cfg(feature = "image")]
fn generate_waterfall_png(samples: &[IQSample], sample_rate: f64, path: &str) {
    let mut generator = WaterfallGenerator::with_hop(256, 128);
    let result = generator.compute(samples, sample_rate);

    // Calculate power range
    let mut min_db = f64::INFINITY;
    let mut max_db = f64::NEG_INFINITY;
    for row in &result.power_db {
        for &p in row {
            if p > max_db {
                max_db = p;
            }
            if p < min_db && p > -200.0 {
                min_db = p;
            }
        }
    }

    let png_data = result.to_png(Colormap::Viridis, min_db, max_db);
    fs::write(path, &png_data).expect("Failed to write PNG");
    println!("    Created: {}", path);
}

#[cfg(not(feature = "image"))]
fn generate_waterfall_png(_samples: &[IQSample], _sample_rate: f64, path: &str) {
    println!("    (Skipping {} - image feature not enabled)", path);
}
