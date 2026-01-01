# R4W Tutorial Notebooks

Interactive Jupyter notebooks for learning SDR concepts with R4W.

## Getting Started

### Prerequisites

1. **Install R4W CLI** (must be in PATH):
   ```bash
   cargo build --release --bin r4w
   export PATH="$PATH:$(pwd)/target/release"
   ```

2. **Create Python environment**:
   ```bash
   cd notebooks
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Start Jupyter**:
   ```bash
   jupyter lab
   ```

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | I/Q Basics | Complex numbers, I/Q samples, visualization |
| 02 | Modulation | AM, FM, PSK fundamentals |
| 03 | Spectrum Analysis | FFT, windowing, spectrograms |
| 04 | Channel Effects | AWGN, fading, Doppler |
| 05 | LoRa Deep Dive | CSS modulation, spreading factors |
| 06 | BER Simulation | Monte Carlo error rate simulation |
| 07 | Waveform Comparison | Performance comparison across waveforms |
| 08 | Mesh Networking | Meshtastic simulation |

## Python API

The `r4w_python` module provides a Python wrapper around the R4W CLI:

```python
from r4w_python import R4W

# Get waveform info
info = R4W.waveform_info("QPSK")

# Generate I/Q samples
samples = R4W.modulate("QPSK", b"Hello World!", sample_rate=48000)

# Analyze spectrum
spectrum = R4W.analyze_spectrum("signal.sigmf-meta", fft_size=1024)

# Run BER simulation
results = R4W.simulate_ber("QPSK", snr_range=range(0, 20))
```

## Directory Structure

```
notebooks/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── r4w_python/
│   ├── __init__.py            # Package init
│   ├── cli.py                 # CLI wrapper
│   └── plotting.py            # Visualization helpers
├── 01_iq_basics.ipynb
├── 02_modulation.ipynb
├── 03_spectrum_analysis.ipynb
├── 04_channel_effects.ipynb
├── 05_lora_deep_dive.ipynb
├── 06_ber_simulation.ipynb
├── 07_waveform_comparison.ipynb
└── 08_mesh_networking.ipynb
```
