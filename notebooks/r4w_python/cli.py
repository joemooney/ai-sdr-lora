"""R4W CLI Wrapper

Provides Python interface to the R4W command-line tool.
"""

import json
import subprocess
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass
class WaveformInfo:
    """Information about a waveform."""
    name: str
    full_name: str
    modulation_type: str
    bits_per_symbol: int
    sample_rate: float
    symbol_rate: float


@dataclass
class SpectrumResult:
    """Result of spectrum analysis."""
    frequencies: np.ndarray
    power_db: np.ndarray
    sample_rate: float
    fft_size: int


@dataclass
class SignalStats:
    """Signal statistics."""
    mean_power_db: float
    peak_power_db: float
    papr_db: float
    dc_offset: complex
    num_samples: int


class R4WError(Exception):
    """Exception raised when R4W CLI fails."""
    pass


class R4W:
    """Python wrapper for R4W CLI."""

    _r4w_path: Optional[str] = None

    @classmethod
    def set_path(cls, path: str):
        """Set the path to the r4w executable."""
        cls._r4w_path = path

    @classmethod
    def _get_cmd(cls) -> str:
        """Get the r4w command."""
        if cls._r4w_path:
            return cls._r4w_path
        return "r4w"

    @classmethod
    def _run(cls, args: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run the r4w CLI with arguments."""
        cmd = [cls._get_cmd()] + args
        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                check=True,
            )
            return result
        except subprocess.CalledProcessError as e:
            raise R4WError(f"R4W command failed: {e.stderr}") from e
        except FileNotFoundError:
            raise R4WError(
                "R4W CLI not found. Install with 'cargo build --release --bin r4w' "
                "and add to PATH, or use R4W.set_path('/path/to/r4w')"
            )

    @classmethod
    def list_waveforms(cls) -> List[str]:
        """List all available waveforms."""
        result = cls._run(["waveform", "--list"])
        # Parse the output to extract waveform names
        waveforms = []
        for line in result.stdout.strip().split("\n"):
            if line.strip() and not line.startswith("Available"):
                # Handle format like "  BPSK - Binary Phase Shift Keying"
                name = line.strip().split(" - ")[0].strip()
                if name:
                    waveforms.append(name)
        return waveforms

    @classmethod
    def waveform_info(cls, name: str) -> WaveformInfo:
        """Get information about a waveform."""
        result = cls._run(["waveform", "--info", name, "--format", "json"])
        try:
            data = json.loads(result.stdout)
            return WaveformInfo(
                name=data.get("name", name),
                full_name=data.get("full_name", name),
                modulation_type=data.get("modulation_type", "unknown"),
                bits_per_symbol=data.get("bits_per_symbol", 1),
                sample_rate=data.get("sample_rate", 48000.0),
                symbol_rate=data.get("symbol_rate", 1000.0),
            )
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return WaveformInfo(
                name=name,
                full_name=name,
                modulation_type="unknown",
                bits_per_symbol=1,
                sample_rate=48000.0,
                symbol_rate=1000.0,
            )

    @classmethod
    def modulate(
        cls,
        waveform: str,
        data: bytes,
        sample_rate: float = 48000.0,
    ) -> np.ndarray:
        """Modulate data and return I/Q samples as numpy array.

        Args:
            waveform: Waveform name (e.g., "QPSK", "BPSK")
            data: Data bytes to modulate
            sample_rate: Sample rate in Hz

        Returns:
            Complex numpy array of I/Q samples
        """
        with tempfile.NamedTemporaryFile(suffix=".sigmf-data", delete=False) as f:
            data_path = Path(f.name)

        meta_path = data_path.with_suffix(".sigmf-meta")

        try:
            # Run modulation
            cls._run([
                "modulate",
                "--waveform", waveform,
                "--message", data.decode("utf-8", errors="replace"),
                "--sample-rate", str(sample_rate),
                "--output", str(meta_path),
            ])

            # Read samples
            samples = cls._read_sigmf(data_path)
            return samples
        finally:
            # Cleanup
            data_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)

    @classmethod
    def simulate(
        cls,
        waveform: str,
        message: str,
        snr_db: float = 20.0,
        sample_rate: float = 48000.0,
        channel_model: str = "awgn",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate transmission with channel effects.

        Args:
            waveform: Waveform name
            message: Message to transmit
            snr_db: SNR in dB
            sample_rate: Sample rate in Hz
            channel_model: Channel model ("awgn", "rayleigh", "rician")

        Returns:
            Tuple of (transmitted samples, received samples)
        """
        with tempfile.NamedTemporaryFile(suffix=".sigmf-data", delete=False) as f:
            tx_path = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".sigmf-data", delete=False) as f:
            rx_path = Path(f.name)

        tx_meta = tx_path.with_suffix(".sigmf-meta")
        rx_meta = rx_path.with_suffix(".sigmf-meta")

        try:
            # Run simulation
            cls._run([
                "simulate",
                "--waveform", waveform,
                "--message", message,
                "--snr", str(snr_db),
                "--sample-rate", str(sample_rate),
                "--channel", channel_model,
                "--output-tx", str(tx_meta),
                "--output-rx", str(rx_meta),
            ])

            tx_samples = cls._read_sigmf(tx_path)
            rx_samples = cls._read_sigmf(rx_path)
            return tx_samples, rx_samples
        finally:
            tx_path.unlink(missing_ok=True)
            rx_path.unlink(missing_ok=True)
            tx_meta.unlink(missing_ok=True)
            rx_meta.unlink(missing_ok=True)

    @classmethod
    def analyze_spectrum(
        cls,
        input_path: Union[str, Path],
        fft_size: int = 1024,
        window: str = "hann",
    ) -> SpectrumResult:
        """Analyze spectrum of a signal file.

        Args:
            input_path: Path to SigMF file
            fft_size: FFT size
            window: Window function

        Returns:
            SpectrumResult with frequencies and power
        """
        result = cls._run([
            "analyze", "spectrum",
            "--input", str(input_path),
            "--fft-size", str(fft_size),
            "--window", window,
            "--output-format", "json",
        ])

        data = json.loads(result.stdout)
        return SpectrumResult(
            frequencies=np.array(data.get("frequencies", [])),
            power_db=np.array(data.get("power_db", [])),
            sample_rate=data.get("sample_rate", 48000.0),
            fft_size=data.get("fft_size", fft_size),
        )

    @classmethod
    def analyze_stats(cls, input_path: Union[str, Path]) -> SignalStats:
        """Get signal statistics.

        Args:
            input_path: Path to SigMF file

        Returns:
            SignalStats with power, PAPR, etc.
        """
        result = cls._run([
            "analyze", "stats",
            "--input", str(input_path),
            "--format", "json",
        ])

        data = json.loads(result.stdout)
        return SignalStats(
            mean_power_db=data.get("mean_power_db", 0.0),
            peak_power_db=data.get("peak_power_db", 0.0),
            papr_db=data.get("papr_db", 0.0),
            dc_offset=complex(
                data.get("dc_offset_i", 0.0),
                data.get("dc_offset_q", 0.0),
            ),
            num_samples=data.get("num_samples", 0),
        )

    @classmethod
    def simulate_ber(
        cls,
        waveform: str,
        snr_range: range,
        num_bits: int = 10000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate BER over a range of SNR values.

        Args:
            waveform: Waveform name
            snr_range: Range of SNR values
            num_bits: Number of bits per SNR point

        Returns:
            Tuple of (snr_values, ber_values)
        """
        snr_values = list(snr_range)
        ber_values = []

        for snr in snr_values:
            result = cls._run([
                "ber",
                "--waveform", waveform,
                "--snr", str(snr),
                "--bits", str(num_bits),
                "--format", "json",
            ])
            data = json.loads(result.stdout)
            ber_values.append(data.get("ber", 0.0))

        return np.array(snr_values), np.array(ber_values)

    @classmethod
    def _read_sigmf(cls, data_path: Path) -> np.ndarray:
        """Read samples from a SigMF data file.

        Args:
            data_path: Path to .sigmf-data file

        Returns:
            Complex numpy array
        """
        with open(data_path, "rb") as f:
            raw_data = f.read()

        # Assume cf32_le format (complex float32 little endian)
        num_samples = len(raw_data) // 8  # 4 bytes I + 4 bytes Q
        samples = np.zeros(num_samples, dtype=np.complex64)

        for i in range(num_samples):
            offset = i * 8
            i_val, q_val = struct.unpack("<ff", raw_data[offset:offset + 8])
            samples[i] = complex(i_val, q_val)

        return samples

    @classmethod
    def generate_samples(
        cls,
        waveform: str,
        data: bytes,
        sample_rate: float = 48000.0,
    ) -> np.ndarray:
        """Generate I/Q samples for data (alias for modulate)."""
        return cls.modulate(waveform, data, sample_rate)
