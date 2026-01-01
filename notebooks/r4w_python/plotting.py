"""R4W Plotting Helpers

Visualization functions for SDR signal analysis in Jupyter notebooks.
"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# Viridis-like colormap for waterfalls
WATERFALL_CMAP = "viridis"


def plot_constellation(
    samples: np.ndarray,
    title: str = "Constellation Diagram",
    ax: Optional[plt.Axes] = None,
    color: str = "#0066CC",
    alpha: float = 0.6,
    marker_size: float = 10,
    show_axes: bool = True,
    show_grid: bool = True,
    figsize: Tuple[float, float] = (6, 6),
) -> plt.Axes:
    """Plot I/Q constellation diagram.

    Args:
        samples: Complex numpy array of I/Q samples
        title: Plot title
        ax: Matplotlib axes (creates new figure if None)
        color: Point color
        alpha: Point transparency
        marker_size: Size of constellation points
        show_axes: Show I/Q axes
        show_grid: Show grid
        figsize: Figure size

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Plot points
    ax.scatter(
        samples.real,
        samples.imag,
        c=color,
        alpha=alpha,
        s=marker_size,
        edgecolors="none",
    )

    # Add axes through origin
    if show_axes:
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.axvline(x=0, color="gray", linewidth=0.5)

    # Styling
    ax.set_xlabel("In-Phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title(title)
    ax.set_aspect("equal")

    if show_grid:
        ax.grid(True, alpha=0.3)

    # Set equal limits
    max_val = max(np.max(np.abs(samples.real)), np.max(np.abs(samples.imag))) * 1.2
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    return ax


def plot_spectrum(
    power_db: np.ndarray,
    frequencies: Optional[np.ndarray] = None,
    sample_rate: float = 48000.0,
    title: str = "Power Spectrum",
    ax: Optional[plt.Axes] = None,
    color: str = "#0066CC",
    fill: bool = True,
    figsize: Tuple[float, float] = (10, 4),
) -> plt.Axes:
    """Plot power spectrum.

    Args:
        power_db: Power in dB
        frequencies: Frequency axis (computed if None)
        sample_rate: Sample rate for frequency axis
        title: Plot title
        ax: Matplotlib axes
        color: Line color
        fill: Fill under curve
        figsize: Figure size

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if frequencies is None:
        n = len(power_db)
        frequencies = np.fft.fftshift(np.fft.fftfreq(n, 1 / sample_rate))

    # Sort by frequency for proper plotting
    sort_idx = np.argsort(frequencies)
    frequencies = frequencies[sort_idx]
    power_db = power_db[sort_idx]

    # Plot
    if fill:
        ax.fill_between(frequencies / 1000, power_db, alpha=0.3, color=color)
    ax.plot(frequencies / 1000, power_db, color=color, linewidth=1)

    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


def plot_waterfall(
    power_db: np.ndarray,
    sample_rate: float = 48000.0,
    fft_size: int = 256,
    hop_size: int = 128,
    title: str = "Waterfall / Spectrogram",
    ax: Optional[plt.Axes] = None,
    cmap: str = WATERFALL_CMAP,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Axes:
    """Plot waterfall/spectrogram display.

    Args:
        power_db: 2D array (time x frequency) of power in dB
        sample_rate: Sample rate
        fft_size: FFT size
        hop_size: Hop size between FFTs
        title: Plot title
        ax: Matplotlib axes
        cmap: Colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        figsize: Figure size

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Compute extents
    time_per_frame = hop_size / sample_rate
    total_time = power_db.shape[0] * time_per_frame

    # Plot
    im = ax.imshow(
        power_db,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[
            -sample_rate / 2000,  # Left (kHz)
            sample_rate / 2000,   # Right (kHz)
            0,                     # Bottom (s)
            total_time * 1000,     # Top (ms)
        ],
    )

    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Time (ms)")
    ax.set_title(title)

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Power (dB)")

    return ax


def plot_time_domain(
    samples: np.ndarray,
    sample_rate: float = 48000.0,
    title: str = "Time Domain",
    ax: Optional[plt.Axes] = None,
    show_i: bool = True,
    show_q: bool = True,
    show_envelope: bool = False,
    figsize: Tuple[float, float] = (10, 4),
) -> plt.Axes:
    """Plot time domain signal.

    Args:
        samples: Complex numpy array
        sample_rate: Sample rate in Hz
        title: Plot title
        ax: Matplotlib axes
        show_i: Show I component
        show_q: Show Q component
        show_envelope: Show amplitude envelope
        figsize: Figure size

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    time_ms = np.arange(len(samples)) / sample_rate * 1000

    if show_i:
        ax.plot(time_ms, samples.real, label="I", alpha=0.8)
    if show_q:
        ax.plot(time_ms, samples.imag, label="Q", alpha=0.8)
    if show_envelope:
        ax.plot(time_ms, np.abs(samples), label="Envelope", color="black", alpha=0.5)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_ber_curve(
    snr_values: np.ndarray,
    ber_values: np.ndarray,
    label: str = "Measured",
    ax: Optional[plt.Axes] = None,
    theoretical: Optional[np.ndarray] = None,
    theoretical_label: str = "Theoretical",
    color: str = "#0066CC",
    figsize: Tuple[float, float] = (8, 6),
) -> plt.Axes:
    """Plot BER vs SNR curve.

    Args:
        snr_values: SNR values in dB
        ber_values: Corresponding BER values
        label: Legend label
        ax: Matplotlib axes
        theoretical: Theoretical BER values (optional)
        theoretical_label: Label for theoretical curve
        color: Line color
        figsize: Figure size

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Plot measured
    ax.semilogy(snr_values, ber_values, "o-", label=label, color=color)

    # Plot theoretical if provided
    if theoretical is not None:
        ax.semilogy(
            snr_values, theoretical, "--",
            label=theoretical_label, color="gray", alpha=0.7
        )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title("BER vs SNR")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # Set reasonable limits
    ax.set_ylim(1e-6, 1)

    return ax


def plot_multi_constellation(
    samples_list: List[np.ndarray],
    labels: List[str],
    title: str = "Constellation Comparison",
    figsize: Tuple[float, float] = (12, 4),
) -> plt.Figure:
    """Plot multiple constellation diagrams side by side.

    Args:
        samples_list: List of complex sample arrays
        labels: Labels for each constellation
        title: Overall title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n = len(samples_list)
    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for ax, samples, label in zip(axes, samples_list, labels):
        plot_constellation(samples, title=label, ax=ax)

    fig.suptitle(title)
    fig.tight_layout()

    return fig


def plot_ber_comparison(
    snr_values: np.ndarray,
    ber_dict: dict,
    title: str = "BER Comparison",
    figsize: Tuple[float, float] = (8, 6),
) -> plt.Axes:
    """Plot multiple BER curves for comparison.

    Args:
        snr_values: SNR values in dB
        ber_dict: Dictionary of {label: ber_values}
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib axes
    """
    _, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(ber_dict)))

    for (label, ber_values), color in zip(ber_dict.items(), colors):
        ax.semilogy(snr_values, ber_values, "o-", label=label, color=color)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(1e-6, 1)

    return ax


def compute_spectrum(
    samples: np.ndarray,
    fft_size: int = 1024,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute power spectrum from samples.

    Args:
        samples: Complex numpy array
        fft_size: FFT size
        window: Window function

    Returns:
        Tuple of (frequencies normalized -0.5 to 0.5, power_db)
    """
    # Apply window
    if window == "hann":
        win = np.hanning(fft_size)
    elif window == "hamming":
        win = np.hamming(fft_size)
    elif window == "blackman":
        win = np.blackman(fft_size)
    else:
        win = np.ones(fft_size)

    # Compute FFT
    windowed = samples[:fft_size] * win
    spectrum = np.fft.fftshift(np.fft.fft(windowed))
    power = np.abs(spectrum) ** 2 / fft_size
    power_db = 10 * np.log10(power + 1e-12)

    frequencies = np.linspace(-0.5, 0.5, fft_size)

    return frequencies, power_db


def compute_waterfall(
    samples: np.ndarray,
    fft_size: int = 256,
    hop_size: int = 128,
    window: str = "hann",
) -> np.ndarray:
    """Compute waterfall/spectrogram.

    Args:
        samples: Complex numpy array
        fft_size: FFT size
        hop_size: Samples between FFTs
        window: Window function

    Returns:
        2D array (time x frequency) of power in dB
    """
    # Apply window
    if window == "hann":
        win = np.hanning(fft_size)
    elif window == "hamming":
        win = np.hamming(fft_size)
    else:
        win = np.ones(fft_size)

    num_frames = (len(samples) - fft_size) // hop_size + 1
    waterfall = np.zeros((num_frames, fft_size))

    for i in range(num_frames):
        start = i * hop_size
        frame = samples[start:start + fft_size] * win
        spectrum = np.fft.fftshift(np.fft.fft(frame))
        power = np.abs(spectrum) ** 2 / fft_size
        waterfall[i] = 10 * np.log10(power + 1e-12)

    return waterfall
