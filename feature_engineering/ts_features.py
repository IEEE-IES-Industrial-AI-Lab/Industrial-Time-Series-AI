"""
Time-series feature engineering for industrial sensor data.

Provides statistical, frequency-domain, wavelet, entropy, and
trend/seasonality decomposition features used in industrial AI literature.

All functions accept window arrays of shape (window_size, num_features)
and return feature dictionaries keyed by feature name, each value being a
1-D array of shape (num_features,).
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Statistical features
# ---------------------------------------------------------------------------

def extract_statistical_features(window_data: np.ndarray) -> dict:
    """Extract time-domain statistical features.

    Args:
        window_data: Shape (window_size, num_features).

    Returns:
        Dict with keys: mean, std, var, min, max, range, rms, skewness, kurtosis,
        median, iqr, zero_crossing_rate, peak_to_peak.
    """
    if window_data.ndim != 2:
        raise ValueError("Expected window_data to be 2-dimensional (window_size, num_features)")

    from scipy.stats import skew, kurtosis as scipy_kurtosis

    features: dict = {}

    features["mean"] = np.mean(window_data, axis=0)
    features["std"] = np.std(window_data, axis=0)
    features["var"] = np.var(window_data, axis=0)
    features["min"] = np.min(window_data, axis=0)
    features["max"] = np.max(window_data, axis=0)
    features["range"] = features["max"] - features["min"]
    features["rms"] = np.sqrt(np.mean(np.square(window_data), axis=0))
    features["median"] = np.median(window_data, axis=0)

    q75 = np.percentile(window_data, 75, axis=0)
    q25 = np.percentile(window_data, 25, axis=0)
    features["iqr"] = q75 - q25

    features["skewness"] = skew(window_data, axis=0, bias=False)
    features["kurtosis"] = scipy_kurtosis(window_data, axis=0, bias=False)

    # Zero-crossing rate (normalised by window length)
    signs = np.sign(window_data)
    zcr = np.sum(np.diff(signs, axis=0) != 0, axis=0).astype(float)
    features["zero_crossing_rate"] = zcr / (window_data.shape[0] - 1)

    features["peak_to_peak"] = np.max(window_data, axis=0) - np.min(window_data, axis=0)

    return features


# ---------------------------------------------------------------------------
# Frequency features (FFT)
# ---------------------------------------------------------------------------

def extract_frequency_features(
    window_data: np.ndarray, sampling_rate: float = 1.0
) -> dict:
    """Extract frequency-domain features using FFT.

    Useful for vibration, current, and voltage industrial sensors.

    Args:
        window_data:   Shape (window_size, num_features).
        sampling_rate: Sensor sampling frequency in Hz.

    Returns:
        Dict with keys: peak_freq_mag, dominant_freq, spectral_energy,
        spectral_entropy, spectral_centroid, spectral_bandwidth.
    """
    if window_data.ndim != 2:
        raise ValueError("Expected window_data to be 2-dimensional (window_size, num_features)")

    window_size = window_data.shape[0]
    fft_vals = np.fft.rfft(window_data, axis=0)
    fft_mag = np.abs(fft_vals) / window_size
    freqs = np.fft.rfftfreq(window_size, d=1.0 / sampling_rate)

    features: dict = {}
    features["peak_freq_mag"] = np.max(fft_mag, axis=0)

    dominant_indices = np.argmax(fft_mag, axis=0)
    features["dominant_freq"] = freqs[dominant_indices]

    features["spectral_energy"] = np.sum(np.square(fft_mag), axis=0)

    # Spectral entropy: normalise power spectrum and compute entropy
    power = np.square(fft_mag) + 1e-12
    power_norm = power / power.sum(axis=0, keepdims=True)
    features["spectral_entropy"] = -np.sum(power_norm * np.log(power_norm + 1e-12), axis=0)

    # Spectral centroid: weighted mean frequency
    features["spectral_centroid"] = np.sum(
        freqs[:, np.newaxis] * fft_mag, axis=0
    ) / (np.sum(fft_mag, axis=0) + 1e-8)

    # Spectral bandwidth
    centroid = features["spectral_centroid"]
    features["spectral_bandwidth"] = np.sqrt(
        np.sum(
            ((freqs[:, np.newaxis] - centroid[np.newaxis, :]) ** 2) * fft_mag,
            axis=0,
        )
        / (np.sum(fft_mag, axis=0) + 1e-8)
    )

    return features


# ---------------------------------------------------------------------------
# Wavelet features
# ---------------------------------------------------------------------------

def extract_wavelet_features(
    window_data: np.ndarray,
    widths: Optional[np.ndarray] = None,
    wavelet: str = "morl",
) -> dict:
    """Extract continuous wavelet transform (CWT) energy features.

    Decomposes each channel with the Morlet wavelet and returns energy
    per scale band — useful for multi-scale industrial fault detection.

    Args:
        window_data: Shape (window_size, num_features).
        widths:      Array of scale widths for CWT. Defaults to 1–32 (32 scales).
        wavelet:     SciPy wavelet name (default "morl" = Morlet).

    Returns:
        Dict with keys: cwt_energy_<scale_idx> for each scale band.
    """
    from scipy.signal import cwt, morlet2

    if window_data.ndim != 2:
        raise ValueError("Expected window_data to be 2-dimensional (window_size, num_features)")

    if widths is None:
        widths = np.arange(1, 33, dtype=float)

    features: dict = {}
    num_features = window_data.shape[1]

    # Bin scales into 4 bands: very fine, fine, medium, coarse
    band_edges = np.array_split(np.arange(len(widths)), 4)

    for feat_idx in range(num_features):
        signal = window_data[:, feat_idx]
        coef = cwt(signal, morlet2, widths)  # (num_scales, window_size)
        energy = np.sum(np.abs(coef) ** 2, axis=1)  # (num_scales,)

        for band_idx, band_indices in enumerate(band_edges):
            key = f"cwt_band{band_idx}_feat{feat_idx}"
            features[key] = float(energy[band_indices].sum())

    # Summarise across features: mean energy per band
    for band_idx in range(4):
        band_keys = [f"cwt_band{band_idx}_feat{i}" for i in range(num_features)]
        features[f"cwt_mean_band{band_idx}"] = np.mean([features[k] for k in band_keys])

    return features


# ---------------------------------------------------------------------------
# Entropy features
# ---------------------------------------------------------------------------

def _approximate_entropy(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Approximate Entropy (ApEn) of a 1-D signal.

    Lower values indicate more regularity; higher values indicate complexity.
    r is specified as a fraction of the signal standard deviation.
    """
    N = len(signal)
    r_threshold = r * np.std(signal) + 1e-8

    def _phi(m_val: int) -> float:
        templates = np.array([signal[i : i + m_val] for i in range(N - m_val + 1)])
        count = np.sum(
            np.max(np.abs(templates[:, np.newaxis] - templates[np.newaxis, :]), axis=-1)
            <= r_threshold,
            axis=-1,
        )
        return float(np.mean(np.log(count / (N - m_val + 1) + 1e-8)))

    return abs(_phi(m) - _phi(m + 1))


def extract_entropy_features(
    window_data: np.ndarray, m: int = 2, r: float = 0.2
) -> dict:
    """Extract entropy-based complexity features.

    Includes Approximate Entropy (ApEn) and Sample Entropy (SampEn proxy).
    These are sensitive to regularity changes caused by industrial faults.

    Args:
        window_data: Shape (window_size, num_features).
        m: Embedding dimension for ApEn.
        r: Tolerance as fraction of signal std.

    Returns:
        Dict with keys: approx_entropy, permutation_entropy.
    """
    if window_data.ndim != 2:
        raise ValueError("Expected window_data to be 2-dimensional (window_size, num_features)")

    num_features = window_data.shape[1]

    apen = np.array([_approximate_entropy(window_data[:, i], m=m, r=r) for i in range(num_features)])

    # Permutation entropy (fast, parameter-free)
    def _perm_entropy(signal: np.ndarray, order: int = 3) -> float:
        N = len(signal)
        patterns: dict = {}
        for i in range(N - order + 1):
            pattern = tuple(np.argsort(signal[i : i + order]))
            patterns[pattern] = patterns.get(pattern, 0) + 1
        total = sum(patterns.values())
        probs = np.array(list(patterns.values())) / total
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    perm_ent = np.array([_perm_entropy(window_data[:, i]) for i in range(num_features)])

    return {
        "approx_entropy": apen,
        "permutation_entropy": perm_ent,
    }


# ---------------------------------------------------------------------------
# Trend / seasonality decomposition
# ---------------------------------------------------------------------------

def extract_trend_seasonality(
    window_data: np.ndarray,
    period: int = 24,
    model: str = "additive",
) -> dict:
    """Decompose each channel into trend, seasonal, and residual components.

    Wraps statsmodels STL decomposition. Returns the mean and variance of
    each component — useful as features for downstream classification.

    Args:
        window_data: Shape (window_size, num_features).
        period:      Seasonal period (e.g. 24 for hourly data with daily cycle).
        model:       "additive" or "multiplicative".

    Returns:
        Dict with keys: trend_mean, trend_std, seasonal_mean, seasonal_std,
        residual_mean, residual_std per feature.
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError as exc:
        raise ImportError("statsmodels is required: pip install statsmodels") from exc

    if window_data.ndim != 2:
        raise ValueError("Expected window_data to be 2-dimensional (window_size, num_features)")

    window_size, num_features = window_data.shape
    if window_size < 2 * period:
        raise ValueError(
            f"window_size ({window_size}) must be >= 2 * period ({period}) for decomposition."
        )

    trend_means, trend_stds = [], []
    seasonal_means, seasonal_stds = [], []
    residual_means, residual_stds = [], []

    for i in range(num_features):
        result = seasonal_decompose(
            window_data[:, i], model=model, period=period, extrapolate_trend="freq"
        )
        trend_means.append(np.mean(result.trend))
        trend_stds.append(np.std(result.trend))
        seasonal_means.append(np.mean(result.seasonal))
        seasonal_stds.append(np.std(result.seasonal))
        residual_means.append(np.mean(result.resid))
        residual_stds.append(np.std(result.resid))

    return {
        "trend_mean": np.array(trend_means),
        "trend_std": np.array(trend_stds),
        "seasonal_mean": np.array(seasonal_means),
        "seasonal_std": np.array(seasonal_stds),
        "residual_mean": np.array(residual_means),
        "residual_std": np.array(residual_stds),
    }


# ---------------------------------------------------------------------------
# Combined extractor
# ---------------------------------------------------------------------------

def extract_all_features(
    window_data: np.ndarray,
    sampling_rate: float = 1.0,
    include_wavelet: bool = False,
    include_entropy: bool = True,
) -> dict:
    """Convenience wrapper that extracts statistical and frequency features.

    Wavelet and entropy features are optional (slower) and can be enabled
    with their respective flags.

    Args:
        window_data:     Shape (window_size, num_features).
        sampling_rate:   Sensor sampling frequency.
        include_wavelet: Compute CWT-based wavelet energy features.
        include_entropy: Compute ApEn and permutation entropy.

    Returns:
        Merged dict of all selected feature groups.
    """
    features = {}
    features.update(extract_statistical_features(window_data))
    features.update(extract_frequency_features(window_data, sampling_rate))
    if include_wavelet:
        features.update(extract_wavelet_features(window_data))
    if include_entropy:
        features.update(extract_entropy_features(window_data))
    return features
