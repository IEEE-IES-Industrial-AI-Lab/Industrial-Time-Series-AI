import numpy as np

def extract_statistical_features(window_data):
    """
    Extracts basic statistical features from a time-series window.
    
    Args:
        window_data (np.ndarray): Shape (window_size, num_features)
        
    Returns:
        dict: A dictionary of statistical metrics mapped by feature index.
    """
    if window_data.ndim != 2:
        raise ValueError("Expected window_data to be 2-dimensional (window_size, num_features)")
        
    features = {}
    
    # Calculate across the temporal dimension (axis=0)
    features['mean'] = np.mean(window_data, axis=0)
    features['std'] = np.std(window_data, axis=0)
    features['var'] = np.var(window_data, axis=0)
    features['min'] = np.min(window_data, axis=0)
    features['max'] = np.max(window_data, axis=0)
    
    # Range
    features['range'] = features['max'] - features['min']
    
    # RMS (Root Mean Square)
    features['rms'] = np.sqrt(np.mean(np.square(window_data), axis=0))
    
    return features

def extract_frequency_features(window_data, sampling_rate=1.0):
    """
    Extracts frequency-domain features using FFT.
    Useful for high-frequency industrial sensors (e.g. vibrations, current).
    
    Args:
        window_data (np.ndarray): Shape (window_size, num_features)
        sampling_rate (float): The sampling frequency of the sensor.
        
    Returns:
        dict: Frequency domain features mapped by feature index.
    """
    window_size = window_data.shape[0]
    fft_vals = np.fft.rfft(window_data, axis=0)
    fft_mag = np.abs(fft_vals) / window_size
    
    freqs = np.fft.rfftfreq(window_size, d=1.0/sampling_rate)
    
    features = {}
    features['peak_freq_mag'] = np.max(fft_mag, axis=0)
    
    # Find the dominant frequency
    dominant_indices = np.argmax(fft_mag, axis=0)
    features['dominant_freq'] = freqs[dominant_indices]
    
    # Spectral energy
    features['spectral_energy'] = np.sum(np.square(fft_mag), axis=0)
    
    return features
