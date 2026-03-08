import numpy as np

def create_sliding_windows(data, window_size, step_size=1):
    """
    Efficiently creates sliding windows from a 2D numpy array.
    Uses numpy's `as_strided` for zero-copy views (memory efficient),
    or fallback indexing if `as_strided` is deemed too complex for generic use.
    
    Args:
        data (np.ndarray): Shape (num_samples, num_features).
        window_size (int): The number of time steps per window.
        step_size (int): The stride between consecutive windows.
        
    Returns:
        np.ndarray: Shape (num_windows, window_size, num_features)
    """
    num_samples, num_features = data.shape
    
    if num_samples < window_size:
        raise ValueError(f"Data length ({num_samples}) is less than window size ({window_size}).")
        
    # Calculate the number of standard windows we can make
    num_windows = (num_samples - window_size) // step_size + 1
    
    # Efficient strided view
    shape = (num_windows, window_size, num_features)
    strides = (data.strides[0] * step_size, data.strides[0], data.strides[1])
    
    windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    
    # Note: `as_strided` returns a view. If the user plans to mutate it heavily
    # or pass it to standard routines expecting contiguous arrays, it's safer to copy.
    # For dataloaders, returning the view directly can save immense RAM on large datasets.
    return windows

def create_forecasting_windows(data, in_window_size, out_window_size, step_size=1):
    """
    Creates input features (X) and target outputs (Y) for forecasting models.
    
    Args:
        data (np.ndarray): Shape (num_samples, num_features).
        in_window_size (int): Length of the input observation window.
        out_window_size (int): Length of the target forecast window.
        step_size (int): The stride between consecutive windows.
        
    Returns:
        tuple: (X, Y)
            X shape: (num_windows, in_window_size, num_features)
            Y shape: (num_windows, out_window_size, num_features)
    """
    num_samples = data.shape[0]
    total_window_size = in_window_size + out_window_size
    
    if num_samples < total_window_size:
        raise ValueError("Data length shorter than combined in/out window sizes.")
        
    # Create windows of length (in_window_size + out_window_size)
    combined_windows = create_sliding_windows(data, total_window_size, step_size)
    
    # Split into X and Y
    X = combined_windows[:, :in_window_size, :]
    Y = combined_windows[:, in_window_size:, :]
    
    return X, Y
