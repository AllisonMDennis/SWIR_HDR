import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import io
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

"""
This module contains functions for High Dynamic Range (HDR) image processing,
including camera response function estimation, radiance map computation,
and saving the linear radiance map as a npy array. 8-bit HDR image generation 
and presentation is handled in a separate code module.
"""

def precompute_zmax(images, Smax, Sd, bias, exposure_times, data_type = "clipdenoise"):
    """
    Precompute Zmax for all pixels and exposure times.
    
    Args:
    Smax (numpy.ndarray): Array of saturation levels for each pixel.
    Sd (numpy.ndarray): Array of dark current slopes for each pixel.
    bias (numpy.ndarray): Array of bias values for each pixel.
    exposure_times (numpy.ndarray): Array of exposure times.
    
    Returns:
    numpy.ndarray: Array of Zmax values with shape [num_exposure_times, height, width]
    """
    is_clipped = "clip" in data_type
    is_cliptop = "cliptop" in data_type
    is_denoised = "denoise" in data_type
    is_raw = "raw" in data_type
    num_exposures = len(exposure_times)
    height, width = Smax.shape

    # Reshape arrays for broadcasting
    Smax = Smax.reshape(1, height, width)
    Sd = Sd.reshape(1, height, width)
    bias = bias.reshape(1, height, width)
    exposure_times = exposure_times.reshape(num_exposures, 1, 1)
    maximages = np.max(images)
    #broadcast maximages
    maximages = np.full([len(exposure_times), height, width], maximages)

    # Compute Zmax for all pixels and exposure times
    if is_clipped and is_denoised:
        Zmax = Smax - (Sd * exposure_times + bias)
        Zmin = np.zeros((num_exposures, height, width))
    elif is_cliptop:
        Zmax = np.repeat(Smax, len(exposure_times), axis=0)
        Zmin = np.zeros((num_exposures, height, width))
    elif is_denoised:
        Zmax = maximages 
        Zmin = np.zeros((num_exposures, height, width))
    elif is_clipped:
        Zmax = np.repeat(Smax, len(exposure_times), axis=0)
        Zmin = Sd*exposure_times+bias
    elif is_raw:
        Zmax = maximages
        Zmin = np.zeros((num_exposures, height, width)) 

    return Zmax, Zmin

def debevec(z, Zmax, Zmin):
    """
    Compute the weighting function for each pixel using the Debevec weighting function.
    """
    
    # Ensure z and Zmax are scalars or have the same shape

    if np.isscalar(z) and np.isscalar(Zmax):
        middle = (Zmax - Zmin) / 2
        if z <= middle:
            return np.divide(z - Zmin, middle - Zmin) 
        else:
            return np.divide(Zmax - z,Zmax - middle)
    else:
        z = np.atleast_2d(z)
        Zmax = np.atleast_2d(Zmax)
        if z.shape != Zmax.shape:
            Zmax = np.full_like(z, Zmax)
        
        weight = np.zeros_like(z, dtype=np.float32)
        middle = (Zmax - Zmin) / 2
        
        weight[z <= middle] = np.divide(z[z <= middle] - Zmin[z <= middle],middle[z <= middle] - Zmin[z <= middle])
        weight[z > middle] = np.divide(Zmax[z > middle] - z[z > middle],Zmax[z > middle] - middle[z > middle])
    return weight

def robertson(z, Zmax, Zmin):
    """
    Compute the weighting function for each pixel using a Gaussian function according to Robertson (2010)
    centered at the midpoint between 0 and Zmax.
    
    Args:
        z: Input pixel values (scalar or array)
        Zmax: Maximum possible pixel value (scalar or array)
        
    Returns:
        Weights between 0 and 1, with maximum weight at the midpoint
    """

    
    # Handle scalar inputs
    if np.isscalar(z) and np.isscalar(Zmax):
        middle = (Zmax + Zmin) / 2
        # Using standard deviation of 1/4 of the range for a reasonable spread
        sigma = (Zmax - Zmin) / 4
        # Gaussian function
        weight = np.exp(-((z - middle) ** 2) / (2 * sigma ** 2))
        return weight
    
    # Handle array inputs
    else:
        z = np.atleast_2d(z)
        Zmax = np.atleast_2d(Zmax)
        if z.shape != Zmax.shape:
            Zmax = np.full_like(z, Zmax)
            
        middle = (Zmax + Zmin) / 2
        # Using standard deviation of 1/4 of the range for a reasonable spread
        sigma = (Zmax - Zmin) / 4
        # Gaussian function
        weight = np.exp(-((z - middle) ** 2) / (2 * sigma ** 2))
        return weight

def broadhat(z, Zmax, Zmin):
    """
    Compute the weighting function for each pixel using a broadhat function.
    
    Args:
    z (numpy.ndarray): Pixel intensity values
    Zmax (numpy.ndarray): Maximum pixel intensity values (same shape as z)
    
    Returns:
    numpy.ndarray: Weights calculated using the broadhat function
    """
    x = np.divide(z-Zmin, Zmax) #

    return np.maximum(0, 1 - ((x / 0.5) - 1)**12)

def vinegoni(z, Zmax, Zmin):
    # Vinegoni et al., Nat Commun 7, 11077 (2016)
    """
    No weighting function, returns 1.
    """
    return 1

def load_data(directory, base_data_folder):
    """Load data based on filename tags."""
    final_data_folder = os.path.join(directory, base_data_folder, "processed_data")
    data_dict = {}
    
    for file in os.listdir(final_data_folder):
        if file.endswith(".npy"):

            is_clipped = "clip" in file
            is_cliptop = "cliptop" in file
            is_denoised = "denoise" in file
            is_raw = "raw" in file
            
            
            if is_clipped:
                key = file.split("_clip")[0]  
            elif is_denoised: 
                key = file.split("_denoise")[0] 
            elif is_raw:
                key = file.split("_raw")[0]
            
            file_path = os.path.join(final_data_folder, file)
            data_type = []
            
            if is_cliptop: data_type.append("cliptop")
            elif is_clipped: data_type.append("clip")
            if is_denoised: data_type.append("denoise")
            if is_raw: data_type.append("raw")
            
            data_dict[key] = {
                'data': np.load(file_path),
                'type': "_".join(data_type)
            }
    
    return data_dict


"""
    Process HDR images from the given directory and experiment title.

    Parameters
    ----------
    directory : str
        The directory containing the experiment data.
    experiment_title : str
        The title of the experiment.
    base_data_folder : str
        The base folder containing the experiment data.
    coefficients_dict : dict
        A dictionary containing the coefficients for the camera response function.
    smoothing_lambda : float, optional
        The smoothing parameter for the camera response function. Default is 1000.
    weighting_function : callable, optional
        The weighting function for the camera response function. Default is debevec.
    num_sets : int, optional
        The number of sets to process. If None, all sets will be processed.

    Returns
    -------
    processed_data : list
        A list of dictionaries containing the processed HDR images for each set.

    Notes
    -----
    This function assumes that the data is stored in the following format:

    directory/experiment_title/base_data_folder/data_type/image001.npy
    directory/experiment_title/base_data_folder/data_type/image001_exposure_time.npy

    The function will create a folder called "final_data" in the base_data_folder and save the
    processed data in the following format:

    final_data/key_radiance_map_data_type_weighting_function.npy

    The processed data is a dictionary containing the following keys:

    key : str
        The key for the set of data.
    data_type : str
        The type of data (e.g. "dark", "bright", etc.).
    radiance_map : numpy array
        The radiance map for the set of data.
    response_curve : numpy array
        The response curve for the set of data.
    z_min : int
        The minimum intensity value for the set of data.
    z_max : int
        The maximum intensity value for the set of data.
    intensity_samples : numpy array
        The intensity samples for the set of data.
    log_exposures : numpy array
        The log exposures for the set of data.

    """


def computeRadianceMap(images, exposure_times, Zmax_precomputed, Zmin_precomputed,
                      return_all=True, crf="default", weighting_function=debevec, method = "default"):
    """
    Compute the radiance map from multiple exposures.

    Args:
        images: Input images
        exposure_times: Exposure times for each image
        Zmax_precomputed: Precomputed maximum intensity values
        return_all: Whether to return additional information
        crf: Pre-computed camera response function (numpy array). Required.
        weighting_function: Function to compute weights
        method: "default" or "adaptive"

    Returns:
        Radiance map and optionally additional information

    Raises:
        ValueError: If a pre-computed CRF array is not provided via the crf parameter
    """
    if not isinstance(crf, np.ndarray):
        raise ValueError(
            "A pre-computed CRF (numpy array) must be provided via the 'crf' parameter. "
            "Use run_calibration in Step0_calibration to generate crf.npy."
        )
    response_curve = crf

    z_min = np.min(Zmin_precomputed)
    z_max = np.median(Zmax_precomputed)

    num_images, height, width = images.shape
    radiance_map = np.zeros((height, width), dtype=np.float32)
    sum_weights = np.zeros((height, width), dtype=np.float32)

    # Per-pixel exposure-usage diagnostics are only populated by the adaptive
    # method; default to None so the return tuple is always well-defined.
    exposure_count = min_exp = max_exp = None

    if method == "default":
        for i in range(num_images):
            w = weighting_function(images[i], Zmax_precomputed[i], Zmin_precomputed[i])
            #clip pixels below z_min
            indices = np.clip(np.round(images[i] - z_min).astype(int), 0, len(response_curve) - 1)
            radiance_map += w * (response_curve[indices] - np.log(exposure_times[i]))
            sum_weights += w
        radiance_map = np.where(sum_weights > 0, radiance_map / sum_weights, 0)
    elif method == "adaptive":
        # Adaptive per-pixel exposure selection (manuscript Section 2.6).
        # Exposures are assumed sorted shortest -> longest (guaranteed by Step 1).
        #   z* > 0.95 Zmax  -> that exposure and all LONGER ones are excluded.
        #   z* < 0.05 Zmax  -> that exposure and all SHORTER ones are excluded
        #                      (Zmin is 0 after DC subtraction).
        # The surviving exposures form a contiguous block per pixel.
        sat_thresh   = 0.95 * Zmax_precomputed
        noise_thresh = 0.05 * Zmax_precomputed

        saturated = images > sat_thresh      # (num_images, H, W)
        too_dark  = images < noise_thresh

        # First saturated exposure (short -> long); everything from it onward is excluded.
        sat_any    = saturated.any(axis=0)
        sat_cutoff = np.where(sat_any, np.argmax(saturated, axis=0), num_images)

        # Last below-noise exposure; everything up to and including it is excluded.
        dark_any     = too_dark.any(axis=0)
        last_dark    = (num_images - 1) - np.argmax(too_dark[::-1], axis=0)
        noise_cutoff = np.where(dark_any, last_dark, -1)

        idx_grid = np.arange(num_images)[:, None, None]
        exposure_masks = (idx_grid > noise_cutoff[None]) & (idx_grid < sat_cutoff[None])

        has_valid     = exposure_masks.any(axis=0)
        fallback_mask = ~has_valid

        # Fallback: use the single exposure closest to the midpoint of the range.
        t_mid        = 0.5 * (exposure_times.min() + exposure_times.max())
        fallback_idx = int(np.argmin(np.abs(exposure_times - t_mid)))
        exposure_masks[fallback_idx] |= fallback_mask

        # Per-pixel exposure-usage diagnostics.
        exposure_count = exposure_masks.sum(axis=0).astype(int)
        min_exp = np.min(np.where(exposure_masks, idx_grid, num_images), axis=0).astype(float)
        max_exp = np.max(np.where(exposure_masks, idx_grid, -1), axis=0).astype(float)

        # HDR fusion on the selected exposures only (same equation as default).
        weight_sum        = np.zeros((height, width), dtype=np.float32)
        radiance_estimate = np.zeros((height, width), dtype=np.float32)
        for i in range(num_images):
            weight  = weighting_function(images[i], Zmax_precomputed[i], Zmin_precomputed[i])
            weight  = np.maximum(weight, 0) * exposure_masks[i].astype(np.float32)
            indices = np.clip(np.round(images[i] - z_min).astype(int), 0, len(response_curve) - 1)
            radiance_estimate += weight * (response_curve[indices] - np.log(exposure_times[i]))
            weight_sum        += weight

        radiance_map = np.where(weight_sum > 0, radiance_estimate / weight_sum, 0)

        total_pixels    = height * width
        valid_pixels    = int(has_valid.sum())
        fallback_pixels = int(fallback_mask.sum())
        print(f"[Step2] Adaptive: {valid_pixels:,} valid "
              f"({100*valid_pixels/total_pixels:.1f}%), "
              f"{fallback_pixels:,} fallback "
              f"({100*fallback_pixels/total_pixels:.1f}%)")

    if return_all:
        return radiance_map, response_curve, z_min, z_max, exposure_count, min_exp, max_exp
    return radiance_map


def get_unique_filename(filepath):
    """
    Generate a unique filename by appending a counter if the file already exists.
    
    Args:
        filepath: Original filepath
        
    Returns:
        Unique filepath that doesn't exist in the target directory
    """
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    
    counter = 1
    new_filepath = filepath
    while os.path.exists(new_filepath):
        new_filepath = os.path.join(directory, f"{base}_{counter}{ext}")
        counter += 1
    
    return new_filepath

def save_as_tiff(radiance_map, filepath, log_scale=False):
    """
    Save radiance map as a TIFF file.
    
    Args:
        radiance_map: The radiance map to save
        filepath: Target filepath
        log_scale: Whether to save in log scale
    """
    import tifffile
    
    # Create a copy to avoid modifying the original
    data = radiance_map.copy()
    
    if log_scale == False:
        # Add small constant to avoid log(0)
        data = np.exp(data)
    
    # Get unique filename
    unique_filepath = get_unique_filename(filepath)
    
    # Save as 32-bit float TIFF with filter and excitation information

    tifffile.imwrite(unique_filepath, data.astype(np.float32))

def process_hdr_images(directory, experiment_title, base_data_folder, coefficients_dict, response_curve="default",
                      smoothing_lambda=1000, weighting_function=debevec, num_sets=None, method = "default"):
    """
    Process HDR images from the given directory and experiment title.
    
    Args:
        directory: Directory containing the experiment data
        experiment_title: Title of the experiment
        base_data_folder: Base folder containing the experiment data
        coefficients_dict: Dictionary containing the coefficients for the camera response function
        response_curve: Optional pre-computed camera response function
        smoothing_lambda: Smoothing parameter for the camera response function
        weighting_function: Function to compute weights
        num_sets: Number of sets to process

    Returns:
        List of dictionaries containing the processed HDR images
    """


    repodirectory = os.getcwd()
    os.chdir(os.path.join(directory, base_data_folder))
    data_dict = load_data(directory, base_data_folder)
    final_data_folder = os.path.join(directory, base_data_folder, "final_data")
    
    os.makedirs(final_data_folder, exist_ok=True)
    
    Smax = coefficients_dict['Smax']
    Sd = coefficients_dict['Sd']
    bias = coefficients_dict['b']
    
    processed_data = []
    
    if num_sets:
        data_dict = dict(list(data_dict.items())[:num_sets])
    
    for key, item in data_dict.items():
        data = item['data']
        data_type = item['type']
        images = data['image']
        exposure_times = data['exposure_time']

        Zmax_precomputed, Zmin_precomputed = precompute_zmax(images, Smax, Sd, bias, exposure_times, data_type = data_type)


        radiance_map, response_curve_computed, z_min, z_max, exposure_count, min_exp, max_exp = computeRadianceMap(
            images, exposure_times, Zmax_precomputed, Zmin_precomputed,
            crf=response_curve, return_all=True, weighting_function=weighting_function,
            method = method
        )

        # Save .npy file with unique filename

        radiance_map_filename = f"{key}_radmap_{data_type}_{weighting_function.__name__}.npy"
        radiance_map_path = os.path.join(final_data_folder, radiance_map_filename)
        unique_npy_path = get_unique_filename(radiance_map_path)
        np.save(unique_npy_path, np.exp(radiance_map))
        
        # Save TIFF files in both linear and log scale
        tiff_base = os.path.splitext(radiance_map_filename)[0]
        linear_tiff_path = os.path.join(final_data_folder, f"{tiff_base}_linear.tif")
        log_tiff_path = os.path.join(final_data_folder, f"{tiff_base}_log.tif")
        
        save_as_tiff(radiance_map, linear_tiff_path, log_scale=False)
        save_as_tiff(radiance_map, log_tiff_path, log_scale=True)
        
        # Save input parameters with unique filename
        input_filename = f'{key}_inputs.txt'
        input_path = os.path.join(final_data_folder, input_filename)
        unique_input_path = get_unique_filename(input_path)


        #save a .txt file with inputs used for processing

        # Get date info
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(unique_input_path, 'w') as f:
            f.write(f"Date: {date}\n")
            f.write(f"Data Directory: {directory}\n")
            f.write(f"Experiment Title: {experiment_title}\n")
            f.write(f"Number of Sets: {num_sets}\n")
            f.write(f"Exposure Times: {exposure_times}\n")
            f.write(f"Smoothing Lambda: {smoothing_lambda}\n")
            f.write(f"Weighting Function: {weighting_function.__name__}\n")
            f.write(f"Method: {method}\n")
            f.write(f"Camera Response Function: {response_curve}\n")
            f.write(f"Output Files:\n")
            f.write(f"  - NPY: {os.path.basename(unique_npy_path)}\n")
            f.write(f"  - TIFF (Linear): {os.path.basename(linear_tiff_path)}\n")
            f.write(f"  - TIFF (Log): {os.path.basename(log_tiff_path)}\n")
        
        processed_data.append({
            'key': key,
            'data_type': data_type,
            'radiance_map': radiance_map,
            'response_curve': response_curve_computed,
            'z_min': z_min,
            'z_max': z_max,
            'exposure_times': exposure_times,
            'exposure_count': exposure_count,
            'min_exp': min_exp,
            'max_exp': max_exp,
            'output_files': {
                'npy': unique_npy_path,
                'tiff_linear': linear_tiff_path,
                'tiff_log': log_tiff_path,
                'inputs': unique_input_path
            }
        })
        #pickle the processed_data object into the final_data folder
        import pickle
        processed_data_path = os.path.join(final_data_folder, f'{key}_processed.pkl')
        unique_processed_data_path = get_unique_filename(processed_data_path)
        with open(unique_processed_data_path, 'wb') as f:
            pickle.dump(processed_data, f)
    return processed_data

def save_radiance_map(radiance_map, directory, experiment_title, base_data_folder):
    """Save the unscaled radiance map."""
    #experiment_folder = os.path.basename(os.path.normpath(directory))
    final_data_folder = os.path.join(directory, base_data_folder, "final_data")
    os.makedirs(final_data_folder, exist_ok=True)

    radiance_file = os.path.join(final_data_folder, f"{experiment_title}_radiance_map.npy")
    np.save(radiance_file, radiance_map)
    print(f"Radiance map saved to: {radiance_file}")


def save_crf_data(processed_data, directory, experiment_title, base_data_folder):
    #experiment_folder = os.path.basename(os.path.normpath(directory))
    data_folder = os.path.join(directory, base_data_folder, "final_data")
    os.makedirs(data_folder, exist_ok=True)

    crf_file = os.path.join(data_folder, f"{experiment_title}_crf_data.npz")
    
    print(f"Number of processed data items: {len(processed_data)}")

    save_dict = {}
    for i, data in enumerate(processed_data):
        print(f"\nProcessing data item {i+1}:")
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"  {key} shape: {value.shape}")
                save_dict[f'{key}_{i}'] = value
            else:
                print(f"  {key}: {value}")
                save_dict[f'{key}_{i}'] = value

    print("\nSaving CRF data...")
    try:
        np.savez(crf_file, **save_dict)
        print(f"CRF data saved to: {crf_file}")
        
        # Verify the saved data
        verify_data = np.load(crf_file, allow_pickle=True)
        print("Verified saved data. Keys in the file:")
        for key in verify_data.keys():
            print(f"  {key}")
    except Exception as e:
        print(f"Error saving CRF data: {str(e)}")

    return crf_file

def load_crf_data(crf_file):
    """Load the saved CRF data."""
    print(f"Loading CRF data from: {crf_file}")
    loaded_data = np.load(crf_file, allow_pickle=True)
    
    print("Keys in the loaded file:")
    for key in loaded_data.keys():
        print(f"  {key}")
    
    processed_data = []
    i = 0
    while f'key_{i}' in loaded_data:
        print(f"Processing data item {i}")
        data_item = {}
        for key in ['key', 'radiance_map', 'response_curve', 'z_min', 'z_max', 'intensity_samples', 'log_exposures']:
            full_key = f'{key}_{i}'
            if full_key in loaded_data:
                data_item[key] = loaded_data[full_key]
                if isinstance(data_item[key], np.ndarray):
                    print(f"  Loaded {full_key}: ndarray with shape {data_item[key].shape}, dtype: {data_item[key].dtype}")
                else:
                    print(f"  Loaded {full_key}: {type(data_item[key])}")
            else:
                print(f"  Warning: {full_key} not found in loaded data")
        if data_item:
            processed_data.append(data_item)
        else:
            print(f"  Warning: No data loaded for item {i}")
        i += 1
    
    print(f"Loaded {len(processed_data)} processed data items")
    
    return processed_data
