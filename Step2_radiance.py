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
    print(data_type)
    num_exposures = len(exposure_times)
    height, width = Smax.shape
    
    # Reshape arrays for broadcasting
    Smax = Smax.reshape(1, height, width)
    Sd = Sd.reshape(1, height, width)
    bias = bias.reshape(1, height, width)
    exposure_times = exposure_times.reshape(num_exposures, 1, 1)
    maximages = np.max(images)
    print("maximages shape is", maximages.shape)
    print("maximages is", maximages)
    #broadcast maximages
    maximages = np.full([len(exposure_times), height, width], maximages)

    print("Smax shape is", Smax.shape)
    
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

def linear(z, Zmax, Zmin):
    """
    Compute the weighting function for each pixel using a linear function.
    
    Args:
    z (numpy.ndarray): Pixel intensity values
    Zmax (numpy.ndarray): Maximum pixel intensity values (same shape as z)
    
    Returns:
    numpy.ndarray: Weights calculated using the linear function
    """
    x = np.divide(z-Zmin, Zmax) #z / Zmax
    if x <0:
        x = 0
    return x

def none(z, Zmax, Zmin):
    """
    No weighting function, returns 1.
    """
    return 1

def square(z, Zmax, Zmin):
    """
    Compute the weighting function for each pixel using a square function between Zmin and Zmax.
    
    Args:
    z (numpy.ndarray): Pixel intensity values
    Zmax (numpy.ndarray): Maximum pixel intensity values (same shape as z)
    
    Returns:
    numpy.ndarray: Weights calculated using the square function
    """
    x = np.zeros_like(z)
    x[z>(Zmin+200)] = 1
    x[z>(Zmax-500)] = 0

    return x

def load_data(directory, base_data_folder):
    """Load data based on filename tags."""
    final_data_folder = os.path.join(directory, base_data_folder, "processed_data")
    data_dict = {}
    
    for file in os.listdir(final_data_folder):
        print(file)
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


def computeRadianceMap(images, exposure_times, Zmax_precomputed, Zmin_precomputed, smoothing_lambda=1000, 
                      return_all=True, crf="default", weighting_function=debevec, key=None, repo=None, method = "default"):
    """
    Compute the radiance map from multiple exposures.
    
    Args:
        images: Input images
        exposure_times: Exposure times for each image
        Zmax_precomputed: Precomputed maximum intensity values
        smoothing_lambda: Smoothing parameter
        return_all: Whether to return additional information
        crf: Camera Response Function (required for mitsunaga_weight and reinhard_weight)
        weighting_function: Function to compute weights
        key: Identifier for saving the response curve
        
    Returns:
        Radiance map and optionally additional information
        
    Raises:
        ValueError: If mitsunaga_weight or reinhard_weight is used without providing a CRF
    """
    # Check if CRF is required but not provided
    if weighting_function.__name__ in ['mitsunaga_weight', 'reinhard_weight'] and crf is None:
        raise ValueError(
            f"Weight function '{weighting_function.__name__}' requires a Camera Response Function (CRF). "
            "Please provide a CRF parameter."
        )
    
    intensity_samples, log_exposures, sample_radiance, z_min, z_max  = sampleIntensities(images, exposure_times, Zmax_precomputed,Zmin_precomputed, weighting_function)
    #check crf input
    if str(crf) == "default":
        response_curve = np.load(os.path.join(repo, "data\\crf.npy"))
    elif crf is None:
        response_curve = computeResponseCurve(intensity_samples, log_exposures, exposure_times, 
                                            smoothing_lambda, weighting_function, z_min, z_max, 
                                            Zmax_precomputed, Zmin_precomputed, key=key)
    else:
        response_curve = crf

    num_images, height, width = images.shape
    radiance_map = np.zeros((height, width), dtype=np.float32)
    sum_weights = np.zeros((height, width), dtype=np.float32)
    
    if method == "default":
        for i in range(num_images):
            w = weighting_function(images[i], Zmax_precomputed[i], Zmin_precomputed[i])
            #clip pixels below z_min
            indices = np.clip(np.round(images[i] - z_min).astype(int), 0, len(response_curve) - 1)
            radiance_map += w * (response_curve[indices] - np.log(exposure_times[i]))
            sum_weights += w
        radiance_map /= sum_weights
    elif method == "adaptive":
        saturation_percentage = 0.80
        noise_percentage = 0.015
        noise_percentage2 = 0.03
        num_images, height, width = images.shape
        noise_thresholds = np.zeros((num_images, height, width), dtype=np.float32)
        noise_thresholds2 = np.zeros((num_images, height, width), dtype=np.float32)
        saturation_thresholds = np.zeros((num_images, height, width), dtype=np.float32)
        for i, exp_time in enumerate(exposure_times):
            # Compute thresholds on RAW datA
            noise_floor = Zmin_precomputed[i]
            noise_thresholds[i] = noise_floor + (Zmax_precomputed[i] - noise_floor) * noise_percentage
            noise_thresholds2[i] = noise_floor + (Zmax_precomputed[i] - noise_floor) * noise_percentage2
            saturation_thresholds[i] = Zmax_precomputed[i] * saturation_percentage
        
        # Per-pixel exposure selection with fallback
        exposure_masks = np.zeros((num_images, height, width), dtype=bool)
        fallback_pixels = 0
        
        #Logging of exposures usage per pixel
        exposure_count = np.zeros((height, width), dtype=int)
        min_exp = np.zeros((height, width), dtype=float)
        max_exp = np.zeros((height, width), dtype=float)
        for y in range(height):
            for x in range(width):
                pixel_values = images[:, y, x]
                
                above_noise = pixel_values > noise_thresholds[:, y, x]
                below_saturation = pixel_values < saturation_thresholds[:, y, x]
                valid_exposures = above_noise & below_saturation
                
                if np.any(valid_exposures):
                    # Normal case: some exposures pass both tests
                    valid_indices = np.where(valid_exposures)[0]
                    first_valid = valid_indices[0]
                    last_valid = valid_indices[-1]
                    
                    selected_indices = list(valid_indices)
                    
                    # Add limited undersaturated
                    undersaturated = np.where(~above_noise)[0]
                    undersaturated = undersaturated[undersaturated < first_valid]
                    if len(undersaturated) > 0:
                        num_to_include = min(1, len(undersaturated))
                        selected_indices.extend(undersaturated[-num_to_include:])
                    
                    # Add limited oversaturated
                    oversaturated = np.where(~below_saturation)[0]
                    oversaturated = oversaturated[oversaturated > last_valid]
                    if len(oversaturated) > 0:
                        num_to_include = min(1, len(oversaturated))
                        selected_indices.extend(oversaturated[:num_to_include])
                    
                    
                    exposure_count[y, x] = len(selected_indices)
                    min_exp[y, x] = min(selected_indices)
                    max_exp[y, x] = max(selected_indices)
                    #if all exposures are within a range above the noise floor, use the longext exposure only
                    if np.all(pixel_values < noise_thresholds2[:, y, x]):
                        selected_indices = np.argmax(exposure_times)
                        max_exp[y, x] = selected_indices
    
                        exposure_count[y, x] = 1
                        min_exp[y, x] = selected_indices
                        
                    exposure_masks[selected_indices, y, x] = True
                else:
                    # FALLBACK: No exposures pass both tests - ensure at least one exposure is selected
                    fallback_pixels += 1
                    
                    if np.all(pixel_values >= saturation_thresholds[:, y, x]):
                        # All saturated: use shortest exposure (best estimate of true signal)
                        best_exposure = np.argmin(exposure_times)
                    else:
                        # All too dark: use longest exposure (highest SNR)
                        best_exposure = np.argmax(exposure_times)
                    
                    exposure_masks[best_exposure, y, x] = True
                    exposure_count[y, x] = 1
                    min_exp[y, x] = best_exposure
                    max_exp[y, x] = best_exposure

        weight_sum = np.zeros((height, width), dtype=np.float32)
        radiance_estimate = np.zeros((height, width), dtype=np.float32)
        mid_val = Zmax_precomputed.mean() / 2
        
        for i, exp_time in enumerate(exposure_times):
            pixel_mask = exposure_masks[i]
            
            if np.any(pixel_mask):
                weight = np.where(
                    images[i] <= mid_val,
                    images[i] / mid_val,
                    (Zmax_precomputed[i] - images[i]) / mid_val
                )
                weight = np.maximum(weight, 0) * pixel_mask.astype(np.float32) #apply pixel mask to weights
                indices = np.clip(np.round(images[i] - z_min).astype(int), 0, len(response_curve) - 1)
                radiance_estimate += weight *(response_curve[indices] -np.log(exp_time))
                
                weight_sum += weight

        radiance_map = radiance_estimate / weight_sum
    
        # Report fallback statistics
        total_pixels = height * width
        if fallback_pixels > 0:
            print(f"      📊 Fallback pixels: {fallback_pixels:,} ({100*fallback_pixels/total_pixels:.1f}%)")

    if return_all:
        return radiance_map, response_curve, z_min, z_max, intensity_samples, log_exposures, sample_radiance, exposure_count, min_exp, max_exp
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
        
    Raises:
        ValueError: If mitsunaga_weight or reinhard_weight is used without providing a response_curve
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
        print(data_type)
        images = data['image']
        exposure_times = data['exposure_time']
        print(exposure_times)

        Zmax_precomputed, Zmin_precomputed = precompute_zmax(images, Smax, Sd, bias, exposure_times, data_type = data_type)


        radiance_map, response_curve_computed, z_min, z_max, intensity_samples, log_exposures, sample_radiance, exposure_count, min_exp, max_exp = computeRadianceMap(
            images, exposure_times, Zmax_precomputed, Zmin_precomputed, smoothing_lambda=smoothing_lambda, 
            crf=response_curve, return_all=True, weighting_function=weighting_function, 
            key=key, repo=repodirectory, method = method
        )
        if response_curve_computed is None: #if no response curve is computed, set response curve to the precomputed one
            response_curve_computed = response_curve

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
            'intensity_samples': intensity_samples,
            'log_exposures': log_exposures,
            'sample_radiance': sample_radiance,
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

def computeResponseCurve(intensity_samples, log_exposures, exposure_times, smoothing_lambda, 
                        weighting_function, z_min, z_max, Zmax_precomputed, Zmin_precomputed, key = None):
    num_samples, num_images = intensity_samples.shape
    print(num_images, num_samples)
    
    # Use actual maximum from samples instead of Zmax_precomputed
    print("zmin", z_min)
    print("zmax",z_max)
    
    actual_max = int(np.max(intensity_samples))
    actual_min = int(np.min(intensity_samples))
    print("actual_min", actual_min)
    print("actual_max", actual_max)
    intensity_range = actual_max - actual_min + 1
    z_mid = int((actual_min + actual_max) // 2)

    total_constraints = (num_samples * num_images + intensity_range - 2 + 
                        intensity_range - 1 + 1)

    mat_A = np.zeros((total_constraints, intensity_range), dtype=np.float64)
    mat_b = np.zeros((total_constraints, 1), dtype=np.float64)

    k = 0
    for i in range(num_samples):
        for j in range(num_images):
            current_zmax = np.median(Zmax_precomputed[j])
            current_zmin = np.median(Zmin_precomputed[j])
            z_ij = intensity_samples[i, j]
            w_ij = weighting_function(z_ij, current_zmax, current_zmin)
            #set negative weights to 0

            if w_ij <= 0:
                w_ij = np.float64(0.0000000001)
            
            
            z_ij_scalar = int(z_ij)
            w_ij_scalar = np.mean(w_ij) if isinstance(w_ij, np.ndarray) else float(w_ij)
            mat_A[k, z_ij_scalar - actual_min] = w_ij_scalar
            mat_b[k, 0] = w_ij_scalar * log_exposures[i, j]
            k += 1
            if k % 1000 == 0:
                print("weighted ",k,"/",total_constraints," samples")
    print("mat_A", mat_A.shape)
    print("mat_A", mat_A)
    print("mat_A min", np.min(mat_A))
    print("mat_A max", np.max(mat_A))
    print("mat_b", mat_b.shape)
    print("mat_b", mat_b)
    print("mat_b min", np.min(mat_b))
    print("mat_b max", np.max(mat_b))
    # Use actual_max instead of Zmax_precomputed for smoothness constraints
    for z_k in range(actual_min + 1, actual_max):
        w_k = weighting_function(z_k, actual_max, actual_min)
        w_k_scalar = np.mean(w_k) if isinstance(w_k, np.ndarray) else float(w_k)
        mat_A[k, z_k - actual_min - 1:z_k - actual_min + 2] = w_k_scalar * smoothing_lambda * np.array([-1, 2, -1])
        k += 1

    for z_k in range(actual_min, actual_max - 1):
        if k < total_constraints - 1:
            mat_A[k, z_k - actual_min] = -1
            mat_A[k, z_k - actual_min + 1] = 1
            mat_b[k, 0] = 0.001
            k += 1
        else:
            break

    mat_A[k, z_mid - actual_min] = 1
    mat_b[k, 0] = 0

    x = np.linalg.lstsq(mat_A, mat_b, rcond=None)[0]
    response_curve = x.flatten()

    filter_info = key
    weight_name = weighting_function.__name__
    currentdate = datetime.now().strftime("%Y%m%d")
    #save CRF to output folder
    np.save(f"{filter_info}_crf_{weight_name}.npy", response_curve)
    print(f"Saved {filter_info}_crf_{weight_name}.npy")
    response_curve = savgol_filter(response_curve, window_length=51, polyorder=3)
    return response_curve

def estimate_radiance(images, exposure_times, Zmax_precomputed, Zmin_precomputed, weighting_function=debevec):
    num_images, height, width = images.shape
    radiance = np.zeros((height, width))
    weight_sum = np.zeros((height, width))
    
    for i in range(num_images):
        weights = weighting_function(images[i], Zmax_precomputed[i], Zmin_precomputed[i])
        weights[weights<=0] = np.float64(0.00000000001)
        radiance += weights * images[i].astype(float) / exposure_times[i]
        weight_sum += weights
    
    return radiance / np.maximum(weight_sum, 1e-6)

def sampleIntensities(images, exposure_times, Zmax_precomputed, Zmin_precomputed, weighting_function=debevec):
    """Sample pixel intensities from the exposure stack, ensuring same pixels are sampled across all exposures."""
    num_images, height, width = images.shape
    z_min = np.min(Zmin_precomputed)
    z_max = np.median(Zmax_precomputed)
    print(z_min)
    print(z_max)
    logger.info(f"z_max: {z_max}; z_min: {z_min}")

    # Find reference image (first image that reaches 95% of Zmax)
    max_intensities = np.array([np.max(img) for img in images])
    threshold = 0.95 * np.max(z_max)
    reference_indices = np.where(max_intensities >= threshold)[0]
    reference_idx = reference_indices[0] if len(reference_indices) > 0 else num_images // 2
    reference_image = images[reference_idx]
    logger.info(f"Using image {reference_idx} as reference (max intensity: {max_intensities[reference_idx]:.2f}, threshold: {threshold:.2f})")
    # Set default num_samples
    num_samples = int(10000)

    # Logarithmic binning setup
    num_bins = min(num_samples // num_images, int(np.max(z_max) - np.min(z_min) + 1))
    num_bins = max(1, int(num_bins))
    logger.info(f"num_bins: {num_bins}")

    bins = np.logspace(np.log10(z_min + 1), np.log10(z_max + 1), num_bins + 1) - 1
    bins = np.unique(bins.astype(int))
    logger.info(f"Number of unique bins: {len(bins)}")

    # Sample pixels using reference image
    sampled_pixel_locations = []
    for j in range(len(bins) - 1):
        bin_mask = (reference_image >= bins[j]) & (reference_image < bins[j+1])
        pixels_in_bin = np.where(bin_mask)
        
        if len(pixels_in_bin[0]) > 0:
            num_to_sample = min(len(pixels_in_bin[0]), num_samples / len(bins))
            num_to_sample = max(1, int(num_to_sample))
            logger.info(f"Sampling {num_to_sample} pixels from bin {j}")
            
            sampled_indices = np.random.choice(len(pixels_in_bin[0]), num_to_sample, replace=False)
            sampled_rows = pixels_in_bin[0][sampled_indices]
            sampled_cols = pixels_in_bin[1][sampled_indices]
            sampled_pixel_locations.extend(list(zip(sampled_rows, sampled_cols)))

    logger.info(f"Total sampled pixel locations: {len(sampled_pixel_locations)}")
    print("Intensities are sampled")

    # Initialize arrays for both types of exposures
    intensity_samples = np.zeros((len(sampled_pixel_locations), num_images), dtype=np.float32)
    log_exposures = np.zeros((len(sampled_pixel_locations), num_images))
    sample_radiance = np.zeros((len(sampled_pixel_locations), num_images))
    
    # Calculate radiance
    radiance = estimate_radiance(images, exposure_times, Zmax_precomputed, Zmin_precomputed, weighting_function)
    
    # Fill arrays
    for i, (row, col) in enumerate(sampled_pixel_locations):
        for j in range(num_images):
            intensity_samples[i, j] = images[j, row, col]
            # Store exposure-time adjusted values (original method)
            log_exposures[i, j] = np.log(radiance[row, col] * exposure_times[j] + 1e-10)
            # Store unadjusted values
            sample_radiance[i, j] = np.log(radiance[row, col] + 1e-10)


    # Validate samples using Zmax_precomputed, if data is clipped
    valid_samples = np.zeros(len(sampled_pixel_locations), dtype=bool)

    for i, (row, col) in enumerate(sampled_pixel_locations):
        valid_exposures = (intensity_samples[i] >= z_min) & (intensity_samples[i] < z_max)
        valid_samples[i] = np.sum(valid_exposures) == num_images  #valid exposures should satisfy all of the images
    
    # Filter out invalid samples
    intensity_samples = intensity_samples[valid_samples]
    log_exposures = log_exposures[valid_samples]
    sample_radiance = sample_radiance[valid_samples]
    print("valid sample max", np.max(intensity_samples))
    print("valid exposures saved")
    logger.info(f"Number of valid samples after filtering: {intensity_samples.shape[0]}")

    return intensity_samples, log_exposures, sample_radiance, z_min, z_max

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
