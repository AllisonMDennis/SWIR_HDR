import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from scipy.signal import savgol_filter
import logging

logger = logging.getLogger(__name__)

"""
Step 0: Instrument calibration.

Dark-frame pipeline:
    import_darkcount_data -> sort_by_exposure_time -> fit_s_curve -> find_linear_range
    -> model_dark_current -> saves Sd.npy, b.npy

CRF and light-response pipeline:
    import_reflectance_data -> analyze_light_response -> saves Smax.npy
    -> sampleIntensities -> computeResponseCurve -> saves crf.npy

Orchestrator:
    run_calibration(dc_dir, reflectance_dir, output_dir, crf_dir, config=None)
"""


# --- Dark frame pipeline ---

def linear_to_asymptote(t, slope, intercept, Smax, smoothness):
    """
    S-curve function that's linear before transitioning to asymptote.
    """
    linear_term = slope * t + intercept
    exp_term = np.exp(-smoothness * (linear_term - Smax))
    exp_term = np.clip(exp_term, -1e15, 1e15)  # Prevent overflow
    return Smax - (1/smoothness) * np.log1p(exp_term)


def is_approximately_monotonic(y, tolerance=0.01):
    """
    Check if the sequence is approximately monotonic.

    Args:
    y : array-like
        The sequence to check.
    tolerance : float, optional
        The relative tolerance for non-monotonicity. Default is 0.01 (1%).

    Returns:
    bool
        True if the sequence is approximately monotonic, False otherwise.
    """
    diff = np.diff(y)
    negative_diffs = diff[diff < 0]
    if len(negative_diffs) == 0:
        return True

    max_allowed_negative = tolerance * np.max(y)
    return np.all(np.abs(negative_diffs) <= max_allowed_negative)


def find_monotonic_range(y, tolerance=0.01):
    """
    Find the largest approximately monotonic range from the beginning of the sequence.
    """
    for i in range(len(y), 0, -1):
        if is_approximately_monotonic(y[:i], tolerance):
            return i
    return 1



def import_darkcount_data(directory):
    """
    Import darkcount data from H5 files in the specified directory.
    """
    # use this version if there is only darkcount data that can all be considered together in the folder (will consider all the files)
    darkcount_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
    # use this version if it's a folder with mixed data -- include the consistent first part of the darkcount file names
    #darkcount_files = [f for f in os.listdir(directory) if f.startswith('darkcount') and f.endswith('.h5')]
    darkcount_files.sort(key=lambda x: float(x.split('_')[-1][:-3]))  # Sort by the number before .h5

    darkcount_data = []
    exposure_times = []

    for file in darkcount_files:
        file_path = os.path.join(directory, file)
        with h5py.File(file_path, 'r') as h5f:
            darkcount = h5f['Cube']['Images'][()]
            exposure_time = h5f['Cube']['TimeExposure'][()].item()
            darkcount_data.append(darkcount)
            exposure_times.append(exposure_time)

    darkcount_array = np.array(darkcount_data)
    darkcount_array = darkcount_array.squeeze()  # This will remove the extra dimension
    exposure_times = np.array(exposure_times)

    return darkcount_array, exposure_times


def analyze_darkcount_data(darkcount_array, exposure_times):
    """
    Analyze darkcount data and return summary statistics.
    """
    mean_values = np.mean(darkcount_array, axis=(1, 2))
    std_values = np.std(darkcount_array, axis=(1, 2))
    min_values = np.min(darkcount_array, axis=(1, 2))
    max_values = np.max(darkcount_array, axis=(1, 2))

    low_outliers = np.sum(darkcount_array < (mean_values[:, np.newaxis, np.newaxis] - 2 * std_values[:, np.newaxis, np.newaxis]), axis=(1, 2))
    high_outliers = np.sum(darkcount_array > (mean_values[:, np.newaxis, np.newaxis] + 2 * std_values[:, np.newaxis, np.newaxis]), axis=(1, 2))

    total_pixels = darkcount_array.shape[1] * darkcount_array.shape[2]
    low_outlier_percentages = (low_outliers / total_pixels) * 100
    high_outlier_percentages = (high_outliers / total_pixels) * 100

    return {
        'mean': mean_values,
        'std': std_values,
        'min': min_values,
        'max': max_values,
        'low_outliers': low_outlier_percentages,
        'high_outliers': high_outlier_percentages
    }


def linear_fit(t, slope, intercept):
    return slope * t + intercept


def fit_s_curve(exposure_times, mean_values):
    """
    Fit the data to the linear-to-asymptote equation and determine the linear range.
    """
    # Improved initial guesses
    initial_slope = (mean_values[5] - mean_values[0]) / (exposure_times[5] - exposure_times[0])
    p0 = [
        initial_slope,  # slope
        mean_values[0],  # intercept
        np.max(mean_values),  # Smax
        0.1  # smoothness
    ]

    # Add bounds to prevent unrealistic values
    bounds = ([0, 0, np.max(mean_values)*0.95, 0], [np.inf, np.inf, np.inf, 10])

    popt, _ = curve_fit(linear_to_asymptote, exposure_times, mean_values,
                        p0=p0, bounds=bounds, maxfev=10000)

    return popt  # Just return popt, we'll calculate linear_range separately


def find_linear_range(exposure_times, mean_values, popt, threshold=0.01):
    """
    Find the linear range of the response curve, focusing on the initial linear part.
    """
    slope, intercept, Smax, smoothness = popt
    linear_prediction = slope * exposure_times + intercept
    actual_response = linear_to_asymptote(exposure_times, *popt)

    relative_deviation = np.abs(actual_response - linear_prediction) / linear_prediction

    # Find where the relative deviation exceeds the threshold
    nonlinear_indices = np.where(relative_deviation > threshold)[0]

    if len(nonlinear_indices) > 0:
        linear_range_end = exposure_times[nonlinear_indices[0]]
    else:
        linear_range_end = exposure_times[-1]

    # For the start, find where the relative deviation goes below the threshold
    linear_start_indices = np.where(relative_deviation < threshold)[0]
    if len(linear_start_indices) > 0:
        linear_range_start = exposure_times[linear_start_indices[0]]
    else:
        linear_range_start = exposure_times[0]

    return linear_range_start, linear_range_end


def model_dark_current(darkcount_array, exposure_times, linear_range, global_popt):
    """Fit per-pixel dark-current slope (Sd) and intercept (b) by linear regression."""

    linear_mask = (exposure_times >= linear_range[0]) & (exposure_times <= linear_range[1])

    if not np.any(linear_mask):
        print("Warning: No exposure times fall within the calculated linear range.")
        print(f"Linear range: {linear_range}")
        print(f"Exposure times: {exposure_times}")
        linear_mask = np.ones_like(exposure_times, dtype=bool)

    linear_exposure_times = exposure_times[linear_mask]
    linear_darkcount_data = darkcount_array[linear_mask]

    if linear_darkcount_data.size == 0:
        raise ValueError("No data points in the linear range. Check your linear_range and exposure_times.")

    num_pixels = darkcount_array.shape[1] * darkcount_array.shape[2]
    print(f"[Step0] Fitting per-pixel dark-current model ({num_pixels} pixels) ...")
    darkcount_data_reshaped = darkcount_array.reshape(darkcount_array.shape[0], -1)

    Sd = np.zeros(num_pixels)
    b = np.zeros(num_pixels)
    non_monotonic_count = 0

    for i in range(num_pixels):
        if i % 10000 == 0:
            print(f"Processing pixel {i}/{num_pixels}")
        pixel_data = darkcount_data_reshaped[:, i]

        if not is_approximately_monotonic(pixel_data, tolerance=0.05):
            non_monotonic_count += 1
            if non_monotonic_count <= 5:
                print(f"[Step0] Non-monotonic pixel at index {i} (count so far: {non_monotonic_count})")
            monotonic_range = find_monotonic_range(pixel_data, tolerance=0.05)
            linear_pixel_data = pixel_data[:monotonic_range]
            linear_times = exposure_times[:monotonic_range]
        else:
            linear_pixel_data = pixel_data[linear_mask]
            linear_times = linear_exposure_times

        try:
            popt_linear, _ = curve_fit(linear_fit, linear_times, linear_pixel_data)
            Sd[i], b[i] = popt_linear
        except:
            Sd[i], b[i], _, _, _ = stats.linregress(linear_exposure_times, linear_pixel_data)

    Sd = Sd.reshape(darkcount_array.shape[1], darkcount_array.shape[2])
    b = b.reshape(darkcount_array.shape[1], darkcount_array.shape[2])

    print(f"[Step0] Dark-current fit complete. Non-monotonic pixels: {non_monotonic_count}")

    return Sd, b, non_monotonic_count


def sort_by_exposure_time(darkcount_array, exposure_times):
    """
    Sort darkcount_array and exposure_times by ascending exposure times.
    """
    sorted_indices = np.argsort(exposure_times)
    sorted_darkcount_array = darkcount_array[sorted_indices]
    sorted_exposure_times = exposure_times[sorted_indices]
    return sorted_darkcount_array, sorted_exposure_times


def save_model_parameters(Sd, b, Smax, output_dir):
    """Save Sd, b, and Smax as NPY files with fixed generic names."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'Sd.npy'), Sd.astype(np.float32))
    np.save(os.path.join(output_dir, 'b.npy'), b.astype(np.float32))
    np.save(os.path.join(output_dir, 'Smax.npy'), Smax.astype(np.float32))
    print(f"[Step0] Saved Sd.npy, b.npy, Smax.npy to {output_dir}")


# --- Light response and CRF pipeline ---

def sigmoid_curve(t, a, b, c, d):
    """
    Generalized sigmoidal function for log-linear fitting.

    Args:
        t: exposure times
        a: amplitude parameter
        b: center point in log space
        c: steepness parameter
        d: vertical offset

    Returns:
        Sigmoidal curve values
    """
    return a / (1 + np.exp((b - np.log10(t))/c)) + d


def log_linear_range_sigmoid(b, c):
    """
    Calculate Slinear point for sigmoid fit based on parameters.

    Args:
        b: center point in log space
        c: steepness parameter

    Returns:
        float: value representing Slinear point in log space
    """
    # Point where sigmoid reaches ~95% of its maximum value
    return b + 1.317 * c


def import_reflectance_data(directory, file_pattern=None):
    """
    Import reflectance measurement data from H5 files in the specified directory.
    """
    if file_pattern:
        reflectance_files = [f for f in os.listdir(directory)
                            if f.startswith(file_pattern) and f.endswith('.h5')]
    else:
        reflectance_files = [f for f in os.listdir(directory) if f.endswith('.h5')]

    if not reflectance_files:
        raise ValueError(f"No H5 files found in {directory}" +
                       (f" matching pattern '{file_pattern}'" if file_pattern else ""))

    try:
        reflectance_files.sort(key=lambda x: float(x.split('_')[-1][:-3]))
    except (IndexError, ValueError):
        print("Warning: Could not sort files by exposure time from filenames.")
        print("Files will be processed in filesystem order.")

    reflectance_data = []
    exposure_times = []
    first_shape = None

    for file in reflectance_files:
        file_path = os.path.join(directory, file)
        try:
            with h5py.File(file_path, 'r') as h5f:
                if 'Cube' in h5f and 'Images' in h5f['Cube']:
                    reflectance = h5f['Cube']['Images'][()]
                else:
                    raise ValueError(f"Required dataset 'Cube/Images' not found in {file}")

                if 'TimeExposure' in h5f['Cube']:
                    exposure_time = h5f['Cube']['TimeExposure'][()].item()
                else:
                    raise ValueError(f"Required dataset 'Cube/TimeExposure' not found in {file}")

                if first_shape is None:
                    first_shape = reflectance.shape
                elif reflectance.shape != first_shape:
                    raise ValueError(f"Inconsistent dimensions in {file}. " +
                                  f"Expected {first_shape}, got {reflectance.shape}")

                reflectance_data.append(reflectance)
                exposure_times.append(exposure_time)

        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            raise

    reflectance_array = np.array(reflectance_data)
    exposure_times = np.array(exposure_times)
    reflectance_array = reflectance_array.squeeze()

    print(f"[Step0] Loaded {len(reflectance_files)} reflectance files")
    print(f"[Step0] Reflectance array shape: {reflectance_array.shape}")
    print(f"[Step0] Exposure time range: {exposure_times.min():.2e} to {exposure_times.max():.2e} seconds")

    return reflectance_array, exposure_times


def analyze_light_response(light_array, exposure_times, threshold=0.01, fit_method='log_linear'):
    """
    Analyze light response data to find per-pixel Smax values.
    Smax is the manuscript calibration output (sigmoid asymptote, Section 2.3 Eq. 2).
    Slinear is returned for QC visualization only and is not saved.
    """
    if fit_method != 'log_linear':
        raise ValueError("fit_method must be 'log_linear'")

    height, width = light_array.shape[1:]
    Slinear = np.zeros((height, width))
    Smax = np.zeros((height, width))

    fit_params = {
        'slope': np.zeros((height, width), dtype=np.float64),
        'intercept': np.zeros((height, width), dtype=np.float64),
        'smoothness': np.zeros((height, width), dtype=np.float64)
    }

    fit_quality = {
        'r_squared': np.zeros((height, width)),
        'fit_error': np.zeros((height, width), dtype=bool),
        'is_monotonic': np.zeros((height, width), dtype=bool)
    }

    log_times = np.log10(exposure_times)

    for i in range(height):
        if i % 50 == 0:
            print(f"[Step0] Smax fitting: row {i}/{height}")
        for j in range(width):
            pixel_data = light_array[:, i, j]

            if not is_approximately_monotonic(pixel_data, tolerance=0.05):
                fit_quality['fit_error'][i, j] = True
                continue

            fit_quality['is_monotonic'][i, j] = True

            try:
                max_val = np.max(pixel_data)
                min_val = np.min(pixel_data)
                range_val = max_val - min_val

                p0 = [
                    range_val,              # amplitude
                    np.median(log_times),   # center in log space
                    0.5,                    # steepness
                    min_val                 # offset
                ]
                bounds = (
                    [0.1*range_val, np.min(log_times), 0.1, min_val],
                    [2.0*range_val, np.max(log_times), 2.0, max_val]
                )

                popt, _ = curve_fit(
                    sigmoid_curve,
                    exposure_times,
                    pixel_data,
                    p0=p0,
                    bounds=bounds,
                    method='trf',
                    maxfev=10000
                )

                fit_params['slope'][i, j]     = popt[0]  # amplitude
                fit_params['intercept'][i, j] = popt[1]  # center
                fit_params['smoothness'][i, j] = popt[2]  # steepness

                Smax[i, j] = popt[0] + popt[3]

                log_slinear = log_linear_range_sigmoid(popt[1], popt[2])
                Slinear[i, j] = sigmoid_curve(10**log_slinear, *popt)

                y_pred = sigmoid_curve(exposure_times, *popt)
                ss_res = np.sum((pixel_data - y_pred) ** 2)
                ss_tot = np.sum((pixel_data - np.mean(pixel_data)) ** 2)
                fit_quality['r_squared'][i, j] = 1 - (ss_res / ss_tot)

            except Exception as e:
                print(f"Fitting failed for pixel ({i}, {j}): {str(e)}")
                fit_quality['fit_error'][i, j] = True
                Slinear[i, j] = np.max(pixel_data)
                Smax[i, j] = np.max(pixel_data) * 1.2

    return Slinear, Smax, fit_params, fit_quality



def select_random_pixels(height, width, n_pixels, seed=None):
    """
    Select random pixel coordinates.
    """
    if seed is not None:
        np.random.seed(seed)

    i_coords = np.random.randint(0, height, n_pixels, dtype=np.int32)
    j_coords = np.random.randint(0, width, n_pixels, dtype=np.int32)
    return [(int(i), int(j)) for i, j in zip(i_coords, j_coords)]


def plot_pixel_responses(light_array, exposure_times, pixel_coords,
                        Slinear, Smax, fit_params, fit_method='log_linear', random_pixels=None):
    """
    Plot light response curves for selected pixels.
    """
    if random_pixels is not None:
        height, width = light_array.shape[1:]
        pixel_coords = select_random_pixels(height, width, random_pixels)

    if not pixel_coords:
        raise ValueError("No pixel coordinates provided or selected")

    n_pixels = len(pixel_coords)
    fig, axes = plt.subplots(n_pixels, 2, figsize=(15, 5*n_pixels))

    if n_pixels == 1:
        axes = axes.reshape(1, -1)

    x_fine = np.logspace(np.log10(min(exposure_times)), np.log10(max(exposure_times)), 1000)

    for idx, (i, j) in enumerate(pixel_coords):
        i, j = int(i), int(j)
        pixel_data = light_array[:, i, j]

        try:
            amplitude = float(fit_params['slope'][i, j])
            center = float(fit_params['intercept'][i, j])
            steepness = float(fit_params['smoothness'][i, j])
            offset = float(Smax[i, j]) - amplitude

            popt = [amplitude, center, steepness, offset]
            y_fine = sigmoid_curve(x_fine, *popt)

            # Linear scale plot
            axes[idx, 0].scatter(exposure_times, pixel_data, label='Data')
            axes[idx, 0].plot(x_fine, y_fine, 'r-', label='Fitted Curve')
            axes[idx, 0].axhline(y=Slinear[i, j], color='b', linestyle=':',
                                label='Slinear')
            axes[idx, 0].axhline(y=Smax[i, j], color='r', linestyle=':',
                                label='Smax')
            axes[idx, 0].set_xlabel('Exposure Time (s)')
            axes[idx, 0].set_ylabel('Pixel Value')
            axes[idx, 0].set_title(f'Pixel ({i}, {j}) - Linear Scale')
            axes[idx, 0].legend()

            # Log scale plot
            axes[idx, 1].scatter(exposure_times, pixel_data, label='Data')
            axes[idx, 1].plot(x_fine, y_fine, 'r-', label='Fitted Curve')
            axes[idx, 1].axhline(y=Slinear[i, j], color='b', linestyle=':',
                                label='Slinear')
            axes[idx, 1].axhline(y=Smax[i, j], color='r', linestyle=':',
                                label='Smax')
            axes[idx, 1].set_xlabel('Exposure Time (s)')
            axes[idx, 1].set_ylabel('Pixel Value')
            axes[idx, 1].set_title(f'Pixel ({i}, {j}) - Log Scale')
            axes[idx, 1].set_xscale('log')
            axes[idx, 1].legend()

        except Exception as e:
            print(f"Error plotting pixel ({i}, {j}): {str(e)}")
            continue

    plt.tight_layout()
    return fig



def _debevec(z, Zmax, Zmin):
    """Triangular (tent) weighting function, used as the default for CRF recovery."""
    if np.isscalar(z) and np.isscalar(Zmax):
        middle = (Zmax + Zmin) / 2.0
        if z <= middle:
            return (z - Zmin) / (middle - Zmin)
        else:
            return (Zmax - z) / (Zmax - middle)
    else:
        z = np.atleast_2d(z)
        Zmax = np.atleast_2d(Zmax)
        if z.shape != Zmax.shape:
            Zmax = np.full_like(z, Zmax)
        weight = np.zeros_like(z, dtype=np.float32)
        middle = (Zmax + Zmin) / 2.0
        weight[z <= middle] = np.divide(
            z[z <= middle] - Zmin[z <= middle],
            middle[z <= middle] - Zmin[z <= middle]
        )
        weight[z > middle] = np.divide(
            Zmax[z > middle] - z[z > middle],
            Zmax[z > middle] - middle[z > middle]
        )
        return weight


def estimate_radiance(images, exposure_times, Zmax_precomputed, Zmin_precomputed,
                      weighting_function=_debevec):
    num_images, height, width = images.shape
    radiance = np.zeros((height, width))
    weight_sum = np.zeros((height, width))

    for i in range(num_images):
        weights = weighting_function(images[i], Zmax_precomputed[i], Zmin_precomputed[i])
        if np.isscalar(weights):
            weights = np.full(images[i].shape, float(weights), dtype=np.float64)
        else:
            weights = np.array(weights, dtype=np.float64)
        weights[weights<=0] = np.float64(0.00000000001)
        radiance += weights * images[i].astype(float) / exposure_times[i]
        weight_sum += weights

    return radiance / np.maximum(weight_sum, 1e-6)


def sampleIntensities(images, exposure_times, Zmax_precomputed, Zmin_precomputed,
                      weighting_function=_debevec):
    """Sample pixel intensities from the exposure stack, ensuring same pixels are sampled across all exposures."""
    num_images, height, width = images.shape
    z_min = np.min(Zmin_precomputed)
    z_max = np.median(Zmax_precomputed)
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
    print("[Step0] Intensities sampled")

    # Initialize arrays
    intensity_samples = np.zeros((len(sampled_pixel_locations), num_images), dtype=np.float32)
    log_exposures = np.zeros((len(sampled_pixel_locations), num_images))
    sample_radiance = np.zeros((len(sampled_pixel_locations), num_images))

    # Calculate radiance
    radiance = estimate_radiance(images, exposure_times, Zmax_precomputed, Zmin_precomputed, weighting_function)

    # Fill arrays
    for i, (row, col) in enumerate(sampled_pixel_locations):
        for j in range(num_images):
            intensity_samples[i, j] = images[j, row, col]
            log_exposures[i, j] = np.log(radiance[row, col] * exposure_times[j] + 1e-10)
            sample_radiance[i, j] = np.log(radiance[row, col] + 1e-10)

    # Validate samples
    valid_samples = np.zeros(len(sampled_pixel_locations), dtype=bool)
    for i, (row, col) in enumerate(sampled_pixel_locations):
        valid_exposures = (intensity_samples[i] >= z_min) & (intensity_samples[i] < z_max)
        valid_samples[i] = np.sum(valid_exposures) == num_images

    intensity_samples = intensity_samples[valid_samples]
    log_exposures = log_exposures[valid_samples]
    sample_radiance = sample_radiance[valid_samples]
    print("[Step0] Valid exposures saved")
    logger.info(f"Number of valid samples after filtering: {intensity_samples.shape[0]}")

    return intensity_samples, log_exposures, sample_radiance, z_min, z_max


def computeResponseCurve(intensity_samples, log_exposures, exposure_times, smoothing_lambda,
                         weighting_function, z_min, z_max, Zmax_precomputed, Zmin_precomputed,
                         key=None):
    """
    Recover the camera response function using a two-step CRF recovery: pixel
    radiance is first estimated from the exposure stack, then g(z) is recovered
    treating ln(Ê_i × t_j) as known.

    The pre-estimated log-exposures ln(Ê_i × t_j) (from sampleIntensities) are
    passed in via log_exposures and used directly as the right-hand side, so the
    only unknowns are the g(z) values:
        Σ_i Σ_j [w(z_ij)(g(z_ij) - ln(Ê_i × t_j))]² + λ Σ [w(z) g''(z)]²

    Returns:
        response_curve: 1-D array of g(z) values, smoothed with Savitzky-Golay filter
    """
    num_samples, num_images = intensity_samples.shape

    actual_max = int(np.max(intensity_samples))
    actual_min = int(np.min(intensity_samples))
    intensity_range = actual_max - actual_min + 1
    z_mid = int((actual_min + actual_max) // 2)

    # Unknowns are the g(z) values only. Constraints: data + smoothness + anchor.
    total_constraints = num_samples * num_images + (intensity_range - 2) + 1

    mat_A = np.zeros((total_constraints, intensity_range), dtype=np.float64)
    mat_b = np.zeros((total_constraints, 1), dtype=np.float64)

    k = 0

    # Data constraints: w_ij * g(z_ij) = w_ij * ln(Ê_i × t_j)
    for i in range(num_samples):
        for j in range(num_images):
            current_zmax = np.median(Zmax_precomputed[j])
            current_zmin = np.median(Zmin_precomputed[j])
            z_ij = intensity_samples[i, j]
            w_ij = weighting_function(z_ij, current_zmax, current_zmin)

            if w_ij <= 0:
                w_ij = np.float64(1e-10)

            z_ij_scalar = int(z_ij)
            w_ij_scalar = np.mean(w_ij) if isinstance(w_ij, np.ndarray) else float(w_ij)

            mat_A[k, z_ij_scalar - actual_min] = w_ij_scalar           # g(z) column
            mat_b[k, 0] = w_ij_scalar * log_exposures[i, j]            # ln(Ê_i × t_j)
            k += 1

    # Smoothness constraints
    for z_k in range(actual_min + 1, actual_max):
        w_k = weighting_function(z_k, actual_max, actual_min)
        w_k_scalar = np.mean(w_k) if isinstance(w_k, np.ndarray) else float(w_k)
        mat_A[k, z_k - actual_min - 1:z_k - actual_min + 2] = (
            w_k_scalar * smoothing_lambda * np.array([-1, 2, -1])
        )
        k += 1

    # Anchor constraint: g(z_mid) = 0
    mat_A[k, z_mid - actual_min] = 1
    mat_b[k, 0] = 0

    x = np.linalg.lstsq(mat_A, mat_b, rcond=None)[0]
    response_curve = x.flatten()

    response_curve = savgol_filter(response_curve, window_length=51, polyorder=3)
    return response_curve


def save_crf_data(processed_data, directory, experiment_title, base_data_folder):
    """Save full CRF dataset for QC purposes."""
    data_folder = os.path.join(directory, base_data_folder, "final_data")
    os.makedirs(data_folder, exist_ok=True)

    crf_file = os.path.join(data_folder, f"{experiment_title}_crf_data.npz")

    print(f"[Step0] Number of processed data items: {len(processed_data)}")

    save_dict = {}
    for i, data in enumerate(processed_data):
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                save_dict[f'{key}_{i}'] = value
            else:
                save_dict[f'{key}_{i}'] = value

    print("[Step0] Saving CRF data...")
    try:
        np.savez(crf_file, **save_dict)
        print(f"[Step0] CRF data saved to: {crf_file}")
        verify_data = np.load(crf_file, allow_pickle=True)
        print("[Step0] Verified saved data. Keys in the file:")
        for key in verify_data.keys():
            print(f"  {key}")
    except Exception as e:
        print(f"Error saving CRF data: {str(e)}")

    return crf_file


def load_crf_data(crf_file):
    """Load the saved CRF data."""
    print(f"[Step0] Loading CRF data from: {crf_file}")
    loaded_data = np.load(crf_file, allow_pickle=True)

    print("[Step0] Keys in the loaded file:")
    for key in loaded_data.keys():
        print(f"  {key}")

    processed_data = []
    i = 0
    while f'key_{i}' in loaded_data:
        data_item = {}
        for key in ['key', 'radiance_map', 'response_curve', 'z_min', 'z_max',
                    'intensity_samples', 'log_exposures']:
            full_key = f'{key}_{i}'
            if full_key in loaded_data:
                data_item[key] = loaded_data[full_key]
            else:
                print(f"  Warning: {full_key} not found in loaded data")
        if data_item:
            processed_data.append(data_item)
        i += 1

    print(f"[Step0] Loaded {len(processed_data)} processed data items")
    return processed_data


# --- Orchestration ---

def run_calibration(dc_dir, reflectance_dir, output_dir, crf_dir, config=None):
    """
    Run the full instrument calibration pipeline.

    Dark-frame pipeline produces Sd.npy and b.npy.
    Light-response pipeline produces Smax.npy (sigmoid asymptote per manuscript Section 2.3).
    CRF recovery produces crf.npy.

    Args:
        dc_dir: Directory containing dark-frame H5 files
        reflectance_dir: Directory containing Teflon-reflectance H5 files (0.001–30 s, for Smax)
        output_dir: Directory to write Sd.npy, b.npy, Smax.npy, crf.npy
        crf_dir: Directory containing checkerboard H5 files (6 exposures, 0.01–0.32 s, for CRF)
        config: Optional dict with keys:
            - dc_file_pattern: filename prefix filter for dark-frame files (default None)
            - reflectance_file_pattern: filename prefix filter for reflectance files (default None)
            - crf_file_pattern: filename prefix filter for CRF files (default None)
            - smoothing_lambda: CRF smoothness parameter (default 1000)
            - weighting_function: callable for CRF weighting (default _debevec)
            - fit_method: light-response fit method; must be 'log_linear' (default 'log_linear')
            - save_full_crf_data: save full CRF dataset for QC (default False)
            - experiment_name: label used in output filenames (default 'calibration')

    Returns:
        dict with arrays 'Sd', 'b', 'Smax', 'crf' and output file paths
    """
    if config is None:
        config = {}

    dc_file_pattern = config.get('dc_file_pattern', None)
    reflectance_file_pattern = config.get('reflectance_file_pattern', None)
    crf_file_pattern = config.get('crf_file_pattern', None)
    smoothing_lambda = config.get('smoothing_lambda', 1000)
    weighting_function = config.get('weighting_function', _debevec)
    fit_method = config.get('fit_method', 'log_linear')
    save_full = config.get('save_full_crf_data', False)
    experiment_name = config.get('experiment_name', 'calibration')

    os.makedirs(output_dir, exist_ok=True)

    # ── Dark-frame pipeline ───────────────────────────────────────────────────
    print(f"[Step0] Loading dark-frame data from {dc_dir}")
    darkcount_array, dc_times = import_darkcount_data(dc_dir)
    darkcount_array, dc_times = sort_by_exposure_time(darkcount_array, dc_times)

    mean_values = analyze_darkcount_data(darkcount_array, dc_times)['mean']
    global_popt = fit_s_curve(dc_times, mean_values)
    linear_range = find_linear_range(dc_times, mean_values, global_popt)

    Sd, b, non_monotonic_count = \
        model_dark_current(darkcount_array, dc_times, linear_range, global_popt)

    # ── Light-response pipeline ───────────────────────────────────────────────
    print(f"[Step0] Loading reflectance data from {reflectance_dir}")
    light_array, light_times = import_reflectance_data(reflectance_dir, reflectance_file_pattern)
    light_array, light_times = sort_by_exposure_time(light_array, light_times)

    print(f"[Step0] Fitting light-response per pixel (method='{fit_method}') ...")
    Slinear, Smax, fit_params, fit_quality = analyze_light_response(
        light_array, light_times, fit_method=fit_method
    )
    print(f"[Step0] Smax range: {Smax.min():.1f} – {Smax.max():.1f}  "
          f"fit errors: {fit_quality['fit_error'].sum()} pixels")

    save_model_parameters(Sd, b, Smax, output_dir)

    # ── CRF recovery ─────────────────────────────────────────────────────────
    print(f"[Step0] Loading CRF calibration data from {crf_dir}")
    crf_array, crf_times = import_reflectance_data(crf_dir, crf_file_pattern)
    crf_array, crf_times = sort_by_exposure_time(crf_array, crf_times)
    print(f"[Step0] CRF exposures: {len(crf_times)} "
          f"({crf_times.min():.3f}\u2013{crf_times.max():.3f} s)")

    # Preprocess: clip to Smax, subtract DC (variant d)
    num_crf_exp = len(crf_times)
    height, width = Smax.shape
    DC_crf   = np.stack([Sd * t + b for t in crf_times]).astype(np.float32)
    Smax_crf = np.broadcast_to(Smax, (num_crf_exp, height, width)).copy().astype(np.float32)
    crf_processed = np.maximum(
        np.minimum(crf_array, Smax_crf) - DC_crf, 0
    ).astype(np.float32)

    Zmax_precomputed = (Smax_crf - DC_crf).astype(np.float32)
    Zmin_precomputed = np.zeros((num_crf_exp, height, width), dtype=np.float32)

    print("[Step0] Sampling intensities for CRF recovery ...")
    intensity_samples, log_exposures, sample_radiance, z_min, z_max = sampleIntensities(
        crf_processed, crf_times, Zmax_precomputed, Zmin_precomputed, weighting_function
    )

    print("[Step0] Computing response curve ...")
    response_curve = computeResponseCurve(
        intensity_samples, log_exposures, crf_times, smoothing_lambda,
        weighting_function, z_min, z_max, Zmax_precomputed, Zmin_precomputed,
        key=experiment_name
    )

    diffs = np.diff(response_curve)
    n_violations = np.sum(diffs < 0)
    if n_violations > 0:
        print(f"[Step0] Warning: CRF has {n_violations} monotonicity violations after smoothing")
    else:
        print("[Step0] CRF monotonicity validated")

    crf_path = os.path.join(output_dir, 'crf.npy')
    np.save(crf_path, response_curve)
    print(f"[Step0] Saved crf.npy to {output_dir}")

    results = {
        'Sd': Sd,
        'b': b,
        'Smax': Smax,
        'crf': response_curve,
        'Sd_path': os.path.join(output_dir, 'Sd.npy'),
        'b_path': os.path.join(output_dir, 'b.npy'),
        'Smax_path': os.path.join(output_dir, 'Smax.npy'),
        'crf_path': crf_path,
    }

    if save_full:
        crf_data_path = save_crf_data([{
            'key': experiment_name,
            'response_curve': response_curve,
            'z_min': z_min,
            'z_max': z_max,
            'intensity_samples': intensity_samples,
            'log_exposures': log_exposures,
            'sample_radiance': sample_radiance,
        }], output_dir, experiment_name, '')
        results['crf_data_path'] = crf_data_path

    return results
