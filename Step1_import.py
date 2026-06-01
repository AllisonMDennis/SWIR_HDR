import os
import numpy as np
import h5py

def load_parameters(param_directory):
    """Load processing parameters from files in specified directory.
    Parameters are to be used with process_and_save function."""
    param_files = {
        'b': 'b.npy',
        'Sd': 'Sd.npy',
        'Smax': 'Smax.npy',
    }
    
    params = {}
    for param_name, filename in param_files.items():
        filepath = os.path.join(param_directory, filename)
        try:
            params[param_name] = np.load(filepath)
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            
    return params


def process_and_save(directory, experiment_title, base_data_folder, operations=[], params=None):
    """
    Process image data with configurable operations.
    """
    
    raw_folder = import_and_save_raw(directory, experiment_title, base_data_folder)
    
        
    # Changed to use processed_data folder
    output_folder = os.path.join(directory, base_data_folder, "processed_data")
    os.makedirs(output_folder, exist_ok=True)
    
    for file in os.listdir(raw_folder):
        if file.endswith("_raw.npy"):
            key = file[:-8]
            data = np.load(os.path.join(raw_folder, file))
            processed_data = np.zeros(data.shape, dtype=data.dtype)
            processed_data['exposure_time'] = data['exposure_time']
            images = data['image']
            
            if 'clip' in operations or 'cliptop' in operations:
                if 'Smax' not in params:
                    raise ValueError("Smax required for clipping")
                images = np.minimum(images, params['Smax'])
                
            if 'denoise' in operations:
                if not all(k in params for k in ['Sd', 'b']):
                    raise ValueError("Sd and b required for denoising")
                images = np.maximum(
                    images - (params['Sd'] * data['exposure_time'][:, np.newaxis, np.newaxis] + params['b']), 
                    0
                )
            
            processed_data['image'] = images
            if operations == []:
                operations = ['raw']
            op_names = '_'.join(operations)
            
            np.save(os.path.join(output_folder, f"{key}_{op_names}.npy"), processed_data)
    
    print(f"Processed data saved in: {output_folder}")
    return output_folder

def import_and_save_raw(directory, experiment_title, base_data_folder):
    """
    Import raw data from H5 files and save as structured numpy arrays.

    Filter and laser identity are read from H5 metadata:
        Cube/GratingID
        Cube/Info/Grating/{grating_id}.attrs['Name']  -> emission filter name
        Cube/Info/Cube.attrs['LaserNm']               -> laser wavelength (nm)

    Output key format: {experiment_title}_{laser_nm}_{filter_name}
    """
    data_folder = os.path.join(directory, base_data_folder, "raw_data")
    os.makedirs(data_folder, exist_ok=True)

    # --- Pass 1: scan all H5 files, read metadata, group by laser/filter ---
    groups = {}  # key -> list of (exposure_time, file_path)

    for filename in os.listdir(directory):
        if not (filename.endswith('.h5') and filename.startswith(f"{experiment_title}_")):
            continue

        file_path = os.path.join(directory, filename)
        try:
            with h5py.File(file_path, 'r') as h5f:
                grating_id  = h5f['Cube/GratingID'][()].item()
                laser_nm    = int(h5f['Cube/Info/Cube'].attrs['LaserNm'].item())
                filter_name = h5f[f'Cube/Info/Grating/{grating_id}'].attrs['Name'].item().decode('utf-8')
                exposure_time = h5f['Cube']['TimeExposure'][()].item()

            key = f"{experiment_title}_{laser_nm}_{filter_name}"
            groups.setdefault(key, []).append((exposure_time, file_path))

        except Exception as e:
            print(f"Error reading metadata from {filename}: {str(e)}")
            continue

    # --- Pass 2: load images for each group, sort by exposure time, save ---
    for key, file_list in groups.items():
        file_list.sort(key=lambda x: x[0])

        image_data = []
        exposure_times = []

        for exposure_time, file_path in file_list:
            try:
                with h5py.File(file_path, 'r') as h5f:
                    image = h5f['Cube']['Images'][()]
                    if image.ndim == 3 and image.shape[0] == 1:
                        image = image.squeeze(0)
                    image_data.append(image)
                    exposure_times.append(exposure_time)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue

        if not image_data:
            print(f"No valid data collected for {key}")
            continue

        image_array = np.array(image_data)
        exposure_times = np.array(exposure_times)
        H, W = image_array.shape[1:]

        print(f"Processing {key} — shape: {image_array.shape}, exposures: {exposure_times}")

        structured_data = np.zeros(len(exposure_times),
                                   dtype=[('exposure_time', float),
                                          ('image', float, (H, W))])
        structured_data['exposure_time'] = exposure_times
        for i in range(len(image_array)):
            structured_data['image'][i] = image_array[i]

        np.save(os.path.join(data_folder, f"{key}_raw.npy"), structured_data)

    print(f"Raw data saved in: {data_folder}")
    return data_folder

