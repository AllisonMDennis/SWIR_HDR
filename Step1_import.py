import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from datetime import datetime
import io


def load_parameters(param_directory):
    """Load processing parameters from files in specified directory.
    Parameters are to be used with process_and_save function."""
    param_files = {
        'b': 'b.npy',
        'Sd': 'Sd.npy',
        'Smax': 'Smax.npy',
        'smooth': 'smooth.npy',
        'Slinear': 'Slinear.npy'
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
            
            if 'clip' or 'cliptop' in operations:
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

def import_and_save_raw(directory, experiment_title, base_data_folder, date_cutoff_1='2024/02/13', date_cutoff_2='2024/11/07'):
    """
    Import raw data from h5 files and save as numpy arrays.
    
    Args:
        directory: Base directory containing the data
        experiment_title: Title of the experiment
        base_data_folder: Name of the folder to store raw data
        date_cutoff_1: First cutoff date string in format 'YYYY/MM/DD' 
        date_cutoff_2: Second cutoff date string in format 'YYYY/MM/DD'
    """
    from datetime import datetime
    
    laser_wavelengths = {'1': '670', '2': '760', '3': '808'}
    
    # Define three sets of emission filters
    early_emission_filters = {
        '12': 'BP1100', '13': 'BP1150', '14': 'BP1300',
        '15': 'BP1350', '16': 'BP1500', '17': 'BP1550'
    }
    
    middle_emission_filters = {
        '12': 'BP1150', '13': 'BP1200', '14': 'BP1250',
        '15': 'BP1300', '16': 'BP1350', '17': 'BP1575'
    }
    
    latest_emission_filters = {
        
        '10': 'NIRIILP', '11': 'LP1250', '12': 'BP1150', '13': 'BP1200', '14': 'BP1300',
        '15': 'BP1350', '16': 'BP1500', '17': 'BP1550'
    }
    
    data_folder = os.path.join(directory, base_data_folder, "raw_data")

    os.makedirs(data_folder, exist_ok=True)
    
    cutoff_datetime_1 = datetime.strptime(date_cutoff_1, '%Y/%m/%d')
    cutoff_datetime_2 = datetime.strptime(date_cutoff_2, '%Y/%m/%d')
    
    for laser_key, laser_value in laser_wavelengths.items():
        for file in os.listdir(directory):
            if not file.startswith(f"{experiment_title}_{laser_key}_"):
                continue
                
            try:
                with h5py.File(os.path.join(directory, file), 'r') as h5f:
                    # Get timestamp and seconds from file
                                         
                    timestamp_bytes = h5f['Cube']['Timestamp'][()]
                    timestamp_str = str(timestamp_bytes.item())
                    timestamp_str = timestamp_str.strip("b'") 
                    file_date = datetime.strptime(timestamp_str, '%Y/%m/%d %H:%M:%S.%f')
                    
                    # Determine which filter set to use based on date and seconds
                    if file_date < cutoff_datetime_1:
                        emission_filters = early_emission_filters
                    elif file_date < cutoff_datetime_2:
                        emission_filters = middle_emission_filters
                    elif file_date >= cutoff_datetime_2:
                        emission_filters = latest_emission_filters
                    
                    # Extract filter key from filename
                    parts = file.split('_')
                    filter_key = next((part for part in parts if any(fk in part for fk in emission_filters.keys())), None)
                    
                    if filter_key is None:
                        print(f"Could not find valid filter key in filename: {file}")
                        continue
                        
                    filter_key = next(fk for fk in emission_filters.keys() if fk in filter_key)
                    filter_value = emission_filters[filter_key]
                    
                    key = f"{experiment_title}_{laser_value}_{filter_value}"
                    
                    # Rest of the processing remains the same as before
                    image_files = [f for f in os.listdir(directory) 
                                 if f.startswith(f"{experiment_title}_{laser_key}_{filter_key}")]
                    
                    if not image_files:
                        print(f"No files found for combination: {key}")
                        continue
                    
                    image_files.sort(key=lambda x: float(x.split('_')[-1][:-3]))
                    image_data = []
                    exposure_times = []
                    
                    for img_file in image_files:
                        file_path = os.path.join(directory, img_file)
                        try:
                            with h5py.File(file_path, 'r') as img_h5f:
                                image = img_h5f['Cube']['Images'][()]
                                exposure_time = img_h5f['Cube']['TimeExposure'][()].item()
                                
                                if image.ndim == 3 and image.shape[0] == 1:
                                    image = image.squeeze(0)
                                
                                image_data.append(image)
                                exposure_times.append(exposure_time)
                        except Exception as e:
                            print(f"Error processing file {img_file}: {str(e)}")
                            continue
                    
                    if not image_data:
                        print(f"No valid data collected for combination: {key}")
                        continue
                    
                    image_array = np.array(image_data)
                    exposure_times = np.array(exposure_times)
                    
                    print(f"Processing {key} - Shape: {image_array.shape}, Exposure times: {exposure_times}")
                    print(f"Used filter set based on date {timestamp_str}")
                    
                    structured_data = np.zeros(len(exposure_times),
                                            dtype=[('exposure_time', float),
                                                  ('image', float, (640, 512))])
                    structured_data['exposure_time'] = exposure_times
                    for i, image in enumerate(image_array):
                        structured_data['image'][i] = np.squeeze(image_array[i])
                    
                    print(f"Final shape of image array for {key}: {structured_data.shape}, "
                          f"dtype: {structured_data.dtype}")
                    np.save(os.path.join(data_folder, f"{key}_raw.npy"), structured_data)
                    
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                continue
    
    print(f"Raw data saved in: {data_folder}")
    return data_folder

def calculate_Slinear_adjusted(directory, experiment_title, Sd, b, Slinear, base_data_folder):
    clipped_denoised_data_folder = os.path.join(directory, base_data_folder, "final_data")
    
    Slinear_adjusted = np.zeros((640, 512))

    for file in os.listdir(clipped_denoised_data_folder):
        if file.endswith("_clipped_denoised.npy"):
            clipped_denoised_data = np.load(os.path.join(clipped_denoised_data_folder, file))
            
            # Access all exposure times at once
            exposure_times = clipped_denoised_data['exposure_time']
            
            for t in exposure_times:
                Slinear_adjusted = np.maximum(Slinear_adjusted, Slinear - Sd * t - b)

    np.save(os.path.join(clipped_denoised_data_folder, f"{experiment_title}_Slinear_adjusted.npy"), Slinear_adjusted)
    print(f"Slinear_adjusted saved in: {clipped_denoised_data_folder}")
