# SWIR_HDR

This repository provides code that combines multiple SWIR image exposures into a single HDR radiance map.


## Installation
To try out this code, you need to [install Anaconda](https://www.anaconda.com/download).

Next, in a terminal or using VS Code, do:

```conda env create -f SWIR_HDR.yml```
```conda activate SWIR_HDR```

## Usage
There are 3 "Step\#_..py" files in this directory. Each file provides functionality to accomplish 3 typical steps of HDR processing. They are intended to be used as follows:

`Step0_DCmodel.py` uses dark count and reflectance or fluorescence images to generate per-pixel models of the exposure time-dependent dark current noise, saturation thresholds, and camera response function. The results of Step 0 are saved as 4 numpy (`file.npy`) arrays that are used in downstream processing steps. The functions in Step 0 only need to be performed once for each detector/imager.

In `Step1_import.py`, functionality is provided so that the h5 files are loaded and the metadata used to extract the relvant exposure times.

In `Step2_radiance.py`, functionality is provided so that the organized data are processed using the dark current model for noise subtraction, the saturation limit to remove non-linear responses, and a weighting function and camera response functions applied for high dynamic range radiance map construction.

## Data
The data folder has some example HDF5 (h5) files and associated images. The `DCcalibration` folder has dark current data at different exposure times. The `CRFrecovery` folder has data for calculating a Camera Response Function. The `Smax_calibration` folder has intensity data for calculating per-pixel saturation.


## CITING 
Our preprint describing this approach is available on bioRxiv.

Amish Patel, Xingjian Zhong, Mallory Moffett, Yidan Sun, Allison M. Dennis, *High dynamic range shortwave infrared (SWIR) imaging of mice with an InGaAs camera* bioRxiv 2025.11.06.687088; doi: https://doi.org/10.1101/2025.11.06.687088


Citation:
```
@article {Patel2025.11.06.687088,
	author = {Patel, Amish and Zhong, Xingjian and Moffett, Mallory and Sun, Yidan and Dennis, Allison M.},
	title = {High dynamic range shortwave infrared (SWIR) imaging of mice with an InGaAs camera},
	year = {2025},
	doi = {10.1101/2025.11.06.687088},
	URL = {https://www.biorxiv.org/content/early/2025/11/08/2025.11.06.687088},
	eprint = {https://www.biorxiv.org/content/early/2025/11/08/2025.11.06.687088.full.pdf},
	journal = {bioRxiv}
}
```
