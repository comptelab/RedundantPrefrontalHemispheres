# Code accompanying the preprint:
## ["Redundant prefrontal hemispheres adapt storage strategy to working memory demands"](https://www.biorxiv.org/content/10.1101/2025.01.15.633176)

This repository contains the code used to generate the main figures and supplementary analyses presented in the preprint. The code includes simulations, data analysis, and plotting routines.

### Getting started:
To reproduce the results, we recommend using the environment provided in environment.yml. You can create the environment using:
```
conda env create --name <env-name> --file=environment.yml
conda activate <env-name>
```
The environment is compatible with both, data analysis and model simulations. (Installation time ~10 min.)

### Demo-Data availability
The Demo-Data (example session) with all analysis run can be found here: 
- Unzip the file
- Move contents to your local ./Results folder
- All monkey analyses can be run with the Demo-Data, though figures may change appearance (for Fig. 7, also download the human data (see below)).

### Data availability
To reproduce the full figures (not just the example session), you can download the full data set (4GB)
To download the data, go to:
- **Monkey data**: https://doi.org/10.1184/R1/31431415 from McDonnell, Umakantha, Williamson et al. (2026) 
- **Human multi-item data**: https://osf.io/67tn3/overview and https://osf.io/krv7g/overview from Schneegans and Bays (2016) and Schneegans and Bays (2018) (Fig. 7)

### Data preprocessing
The full monkey data is downloaded as Matlab-files. 
To combine these files into the here used data frames, run the scripts from the main folder:
- **read_MatFiles.py**: transforms the individual files into one data frame and aligns the eye tracker.  
- **create_sequentialDataframe.py**: creates a behavioral dataframe from this information using only correct, sequential trials.

### Repository structure and analyses:
- **Folders**: Folders contain the available code to replicate each figure.
- **Jupyter notebooks**: Notebooks replicate all panels of a figure. The notebooks create figures directly, using preprocessed data and simulation results.
- **Python scripts**: Scripts contain more computationally expensive processing steps or simulations, which generate the results used for plotting. Therefore the scripts should typically be run before the jupyter notebooks.
- **Results**: The Results folder is empty for now, fill it with files from the Demo-Data or run yourself.

### Contact
If you have any questions, encounter issues, or would like to discuss the methods or paper, feel free to reach out via github or email (mel.tschiersch@gmail.com)
