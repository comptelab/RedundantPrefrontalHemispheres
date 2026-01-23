# Code accompanying the preprint:
## ["Redundant prefrontal hemispheres adapt storage strategy to working memory demands"](https://www.biorxiv.org/content/10.1101/2025.01.15.633176)

This repository contains the code used to generate the main figures and supplementary analyses presented in the preprint. The code includes simulations, data analysis, and plotting routines.

🔒 Note: The dataset used in the study is not yet publicly available, but will be shared via this repository as soon as possible.

### Getting started:
To reproduce the results, we recommend using the environment provided in environment.yml. You can create the environment using:
```
conda env create --name <env-name> --file=environment.yml
conda activate <env-name>
```
The environment is compatible with both, data analysis and model simulations.

### Repository structure:
- **Folders**: Folders contain the available code for each figure.
- **Jupyter notebooks**: Notebooks replicate panels of specific figures. The notebooks create figures directly, using preprocessed data and simulation results.
- **Python scripts**: Scripts contain more computationally expensive preprocessing steps or simulations.
- **Results**: The Results folder is empty for now. I can send an example session upon request but it is slightly too large for simple uploading.

### Notes
While the notebooks should run as-is (once the data is available), some files assume a specific folder structure and data location, which will be clarified once the dataset is released.
All figures should be reproducible from the notebooks. Though figures will change when only running with the currently available single session.

### Contact
If you have any questions, encounter issues, or would like to discuss the methods or paper, feel free to reach out via github or email (mel.tschiersch@gmail.com)
