# Github repository to project From naive to native form: Shared kinematic signatures of conventionalization and acquisition in manual language

This repository stores coding pipeline to process and analyze data associated with project "From naive to native form: Shared kinematic signatures of conventionalization and acquisition in manual language". This project investigates what kinematic features distinguish gesturers from signers ("conventionalization" trajectory) and sign learners from native signers ("acquisition" trajectory).

## Overview

The pipeline consists of several processing and analysis steps, whereby each step works on the output of the previous step. However, they are build in modular way such that one can implement individual scripts for their own purposes.

You can browse through the pipeline as a [website]([https://sarkadava.github.io/FLESH_Effort/](https://sarkadava.github.io/GestureToSign_KinematicTrajectory/)).

The pipeline is divided into the following steps:

- Pre-processing: Preparing video clips for analysis
- Motion tracking: 2D pose estimation using Sapiens2 
- Processing of motion into workable csv files
- Extraction of kinematic features
- Principal Component Analysis to identify main kinematic dimensions
- Statistical analysis: Modelling each component identified by PCA as a function of group identity (gesturer, learner, signer)
## Prerequisites

If you wish to use only some steps of the pipelineor the entire pipeline, you can follow the steps below. Note that this project mostly in Python, but implements also some steps in R. Note that, for example, Visual Studio Code allows one to run both Python and R scripts. 

```bash
# 1 - Clone the Repository
git clone https://github.com/sarkadava/GestureToSign_KinematicTrajectory.git
cd GestureToSign_KinematicTrajectory

# 2 - Create a TSPROCESS Conda Environment
conda env create -f environment.yml

# 3 - Add the Conda Environment to Jupyter Notebook
conda activate TSPROCESS
python -m ipykernel install --user --name=TSPROCESS --display-name "Python (TSPROCESS)"

# 4 - Run the Jupyter Notebook (Optional: You can also open the scripts in Visual Studio Code. Always make sure you are in the folder of the script you are running)
jupyter notebook
```

## How to cite

If you want to use and cite and part of the **coding pipeline**, cite:

Kadavá, Š. (2026). Supplementary Material to the project From naive to native form: Shared kinematic signatures of conventionalization and acquisition in manual language (Version 1.0.0) [Computer software]. https://github.com/sarkadava/GestureToSign_KinematicTrajectory

## Contact

kadava[at]leibniz-zas[dot]de (Šárka Kadavá)
