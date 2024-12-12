# Automated Generation of Natural Product Drug Structures: A VAE-BO Framework for Molecular Design

- Jingwen Du, jingwen.du@vanderbilt.edu
- Carol He, carol.he@vanderbilt.edu

This repository contains the data, code, and results for our CS 6362: Advanced Machine Learning project.

## Datasets

The `data` folder contains various files with different formats of 404,318 natural products (NPs) from the NPAtlas and COCONUT databases.

- `data/np_selfies_dataset.csv`: SELFIES strings for all 400k+ NPs
- `data/np_smiles_dataset.csv`: SMILES strings for all 400k+ NPs
- `data/np_smiles_logp_qed_sas.csv`: SMILES strings and property values (logP, QED, SAS) for all 400k+ NPs

## Chemical VAE

Our very first experiments involve training the Chemical VAE designed by [Gómez-Bombarelli et al.](https://pubs.acs.org/doi/epdf/10.1021/acscentsci.7b00572?ref=article_openPDF) on NPs and comparing it (as a baseline) to our SELFIES VAE. We trained and evaluated the Chemical VAE directly using the authors' original code, which are included in the `chemical_vae` and `chemvae` folders.

- `chemvae`: Contains the model implementation of the Chemical VAE
- `chemical_vae/models`
  - `np_properties`: Contains the hyperparameters for and weights from training the Chemical VAE (with property prediction model) on 10k NPs on the CPU
  - `zinc_properties`: Contains the hyperparameters for and pretrained weights from training the Chemical VAE (with property prediction model) on 250k molecules from ZINC dataset
- `chemical_vae/analysis`
  - `intro_to_chemvae.ipynb`: Jupyter notebook from original Chemical VAE github that does some basic analysis on the pretrained VAE. Updated to include validity, diversity, and reconstruction accuracy calculation
  - `intro_to_chemvae_np.ipynb`: Same Jupyter notebook except the contents are adapted to analyzing the VAE trained on NPs

## SELFIES VAE

We implemented a standard VAE that encodes from and decodes to one-hot SELFIES encoding and drew inspiration from the SELFIES paper written by [Krenn et al.](https://iopscience.iop.org/article/10.1088/2632-2153/aba947/pdf). We compared this VAE with the Chemical VAE, tuned its hyperparameters, and ran BO on its latent space.

- `selfiesvae`: Contains the model implementation of the SELFIES VAE

### Initial Experiments

- `selfies_vae/initial_exps`
  - `datasets/`: 200k NP and QM9 datasets used
  - `settings*.yml`: Contains hyperparameters that the SELFIES paper used (fitted for the QM9 dataset)
  - `preprocess*.py`: Scripts used to preprocess the NP and QM9 datasets to generate SELFIES encodings
  - `train*.py`: Scripts used to train the SELFIES VAE on NP and QM9 datasets
  - `results*.dat`: Files containing the validity, diversity, and reconstruction accuracy results for both VAE models

### Hyperparameter Tuning

- `selfies_vae/hp_tuning`: Code used to tune the SELFIES VAE on 50k NPs using 25 trials
  - `datasets/np_50k.csv`: 50k NP dataset
  - `settings_50k_hp_tuning.yml`: Contains hyperparameters that will not be tuned (e.g. output file names) and the names of hyperparameters that will be tuned
  - `preprocess_50k.py`: Script for preprocessing 50k NP dataset
  - `np_50k_preprocessed.zip`: Zip file containing the outputs of the preprocessing script (one-hot encodings for all 50k NPs, training dataset, validation dataset, and SELFIES alphabet)
  - `tune_hp_50k_25.py`: Script for hyperparameter tuning
  - `results50k_25trials/*`: Validity, diversity, and reconstruction accuracy values for each trial

### Training with Optimal Hyperparameters

- `selfies_vae/train_50k_optim`: Code used for training the SELFIES VAE on the 50k NP dataset using the optimal hyperparameters, while saving the weights and logging loss values
  - `datasets/np_50k.csv`: 50k NP dataset
  - `settings_50k_optim.yml`: Optimal hyperparameters and output file names
  - `np_50k_preprocessed.zip`: Zip file containing the outputs of the preprocessing script (one-hot encodings for all 50k NPs, training dataset, validation dataset, and SELFIES alphabet)
  - `train_50k.py`: Script for training the model
  - `results_50k_optim.dat`: Validity, diversity, and reconstruction accuracy values during training
  - `saved_models_50k_optim/*`: Encoder and decoder weights for each epoch
  - `training_logs_50k_optim/*`: Logs containing the training loss, validation loss, KL divergence, and reconstruction loss per epoch
  - `train_plots.py`: Script used for generating the 4 loss vs epoch plots shown in `Training_Logs_Grid_50k_optim.png`

### Bayesian Optimization

- `selfies_vae/bayesian_optimization`: Code used for running BO on the SELFIES VAE model trained on the 50k dataset with optimal hyperparameters
  - `decoder.pt`: Saved decoder weights from last epoch of training
  - `settings_50k_optim.yml`: Hyperparameters needed for BO, such as latent dimension
  - `np_50k_preprocessed.zip`: Zip file containing one-hot encodings for all 50k NPs, training dataset, validation dataset, and SELFIES alphabet
  - `bo.py`: BO implementation
  - `bayesian_opt.py`: Script for setting up and running BO on trained VAE's latent space
  - `bo_progress_50k_optim*.yml`: Files containing the latent vectors sampled and their corresponding objective function value (5 \* QED - SAS). Each file corresponds to a different BO trial, and each trial may differ in the number of BO iterations, number of initial training samples, and bounds from which to sample.

### Notes on BO

- From a high-level qualitative inspection of the BO trial results, BO mainly sampled molecules that had similar low objective values across all trials (we wanted BO to maximize the objective value).
- Different combinations of BO hyperparameters were experimented with. Changing the sampling bounds greatly affected the effectiveness of the sampling process. However, more time is needed to determine which bounds are more useful.
- The VAE model used was only trained on 50k NPs, so the model's latent space is most likely not informative enough for BO to find very optimal molecules.
- The objective function used for BO here (5 \* QED - SAS) is often used in literature but is targeted towards synthetic molecules, but NPs are not easily synthesized.
