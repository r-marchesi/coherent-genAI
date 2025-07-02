# Coherent Cross-modal Generation of Synthetic Cancer Data Enables Robust Multimodal Modeling to Advance Precision Oncology

Repository for the paper **"Coherent Cross-modal Generation of Synthetic Cancer Data Enables Robust Multimodal Modeling to Advance Precision Oncology"**, by Raffaele Marchesi, Nicolò Lazzaro, Walter Endrizzi, Gianluca Leonardi, Matteo Pozzi, Flavio Ragni, Stefano Bovo, Monica Moroni, Venet Osmani, Giuseppe Jurman.

## Abstract

The integration of multimodal data is critical for precision oncology, but its clinical application is challenged by the issue of missing modalities. To address this, we developed a generative framework operating on a unified embedding level to synthesize any missing modality—copy-number alterations (CNA), transcriptomics (RNA-Seq), proteomics (RPPA), or histopathology (WSI)—from any available subset of this data. We introduce and benchmark two diffusion-based architectures: a flexible multi-condition model with a masking strategy that seamlessly handles arbitrary subsets of inputs, and Coherent Denoising, a novel ensemble method which aggregates predictions from multiple single-condition models and enforces consensus during the reverse diffusion process. On a cohort of over 10,000 TCGA samples spanning across 20 different tumor types, we demonstrate that both methods generate high-fidelity data that preserves the complex biological signals required for downstream cancer classification, stage prediction, and survival analysis. Furthermore, we show this framework can be used at inference-time to rescue the performance of predictive models on incomplete patient profiles and leverage counterfactual analysis to guide the prioritization of diagnostic tests. Our work establishes a robust and flexible generative framework for completing sparse multimodal, multi-omics datasets, providing a key step toward enhancing data-driven clinical decision support in oncology.


## Project Pipeline

The overall methodology of our work is captured in the figure below, detailing the journey from raw data preprocessing to generative model training and downstream application.

![Project Pipeline Overview](pipeline.png)


## Repository Structure

The repository is organized into the main analysis codebase and the data preprocessing pipeline.

* `code/`: Contains the core logic for model training, downstream analysis, and visualization.
    * `autoencoders/`: Scripts for training and using the modality-specific autoencoders to create the harmonized latent space.
    * `downstream/`: Contains all scripts and notebooks to reproduce the downstream analyses and figures presented in the paper (e.g., reconstruction fidelity, performance rescue, counterfactual inference).
    * `lib/`: A library of shared modules, including dataset handlers, diffusion model definitions, and training/sampling loops.
    * `main_train.py` & `main_test.py`: Main entry points for training the generative models and running evaluations.
    * `ZZZ_*.ipynb`: Exploratory notebooks for visualizing results and intermediate analyses.
* `preprocessing_TCGA/`: A sequential, numbered pipeline of scripts to download and process the raw TCGA data into the format required for the analysis.

## Requirements

Enviroment requirements for this project can be found in `requirements.txt`

## Contact

For any questions, please contact Raffaele Marchesi at rmarchesi@fbk.eu.
