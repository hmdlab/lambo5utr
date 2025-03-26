# MOBO-5UTR
## Overview
MOBO-5UTR is a sequence optimization model that combines [LaMBO](https://github.com/samuelstanton/lambo) and [DNABERT](https://github.com/jerryji1993/DNABERT). From the original LaMBO architecture, we incorporated the pre-trained DNA language model as a denoising autoencoder to leverage its learned genomic information. Our model demonstrated improved performance in optimizing human 5' UTR sequences across up to four computational objectives, including AGC content, mean ribosome load (MRL), in vitro stability, and reduced G4 tendency.

## Installation
### Environment
We have developed our model in the following environment. We strongly recommend using the same Python and CUDA versions, as our pipeline integrates multiple models with potentially conflicting dependencies.
- Linux architecture: x86_64
- Ubuntu: 20.04.6 LTS, focal
- CUDA: 11.3, V11.3.58
- Python: 3.8.10
- R: 4.1.2 (only for G4 prediction)

### Installation of MOBO-5UTR
Clone this repository to the local environment.
```
git clone https://github.com/hmdlab/mobo5utr.git
```
Install the necessary python packages along with the `lambo5utr` package as shown below.
```
pip3 install -r requirements.txt
pip3 install -e .
```

### Installation of other models
The following models are used in our pipeline. Optimus5Prime is included under `./misc/retrained_main_MRL_model.hdf5` as a retrained model using the same model and data provided in the original work. Additional models needed to be downloaded and installed separately.
- [DNABERT](https://github.com/jerryji1993/DNABERT): Pre-trained BERT with human genome used as the encoder and decoder of LaMBO
- [Optimus5Prime](https://github.com/pjsample/human_5utr_modeling): For predicting MRL
- [Nullrecurrent](https://github.com/eternagame/KaggleOpenVaccine): For predicting in vitro degradation
- [DeepG4](https://github.com/raphaelmourad/DeepG4): For predicting G4 score
1. Download DNABERT 3mer model
Download the DNABERT 3mer model from [here](https://github.com/jerryji1993/DNABERT).
Specify your local path in `./hydra_config/black_box_opt.yaml` as shown below.
```
pretrained_path: /path/to/your/DNABERT/3-new-12w-0
```
2. Installation of Nullrecurrent
The original instructions can be found [here](https://github.com/eternagame/KaggleOpenVaccine/tree/main?tab=readme-ov-file#setup).
   - Git clone the KaggleOpenVaccine with git-lfs. Available under MIT license.
```
sudo apt-get install git-lfs
git clone https://github.com/eternagame/KaggleOpenVaccine.git
```
   - Install EternaFold from [here](https://eternagame.org/tech). Freely available for non-commercial use upon registration.
   - Git clone Arnie from [here](https://github.com/DasLab/arnie) and link it to EternaFold. See detailed instructions [here](https://daslab.github.io/arnie/#/setup/environment).
```
git clone https://github.com/DasLab/arnie.git
```
   - Add path to the KOV direcotry so that the script can find the model parameters. It is also recommended to add it to `.bash_profile`.
```
export KOV_PATH="/path/to/KaggleOpenVaccine"
```
3. Installation of DeepG4
Build the R environment from renv.lock. The DeepG4 package will be included in the environment.
```
R
> renv::restore()
> q()
```
4. Quick test
Use the following scripts to test your installation. This test took ~20min with 2GPUs in our environment.
```
source run_test.sh
```

## Running MOBO-5UTR
Use the following script for running LaMBO-DNABERT.  
The entire training took ~12h with 2 GPUs in our environment.
```
source run_script_dnabert.sh
```

## Analyzing and ranking generated sequences
This section includes two notebooks, `notebooks/compare_ranking.ipynb` and `sequence_selection.ipynb`, for analyzing and ranking genrated sequences. You can use those sequences to test this section without actually running the model.  
`compare_ranking.ipynb` compares four candidate selection methods, including the simplely defined Pareto frontiers, Pareto ranking, MIPS scores, and R-method. `sequence_selection.ipynb` describes how we eventually selected sequences for wet-lab experiments. The selection step requires [USEARCH](https://www.drive5.com/usearch/) to be pre-installed as a command line tool in your environment.

## Citation
If you use this code for your research, please cite:
```
citation info to be provided.
```
