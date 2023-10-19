# Data-centric Synthetic Data Generation

Code for the paper [*"Reimagining Synthetic Data Generation through Data-Centric AI: A Comprehensive Benchmark"*](https://neurips.cc/virtual/2023/poster/73527) accepted at NeurIPS 2023.


## Replicating the paper's results


### Installation
Clone the repository to your computer, create a new virtual environment, and install the package.


```bash
git clone https://github.com/HLasse/data-centric-synthetic-data
cd data-centric-synth
pip install -e .
## if this fails, install torch before installing the library
pip install torch torchaudio torchvision
pip install -e .
```

### Running experiments

To run the main experiment, run the following file:

```python
python src/application/main_experiment/run_main_experiment.py
```

Note, this will take a long time and require a GPU (~600-1.000 GPU hours). 
Modify the number of seeds/generative models in the file for faster training.


To run the experiment to produce figure 1, run the following file:

```python
python src/application/figure1/run_figure1_exp.py
```


To run the experiment on adding label noise to the Covid mortality dataset, run the following file:

```python
python src/application/main_experiment/run_noise_experiment.py
```

To run hyperparameter tuning of generative models, run the following file:

```python
python src/application/synthcity_hparams/optimize_model_hparams.py
```


### Replicating graphs and tables

To replicate the plots and tables from the main paper, run the following bash script:

```bash
sh replicate_main_paper.sh
```

To replicate the plots and tables from the appendix, run the following bash script:

```bash
sh replicate_appendix.sh
```

Tables will printed to the terminal and plots saved to `results/figure1|main_experiment'



### Repository structure
The `data` folder in the root of the repo contains the output of the experiments run in the benchmark. The `results` folder contains the processed outputs of the `data` folder such as plots.

All source code can be found in the `src` folder. `src` contains two subfolders, `data_centric_synth` which contains the bulk of the code used to do data profiling, train generative models and generate data, etc, and `application` which contains code for running the specific experiments.

An overview of the content of `src` can be found below

```
├── application/
│   ├── data_centric_thresholds/ # scripts for running and finding the optimal data-centric thresholds
│   ├── figure1/ # code for creating figure 1 in the paper
│   ├── main_experiment/
│   │   ├── eval/ # scripts related to evaluation, i.e. plots and tables
│   │   ├── run_main_experiment.py # run the main experiment
│   │   ├── run_noise_experiment.py # run the label noise experiment
│   │   └── run_org_data_postprocessing_experiment.py # run the postprocessing of real data experiment
│   ├── stat_dist/ # code related to extracting statistical fidelity
│   ├── synthcity_hparams/ # scripts for optimizing hyperparameters of the generative models
│   └── constants.py # constants such as directories

├── data_centric_synth/
│   ├── causal_discovery/ # code related to the experiment on structure learning
│   ├── data_models/ # data classes for the different experiments
│   ├── data_sculpting/ # code to do data-profiling/sculpting
│   ├── dataiq/ # implementation of data iq and data maps
│   ├── datasets/ # loaders for the different datasets
│   ├── evaluation/ # helpers for evaluation of the experiments
│   ├── experiments/ # main experimental loops
│   ├── serialization/ # helper functions for saving/loading pickle files
│   ├── synthetic_data/ # functinos for generation synthetic data
│   └── utils.py # utility for setting random seed globally
```



