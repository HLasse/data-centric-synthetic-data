# Data-centric Synthetic Data Generation

Code for the paper *"Reimagining Synthetic Data Generation through Data-Centric AI: A Comprehensive Benchmark"*. 


## Replicating the paper's results


### Installation
Clone the repository to your computer, create a new virtual environment, and install the package.


```bash
git clone <URL OF REPO>
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
