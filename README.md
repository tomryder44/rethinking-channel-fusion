# Rethinking channel fusion for robust multivariate time series classification under distribution shift

This repository contains the code for our TMLR paper, which is available [here](https://openreview.net/forum?id=lbAzMRlrFC).

If something doesn't work or isn't clear (about the code or the paper), feel free to open an issue or reach out at `tmer20@bath.ac.uk`.

## Set up
First, clone the repo:
```bash
git clone https://github.com/tomryder44/rethinking-channel-fusion.git
cd rethinking-channel-fusion
```
Then install the required packages:
```bash
pip install -r requirements.txt
```

## Datasets
The datasets themselves are not included in this repository, so they need to be downloaded and preprocessed.

### HAR datasets
Download each dataset from the link below and place its files in the folder shown.

| Dataset | Source | Folder | Contents |
|---|---|---|---|
| DSADS | [UCI](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities) | `datasets/raw/dsads/` | `a01/` ... `a19/` |
| MHEALTH | [UCI](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) | `datasets/raw/mhealth/` | `mHealth_subject1.log` ... |
| PAMAP2 | [UCI](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) | `datasets/raw/pamap/` | `subject101.dat` ...|
| WISDM | [WISDM](https://www.cis.fordham.edu/wisdm/dataset.php) | `datasets/raw/wisdm/` | `WISDM_ar_v1.1_raw.txt` |

### MI datasets
These are loaded through [MOABB](https://moabb.neurotechx.com/) (Mother of all BCI Benchmarks).
They are downloaded automatically the first time they are preprocessed, so no manual download is needed.

| Dataset | Source |
|---|---|
| BNCI-1 | [MOABB](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html) |
| BNCI-2 | [MOABB](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_002.html) | 
| BNCI-4 | [MOABB](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_004.html) | 
| ZHOU | [MOABB](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Zhou2016.html) |

### Preprocessing
To preprocess a single dataset, run (e.g. for DSADS):
```bash
python -m data_processing.preprocessing.run_preprocessing --dataset dsads
```
The data is saved to `datasets/processed/dsads/`, with one input file and one label file per subject (`0_x.npy`, `0_y.npy`, `1_x.npy`, `1_y.npy`, ...).

The valid dataset names are: `dsads`, `mhealth`, `pamap`, `wisdm`, `bnci1`, `bnci2`, `bnci4`, `zhou`.

## Running experiments
The various experiment entry points are in `experiments/`, and each can be run with command line arguments.
For example:
```bash
python -m experiments.experiment_main --exp_name test --cuda 0 --num_runs 3 --dataset dsads --algorithm erm --fusion early --encoder cnn
```
- `exp_name` is the name of the experiment, and sets the folder in `outputs/` where the log files and config/results files are saved to
- Hyperparameters can be overridden with `--set`, e.g. `--set num_filters=16`
- The valid encoder/fusion/algorithm combinations are listed in `mapping.py`

The following hyperparameters can be set: `lr` (learning rate), `wd` (weight decay), `num_epochs`, `batch_size`, `num_filters`.
There are also algorithm-specific hyperparameters that can be found in their trainer class (e.g. in `trainers/dg/`).

The `scripts/` folder contains the bash scripts used to generate the results in the paper.
They call the experiment scripts, looping over datasets and experiment settings.
Each script takes the GPU index and the number of runs (seeds) as arguments:
```bash
bash scripts/core/run_experiment_main.sh 0 3
```
Here `0` is the GPU and `3` is the number of runs.

## Plotting results
Results are saved in the config `.yaml` files in `outputs/configs/<exp_name>/`, and logs in `outputs/logs/<exp_name>/`.

Once an experiment has finished, the figures can be generated with the scripts in `plots/`, e.g.:
```bash
python -m plots.plot_ood_performance
```
At present, the plotting scripts have the experiment names we used hardcoded.
Either use the same experiment names (as they are in `scripts/`), or change the name of the results folder to match the code or the code to match the results folder (sorry).

Figures are saved to `outputs/plots/`.

Note: the figures use LaTeX for text rendering, so a LaTeX installation is required.
