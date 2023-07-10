
Synthetic Dataset Generator and Evaluator with UniformGAN

This repository is a toolset to the generation and evaluation of different synthetic datasets. The source code is easily modifiable and adaptable to new dataset generation techniques and also evaluation metrics. 

We hope that this will help people create and evaluate new synthetic dataset generation techniques.

## Installation

Prereqs: Python 3.7 (I think 3.8 will also work). Would recommend using [pyenv](https://github.com/pyenv/pyenv) to install. 

This project was tested with python 3.7.6.

First create a python environment.
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Remember to `source env/bin/activate` every time you want to run this! (Or run sh files i.e. `./run.sh` for simplified usage and automatic sourcing)

You can also just run `./install.sh` that will run this code. Just make sure you have the right python  version installed.

### MIT Supercloud addendum
Run these instructions before continuing.
```
 module load anaconda/2021a
 mkdir /state/partition1/user/$USER
 export TMPDIR=/state/partition1/user/$USER
```

## Instructions for using the CLI
There are two modes to using the cli: production and testing. In production, the original dataset is not split into training and testing datasets. In testing, the dataset is split into training and testing datasets.
To test, simply add the `--test` attribute to the command line arguments.

## tl;dr example with Adult dataset
```
./upload.sh --test # First time to generate a .json file
./upload.sh --test # Once you edit the .json file you can finally upload it!
./run.sh --test --datasets=Adult --generators=ctgan,tablegan --actions=gen
./run.sh --test --datasets=Adult --generators=uniformgan_003 --actions=gen --additionals=use_dp:True --postfix=_dp
./run.sh --test --datasets=Adult --generators=ctgan,tablegan,uniformgan_003 --actions=stats,privacy,eval
./run.sh --test --datasets=Adult --generators=ctgan,tablegan,uniformgan_003 --actions=aggregate
python complete_analysis.py.py
```

Finally check out `SUMMARY.csv`

### Uploading datasets and config files
In order to upload a dataset, simply add a csv file into the uploads folder. Next run `./upload.sh` and select the number corresponding to the dataset. During the initial setup, a companion json file will be generated that will have to be edited by the user. Name the dataset as well as specific attributes correlated with each dataset (ordinal and discrete columns). Once satisfied with settings, rerun `./upload.sh` to generate the dataset.

You can specify specifically the `--test_split` ratio and also the number of randomly split datasets.

You will have to run `./upload.sh` twice for each dataset you want to add (one to initialize the config file and one to copy over everything)

Example: `./upload.sh --test --test_split=0.2`

### Generating and evaluating datasets
After uploading the dataset, we can begin generating synthetic datasets. It is possible to generate multiple datasets at the same time.

`./run.sh --test --actions=gen --datasets=Adult --generators=ctgan,tablegan,uniformgan`

After generation, stats can be generated as well.
`./run.sh --test --actions=stats,privacy,eval --datasets=Adult --generators=ctgan,tablegan,uniformgan`

All these stats can be manually viewed in the `datasets_analysis/[DATASET]` folder.

### Adding your own GAN 
Adding a new synthetic training method is simple! 

All GAN generators should have a naming convention of  `generators/[generatorname].py`. Use `generator/[generatorname]_helper.py` to define any helper functions and imports.

Look at generators/README.md for more information

### Evaluating your own datasets
Upload your dataset using `./upload.sh` (you will have to call the function twice -- once to create the `.json` file and one to fully upload).

For the synthetic data in training mode (`--test` because you want to use `eval`), place them in `datasets_synthetic/[Datasetname]/[datasetname_lowercased]_train1_[GANname]-00.csv]` `datasets_synthetic/[Datasetname]/[datasetname_lowercased]_train1_[GANname]-01.csv]`

If your using production mode (`--production`), name it `datasets_synthetic/[Datasetname]/[datasetname_lowercased]_[GANname].csv]`

Now you can continue evaluating the synthetic dataset!! `./run.sh --dataset=[Datasetname] --generators=[GANname] --actions=stats` (for `--test or --production`)

### Adding your own evaluation metric
Adding an evaluation metric look at the `actions.py` file. Helper functions are located in the `analysis` subfolder.

### Constraints
There are a few known constraints with this framework. 
Strings and numbers only. Unfortunately, time series and timestamps will not work.

Specifically, when generating with TableGAN, it is required that the target column must be the last column in the csv and also binary (must have exactly 2 different options).

Furthermore, datasets should not contain NaN rows. NaNs can either be replaced in preprocessing with a "None" string substitution or for continuous rows, a secondary row where one column is a discrete column of whether the original column is NaN.

## Possible bugs
Astype issues with real and synthetic