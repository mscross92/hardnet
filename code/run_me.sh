#!/bin/bash

RUNPATH="$( cd "$(dirname "$0")" ; pwd -P )/.."
DATASETS="$RUNPATH/data/sets"
DATALOGS="$RUNPATH/data/logs"

mkdir -p "$DATASETS"
mkdir -p "$DATALOGS"

( # Run the code
    cd "$RUNPATH"
    python ./code/HardNet.py --w1bsroot "$DATASETS/" --fliprot=False --training-set=turbid_milk --imageSize=29 --experiment-name=exp_train_random/ $@ | tee -a "$DATALOGS/log_HardNet_A.log"
)

