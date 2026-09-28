#!/bin/bash

cuda="${1:-0}"
num_runs="${2:-1}"

exp_name="encoders"
datasets=("dsads" "mhealth" "pamap" "wisdm" "bnci1" "bnci2" "bnci4" "zhou")
encoders=("transformer" "cnn_lstm")
fusions=("early" "middle" "late")
algorithm="erm"

for dataset in "${datasets[@]}"; do
  for encoder in "${encoders[@]}"; do
    for fusion in "${fusions[@]}"; do
      python3 -m experiments.experiment_main --exp_name "$exp_name" --cuda "$cuda" --num_runs "$num_runs" --dataset "$dataset" --algorithm "$algorithm" --fusion "$fusion" --encoder "$encoder"
    done
  done
done
