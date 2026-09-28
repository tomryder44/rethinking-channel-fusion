#!/bin/bash

cuda="${1:-0}"
num_runs="${2:-1}"

exp_name="early_ens"
datasets=("dsads" "mhealth" "pamap" "wisdm" "bnci1" "bnci2" "bnci4" "zhou")
fusion="early"
encoder="cnn"
algorithm="erm_ens"
num_models=(5 10 "C")

for dataset in "${datasets[@]}"; do
  for nm in "${num_models[@]}"; do
    python3 -m experiments.experiment_main --exp_name "$exp_name" --cuda "$cuda" --num_runs "$num_runs" --dataset "$dataset" --algorithm "$algorithm" --fusion "$fusion" --encoder "$encoder" --set num_models="$nm"
  done
done
