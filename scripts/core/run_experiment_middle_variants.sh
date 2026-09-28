#!/bin/bash

cuda="${1:-0}"
num_runs="${2:-1}"

exp_name="middle_variants"
datasets=("dsads" "mhealth" "pamap" "wisdm" "bnci1" "bnci2" "bnci4" "zhou")
fusions=("middle_e" "middle_n" "middle_en")
encoder="cnn"
algorithm="erm"

for fusion in "${fusions[@]}"; do
  for dataset in "${datasets[@]}"; do
    python3 -m experiments.experiment_main --exp_name "$exp_name" --cuda "$cuda" --num_runs "$num_runs" --dataset "$dataset" --algorithm "$algorithm" --fusion "$fusion" --encoder "$encoder"
  done
done
