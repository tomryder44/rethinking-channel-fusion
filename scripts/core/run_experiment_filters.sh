#!/bin/bash

cuda="${1:-0}"
num_runs="${2:-1}"

exp_name="filters"
datasets=("dsads" "mhealth" "pamap" "wisdm" "bnci1" "bnci2" "bnci4" "zhou")
encoder="cnn"
algorithm="erm"
fusion="early"
num_filters=(16 32 64)

for dataset in "${datasets[@]}"; do
  for num_filt in "${num_filters[@]}"; do
    python3 -m experiments.experiment_main --exp_name "$exp_name" --cuda "$cuda" --num_runs "$num_runs" --dataset "$dataset" --algorithm "$algorithm" --fusion "$fusion" --encoder "$encoder" --set num_filters=$num_filt
  done
done
