#!/bin/bash

cuda="${1:-0}"
num_runs="${2:-1}"

exp_name="late_weights"
datasets=("dsads" "mhealth" "pamap" "wisdm" "bnci1" "bnci2" "bnci4" "zhou")


for dataset in "${datasets[@]}"; do
  python3 -m experiments.experiment_late_weights --exp_name "$exp_name" --cuda "$cuda" --num_runs "$num_runs" --dataset "$dataset"
done