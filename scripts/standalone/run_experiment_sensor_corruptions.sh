#!/bin/bash

cuda="${1:-0}"
num_runs="${2:-1}"

exp_name="sensor_corruptions"
datasets=("dsads" "mhealth" "pamap" "bnci1" "bnci2" "zhou")


for dataset in "${datasets[@]}"; do
  python3 -m experiments.experiment_sensor_corruptions --exp_name "$exp_name" --cuda "$cuda" --num_runs "$num_runs" --dataset "$dataset"
done