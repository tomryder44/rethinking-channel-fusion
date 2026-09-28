#!/bin/bash

cuda="${1:-0}"
num_runs="${2:-1}"

exp_name="dg"
datasets=("dsads" "mhealth" "pamap" "wisdm" "bnci1" "bnci2" "bnci4" "zhou")
fusion="early"
encoder="cnn"


algorithm="mmd"
penalties=(0.1 0.5 1 2 5)
for dataset in "${datasets[@]}"; do
  for pen in "${penalties[@]}"; do
    python3 -m experiments.experiment_main --exp_name "$exp_name" --cuda "$cuda" --num_runs "$num_runs" --dataset "$dataset" --algorithm "$algorithm" --fusion "$fusion" --encoder "$encoder" --set penalty="$pen"
  done
done


algorithm="vrex"
penalties=(1 5 10 20 50)
for dataset in "${datasets[@]}"; do
  for pen in "${penalties[@]}"; do
    python3 -m experiments.experiment_main --exp_name "$exp_name" --cuda "$cuda" --num_runs "$num_runs" --dataset "$dataset" --algorithm "$algorithm" --fusion "$fusion" --encoder "$encoder" --set penalty="$pen"
  done
done


algorithm="nnr"
penalties=(0.0001 0.0005 0.001 0.002 0.005)
for dataset in "${datasets[@]}"; do
  for pen in "${penalties[@]}"; do
    python3 -m experiments.experiment_main --exp_name "$exp_name" --cuda "$cuda" --num_runs "$num_runs" --dataset "$dataset" --algorithm "$algorithm" --fusion "$fusion" --encoder "$encoder" --set penalty="$pen"
  done
done


algorithm="irm"
penalties=(0.0001 0.001 0.01 0.1 1)
for dataset in "${datasets[@]}"; do
  for pen in "${penalties[@]}"; do
    python3 -m experiments.experiment_main --exp_name "$exp_name" --cuda "$cuda" --num_runs "$num_runs" --dataset "$dataset" --algorithm "$algorithm" --fusion "$fusion" --encoder "$encoder" --set penalty="$pen"
  done
done


algorithm="groupdro"
penalties=(0.0001 0.001 0.01 0.1 1)
for dataset in "${datasets[@]}"; do
  for pen in "${penalties[@]}"; do
    python3 -m experiments.experiment_main --exp_name "$exp_name" --cuda "$cuda" --num_runs "$num_runs" --dataset "$dataset" --algorithm "$algorithm" --fusion "$fusion" --encoder "$encoder" --set penalty="$pen"
  done
done


algorithm="diversify"
Ks=(2 3 5)
for dataset in "${datasets[@]}"; do
  for K in "${Ks[@]}"; do
    python3 -m experiments.experiment_main --exp_name "$exp_name" --cuda "$cuda" --num_runs "$num_runs" --dataset "$dataset" --algorithm "$algorithm" --fusion "$fusion" --encoder "$encoder" --set K="$K"
  done
done


algorithm="phaser"
ks=(0.0625 0.125 0.25 0.5)
for dataset in "${datasets[@]}"; do
  for k in "${ks[@]}"; do
    python3 -m experiments.experiment_main --exp_name "$exp_name" --cuda "$cuda" --num_runs "$num_runs" --dataset "$dataset" --algorithm "$algorithm" --fusion "$fusion" --encoder "$encoder" --set nperseg_k="$k"
  done
done

