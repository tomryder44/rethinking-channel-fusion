#!/bin/bash

cuda=4
num_runs=5

start=$(date +%s)

bash scripts/core/run_experiment_main.sh "$cuda" "$num_runs"
bash scripts/core/run_experiment_early_ensemble.sh "$cuda" "$num_runs"
bash scripts/core/run_experiment_middle_variants.sh "$cuda" "$num_runs"
bash scripts/core/run_experiment_filters.sh "$cuda" "$num_runs"
bash scripts/core/run_experiment_encoders.sh "$cuda"  "$num_runs"
bash scripts/core/run_experiment_dg.sh "$cuda" "$num_runs"


end=$(date +%s)
echo "Total time: $((end - start)) seconds"
