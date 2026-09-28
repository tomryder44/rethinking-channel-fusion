#!/bin/bash

cuda=4
num_runs=5

start=$(date +%s)

bash scripts/standalone/run_experiment_late_weights.sh "$cuda" "$num_runs"
bash scripts/standalone/run_experiment_fusion_selection.sh "$cuda" "$num_runs"
bash scripts/standalone/run_experiment_sensor_corruptions.sh "$cuda" "$num_runs"


end=$(date +%s)
echo "Total time: $((end - start)) seconds"
