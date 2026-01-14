#!/bin/bash

echo "STARTING ABLATION TRAIN SWEEP"

# Original model with original prompt baseline with lambda = 1.0
python src/train_runner.py --model_version original --lambda_cl 1.0

# Original model with stats prompt baseline with lambda = 1.0
python src/train_runner.py --model_version original --use_stats_prompts --lambda_cl 1.0

# Sweep through five Lambda values for Optimized Model with Original Prompts
for l in 0.0 0.5 1.0 2.0
do
   python src/train_runner.py --model_version optimized --lambda_cl $l
done

# Sweep through five Lambda values for Optimized Model with Stats Prompts
for l in 0.0 0.5 1.0 2.0 5.0
do
   python src/train_runner.py --model_version optimized --use_stats_prompts --lambda_cl $l
done

echo "ABLATION TRAIN SWEEP COMPLETE"

