#!/bin/bash

echo "STARTING ABLATION TRAIN SWEEP"

# Sweep through five Lambda values for Optimized Model with Stats Prompts
for l in 0.0 0.5 1.0 2.0 5.0
do
   python src/train_runner.py --model_version optimized --use_stats_prompts --lambda_cl $l
done

echo "ABLATION TRAIN SWEEP COMPLETE"


