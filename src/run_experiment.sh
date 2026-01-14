#!/bin/bash

# Sweep through all five Lambda values for Optimized Model
for l in 0.0 0.5 1.0 2.0 5.0
do
   python src/train_runner.py --model_version optimized --use_stats_prompts --lambda_cl $l
done

# Run the Original Prompt Baseline with lambda = 1.0
python src/train_runner.py --model_version original --lambda_cl 1.0


