#!/bin/bash

# Define the same Lambda values used in training
LAMBDAS="0.0 0.5 1.0 2.0 5.0"

echo "STARTING ABLATION TEST SWEEP"

# Test Optimized Model with Statistical Prompts (Phase 2 & 3)
for l in $LAMBDAS
do
   echo "Evaluating Statistical Prompts | Lambda: $l"
   python src/test_runner.py --model_version optimized --use_stats_prompts --lambda_cl $l --num_runs 3
done

# Test Optimized Model with Original Prompts (Phase 3 Architecture Only)
for l in $LAMBDAS
do
   echo "Evaluating Original Prompts | Lambda: $l"
   python src/test_runner.py --model_version optimized --lambda_cl $l --num_runs 3
done

echo "ABLATION TEST SWEEP COMPLETE"