#!/bin/bash

echo "STARTING GRAD-CAM EXPLAINABILITY DIAGNOSTIC"

python src/explainability_runner.py  --lambda_cl 2.0 --model_version optimized --conflicts BitTorrent Gmail Gmail Skype 

echo "GRAD-CAM EXPLAINABILITY DIAGNOSTIC COMPLETE"