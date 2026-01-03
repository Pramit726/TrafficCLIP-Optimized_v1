#!/bin/bash
set -e  # Stop on error

echo "=== Step 1: Download Miniconda ==="
cd /content
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

echo "Miniconda installer downloaded."

echo "=== Step 2: Install Miniconda ==="
bash Miniconda3-latest-Linux-x86_64.sh -b -p /content/miniconda

echo "=== Step 3: Initialize Conda ==="
/content/miniconda/bin/conda init bash

echo "=== Step 4: Reload shell ==="
source ~/.bashrc

echo "Conda version:"
conda --version

echo "=== Step 5: Clone GitHub repo ==="
read -s -p "GitHub PAT: " GITHUB_TOKEN
echo ""

git clone https://${GITHUB_TOKEN}@github.com/Pramit726/TrafficCLIP-Optimized.git

echo "=== Setup completed successfully ==="

cd TrafficCLIP-Optimized/