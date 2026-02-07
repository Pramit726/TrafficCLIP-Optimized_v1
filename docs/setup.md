# Miniconda Setup & GitHub PAT Configuration (Terminal)

## Step 1: Download Miniconda

Navigate to the working directory and download the Miniconda installer.

```bash
cd /content
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Validate the installer download
```bash
ls -lh Miniconda3-latest-Linux-x86_64.sh
```
## Step 2: Install Miniconda
Install Miniconda in batch mode at the specified location
```bash
bash Miniconda3-latest-Linux-x86_64.sh -b -p /content/miniconda
```

## Step 3: Initialize Conda for Bash
Initialize Conda so it works with the Bash shell.
```bash
/content/miniconda/bin/conda init bash
```

## Step 4: Reload Shell Environment
Reload the shell configuration and verify the Conda installation.
```bash
source ~/.bashrc
conda --version
```

## Step 5: Enter GitHub Personal Access Token (PAT)
Securely enter your GitHub PAT (input will be hidden).
```bash
read -s -p "GitHub PAT: " GITHUB_TOKEN
```

## Shell script
to run the training
```bash
wsl
```

```bash
source ~/miniconda3/etc/profile.d/conda.sh
```

```bash
bash src/run_train.sh
```

