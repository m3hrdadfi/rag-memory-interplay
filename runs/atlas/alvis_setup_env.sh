#!/bin/bash

# Load modules
module load git-lfs/3.5.1
module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.3.1

# Check if Python is installed
if ! command -v python &> /dev/null
then
    echo "Python is not installed. Please install Python 3.9 or higher and try again."
    exit 1
fi

# Check if the Python version is 3.9 or greater
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
REQUIRED_VERSION="3.9"
if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "Python version 3.9 or greater is required. Your current version is $PYTHON_VERSION."
    exit 1
fi

# Check if the virtual environment 'venv' exists
if [ -d "venv" ]; then
    echo "Virtual environment 'venv' already exists."
else
    echo "Creating virtual environment..."
    virtualenv --system-site-packages venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if requirements.txt exists in the current directory
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
    pip install -r requirements.txt
    # pip install --no-cache-dir -r requirements.txt

else
    echo "requirements.txt not found in the current directory."
    deactivate
    exit 1
fi

# Deactivate the virtual environment
echo "Deactivating virtual environment..."
deactivate

echo "Setup complete."