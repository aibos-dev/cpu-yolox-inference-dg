#!/bin/bash

# Create weights directory if it doesn't exist
WEIGHTS_DIR="weights"
mkdir -p $WEIGHTS_DIR

# Change to weights directory
cd $WEIGHTS_DIR

echo "Downloading YOLOX-Nano..."
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth

echo "Downloading YOLOX-Tiny..."
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth


echo "All downloads completed!"
echo "Models have been downloaded to the '$WEIGHTS_DIR' directory"

# List all downloaded files with their sizes
echo -e "\nDownloaded models:"
ls -lh

# Verify all files were downloaded successfully
echo -e "\nVerifying downloads..."
for file in *.pt *.pth; do
    if [ -f "$file" ]; then
        echo "✓ $file downloaded successfully"
    else
        echo "✗ Failed to download $file"
    fi
done