#!/bin/bash

# Face Recognition System - Setup Script for Raspberry Pi 5
# This script automates the installation process

echo "=========================================="
echo "Face Recognition System Setup"
echo "For Raspberry Pi 5"
echo "=========================================="
echo ""

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Warning: This script is designed for Linux (Raspberry Pi OS)"
fi

# Update system
echo "[1/6] Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install system dependencies
echo "[2/6] Installing system dependencies..."
sudo apt install -y python3-pip python3-opencv
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
sudo apt install -y libharfbuzz0b libwebp7 libtiff5 libjasper-dev
sudo apt install -y libqtgui4 libqt4-test libqtcore4 || true

# Install Python packages
echo "[3/6] Installing Python packages..."
pip3 install --upgrade pip
pip3 install numpy>=1.21.0
pip3 install opencv-python>=4.5.0

# Install TensorFlow Lite Runtime
echo "[4/6] Installing TensorFlow Lite..."
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

# Create project structure
echo "[5/6] Creating project structure..."
mkdir -p mrdavid
mkdir -p captured_frames

# Check for model files
echo "[6/6] Checking for model files..."

if [ ! -f "blazeface.tflite" ]; then
    echo "⚠️  Warning: blazeface.tflite not found!"
    echo "   You need to download the BlazeFace model"
    echo "   Attempting to download from MediaPipe..."
    wget https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite -O blazeface.tflite 2>/dev/null || echo "   Download failed. Please download manually."
fi

if [ ! -f "mobilefacenet.tflite" ]; then
    echo "⚠️  Warning: mobilefacenet.tflite not found!"
    echo "   You need to download the MobileFaceNet model manually"
fi

# Check for training images
if [ -z "$(ls -A mrdavid 2>/dev/null)" ]; then
    echo "⚠️  Warning: No images found in 'mrdavid' folder!"
    echo "   Please add Mr. David's photos to the 'mrdavid' folder"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Place blazeface.tflite in the current directory (if not downloaded)"
echo "2. Place mobilefacenet.tflite in the current directory"
echo "3. Add Mr. David's photos to the 'mrdavid' folder"
echo "4. Run the system: python3 face_recognition_system.py"
echo ""
echo "For more information, see README.md"
echo ""
