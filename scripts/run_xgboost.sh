#!/bin/bash

# XGBoost DDoS Defender Launcher Script
# This script activates the virtual environment and starts the XGBoost application

echo "🚀 Starting XGBoost DDoS Defender"
echo "=================================="

# Check if running on Windows (Git Bash/MSYS2)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    echo "Windows environment detected"
    VENV_PATH="/c/Users/burak/Desktop/nenv"
    PYTHON_CMD="python"
else
    # Linux/Pi environment
    echo "Linux environment detected"
    VENV_PATH="/home/pi/nenv"
    PYTHON_CMD="python3"
fi

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Error: Virtual environment not found at $VENV_PATH"
    echo "Please create it first with: python3 -m venv $VENV_PATH"
    exit 1
fi

echo "📦 Activating virtual environment from $VENV_PATH"

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows activation
    source "$VENV_PATH/Scripts/activate"
else
    # Linux activation
    source "$VENV_PATH/bin/activate"
fi

# Verify activation
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Error: Failed to activate virtual environment"
    exit 1
fi

echo "✅ Virtual environment activated: $VIRTUAL_ENV"

# Navigate to project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "📁 Working directory: $(pwd)"

# Check if XGBoost model exists
if [ ! -f "models/xgboost_model.bin" ]; then
    echo "⚠️  Warning: XGBoost model not found at models/xgboost_model.bin"
    echo "The application will run in test mode"
fi

# Check Python dependencies
echo "🔍 Checking dependencies..."
$PYTHON_CMD -c "import xgboost, numpy, scapy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Missing dependencies. Installing..."
    pip install xgboost numpy scapy-python3
fi

echo "🎯 Starting XGBoost Application..."
echo "   - Model: XGBoost (Fast & Lightweight)"
echo "   - Lookback: 5 time windows"
echo "   - Config server: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Start the application
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows - run directly
    $PYTHON_CMD xgboost_app/main.py
else
    # Linux - run with sudo to capture packets
    echo "🔐 Running with elevated privileges for packet capture..."
    sudo -E $PYTHON_CMD xgboost_app/main.py
fi