#!/bin/bash

# TST DDoS Defender Launcher Script
# This script activates the virtual environment and starts the TST application

echo "🚀 Starting TST (Time Series Transformer) DDoS Defender"
echo "========================================================"

# Check if running on Windows (Git Bash/MSYS2)
# Use POSIX-compatible syntax instead of bash [[ ]]
if [ "$OSTYPE" = "msys" ] || [ "$OSTYPE" = "cygwin" ] || [ "$(uname -o 2>/dev/null)" = "Msys" ]; then
    echo "Windows environment detected"
    VENV_PATH="/c/Users/burak/Desktop/nenv"
    PYTHON_CMD="python"
else
    # Linux/Pi environment
    echo "Linux environment detected"
    VENV_PATH="/home/dev/nenv"
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
# Use POSIX-compatible syntax
if [ "$OSTYPE" = "msys" ] || [ "$OSTYPE" = "cygwin" ] || [ "$(uname -o 2>/dev/null)" = "Msys" ]; then
    # Windows activation
    . "$VENV_PATH/Scripts/activate"
else
    # Linux activation
    . "$VENV_PATH/bin/activate"
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

# Check if TST model exists
MODEL_FOUND=false
if [ -f "models/tst_model_fp32.pth" ]; then
    echo "✅ Found FP32 TST model (Full Precision)"
    MODEL_FOUND=true
elif [ -f "models/tst_model_int8.pth" ]; then
    echo "✅ Found INT8 TST model (Quantized)"
    echo "ℹ️  Note: The application is configured to use FP32 model for better accuracy"
    MODEL_FOUND=true
else
    echo "⚠️  Warning: TST model not found. The application will run in test mode"
fi

# Check Python dependencies
echo "🔍 Checking dependencies..."
$PYTHON_CMD -c "import torch, numpy, scapy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Missing dependencies. Installing..."
    
    # Install PyTorch for the appropriate platform
    if [ "$OSTYPE" = "msys" ] || [ "$OSTYPE" = "cygwin" ] || [ "$(uname -o 2>/dev/null)" = "Msys" ]; then
        # Windows - install CPU version of PyTorch
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    else
        # Linux/Pi - install appropriate version
        if [ "$(uname -m)" = "aarch64" ]; then
            # ARM64/Pi - install from wheel
            echo "ARM64 detected - installing PyTorch for ARM64"
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        else
            # x86_64 Linux
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    fi
    
    # Install other dependencies
    pip install numpy scapy-python3 pandas
fi

# Check PyTorch version
TORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "✅ PyTorch version: $TORCH_VERSION"
else
    echo "❌ Error: PyTorch installation failed"
    exit 1
fi

echo "🎯 Starting TST Application..."
echo "   - Model: Time Series Transformer (Configurable Performance)"

# Show current performance configuration
echo "📊 Checking performance configuration..."
$PYTHON_CMD scripts/config_tst.py --current > /dev/null 2>&1
if [ $? -eq 0 ]; then
    CURRENT_PROFILE=$($PYTHON_CMD scripts/config_tst.py --current 2>/dev/null)
    echo "   - Current profile: $CURRENT_PROFILE"
    
    # Show profile info
    case $CURRENT_PROFILE in
        "light")
            echo "   - CPU Usage: ~5% (Production ready)"
            ;;
        "medium") 
            echo "   - CPU Usage: ~30% (Balanced)"
            ;;
        "heavy")
            echo "   - CPU Usage: ~60% (Research grade)"
            ;;
        "ultra")
            echo "   - CPU Usage: ~90% (Maximum research)"
            ;;
        *)
            echo "   - CPU Usage: Variable (Custom profile)"
            ;;
    esac
else
    echo "   - Using default configuration"
fi

echo "   - Sequence length: 400 time windows"
echo "   - Config server: http://localhost:8001"
echo ""
echo "⚙️  To change performance profile:"
echo "   python scripts/config_tst.py --set <light|medium|heavy|ultra>"
echo ""
echo "� Available profiles:"
echo "   - light  (5% CPU)  - Production ready"
echo "   - medium (30% CPU) - Balanced performance"  
echo "   - heavy  (60% CPU) - Research grade"
echo "   - ultra  (90% CPU) - Maximum research quality"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Start the application
if [ "$OSTYPE" = "msys" ] || [ "$OSTYPE" = "cygwin" ] || [ "$(uname -o 2>/dev/null)" = "Msys" ]; then
    # Windows - run directly
    $PYTHON_CMD tst_app/main.py
else
    # Linux - run with sudo to capture packets
    echo "🔐 Running with elevated privileges for packet capture..."
    sudo -E $PYTHON_CMD tst_app/main.py
fi