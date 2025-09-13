# DDoS Defender: Modular Detection System

A high-performance, modular DDoS detection system designed specifically for Raspberry Pi environments. The system supports two mutually exclusive detection modes using either XGBoost (fast, lightweight) or Time Series Transformer (high accuracy, deep learning) models.

## 🏗️ Architecture Overview

The system is built with a strict modular architecture where two separate applications share common pipeline components but never run simultaneously:

```
/home/pi/ddos_defender/
├── common/
│   └── pipeline_components.py    # All shared, reusable logic
├── xgboost_app/
│   └── main.py                   # XGBoost application entrypoint
├── tst_app/
│   ├── main.py                   # TST application entrypoint
│   └── tstplus.py                # TST model class definition
├── models/
│   ├── xgboost_model.bin         # Pre-trained XGBoost model
│   ├── tst_model_fp32.pth        # Original TST model (FP32)
│   └── tst_model_int8.pth        # Quantized TST model (INT8) - generated
├── db/                           # SQLite databases for data storage
├── logs/                         # Application logs and alerts
└── scripts/
    ├── quantize_tst.py           # TST model optimization utility
    ├── run_xgboost.sh            # XGBoost application launcher
    └── run_tst.sh                # TST application launcher
```

**External Dependencies:**
- Virtual Environment: `/home/pi/nenv` (Linux/Pi) or `/c/Users/burak/Desktop/nenv` (Windows)

## 🎯 Detection Modes

### XGBoost Mode (Fast & Lightweight)
- **Model**: Gradient Boosting Trees
- **Lookback**: 5 time windows
- **Inference Time**: ~0.5-2ms
- **Memory Usage**: Low (~50MB)
- **Use Case**: Real-time detection, resource-constrained environments

### TST Mode (High Accuracy)
- **Model**: Time Series Transformer
- **Lookback**: 400 time windows (sequence modeling)  
- **Inference Time**: ~10-50ms (INT8) / ~20-100ms (FP32)
- **Memory Usage**: High (~200-500MB)
- **Use Case**: Maximum accuracy, sufficient computational resources

## 🚀 Quick Start

### Prerequisites

**Hardware Requirements:**
- Raspberry Pi 4/5 (8GB RAM recommended for TST mode)
- Raspberry Pi OS (64-bit, Bookworm)

**System Dependencies:**
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-venv python3-pip libpcap-dev macchanger conntrack -y
```

### Installation

1. **Create Virtual Environment:**
   ```bash
   python3 -m venv /home/pi/nenv
   source /home/pi/nenv/bin/activate
   ```

2. **Install Python Dependencies:**
   ```bash
   # Basic dependencies
   pip install scapy xgboost numpy pandas scikit-learn
   
   # For TST mode, install PyTorch for ARM64:
   wget https://download.pytorch.org/whl/cpu/torch-2.3.0%2Bcpu-cp311-cp311-linux_aarch64.whl
   pip install torch-2.3.0+cpu-cp311-cp311-linux_aarch64.whl
   rm torch-2.3.0+cpu-cp311-cp311-linux_aarch64.whl
   ```

3. **Optimize TST Model (Recommended):**
   ```bash
   cd /home/pi/ddos_defender
   python scripts/quantize_tst.py
   ```

## 🏃 Running the Applications

**Never run both applications simultaneously!** Choose one:

### Option 1: XGBoost (Recommended for most users)
```bash
cd /home/pi/ddos_defender
./scripts/run_xgboost.sh
```

### Option 2: TST (For maximum accuracy)
```bash
cd /home/pi/ddos_defender
./scripts/run_tst.sh
```

## 📊 Monitoring and Configuration

### Configuration Server
Each application runs a configuration server:
- XGBoost: `http://localhost:8000/status`
- TST: `http://localhost:8001/status`

### Update Detection Parameters
```bash
# Change labeling flag (0=normal, 1=attack)
curl "http://localhost:8000/update-parameters?flag=1"
```

### Database Storage
- Input features: `db/input.db`
- Predictions: `db/output.db`  
- DDoS alerts: `logs/ddos_alert_*.json`

## 🔧 Advanced Configuration

### TST Model Optimization
The TST model can be quantized from FP32 to INT8 for 2-4x performance improvement:

```bash
python scripts/quantize_tst.py
```

This creates `models/tst_model_int8.pth` with:
- ~75% size reduction
- 2-4x inference speedup
- Minimal accuracy loss (<1%)

### Mitigation Settings
Edit `common/pipeline_components.py`:
```python
NO_MITIGATE = False  # Enable actual mitigation (default: True for safety)
FAST_MITIGATE = True  # Enable aggressive mitigation
```

**⚠️ Warning**: Production mitigation affects network configuration. Test thoroughly!

### Performance Tuning
Adjust detection sensitivity in the mitigation component:
```python
attack_ratio > 0.7  # Trigger threshold (70% of recent predictions)
context_length = 40  # Number of predictions to consider
mitigation_cooldown = 30  # Seconds between mitigation actions
```

## 🛠️ Development and Testing

### Windows Development
The system works on Windows for development with synthetic data:
```bash
# From Git Bash or WSL
cd /c/Users/burak/Desktop/ddos_defender
./scripts/run_xgboost.sh
```

### Component Testing
Test individual components:
```python
# Test packet capture
python -c "from common.pipeline_components import capture; capture(queue.Queue())"

# Test TST model
python scripts/quantize_tst.py

# Test database storage
python -c "from common.pipeline_components import create_storage_tables; create_storage_tables(5, 0.6)"
```

## 📈 Performance Benchmarks

| Mode | Avg Inference | Memory Usage | CPU Usage | Best Use Case |
|------|---------------|--------------|-----------|---------------|
| XGBoost | 0.5-2ms | ~50MB | ~10-20% | Real-time, low-power |
| TST (INT8) | 10-50ms | ~200MB | ~30-60% | High accuracy |
| TST (FP32) | 20-100ms | ~500MB | ~50-80% | Research/development |

*Benchmarks on Raspberry Pi 5 (8GB)*

## 🔍 Troubleshooting

### Common Issues

**1. Permission Denied (Packet Capture)**
```bash
sudo -E python3 xgboost_app/main.py
```

**2. Virtual Environment Not Found**
```bash
# Verify path
ls -la /home/pi/nenv
# Or recreate
python3 -m venv /home/pi/nenv
```

**3. PyTorch Not Found (TST)**
```bash
# Check architecture
uname -m  # Should show aarch64 for Pi
# Install correct wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**4. Model Loading Errors**
```bash
# Check model files
ls -la models/
# Verify model integrity
python scripts/quantize_tst.py  # Will test model loading
```

### Logs and Debugging
- Application logs: Console output with timestamps
- DDoS alerts: `logs/ddos_alert_*.json`
- Database logs: SQLite databases in `db/`

Enable debug logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📚 System Architecture Details

### Pipeline Components (Shared)
1. **Packet Capture**: Monitors network traffic for MAVLink packets
2. **Preprocessing**: Converts packet timestamps to feature vectors  
3. **Storage**: SQLite databases for input features and predictions
4. **Mitigation**: Safe DDoS countermeasures with configurable actions
5. **Configuration**: HTTP server for runtime parameter updates

### Detection Flow
```
Network Packets → Feature Extraction → Model Inference → Mitigation Decision → Storage
```

### Thread Architecture
Each application runs 7 concurrent threads:
1. Packet capture
2. Preprocessing  
3. Model-specific detection
4. Mitigation
5. Input storage
6. Output storage
7. Configuration server

## 🤝 Contributing

This is a production system with strict architectural requirements:
- Maintain modular separation between XGBoost and TST applications
- All shared code must go in `common/pipeline_components.py`
- Never modify the established directory structure
- Test thoroughly on Raspberry Pi before production deployment

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

Built for drone security applications with a focus on real-time performance and accuracy on embedded systems.